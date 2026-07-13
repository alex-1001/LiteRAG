"""wires together modules and exposes endpoints of API"""
from fastapi import FastAPI, Depends, Request, HTTPException, status
import time
from typing import Dict
from transformers import AutoTokenizer

from app.models import QueryRequest, QueryResponse, IngestRequest, IngestResponse, Source
from app.config import load_settings, Settings
from app.embed import Embedder
from app.vectorstore import VectorStore
from app.ingest import ingest_folder, resolve_ingest_path
from app.retrieve import retrieve_chunks
from app.rag import answer_query
from contextlib import asynccontextmanager
from openai import OpenAI
from threading import Lock
import logging

def get_config(request: Request) -> Settings:
    return request.app.state.config

def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder

def get_vectorstore(request: Request) -> VectorStore:
    return request.app.state.vectorstore

def get_tokenizer(request: Request) -> AutoTokenizer:
    return request.app.state.tokenizer

def get_client(request: Request) -> OpenAI:
    return request.app.state.client

def get_lock(request: Request) -> Lock:
    return request.app.state.lock

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    settings = load_settings()
    if not all([settings.openrouter_base_url, settings.openrouter_api_key, settings.openrouter_model]):
        raise ValueError(
            "OPENROUTER_BASE_URL, OPENROUTER_API_KEY, and OPENROUTER_MODEL must be set for the app. "
            "Check your .env or environment."
        )
    app.state.config = settings
    
    embedder = Embedder(settings.embed_model_name)
    app.state.embedder = embedder
    
    tokenizer_name = settings.tokenizer_name or settings.embed_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    app.state.tokenizer = tokenizer
    
    vectorstore = VectorStore(dim=embedder.dim, storage_dir=settings.storage_dir, embed_model_name=settings.embed_model_name)
    vectorstore.load_or_create()
    app.state.vectorstore = vectorstore
    
    client = OpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)
    app.state.client = client
    
    lock = Lock()
    app.state.lock = lock
    yield
    
    # teardown
    try:
        vectorstore.save()
    except Exception as e:
        # prevents masking other errors
        logging.exception("Error saving vectorstore on shutdown: %s", e)

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health() -> Dict[str, str]:
    #TODO: run more diagnostics
    return {"status": "ok"}

@app.post("/ingest")
def ingest(
    req: IngestRequest, 
    config: Settings = Depends(get_config), 
    embedder: Embedder = Depends(get_embedder),
    tokenizer: AutoTokenizer = Depends(get_tokenizer), 
    vectorstore: VectorStore = Depends(get_vectorstore),
    lock: Lock = Depends(get_lock),
) -> IngestResponse:
    start = time.perf_counter()
    
    try:
        source_path = resolve_ingest_path(config.data_dir, req.source_path)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    
    try:
        chunks = ingest_folder(source_path, tokenizer, config.chunk_size, config.chunk_overlap)
        if not chunks:
                if req.force_rebuild:
                    vectorstore.clear()
                    vectorstore.create()
                
                return IngestResponse(
                    documents_processed=0,
                    chunks_created=0,
                    document_ids=[],
                    processing_time_seconds=time.perf_counter() - start,
                )
            
        vectors = embedder.embed_chunks(chunks)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except NotADirectoryError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied while reading the data folder",
        ) from e
    except Exception as e:
        logging.exception("Unexpected ingest failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest documents",
        ) from e
    
    try:
        with lock:
            if req.force_rebuild:
                vectorstore.clear()
                vectorstore.create()
            
            vectorstore.add(vectors, chunks)
            vectorstore.save()
    except Exception as e:
        logging.exception("Failed to update vectorstore during ingest")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save ingested documents",
        ) from e
        
    doc_ids = list({c.document_id for c in chunks})
    
    return IngestResponse(
        documents_processed=len(doc_ids),
        chunks_created=len(chunks),
        document_ids=doc_ids,
        processing_time_seconds=time.perf_counter() - start,
    )

@app.post("/query")
def query(
    req: QueryRequest,
    config: Settings = Depends(get_config),
    embedder: Embedder = Depends(get_embedder),
    client: OpenAI = Depends(get_client),
    vectorstore: VectorStore = Depends(get_vectorstore),
    lock: Lock = Depends(get_lock),
) -> QueryResponse:
    top_k = req.top_k or config.top_k
    
    try:
        with lock:
            chunks, distances = retrieve_chunks(
                req.question,
                vectorstore,
                embedder,
                top_k=top_k,
            )
    except RuntimeError as e:
        logging.exception("Vectorstore runtime failure")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search index is not ready"
        ) from e
    except Exception as e:
        logging.exception("Unexpected retrieval failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve relevant documents",
        ) from e

    try:
        answer, cited_ids = answer_query(
            req.question,
            chunks,
            client=client,
            llm_model=config.openrouter_model
        )
    except (ValueError, TypeError) as e:
        logging.exception("LLM returned an invalid response")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM returned an invalid response",
        ) from e
    except Exception as e:
        logging.exception("Unexpected query failure")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate answer",
        ) from e
            
    # chunks actually cited in answer
    sources = []
    for cid in cited_ids:
        chunk = chunks[cid - 1] # comes as 1-indexed
        sources.append(
            Source(
                document_name=chunk.document_name,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                snippet=chunk.text,
                score=1.0 - distances[cid - 1],
            )
        )
        
    return QueryResponse(
        answer=answer,
        used_retrieval=len(chunks) > 0,
        sources=sources,
    )
