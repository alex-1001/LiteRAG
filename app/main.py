"""wires together modules and exposes endpoints of API"""
from fastapi import FastAPI, Depends, Request, HTTPException, status
import time
from typing import Dict
from transformers import AutoTokenizer

from app.models import QueryRequest, QueryResponse, IngestRequest, IngestResponse, Source
from app.config import Settings, load_settings
from app.embed import Embedder
from app.vectorstore import VectorStore, ChunkConflictError
from app.ingest import ingest_folder, resolve_ingest_path
from app.retrieve import retrieve_chunks
from app.rag import InvalidModelResponse, answer_query
from app.llm import ChatModel, build_chat_model
from contextlib import asynccontextmanager
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

def get_chat_model(request: Request) -> ChatModel:
    return request.app.state.chat_model

def get_lock(request: Request) -> Lock:
    return request.app.state.lock

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    settings = load_settings()
    app.state.config = settings
    
    embedder = Embedder(settings.embed_model_name)
    app.state.embedder = embedder
    
    tokenizer_name = settings.tokenizer_name or settings.embed_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    app.state.tokenizer = tokenizer
    
    vectorstore = VectorStore(dim=embedder.dim, storage_dir=settings.storage_dir, embed_model_name=settings.embed_model_name)
    vectorstore.load_or_create()
    app.state.vectorstore = vectorstore

    chat_model = build_chat_model(settings)
    app.state.chat_model = chat_model
    
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
        chunks = ingest_folder(
            source_path,
            tokenizer,
            config.chunk_size,
            config.chunk_overlap,
            document_root=config.data_dir,
        )
        if not chunks:
            if req.force_rebuild:
                with lock:
                    vectorstore.clear()
                    vectorstore.create()
                    vectorstore.save()

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
            
            result = vectorstore.add_idempotent(vectors, chunks)
            vectorstore.save()
    except ChunkConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "One or more chunks changed since they were ingested. Use force_rebuild=true to rebuild the index.",
                "changed_chunk_ids": e.chunk_ids,
            },
        ) from e
    except Exception as e:
        logging.exception("Failed to update vectorstore during ingest")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save ingested documents",
        ) from e
        
    doc_ids = list({c.document_id for c in chunks})
    
    return IngestResponse(
        documents_processed=len(doc_ids),
        chunks_created=result.chunks_added,
        document_ids=doc_ids,
        processing_time_seconds=time.perf_counter() - start,
    )

@app.post("/query")
def query(
    req: QueryRequest,
    config: Settings = Depends(get_config),
    embedder: Embedder = Depends(get_embedder),
    chat_model: ChatModel = Depends(get_chat_model),
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
            chat_model=chat_model,
        )
    except InvalidModelResponse as e:
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
                cosine_similarity=1.0 - distances[cid - 1],
            )
        )
        
    return QueryResponse(
        answer=answer,
        used_retrieval=len(chunks) > 0,
        sources=sources,
    )
