"""handles retrieval of chunks from a query (interface between embed and vectorstore tools)"""
from app.vectorstore import VectorStore
from app.embed import Embedder
from app.models import DocumentChunk
from typing import List, Tuple

def retrieve_chunks(question: str, vectorstore: VectorStore, embedder: Embedder, top_k: int = 5) -> Tuple[List[DocumentChunk], List[float]]:
    """retrieves top k most similar chunks from vectorstore to a question"""
    query_vector = embedder.embed_query(question)
    ids, distances, chunks = vectorstore.search(query_vector, top_k)
    return chunks, distances

# TODO: de-dupe, similarity threshold, reranking