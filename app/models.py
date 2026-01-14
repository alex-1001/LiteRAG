"""Data models for the application"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime, timezone

class DocumentChunk(BaseModel):
    """
    Atomic unit of RAG retrieval
    """
    document_id: UUID = Field(..., description="Unique, stable identifier of document that a chunk originates from.")
    document_name: str = Field(..., description="Human-readable name of document.")
    chunk_id: str = Field(..., description="Unique identifier for a chunk within its document.")
    text: str = Field(..., description="Raw text from document, used for retrieval and augmentation.", min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional information about the chunk.")
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when chunk was created.")
    
class Source(BaseModel):
    """
    A citation-object for information used in RAG.
    """
    document_name: str = Field(..., description="Human-readable name of document.")
    document_id: UUID = Field(..., description="Unique, stable identifier of document that a chunk originates from.")
    chunk_id: str = Field(..., description="Unique identifier for a chunk within its document.")
    snippet: str = Field(..., description="An excerpt of the chunk.", min_length=1)
    score: float = Field(..., description="Similarity score between the query and the chunk.", ge=0.0, le=1.0)
    
class IngestRequest(BaseModel):
    """
    Request to ingest a folder of documents (.txt/.md) into vector database.
    Will default to DATA_DIR environment variable if not provided.
    """
    folder_path: Optional[str] = Field(None, description="Path to folder containing documents to ingest.")
    force_rebuild: bool = Field(False, description="Whether to rebuild the vector database from scratch.")
    
class IngestResponse(BaseModel):
    """
    Response after ingesting folder of documents.
    """
    documents_processed: int = Field(..., description="Number of documents processed.", ge=0)
    chunks_created: int = Field(..., description="Number of chunks created.", ge=0)
    document_ids: List[UUID] = Field(..., description="List of unique identifiers for each document processed.")
    processing_time_seconds: float = Field(..., description="Time taken to process the documents.", ge=0.0)
    
class QueryRequest(BaseModel):
    """
    Natural language user query to be answered with RAG (if applicable)
    """
    question: str = Field(..., description="Natural language user query to be answered with RAG (if applicable).", min_length=1)
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve from the vector database. Will default to TOP_K environment variable.", ge=1)
    
class QueryResponse(BaseModel):
    """
    Response to user query
    """
    answer: str = Field(..., description="Answer to the user query.", min_length=1)
    used_retrieval: bool = Field(..., description="Whether the answer was retrieved from the vector database.")
    sources: List[Source] = Field(..., description="List of sources used to answer the query.")
    