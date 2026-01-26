from app.models import DocumentChunk
from uuid import uuid4

def _make_chunk(text: str, document_name: str = "doc", chunk_id: str = "0") -> DocumentChunk:
    """Build a DocumentChunk for tests with required fields and defaults."""
    return DocumentChunk(
        document_id=uuid4(),
        document_name=document_name,
        chunk_id=chunk_id,
        text=text,
        metadata={},
    )