"""
Process a folder of documents into chunks (DocumentChunks)
"""
# TODO: use dynamic chunk and overlap sizes based on document size
from app.models import DocumentChunk
from typing import List, Tuple, Dict, Any
from pathlib import Path
from uuid import uuid5, NAMESPACE_DNS
import os 
from dotenv import load_dotenv
from transformers import PreTrainedTokenizerBase

load_dotenv() # maybe move this elsewhere later
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

def _validate_data_dir(data_dir: str) -> Path:
    """Validate and return Path object for data directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory {data_dir} is not a directory.")
    return data_dir

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespaces in a text string. 
    - convert special white space characters (e.g. "\\r\\n" or "\\r") to "\\n"
    - within text, remove white space at end of each line (trailing whitespace)
    - at text-level, remove leading AND trailing whitespaces
    
    Returns the normalized text string.
    """
    text = text.replace("\a", "").replace("\b", "") # remove control characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = text.strip()
    return text

def load_text_documents(data_dir: str) -> List[Tuple[Path, str]]:
    """
    Load all text documents (.txt/.md) from the data directory.
    Returns a list of tuples containing (document Path, content).
    """
    data_dir = _validate_data_dir(data_dir)

    acceptable_extensions = {".txt", ".md"}
    paths = [file for file in data_dir.rglob("*") if file.is_file() and file.suffix.lower() in acceptable_extensions]
    paths = sorted(paths, key=lambda x: x.as_posix().lower()) # sort by posix-formatted path (case-insensitive)

    documents = []
    for path in paths:
        raw_text = path.read_text(encoding="utf-8", errors="replace") # note that replace to handle any encoding errors
        text = normalize_whitespace(raw_text)
        if not text: # skip empty documents 
            continue
        documents.append((path, text)) # document Path, text
        
    return documents

def chunk_text(text: str, chunk_size: int, chunk_overlap: int, tokenizer: PreTrainedTokenizerBase) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Chunk the text into chunks of the given size, with the given overlap.
    Returns a list of tuples containing (chunk text, metadata).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size.")
    
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap must be greater or equal to 0.")
    
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    
    if not text:
        return []

    tokens = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"] # list of token ids (integers)
    
    chunks = []
    if chunk_size >= len(tokens):
        chunks.append((text, {
            "start_token_index": 0,
            "end_token_index": len(tokens) - 1,
            "chunk_size": len(tokens),
        })) # returns single chunk if text is less than chunk size
    else: 
        start = 0
        step_size = chunk_size - chunk_overlap
        window_span = chunk_size - 1
        
        while start + window_span < len(tokens):
            end = start + window_span
            chunk_tokens = tokens[start:end+1]
            text_chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append((text_chunk, {
                "start_token_index": start,
                "end_token_index": end,
                "chunk_size": len(chunk_tokens),
            }))
            
            start +=  step_size

        # handle tail
        num_remaining_tokens = len(tokens) - start 
        if num_remaining_tokens > chunk_overlap:
            chunk_tokens = tokens[start:]
            text_chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append((text_chunk, {
                "start_token_index": start,
                "end_token_index": len(tokens) - 1,
                "chunk_size": num_remaining_tokens,
            }))
    
    return chunks

def ingest_folder(data_dir: str, tokenizer: PreTrainedTokenizerBase) -> List[DocumentChunk]:
    """
    Processes the folder of documents into chunks.
    Returns a list of DocumentChunks.
    """
    data_dir = _validate_data_dir(data_dir)
    
    document_chunks = []
    
    documents = load_text_documents(data_dir)
    for doc_path, text in documents:
        document_name = doc_path.name
        document_id = uuid5(NAMESPACE_DNS, doc_path.as_posix().lower())
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP, tokenizer)
        for i, (text_chunk, metadata) in enumerate(chunks):
            chunk_id = f"{str(document_id)}-{i}"
            document_chunks.append(DocumentChunk(
                document_id=document_id,
                document_name = document_name,
                chunk_id = chunk_id,
                text = text_chunk,
                metadata = metadata,
            ))
            
    return document_chunks