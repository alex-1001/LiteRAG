"""handles vector storage and search"""

from app.models import DocumentChunk
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Dict, Tuple
from hnswlib import Index
import json
from pathlib import Path
import warnings

class VectorStore:
    """Wrapper class that couples vector index to metadata database; supports vector storage and search"""
    def __init__(self, dim: int, storage_dir: str, embed_model_name: Optional[str] = None, initial_max_elements: int = 10000):
        self.dim = dim # dimension of vectors (fixed)
        self.storage_dir = Path(storage_dir) # directory to store vector index and metadata
        self.storage_dir.mkdir(parents=True, exist_ok=True) # create directory if it doesn't exist
        self.embed_model_name = embed_model_name # optional validation of embedding model used
        
        self.space = "cosine" # vector space (cosine similarity)
        self.index: Optional[Index] = None # vector index
        self.hsnw_params = {
            "max_elements": initial_max_elements,
            "M": 16,
            "ef_construction": 128,
            "allow_replace_deleted": True,
        }
        
        self.next_id = 0 # next chunk id to assign in index
        self.id_to_chunk: Dict[int, DocumentChunk] = {} # maps chunk ids (stored in index) to Document Chunks
        
    
    def create(self):
        """initializes (or resets) vector index and metadata database in memory. does not remove from disk."""
        # initialize vector index
        self.index = Index(space=self.space, dim = self.dim)
        self.index.init_index(**self.hsnw_params)
        
        self.id_to_chunk.clear() # clear id -> chunk mapping
        self.next_id = 0 # restart counter
    
    def load(self):
        """loads vector index and metadata into memory"""
        index_path = Path(self.storage_dir) / "index.bin"
        metadata_path = Path(self.storage_dir) / "metadata.jsonl"
        index_metadata_path = Path(self.storage_dir) / "index_metadata.json"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Vector index file {index_path} not found")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file {metadata_path} not found")
        if not index_metadata_path.exists():
            raise FileNotFoundError(f"Index metadata file {index_metadata_path} not found")
        
        # load index metadata
        try:
            with open(index_metadata_path, "r") as f:
                index_metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load index metadata from {index_metadata_path}: {e}") from e
        
        # validate index metadata
        if index_metadata["dim"] != self.dim:
            raise ValueError(f"Dimension in index metadata {index_metadata['dim']} does not match expected dimension {self.dim}")
        if index_metadata["space"] != self.space:
            raise ValueError(f"Space in index metadata {index_metadata['space']} does not match expected space {self.space}")
        if self.embed_model_name is not None:
            if index_metadata["embed_model_name"] is None:
                warnings.warn(f"Loaded embed_model_name is None while current embed_model_name is {self.embed_model_name}. Using None for embed_model_name.")
                self.embed_model_name = None
            elif index_metadata["embed_model_name"] != self.embed_model_name:
                raise ValueError(f"Loaded embed_model_name {index_metadata['embed_model_name']} does not match expected embed_model_name {self.embed_model_name}")
        else:
            self.embed_model_name = index_metadata["embed_model_name"]
            
        if index_metadata["hsnw_params"] != self.hsnw_params:
            warnings.warn(f"Loaded hsnw_params {index_metadata['hsnw_params']} does not match expected hsnw_params {self.hsnw_params}. Using new hsnw_params.")
            self.hsnw_params = index_metadata["hsnw_params"]
        
        # load index
        try:
            loaded_index = Index(space=index_metadata["space"], dim=index_metadata["dim"])
            loaded_index.load_index(index_path.as_posix())
        except Exception as e:
            # Catch all exceptions when loading index
            raise RuntimeError(f"Failed to load vector index from {index_path}: {e}") from e

        # extra check: ensure index is properly initialized and not mutated
        if loaded_index.dim != index_metadata["dim"]:
            raise ValueError(f"Loaded index dimension {loaded_index.dim} does not match expected dimension {index_metadata['dim']} after loading")
        if loaded_index.space != index_metadata["space"]:
            raise ValueError(f"Loaded index space {loaded_index.space} does not match expected space {index_metadata['space']} after loading")

        # load metadata
        try:
            loaded_id_to_chunk = {}
            with open(metadata_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    id = data["id"]
                    chunk = DocumentChunk.model_validate(data["chunk"])
                    loaded_id_to_chunk[id] = chunk
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {metadata_path}: {e}") from e
        
        # validate index ids
        index_ids = set(loaded_index.get_ids_list())
        metadata_ids = set(int(id) for id in loaded_id_to_chunk.keys())
        if index_ids != metadata_ids:
            missing_in_metadata = index_ids - metadata_ids
            missing_in_index = metadata_ids - index_ids
            
            raise RuntimeError(
                f"ID mismatch between index and metadata:\n"
                f"  IDs in index but not in metadata: {sorted(missing_in_metadata)}\n"
                f"  IDs in metadata but not in index: {sorted(missing_in_index)}"
            )
        # from now on, we can assume that the loaded index and metadata are consistent

        # fetch next_id
        max_index_id = max(index_ids) if index_ids else -1
        self.next_id = max_index_id + 1
        
        # update vector index and metadata in memory
        self.index = loaded_index
        self.id_to_chunk = loaded_id_to_chunk
        
    def load_or_create(self):
        """loads vector index and metadata from disk (if it exists in storage_dir) or initializes new ones"""
        index_path = self.storage_dir / "index.bin"
        metadata_path = self.storage_dir / "metadata.jsonl"
        index_metadata_path = self.storage_dir / "index_metadata.json"

        # If any are missing, initialize a new index and metadata
        if (index_path.exists() and metadata_path.exists() and index_metadata_path.exists()):
            self.load()
        else: 
            self.create()
        
    
    def add(self, vectors: NDArray[np.float32], chunks: List[DocumentChunk]) -> List[int]:
        """adds vectors (N, dim) or single vector (dim,) to index and updates metadata database. expands index if necessary.
        chunks is a list of DocumentChunks that correspond to the vectors. must be same length as vectors.
        returns a list of ids assigned to the vectors."""
        if not self.is_initialized():
            raise RuntimeError("Vector index and metadata database are not properly initialized. Please call create() first.")
        
        # check if vectors are correct shape
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim != 2:
            raise ValueError(f"Vectors must be (N, dim) or (dim,) but got {vectors.shape}")
        
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match expected dimension {self.dim}")
        
        num_vectors = vectors.shape[0]
        # check if chunks are correct shape
        if len(chunks) != num_vectors: 
            raise ValueError(f"Chunks must be same length as vectors but got {len(chunks)} chunks and {num_vectors} vectors")
        for chunk in chunks:
            if not isinstance(chunk, DocumentChunk):
                raise ValueError(f"Chunk {chunk} is not a DocumentChunk")
        
        # resize if index has space
        if self.index.get_current_count() + num_vectors > self.index.get_max_elements():
            self.resize((self.index.get_current_count() + num_vectors) * 2) # resize to double what we would need
            
        # add vectors to index
        ids = list(range(self.next_id, self.next_id + num_vectors))
        self.index.add_items(data = vectors, ids = ids, replace_deleted = True) # note that vectors marked as deleted are replaced
        self.next_id += num_vectors
        
        # update metadata
        for id, chunk in zip(ids, chunks):
            self.id_to_chunk[id] = chunk
        
        return ids
            
    def search(self, query: NDArray[np.float32], top_k: int) -> Tuple[List[int], List[float], List[DocumentChunk]]:
        """searches vector index for top k most similar vectors to a single query vector (dim,) or (1, dim).
        returns a tuple of (ids, distances, chunks)"""
        if not self.is_initialized():
            raise RuntimeError("Vector index and metadata database are not properly initialized. Please call create() first.")
        if top_k <= 0:
            raise ValueError(f"Top k must be greater than 0 but got {top_k}")
        
        # check if query is correct shape 
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        elif query.ndim != 2: 
            raise ValueError(f"Query must be (dim,) or (N, dim) but got {query.shape}")
        if query.shape[1] != self.dim:
            raise ValueError(f"Query dimension {query.shape[1]} does not match expected dimension {self.dim}")
        if query.shape[0] != 1:
            raise ValueError(f"Query must have single first dimension but got {query.shape[0]}")
        
        # query index
        max_vectors = self.index.get_current_count()
        if max_vectors == 0:
            return [], [], []
        
        if max_vectors <= top_k:
            # return all vectors
            top_k = max_vectors
        
        top_k_ids, top_k_distances = self.index.knn_query(data = query, k = top_k) # (1, k)
        # TODO: allow for controlling ef
        
        # get chunks
        chunks = [self.id_to_chunk[int(id)] for id in top_k_ids[0]]
        return top_k_ids[0].tolist(), top_k_distances[0].tolist(), chunks
    
    def resize(self, new_size: int):
        """resizes vector index to new size. new size must be greater than or equal to number of vectors in index."""
        if not self.is_initialized():
            raise RuntimeError("Vector index and metadata database are not properly initialized. Please call create() first.")

        num_vectors = self.index.get_current_count()
        
        if new_size < num_vectors:
            raise ValueError(f"New size {new_size} is less than number of vectors {num_vectors}")
        
        self.index.resize_index(new_size)
        self.hsnw_params["max_elements"] = new_size
        
    def save(self):
        """saves vector index and metadata to disk"""
        if not self.is_initialized():
            raise RuntimeError("Vector index and metadata database are not properly initialized. Please call create() first.")
        # TODO: atomic saves
        index_path = Path(self.storage_dir) / "index.bin"
        metadata_path = Path(self.storage_dir) / "metadata.jsonl"
        index_metadata_path = Path(self.storage_dir) / "index_metadata.json"
        
        # save index
        self.index.save_index(index_path.as_posix()) 
        
        # save metadata
        with open(metadata_path, "w") as f:
            for id, chunk in self.id_to_chunk.items():
                line = {
                    "id": id,
                    "chunk": chunk.model_dump()
                }
                f.write(json.dumps(line) + "\n")
        
        # save index metadata
        index_metadata = {
            "embed_model_name": self.embed_model_name,
            "dim": self.dim,
            "space": self.space,
            "hsnw_params": self.hsnw_params, # not strictly necessary
            "next_id": self.next_id, # not strictly necessary
        }
        with open(index_metadata_path, "w") as f:
            json.dump(index_metadata, f)
        
    def __len__(self) -> int:
        """returns number of vectors in index"""
        if not self.is_initialized():
            raise RuntimeError("Vector index and metadata database are not properly initialized. Please call create() first.")
        
        count = self.index.get_current_count()
        if count != len(self.id_to_chunk): # maybe want to also check next_id? if we want to enforce 0 to n id's
            raise RuntimeError(
                f"Data inconsistency in number of vectors: index={count}, metadata={len(self.id_to_chunk)}, next_id={self.next_id}"
            )
        return self.index.get_current_count()
    
    def is_initialized(self) -> bool:
        """returns True if vector index and metadata database are properly initialized"""
        return self.index is not None and self.id_to_chunk is not None
    
    def clear(self):
        """clears vector index and metadata database from memory and disk. requires calling create() again to re-initialize."""
        # clear index and metadata in memory
        self.index = None
        self.id_to_chunk.clear()
        self.next_id = 0
        
        # clear from disk
        index_path = Path(self.storage_dir) / "index.bin"
        metadata_path = Path(self.storage_dir) / "metadata.jsonl"
        index_metadata_path = Path(self.storage_dir) / "index_metadata.json"
        
        index_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        index_metadata_path.unlink(missing_ok=True)