import pytest
from pathlib import Path
from app.vectorstore import VectorStore
from app.models import DocumentChunk
import numpy as np
from numpy.typing import NDArray
from uuid import uuid4
from typing import List, Tuple

# Tolerance for "effectively zero" cosine distance: hnswlib uses float32 and
# floating-point rounding in L2 normalization and 1-cos_sim can yield ~1e-7.
DISTANCE_ATOL = 1e-5

@pytest.fixture
def vectorstore_path(tmp_path: Path) -> Path:
    path = tmp_path / "vectorstore"
    path.mkdir()
    yield path
    
def create_ordered_test_vectors(num_vectors: int, dim: int) -> Tuple[NDArray[np.float32], List[DocumentChunk]]:
    # create a L2-normalized, lower triangular matrix of ones
    # gives us a pre-defined ordering when doing cosine similarity with the first elementary vector
    if num_vectors > dim:
        raise ValueError(f"Number of vectors {num_vectors} must be less than or equal to dimension {dim} to ensure ordering.")
    
    rows = np.reshape(np.arange(1, num_vectors + 1, dtype=np.float32), (num_vectors, 1))
    cols = np.reshape(np.arange(dim, dtype=np.float32), (1, dim))
    vectors = np.where(rows > cols, 1.0 / np.sqrt(rows), 0.0).astype(np.float32)
    
    # create chunks 
    chunks = []
    for i in range(num_vectors):
        id = uuid4()
        chunks.append(DocumentChunk(
            document_id=id,
            document_name=f"doc_{i}",
            chunk_id=f"chunk_{i}",
            text=f"Test Chunk {i}",
            metadata={}
        ))
    
    return vectors, chunks

def create_identical_test_vectors(num_vectors: int, dim: int) -> Tuple[NDArray[np.float32], List[DocumentChunk]]:
    vectors = np.ones((num_vectors, dim), dtype=np.float32)
    chunks = [DocumentChunk(
        document_id=uuid4(),
        document_name=f"doc_{i}",
        chunk_id=f"chunk_{i}",
        text=f"Test Chunk {i}",
        metadata={}
    ) for i in range(num_vectors)]
    return vectors, chunks

class TestVectorStoreInitialization:
    """Test suite for VectorStore proper initializiation and creation"""
    def test_create_initializes_vector_index_and_metadata(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path)
        vs.create()
        assert vs.index is not None
        assert vs.id_to_chunk is not None
        assert vs.next_id == 0
    
    def test_is_initialized_returns_true_after_create(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path)
        vs.create()
        assert vs.is_initialized()
    
    def test_is_initialized_returns_false_before_create(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path)
        assert not vs.is_initialized()
    
    def test_is_initialized_returns_false_after_clear(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path)
        vs.create()
        vs.clear()
        assert not vs.is_initialized()

class TestVectorStoreAdd:
    """Test suite for VectorStore add functionality"""
    def test_add_adds_vectors_to_index(self, vectorstore_path: Path):
        # Note: Explicitly specify initial_max_elements as a best practice for clarity and maintainability.
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=10, dim=10)
        added_ids = vs.add(vectors=vectors, chunks=chunks)
        
        assert len(added_ids) == 10
        assert len(vs) == 10
        assert all(isinstance(i, int) for i in added_ids)
        assert added_ids == list(range(10))

        # Add a second batch and check that everything works as expected
        vectors2, chunks2 = create_ordered_test_vectors(num_vectors=5, dim=10)
        added_ids2 = vs.add(vectors=vectors2, chunks=chunks2)
        assert len(added_ids2) == 5
        # Length should now be 15
        assert len(vs) == 15
        # IDs for the second batch should be sequential after the first
        assert added_ids2 == list(range(10, 15))
        # Confirm all IDs are ints and in expected range
        all_ids = added_ids + added_ids2
        assert all(isinstance(i, int) for i in all_ids)
        assert set(all_ids) == set(range(15))

    def test_add_adds_single_vector_to_index(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vector, chunk = create_identical_test_vectors(num_vectors=1, dim=10)
        added_ids = vs.add(vectors=vector, chunks=chunk)
        assert len(added_ids) == 1
        assert len(vs) == 1
        assert isinstance(added_ids[0], int)
        assert added_ids[0] == 0

        # Add another single vector and check everything again
        vector2, chunk2 = create_identical_test_vectors(num_vectors=1, dim=10)
        added_ids2 = vs.add(vectors=vector2, chunks=chunk2)
        assert len(added_ids2) == 1
        assert len(vs) == 2
        assert isinstance(added_ids2[0], int)
        assert added_ids2[0] == 1
        # All IDs should be unique and range 0, 1
        assert set(added_ids + added_ids2) == {0, 1}

    def test_add_adds_vectors_to_metadata(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=10, dim=10)
        added_ids = vs.add(vectors=vectors, chunks=chunks)
        assert len(vs.id_to_chunk) == 10
        assert all(id in vs.id_to_chunk for id in added_ids)
        assert all(vs.id_to_chunk[id] == chunks[i] for i, id in enumerate(added_ids))

        # Add again, test metadata correctness for total
        vectors2, chunks2 = create_ordered_test_vectors(num_vectors=5, dim=10)
        added_ids2 = vs.add(vectors=vectors2, chunks=chunks2)
        assert len(vs.id_to_chunk) == 15
        assert all(id in vs.id_to_chunk for id in added_ids2)
        # Check all newly added items in metadata map to correct chunk
        for i, id in enumerate(added_ids2):
            assert vs.id_to_chunk[id] == chunks2[i]

    def test_add_adds_single_vector_to_metadata(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vector, chunk = create_identical_test_vectors(num_vectors=1, dim=10)
        added_ids = vs.add(vectors=vector, chunks=chunk)
        assert len(vs.id_to_chunk) == 1
        assert added_ids[0] in vs.id_to_chunk
        assert vs.id_to_chunk[added_ids[0]] == chunk[0]

        # Add a second single vector and check again
        vector2, chunk2 = create_identical_test_vectors(num_vectors=1, dim=10)
        added_ids2 = vs.add(vectors=vector2, chunks=chunk2)
        assert len(vs.id_to_chunk) == 2
        assert added_ids2[0] in vs.id_to_chunk
        assert vs.id_to_chunk[added_ids2[0]] == chunk2[0]
    
    @pytest.mark.parametrize("num_initial_vectors, num_added_vectors", [(10, 100), (100, 10), (100, 1), (99, 2)])
    def test_add_resizes_index_if_necessary(self, vectorstore_path: Path, num_initial_vectors: int, num_added_vectors: int):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_identical_test_vectors(num_vectors=num_initial_vectors, dim=10)
        added_ids = vs.add(vectors=vectors, chunks=chunks)
        
        vectors2, chunks2 = create_identical_test_vectors(num_vectors=num_added_vectors, dim=10)
        added_ids2 = vs.add(vectors=vectors2, chunks=chunks2)
        assert len(vs) == num_initial_vectors + num_added_vectors
        assert all(isinstance(i, int) for i in added_ids2)
        assert added_ids2 == list(range(num_initial_vectors, num_initial_vectors + num_added_vectors))
        assert vs.hsnw_params["max_elements"] >= num_initial_vectors + num_added_vectors # ensure index was resized 
    
    def test_add_raises_error_if_not_initialized(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vec, chunk = create_identical_test_vectors(num_vectors=1, dim=10)
        with pytest.raises(RuntimeError):
            vs.add(vectors=vec, chunks=chunk)
    
    def test_add_raises_error_if_vectors_are_not_correct_dimension(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vec, chunk = create_identical_test_vectors(num_vectors=1, dim=5)
        with pytest.raises(ValueError):
            vs.add(vectors=vec, chunks=chunk)
    
    def test_add_raises_error_if_vectors_are_not_correct_shape(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_identical_test_vectors(num_vectors=2, dim=10)
        vectors = vectors[:, np.newaxis]
        with pytest.raises(ValueError):
            vs.add(vectors=vectors, chunks=chunks)

class TestOrderedTestVectorsHelper:
    """Verifies create_ordered_test_vectors ordering assumption for [1,0,...,0] query."""

    def test_ordered_vectors_ascending_distance_under_first_elementary_query(self):
        # Cosine distance = 1 - cosine_sim. For unit q=[1,0,...,0] and unit rows,
        # cos_sim = dot(row, q) = row[0] = 1/sqrt(i+1). Ascending distance => order 0,1,2,...
        vectors, _ = create_ordered_test_vectors(5, 5)
        q = np.zeros(5, dtype=np.float32)
        q[0] = 1.0
        sims = np.dot(vectors, q)
        dists = 1.0 - sims
        order = np.argsort(dists)
        assert list(order) == [0, 1, 2, 3, 4]


class TestVectorStoreSearch:
    """Test suite for VectorStore search functionality"""

    def test_search_ordered_vectors_returns_ids_ascending_by_distance(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=5, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        q = np.zeros(5, dtype=np.float32)
        q[0] = 1.0
        ids, distances, _ = vs.search(q, top_k=5)
        assert ids == [0, 1, 2, 3, 4]
        assert distances == sorted(distances)

    def test_search_identical_query_gives_zero_distance(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_identical_test_vectors(num_vectors=1, dim=10)
        vs.add(vectors=vectors, chunks=chunks)
        ids, distances, _ = vs.search(vectors[0], top_k=1)
        assert len(ids) == 1
        assert distances[0] == pytest.approx(0.0, abs=DISTANCE_ATOL)

    def test_search_identical_vectors_all_zero_distance(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_identical_test_vectors(num_vectors=5, dim=10)
        vs.add(vectors=vectors, chunks=chunks)
        ids, distances, _ = vs.search(vectors[0], top_k=5)
        assert all(d == pytest.approx(0.0, abs=DISTANCE_ATOL) for d in distances)

    def test_search_returns_matching_chunks(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        q = np.zeros(5, dtype=np.float32)
        q[0] = 1.0
        ids, _, result_chunks = vs.search(q, top_k=3)
        for i, id in enumerate(ids):
            assert result_chunks[i] == vs.id_to_chunk[id]

    def test_search_empty_index_returns_empty(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        q = np.zeros(5, dtype=np.float32)
        q[0] = 1.0
        ids, distances, chunks = vs.search(q, top_k=5)
        assert ids == []
        assert distances == []
        assert chunks == []

    def test_search_top_k_capped_when_fewer_vectors(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        q = np.zeros(5, dtype=np.float32)
        q[0] = 1.0
        ids, distances, _ = vs.search(q, top_k=10)
        assert len(ids) == 3
        assert len(distances) == 3

    def test_search_returns_only_closest_k_vectors_and_chunks(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=5, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        q = np.zeros(5, dtype=np.float32)
        q[0] = 1.0
        ids, distances, result_chunks = vs.search(q, top_k=3)
        assert len(ids) == 3
        assert ids == [0, 1, 2]
        assert distances == sorted(distances)
        for i, id in enumerate(ids):
            assert result_chunks[i] == vs.id_to_chunk[id]

    def test_search_raises_if_not_initialized(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        with pytest.raises(RuntimeError):
            vs.search(np.zeros(10, dtype=np.float32), top_k=1)

    @pytest.mark.parametrize("k", [0, -1])
    def test_search_raises_for_invalid_top_k(self, vectorstore_path: Path, k: int):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vec, chunk = create_identical_test_vectors(num_vectors=1, dim=10)
        vs.add(vectors=vec, chunks=chunk)
        with pytest.raises(ValueError):
            vs.search(vec[0], top_k=k)

    def test_search_raises_for_wrong_query_dimension(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vec, chunk = create_identical_test_vectors(num_vectors=1, dim=10)
        vs.add(vectors=vec, chunks=chunk)
        with pytest.raises(ValueError):
            vs.search(np.zeros(5, dtype=np.float32), top_k=1)

    def test_search_raises_for_wrong_query_shape(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vec, chunk = create_identical_test_vectors(num_vectors=1, dim=10)
        vs.add(vectors=vec, chunks=chunk)
        with pytest.raises(ValueError):
            vs.search(np.zeros((2, 10), dtype=np.float32), top_k=1)

    def test_search_accepts_query_1d_and_2d(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        ids_1d, _, _ = vs.search(vectors[0], top_k=1)
        ids_2d, _, _ = vs.search(vectors[:1], top_k=1)
        assert len(ids_1d) == 1
        assert len(ids_2d) == 1
        assert ids_1d[0] == ids_2d[0]

class TestVectorStoreSaveLoad:
    """Test suite for VectorStore save and load round trips"""

    def test_save_creates_expected_files(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        vs.save()
        assert (vectorstore_path / "index.bin").exists()
        assert (vectorstore_path / "metadata.jsonl").exists()
        assert (vectorstore_path / "index_metadata.json").exists()

    def test_save_raises_if_not_initialized(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        with pytest.raises(RuntimeError):
            vs.save()

    def test_load_restores_index_and_metadata(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        added_ids = vs.add(vectors=vectors, chunks=chunks)
        vs.save()
        loaded = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        loaded.load()
        assert len(loaded) == 3
        assert set(loaded.id_to_chunk.keys()) == set(added_ids)
        for i, id in enumerate(added_ids):
            assert loaded.id_to_chunk[id] == chunks[i]

    def test_load_preserves_search_results(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        ids_before, dists_before, _ = vs.search(vectors[0], top_k=3)
        vs.save()
        loaded = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        loaded.load()
        ids_after, dists_after, _ = loaded.search(vectors[0], top_k=3)
        assert ids_after == ids_before
        assert dists_after == dists_before

    def test_load_raises_when_index_file_missing(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        vs.save()
        (vectorstore_path / "index.bin").unlink()
        loaded = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        with pytest.raises(FileNotFoundError):
            loaded.load()

    def test_load_raises_when_metadata_file_missing(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        vs.save()
        (vectorstore_path / "metadata.jsonl").unlink()
        loaded = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        with pytest.raises(FileNotFoundError):
            loaded.load()

    def test_load_raises_for_dimension_mismatch(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        vs.save()
        loaded = VectorStore(dim=7, storage_dir=vectorstore_path, initial_max_elements=100)
        with pytest.raises(ValueError, match="dimension"):
            loaded.load()

    def test_load_or_create_loads_when_all_files_exist(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        vs.save()
        loaded = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        loaded.load_or_create()
        assert len(loaded) == 3
        ids, _, _ = loaded.search(vectors[0], top_k=3)
        assert len(ids) == 3

    def test_load_or_create_creates_when_any_file_missing(self, vectorstore_path: Path):
        # No save: all three files are missing, so load_or_create runs create()
        loaded = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        loaded.load_or_create()
        assert loaded.is_initialized()
        assert len(loaded) == 0

class TestVectorStoreUtilities:
    """Test suite for VectorStore utility functions"""

    def test_len_raises_if_not_initialized(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        with pytest.raises(RuntimeError):
            len(vs)

    def test_clear_removes_files_from_disk(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=3, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        vs.save()
        vs.clear()
        assert not (vectorstore_path / "index.bin").exists()
        assert not (vectorstore_path / "metadata.jsonl").exists()
        assert not (vectorstore_path / "index_metadata.json").exists()

    def test_clear_allows_create_and_add_afterward(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_ordered_test_vectors(num_vectors=2, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        vs.clear()
        vs.create()
        vec, chunk = create_identical_test_vectors(num_vectors=1, dim=5)
        vs.add(vectors=vec, chunks=chunk)
        assert len(vs) == 1

    def test_resize_raises_if_not_initialized(self, vectorstore_path: Path):
        vs = VectorStore(dim=10, storage_dir=vectorstore_path, initial_max_elements=100)
        with pytest.raises(RuntimeError):
            vs.resize(100)

    def test_resize_raises_when_new_size_less_than_count(self, vectorstore_path: Path):
        vs = VectorStore(dim=5, storage_dir=vectorstore_path, initial_max_elements=100)
        vs.create()
        vectors, chunks = create_identical_test_vectors(num_vectors=5, dim=5)
        vs.add(vectors=vectors, chunks=chunks)
        with pytest.raises(ValueError, match="less than"):
            vs.resize(3)