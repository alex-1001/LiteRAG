"""Test suite for retrieve module."""
import pytest
import numpy as np
from unittest.mock import Mock

from app.retrieve import retrieve_chunks
from app.vectorstore import VectorStore
from app.embed import Embedder
from tests.conftest import _make_chunk


@pytest.fixture
def mock_embedder():
    """Create a mock Embedder for testing."""
    embedder = Mock(spec=Embedder)
    embedder.dim = 8
    embedder.embed_query.return_value = np.zeros((1, 8), dtype=np.float32)
    return embedder


@pytest.fixture
def mock_vectorstore():
    """Create a mock VectorStore for testing."""
    vectorstore = Mock(spec=VectorStore)
    vectorstore.search.return_value = ([], [], []) # default
    return vectorstore


class TestRetrieveChunks:
    """Test suite for retrieve_chunks functionality."""

    def test_calls_embedder_with_question(self, mock_embedder, mock_vectorstore):
        """retrieve_chunks should call embedder.embed_query with the question string."""
        question = "What is the meaning of life?"
        retrieve_chunks(question, mock_vectorstore, mock_embedder, top_k=5)
        mock_embedder.embed_query.assert_called_once_with(question)

    def test_calls_vectorstore_with_embedding_and_top_k(self, mock_embedder, mock_vectorstore):
        """retrieve_chunks should call vectorstore.search with the query embedding and top_k."""
        query_embedding = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
        mock_embedder.embed_query.return_value = query_embedding
        
        retrieve_chunks("test question", mock_vectorstore, mock_embedder, top_k=3)
        
        call_args = mock_vectorstore.search.call_args
        np.testing.assert_array_equal(call_args[0][0], query_embedding)
        assert call_args[0][1] == 3

    @pytest.mark.parametrize("top_k", [None, 1, 3, 7, 10])
    def test_retrieve_chunks_top_k_values(self, mock_embedder, mock_vectorstore, top_k):
        """retrieve_chunks should use the correct top_k parameter, defaulting to 5 when None is given."""
        expected_k = 5 if top_k is None else top_k
        if top_k is None:
            retrieve_chunks("test", mock_vectorstore, mock_embedder)
        else:
            retrieve_chunks("test", mock_vectorstore, mock_embedder, top_k=top_k)
        call_args = mock_vectorstore.search.call_args
        assert call_args[0][1] == expected_k

    def test_returns_chunks_and_distances(self, mock_embedder, mock_vectorstore):
        """retrieve_chunks should return the chunks and distances from vectorstore.search."""
        chunks = [_make_chunk(f"chunk_{i}") for i in range(3)]
        distances = [0.1, 0.2, 0.3]
        mock_vectorstore.search.return_value = ([0, 1, 2], distances, chunks)
        
        result_chunks, result_distances = retrieve_chunks("test", mock_vectorstore, mock_embedder, top_k=3)
        
        assert result_chunks == chunks
        assert result_distances == distances

    def test_handles_empty_results(self, mock_embedder, mock_vectorstore):
        """retrieve_chunks should handle empty search results gracefully."""
        mock_vectorstore.search.return_value = ([], [], [])
        chunks, distances = retrieve_chunks("test", mock_vectorstore, mock_embedder, top_k=5)
        assert chunks == []
        assert distances == []
