import pytest
import numpy as np
from unittest.mock import patch, Mock

from app.embed import Embedder
from tests.conftest import _make_chunk

@pytest.fixture
def mock_sentence_transformer():
    """Patch SentenceTransformer; yield (MockST, mock_instance) for tests to configure and assert."""
    with patch("app.embed.SentenceTransformer") as MockST:
        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 8
        mock_instance.get_max_seq_length.return_value = 512
        mock_instance.encode.return_value = np.zeros((1, 8), dtype=np.float32)
        MockST.return_value = mock_instance
        yield MockST, mock_instance


class TestEmbedderInitialization:
    """Test suite for Embedder initialization."""

    def test_init_stores_model_name_and_creates_model(self, mock_sentence_transformer):
        """Init stores model_name and constructs SentenceTransformer with it."""
        MockST, _ = mock_sentence_transformer
        embedder = Embedder("all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"
        MockST.assert_called_once_with("all-MiniLM-L6-v2")


class TestEmbedderEmbedTexts:
    """Test suite for embed_texts()."""

    def test_embed_texts_empty_list_returns_empty_array(self, mock_sentence_transformer):
        """Empty list returns (0, dim) zeros and does not call encode."""
        _, mock_instance = mock_sentence_transformer
        embedder = Embedder("any-model")
        result = embedder.embed_texts([])
        assert result.shape == (0, 8)
        assert result.dtype == np.float32
        mock_instance.encode.assert_not_called()

    def test_embed_texts_single_text(self, mock_sentence_transformer):
        """Single text returns (1, dim) and encode called with correct args."""
        _, mock_instance = mock_sentence_transformer
        embedder = Embedder("any-model")
        result = embedder.embed_texts(["hello"])
        assert result.shape == (1, 8)
        assert result.dtype == np.float32
        mock_instance.encode.assert_called_once_with(
            ["hello"], normalize_embeddings=True, convert_to_numpy=True
        )

    def test_embed_texts_multiple_texts(self, mock_sentence_transformer):
        """Multiple texts return (N, dim) and encode called with the list."""
        _, mock_instance = mock_sentence_transformer
        mock_instance.encode.return_value = np.zeros((3, 8), dtype=np.float32)
        embedder = Embedder("any-model")
        result = embedder.embed_texts(["a", "b", "c"])
        assert result.shape == (3, 8)
        mock_instance.encode.assert_called_once_with(
            ["a", "b", "c"], normalize_embeddings=True, convert_to_numpy=True
        )

    def test_embed_texts_returns_float32(self, mock_sentence_transformer):
        """Output is float32 even when encode returns float64."""
        _, mock_instance = mock_sentence_transformer
        mock_instance.encode.return_value = np.zeros((1, 8), dtype=np.float64)
        embedder = Embedder("any-model")
        result = embedder.embed_texts(["x"])
        assert result.dtype == np.float32

    def test_embed_texts_calls_encode_with_normalize_and_convert_to_numpy(
        self, mock_sentence_transformer
    ):
        """encode is called with normalize_embeddings=True and convert_to_numpy=True."""
        _, mock_instance = mock_sentence_transformer
        embedder = Embedder("any-model")
        embedder.embed_texts(["x"])
        call_kw = mock_instance.encode.call_args[1]
        assert call_kw["normalize_embeddings"] is True
        assert call_kw["convert_to_numpy"] is True


class TestEmbedderEmbedChunks:
    """Test suite for embed_chunks()."""

    def test_embed_chunks_valid_chunks(self, mock_sentence_transformer):
        """Valid DocumentChunks produce (N, dim) and encode receives their texts."""
        _, mock_instance = mock_sentence_transformer
        mock_instance.encode.return_value = np.zeros((2, 8), dtype=np.float32)
        embedder = Embedder("any-model")
        chunks = [_make_chunk("t1", chunk_id="0"), _make_chunk("t2", chunk_id="1")]
        result = embedder.embed_chunks(chunks)
        assert result.shape == (2, 8)
        mock_instance.encode.assert_called_once_with(
            ["t1", "t2"], normalize_embeddings=True, convert_to_numpy=True
        )

    def test_embed_chunks_empty_list(self, mock_sentence_transformer):
        """Empty chunks list returns (0, dim) and encode is not called."""
        _, mock_instance = mock_sentence_transformer
        embedder = Embedder("any-model")
        result = embedder.embed_chunks([])
        assert result.shape == (0, 8)
        mock_instance.encode.assert_not_called()

    @pytest.mark.parametrize("chunks", [None, 1, "x", {}])
    def test_embed_chunks_raises_if_not_list(self, mock_sentence_transformer, chunks):
        """Raises ValueError when chunks is not a list."""
        embedder = Embedder("any-model")
        with pytest.raises(ValueError, match="Chunks must be a list"):
            embedder.embed_chunks(chunks)

    def test_embed_chunks_raises_if_chunk_not_document_chunk(self, mock_sentence_transformer):
        """Raises ValueError when a list element is not a DocumentChunk."""
        embedder = Embedder("any-model")
        chunks = [_make_chunk("t1"), "not a chunk"]
        with pytest.raises(ValueError, match="is not a DocumentChunk"):
            embedder.embed_chunks(chunks)


class TestEmbedderEmbedQuery:
    """Test suite for embed_query()."""

    def test_embed_query_valid_string(self, mock_sentence_transformer):
        """Valid string returns (1, dim) and encode called with [query]."""
        _, mock_instance = mock_sentence_transformer
        embedder = Embedder("any-model")
        result = embedder.embed_query("q")
        assert result.shape == (1, 8)
        mock_instance.encode.assert_called_once_with(
            ["q"], normalize_embeddings=True, convert_to_numpy=True
        )

    @pytest.mark.parametrize("query", [None, 1, 3.14, [], {}])
    def test_embed_query_raises_if_not_string(self, mock_sentence_transformer, query):
        """Raises ValueError when query is not a string."""
        embedder = Embedder("any-model")
        with pytest.raises(ValueError, match="Query must be a string"):
            embedder.embed_query(query)


class TestEmbedderDim:
    """Test suite for dim property."""

    def test_dim_returns_model_embedding_dimension(self, mock_sentence_transformer):
        """dim returns model.get_sentence_embedding_dimension()."""
        _, mock_instance = mock_sentence_transformer
        embedder = Embedder("any-model")
        assert embedder.dim == 8
        mock_instance.get_sentence_embedding_dimension.assert_called()


class TestEmbedderGetMaxEmbeddingInputLength:
    """Test suite for get_max_embedding_input_length()."""

    def test_get_max_embedding_input_length_returns_model_max_seq_length(
        self, mock_sentence_transformer
    ):
        """get_max_embedding_input_length returns model.get_max_seq_length()."""
        _, mock_instance = mock_sentence_transformer
        embedder = Embedder("any-model")
        assert embedder.get_max_embedding_input_length() == 512
        mock_instance.get_max_seq_length.assert_called_once()
