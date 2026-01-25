import pytest
from typing import List
from app.ingest import normalize_whitespace, load_text_documents, chunk_text, ingest_folder
from app.models import DocumentChunk
from pathlib import Path
from uuid import UUID

class TestNormalizeWhitespace:
    """Test suite for normalize_whitespace()"""
    
    @pytest.mark.parametrize("input, expected", [
        ("line1\r\nline2\r\nline3", "line1\nline2\nline3"),
        ("line1\rline2\nline3", "line1\nline2\nline3"),
        ("line1\nline2\r\nline3", "line1\nline2\nline3"),
    ])
    def test_whitespace_formats(self, input, expected):
        """special white space characters (e.g. "\\r\\n" or "\\r") converted to "\\n"""
        assert normalize_whitespace(input) == expected
        
    @pytest.mark.parametrize("input, expected", [
        ("line1 \nline2\nline3", "line1\nline2\nline3"),
        ("line1\nline2\nline3 ", "line1\nline2\nline3"),
        ("line1\nline2  \t \nline3 ", "line1\nline2\nline3"),
    ])
    def test_trailing_whitespace(self, input, expected):
        """trailing whitespace removed from each line"""
        assert normalize_whitespace(input) == expected
    
    @pytest.mark.parametrize("input, expected", [
        (" line1\nline2\nline3 ", "line1\nline2\nline3"),
        (" line1\nline2\nline3\n ", "line1\nline2\nline3"),
        (" line1\nline2\nline3\n\t\t ", "line1\nline2\nline3"),
    ])
    def test_doc_level_whitespace(self, input, expected):
        """leading and trailing whitespace removed from document"""
        assert normalize_whitespace(input) == expected
        
    def test_empty_string(self):
        """Test that empty string returns empty string"""
        assert normalize_whitespace("") == ""
    
    @pytest.mark.parametrize("input, expected", [
        (" ", ""),
        ("\t", ""),
        ("\t \t", ""),
        ("\n \n", ""),
        ("\r", ""),
        ("\f", ""),
        ("\b", ""),
        ("\a", ""),
    ])
    def test_only_whitespace(self, input, expected):
        """Test that only whitespace returns empty string"""
        assert normalize_whitespace(input) == expected

@pytest.fixture
def temp_doc_dir(tmp_path):
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create test files
    (data_dir / "simple.txt").write_text("Simple document with just one line.")
    (data_dir / "multiline.txt").write_text("Line 1\nLine 2\nLine 3")
    (data_dir / "markdown.md").write_text("# Title\n\nContent here.")
    (data_dir / "empty.txt").write_text("") 
    (data_dir / "python.py").write_text("print('this is not a text file')")

    yield data_dir
    
class TestLoadTextDocuments:
    """Test suite for load_text_documents()"""
    def test_finds_text_files(self, temp_doc_dir):
        """.txt and .md files are found"""
        docs = load_text_documents(str(temp_doc_dir))
        assert isinstance(docs[0][0], Path)
        doc_names = [doc[0].name for doc in docs]
        assert "simple.txt" in doc_names
        assert "multiline.txt" in doc_names
        assert "markdown.md" in doc_names
        
        
    def test_skips_nontext_files(self, temp_doc_dir):
        """skips files of wrong type"""
        docs = load_text_documents(str(temp_doc_dir))
        doc_types = [doc[0].suffix for doc in docs]
        assert all([doc_type in {".md", ".txt"} for doc_type in doc_types])
    
    def test_skips_empty_files(self, temp_doc_dir):
        """skips empty files"""
        docs = load_text_documents(str(temp_doc_dir))
        doc_names = [doc[0].name for doc in docs]
        assert "empty.txt" not in doc_names
    
    def test_nonexistent_directory(self):
        """raises FileNotFoundError for non-existent directory"""
        with pytest.raises(FileNotFoundError):
            load_text_documents("/nonexistent/path/12345")

    def test_file_not_directory(self, temp_doc_dir):
        """raises NotADirectoryError for file path"""
        file_path = temp_doc_dir / "not_a_dir.txt"
        file_path.write_text("test")
        with pytest.raises(NotADirectoryError):
            load_text_documents(str(file_path))

class MockTokenizer:
    def __init__(self):
        self._words = []
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Split text into words, return as token IDs"""
        self._words = text.split()
        word_ids = list(range(len(self._words))) # [0, 1, 2, ...]
        return {"input_ids": word_ids} # expected format for hugging face tokenizers  
    
    def decode(self, tokens: List[int], **kwargs) -> str:
        """Reconstruct text from token IDs"""
        words = [self._words[i] for i in tokens if 0 <= i < len(self._words)]
        return " ".join(words)
    
    def __call__(self, text: str, **kwargs) -> List[int]:
        return self.encode(text, **kwargs)

@pytest.fixture
def mock_tokenizer():
    yield MockTokenizer()
        
class TestChunkText:
    """Test suite for chunk_text()"""
    def test_empty_text(self, mock_tokenizer):
        """returns empty list if text is empty"""
        text = ""
        chunks = chunk_text(text, 100, 1, mock_tokenizer)
        assert chunks == []
    
    @pytest.mark.parametrize("text, chunk_size, overlap, expected", [
        ("word word word", 10, 1, [("word word word", {"start_token_index": 0, "end_token_index": 2, "chunk_size": 3})]),
        ("word word word", 3, 1, [("word word word", {"start_token_index": 0, "end_token_index": 2, "chunk_size": 3})]),
        ("word word word", 3, 0, [("word word word", {"start_token_index": 0, "end_token_index": 2, "chunk_size": 3})]),
    ])
    def test_single_chunk(self, mock_tokenizer, text, chunk_size, overlap, expected):
        """returns single chunk if text is less than chunk size"""
        chunks = chunk_text(text, chunk_size, overlap, mock_tokenizer)
        assert chunks == expected
    
    @pytest.mark.parametrize("text, chunk_size, overlap, num_chunks", [
        ("word " * 11, 10, 1, 2),
        ("word " * 10, 3, 1, 5),
        ("word " * 10, 3, 2, 8),
        ("word " * 10, 3, 0, 4),
        ("word " * 5, 4, 3, 2),
        ("word " * 4, 3, 2, 2),
        ("word " * 9, 4, 0, 3),
    ])
    def test_multiple_chunks(self, mock_tokenizer, text, chunk_size, overlap, num_chunks):
        """returns multiple chunks if text is greater than chunk size"""
        chunks = chunk_text(text, chunk_size, overlap, mock_tokenizer)
        assert len(chunks) == num_chunks
    
    @pytest.mark.parametrize("text, chunk_size, overlap, expected", [
        ("word " * 5, 3, 1, {"start_token_index": 0, "end_token_index": 2, "chunk_size": 3}),
        ("word " * 5, 3, 0, {"start_token_index": 0, "end_token_index": 2, "chunk_size": 3}),
    ])
    def test_metadata(self, mock_tokenizer, text, chunk_size, overlap, expected):
        """includes correct metadata about chunk boundaries and size"""
        chunks = chunk_text(text, chunk_size, overlap, mock_tokenizer)
        assert chunks[0][1] == expected
        for _, metadata in chunks:
            assert "start_token_index" in metadata
            assert "end_token_index" in metadata
            assert "chunk_size" in metadata
            assert metadata["chunk_size"] <= chunk_size, (
                f'Expected metadata["chunk_size"] <= chunk_size ({metadata["chunk_size"]} <= {chunk_size}), '
                f"but chunk metadata reports a larger chunk size than allowed."
            )
            assert metadata["start_token_index"] <= metadata["end_token_index"], (
                f'Expected metadata["start_token_index"] ({metadata["start_token_index"]}) to be <= '
                f'metadata["end_token_index"] ({metadata["end_token_index"]}), '
                "but the start index is after the end index in chunk metadata."
            )
            
    def test_chunks_have_overlap(self, mock_tokenizer):
        """Verify that consecutive chunks actually overlap"""
        text = "word " * 20
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=2, tokenizer=mock_tokenizer)
        
        if len(chunks) > 1:
            first_end = chunks[0][1]["end_token_index"]
            second_start = chunks[1][1]["start_token_index"]
            overlap = first_end - second_start + 1
            assert overlap == 2  # Should have 2 token overlap
    
    @pytest.mark.parametrize("chunk_size, overlap", [
        (10, 11),
        (10, 10),
        (10, -1),
        (0, 1),
    ])
    def test_invalid_overlap(self, chunk_size, overlap, mock_tokenizer):
        """raises ValueError if overlap is greater than or equal to chunk size"""
        with pytest.raises(ValueError):
            chunk_text("word " * 10, chunk_size, overlap, mock_tokenizer)

class TestIngestFolder:
    """Test suite for ingest_folder()"""
    
    def test_returns_document_chunks(self, temp_doc_dir, mock_tokenizer):
        """returns list of DocumentChunk objects"""
        chunks = ingest_folder(str(temp_doc_dir), mock_tokenizer)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    
    def test_chunks_have_required_fields(self, temp_doc_dir, mock_tokenizer):
        """all chunks have required DocumentChunk fields"""
        chunks = ingest_folder(str(temp_doc_dir), mock_tokenizer)
        for chunk in chunks:
            assert chunk.document_id is not None
            assert isinstance(chunk.document_id, UUID)
            assert chunk.document_name
            assert chunk.chunk_id
            assert chunk.text
            assert len(chunk.text) > 0  # Not empty
            assert isinstance(chunk.metadata, dict)
    
    def test_document_id_deterministic(self, temp_doc_dir, mock_tokenizer):
        """same document always gets same document_id"""
        chunks1 = ingest_folder(str(temp_doc_dir), mock_tokenizer)
        chunks2 = ingest_folder(str(temp_doc_dir), mock_tokenizer)
        
        # Group chunks by document_name
        doc_ids1 = {chunk.document_name: chunk.document_id for chunk in chunks1}
        doc_ids2 = {chunk.document_name: chunk.document_id for chunk in chunks2}
        
        # Same documents should have same IDs
        assert doc_ids1 == doc_ids2
    
    def test_chunk_id_format(self, temp_doc_dir, mock_tokenizer):
        """chunk_id follows expected format: {document_id}-{index}"""
        chunks = ingest_folder(str(temp_doc_dir), mock_tokenizer)
        
        # Group chunks by document
        from collections import defaultdict
        chunks_by_doc = defaultdict(list)
        for chunk in chunks:
            chunks_by_doc[chunk.document_id].append(chunk)
        
        # Check chunk_id format for each document
        for doc_id, doc_chunks in chunks_by_doc.items():
            for i, chunk in enumerate(sorted(doc_chunks, key=lambda c: c.chunk_id)):
                expected_chunk_id = f"{doc_id}-{i}"
                assert chunk.chunk_id == expected_chunk_id, (
                    f"Expected chunk_id '{expected_chunk_id}', got '{chunk.chunk_id}'"
                )
    
    def test_processes_all_documents(self, temp_doc_dir, mock_tokenizer):
        """processes all valid documents in directory"""
        chunks = ingest_folder(str(temp_doc_dir), mock_tokenizer)
        
        # Get unique document names
        doc_names = set(chunk.document_name for chunk in chunks)
        
        # Should have processed the 3 non-empty text files
        assert "simple.txt" in doc_names
        assert "multiline.txt" in doc_names
        assert "markdown.md" in doc_names
        assert "empty.txt" not in doc_names  # Should be skipped
        assert "python.py" not in doc_names  # Should be ignored
    
    def test_each_document_has_unique_id(self, temp_doc_dir, mock_tokenizer):
        """each document gets a unique document_id"""
        chunks = ingest_folder(str(temp_doc_dir), mock_tokenizer)
        
        # Group by document_name
        from collections import defaultdict
        chunks_by_name = defaultdict(list)
        for chunk in chunks:
            chunks_by_name[chunk.document_name].append(chunk)
        
        # Each document should have same document_id for all its chunks
        doc_ids = {}
        for doc_name, doc_chunks in chunks_by_name.items():
            # All chunks from same document should have same document_id
            chunk_ids = set(chunk.document_id for chunk in doc_chunks)
            assert len(chunk_ids) == 1, (
                f"Document '{doc_name}' has multiple document_ids: {chunk_ids}"
            )
            doc_ids[doc_name] = list(chunk_ids)[0]
        
        # Different documents should have different IDs
        assert len(doc_ids) == len(set(doc_ids.values())), (
            "Multiple documents share the same document_id"
        )
    
    def test_chunks_contain_text(self, temp_doc_dir, mock_tokenizer):
        """chunks contain actual text content"""
        chunks = ingest_folder(str(temp_doc_dir), mock_tokenizer)
        
        for chunk in chunks:
            assert chunk.text
            assert len(chunk.text.strip()) > 0  # Not just whitespace
            # Text should be from the original document
            assert isinstance(chunk.text, str)
    
    def test_empty_directory(self, tmp_path, mock_tokenizer):
        """returns empty list for directory with no text files"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        chunks = ingest_folder(str(empty_dir), mock_tokenizer)
        assert chunks == []
    
    def test_single_document(self, tmp_path, mock_tokenizer):
        """handles directory with single document"""
        data_dir = tmp_path / "single_doc"
        data_dir.mkdir()
        (data_dir / "only.txt").write_text("Single document content.")
        
        chunks = ingest_folder(str(data_dir), mock_tokenizer)
        assert len(chunks) > 0
        assert all(chunk.document_name == "only.txt" for chunk in chunks)
    
    def test_propagates_errors(self, tmp_path, mock_tokenizer):
        """propagates errors from load_text_documents"""
        # Test that FileNotFoundError is raised
        with pytest.raises(FileNotFoundError):
            ingest_folder("/nonexistent/path/12345", mock_tokenizer)
        
        # Test that NotADirectoryError is raised
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")
        with pytest.raises(NotADirectoryError):
            ingest_folder(str(file_path), mock_tokenizer)