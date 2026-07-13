import pytest
from app.main import app, lifespan, get_client, get_config, get_embedder, get_vectorstore, get_lock, get_tokenizer
from app.config import Settings
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from tests.conftest import _make_chunk
from types import SimpleNamespace
from threading import Lock

@pytest.fixture
def main_app_overrides():
    app.dependency_overrides.clear()
    yield app.dependency_overrides
    app.dependency_overrides.clear()

@pytest.fixture
def anyio_backend():
    return "asyncio"

class TestHealth:
    def test_health_returns_ok(self):
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

class TestIngest:
    @pytest.fixture
    def ingest_app_context(self, main_app_overrides):
        settings = Settings(
            embed_model_name="test-embed",
            openrouter_base_url="https://example.com",
            openrouter_api_key="test-key",
            openrouter_model="test-llm",
            data_dir="/default/data",
            chunk_size=100,
            chunk_overlap=10,
        )

        mocks = SimpleNamespace(
            settings=settings,
            embedder=Mock(),
            tokenizer=Mock(),
            vectorstore=Mock(),
            lock=Lock(),
            test_app=TestClient(app),
        )

        # note the below mutates app object from main_app_overrides()
        main_app_overrides[get_config] = lambda: mocks.settings
        main_app_overrides[get_embedder] = lambda: mocks.embedder
        main_app_overrides[get_tokenizer] = lambda: mocks.tokenizer
        main_app_overrides[get_vectorstore] = lambda: mocks.vectorstore
        main_app_overrides[get_lock] = lambda: mocks.lock

        return mocks
    
    def test_ingest_requires_folder(self, ingest_app_context):
        ingest_mocks = ingest_app_context
        
        ingest_mocks.settings.data_dir = None
        response = ingest_mocks.test_app.post("/ingest", json={})
        
        assert response.status_code == 400
        assert response.json()["detail"] == "folder_path required when DATA_DIR is not set"
    
    def test_ingest_uses_request_folder(self, ingest_app_context):
        ingest_mocks = ingest_app_context
        chunk = _make_chunk("test chunk")
        
        with patch("app.main.ingest_folder") as mock_ingest_folder:
            mock_ingest_folder.return_value = [chunk]
            response = ingest_mocks.test_app.post("/ingest", json={"folder_path": "/request/data"})
        
        assert response.status_code == 200
        mock_ingest_folder.assert_called_once_with(
            "/request/data", 
            ingest_mocks.tokenizer, 
            ingest_mocks.settings.chunk_size, 
            ingest_mocks.settings.chunk_overlap
        )

    def test_ingest_defaults_to_config_data_dir(self, ingest_app_context):
        ingest_mocks = ingest_app_context
        chunk = _make_chunk("test chunk")

        with patch("app.main.ingest_folder") as mock_ingest_folder:
            mock_ingest_folder.return_value = [chunk]
            response = ingest_mocks.test_app.post("/ingest", json={})

        assert response.status_code == 200
        mock_ingest_folder.assert_called_once_with(
            "/default/data",
            ingest_mocks.tokenizer,
            ingest_mocks.settings.chunk_size,
            ingest_mocks.settings.chunk_overlap,
        )
    
    def test_ingest_embeds_chunks_and_saves_to_vectorstore(self, ingest_app_context):
        ingest_mocks = ingest_app_context
        chunks = [_make_chunk(f"chunk{i} text", f"doc{i}.md", f"{i}") for i in range(2)]
        
        vectors = Mock()
        ingest_mocks.embedder.embed_chunks.return_value = vectors
        
        with patch("app.main.ingest_folder") as mock_ingest_folder:
            mock_ingest_folder.return_value = chunks
            response = ingest_mocks.test_app.post("/ingest", json={"folder_path": "/default/test", "force_rebuild": False})
        
        assert response.status_code == 200
        ingest_mocks.embedder.embed_chunks.assert_called_once_with(chunks)
        ingest_mocks.vectorstore.add.assert_called_once_with(
            vectors,
            chunks,
        )
        ingest_mocks.vectorstore.save.assert_called_once()
        ingest_mocks.vectorstore.clear.assert_not_called()
        ingest_mocks.vectorstore.create.assert_not_called()

    def test_ingest_force_rebuild_clears_and_creates_vectorstore(self, ingest_app_context):
        ingest_mocks = ingest_app_context
        chunks = [_make_chunk("chunk text")]
        vectors = Mock()
        ingest_mocks.embedder.embed_chunks.return_value = vectors

        with patch("app.main.ingest_folder") as mock_ingest_folder:
            mock_ingest_folder.return_value = chunks
            response = ingest_mocks.test_app.post(
                "/ingest",
                json={"folder_path": "/request/data", "force_rebuild": True},
            )

        assert response.status_code == 200
        ingest_mocks.vectorstore.clear.assert_called_once()
        ingest_mocks.vectorstore.create.assert_called_once()
        ingest_mocks.vectorstore.add.assert_called_once_with(vectors, chunks)
        ingest_mocks.vectorstore.save.assert_called_once()

    def test_ingest_response_reports_documents_chunks_and_ids(self, ingest_app_context):
        ingest_mocks = ingest_app_context
        chunk_a = _make_chunk("chunk A", "doc.md", "0")
        chunk_b = chunk_a.model_copy(update={"text": "chunk B", "chunk_id": "1"})
        chunks = [chunk_a, chunk_b]
        ingest_mocks.embedder.embed_chunks.return_value = Mock()

        with patch("app.main.ingest_folder") as mock_ingest_folder:
            mock_ingest_folder.return_value = chunks
            response = ingest_mocks.test_app.post("/ingest", json={"folder_path": "/request/data"})

        body = response.json()

        assert response.status_code == 200
        assert body["documents_processed"] == 1
        assert body["chunks_created"] == 2
        assert set(body["document_ids"]) == {str(chunk_a.document_id)}
        assert body["processing_time_seconds"] >= 0

class TestQuery:
    @pytest.fixture
    def query_app_context(self, main_app_overrides):
        settings = Settings(
            embed_model_name="test-embed",
            openrouter_base_url="https://example.com",
            openrouter_api_key="test-key",
            openrouter_model="test-llm",
            top_k=5,
        )
        
        mocks = SimpleNamespace(
            settings=settings,
            embedder=Mock(),
            client=Mock(),
            vectorstore=Mock(),
            test_app=TestClient(app),
        )
        
        main_app_overrides[get_config] = lambda: mocks.settings
        main_app_overrides[get_embedder] = lambda: mocks.embedder
        main_app_overrides[get_client] = lambda: mocks.client
        main_app_overrides[get_vectorstore] = lambda: mocks.vectorstore
        
        return mocks
    
    def test_query_uses_request_top_k(self, query_app_context):
        query_mocks = query_app_context
        
        with patch("app.main.retrieve_chunks") as mock_retrieve, patch("app.main.answer_query") as mock_answer:  
            mock_retrieve.return_value = ([_make_chunk("retrieved text")], [0.25])
            mock_answer.return_value = ("Answer", [1])
            
            response = query_mocks.test_app.post("/query", json={"question": "What is this?", "top_k": 3})
            
        assert response.status_code == 200
        mock_retrieve.assert_called_once_with(
            "What is this?",
            query_mocks.vectorstore,
            query_mocks.embedder,
            top_k=3,
        )

    def test_query_defaults_to_config_top_k(self, query_app_context):
        query_mocks = query_app_context

        with patch("app.main.retrieve_chunks") as mock_retrieve, patch("app.main.answer_query") as mock_answer:
            mock_retrieve.return_value = ([_make_chunk("retrieved text")], [0.25])
            mock_answer.return_value = ("Answer", [1])

            response = query_mocks.test_app.post("/query", json={"question": "What is this?"})

        assert response.status_code == 200
        mock_retrieve.assert_called_once_with(
            "What is this?",
            query_mocks.vectorstore,
            query_mocks.embedder,
            top_k=5,
        )

    def test_query_calls_answer_query_with_client_and_model(self, query_app_context):
        query_mocks = query_app_context
        chunks = [_make_chunk("retrieved text")]

        with patch("app.main.retrieve_chunks") as mock_retrieve, patch("app.main.answer_query") as mock_answer:
            mock_retrieve.return_value = (chunks, [0.25])
            mock_answer.return_value = ("Answer", [1])

            response = query_mocks.test_app.post("/query", json={"question": "What is this?"})

        assert response.status_code == 200
        mock_answer.assert_called_once_with(
            "What is this?",
            chunks,
            client=query_mocks.client,
            llm_model="test-llm",
        )
        
    def test_query_cites_correct_chunks(self, query_app_context):
        query_mocks = query_app_context
        chunks = [_make_chunk(f"chunk{i} text", f"doc{i}.md", f"{i}") for i in range(3)]
        
        with patch("app.main.retrieve_chunks") as mock_retrieve, patch("app.main.answer_query") as mock_answer:  
            mock_retrieve.return_value = (chunks, [0.4, 0.6, 0.7])
            mock_answer.return_value = ("Answer", [2, 3])
            
            response = query_mocks.test_app.post("/query", json={"question": "What is this?", "top_k": 3})
            
        body = response.json()
        
        assert body["answer"] == "Answer"
        assert body["used_retrieval"] is True
        assert len(body["sources"]) == 2
        assert [s["document_name"] for s in body["sources"]] == ["doc1.md", "doc2.md"]
        assert [s["snippet"] for s in body["sources"]] == ["chunk1 text", "chunk2 text"]
        assert [s["score"] for s in body["sources"]] == pytest.approx([0.4, 0.3])

    def test_query_returns_no_sources_when_no_chunks_retrieved(self, query_app_context):
        query_mocks = query_app_context

        with patch("app.main.retrieve_chunks") as mock_retrieve, patch("app.main.answer_query") as mock_answer:
            mock_retrieve.return_value = ([], [])
            mock_answer.return_value = ("No relevant documents are associated with this query.", [])

            response = query_mocks.test_app.post("/query", json={"question": "What is this?"})

        assert response.status_code == 200
        assert response.json() == {
            "answer": "No relevant documents are associated with this query.",
            "used_retrieval": False,
            "sources": [],
        }

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": ""},
            {"question": "What is this?", "top_k": 0},
        ],
    )
    def test_query_rejects_invalid_request_body(self, query_app_context, payload):
        response = query_app_context.test_app.post("/query", json=payload)

        assert response.status_code == 422

class TestLifespan:
    @pytest.fixture
    def lifespan_settings(self):
        return Settings(
            embed_model_name="test-embed",
            tokenizer_name="test-tokenizer",
            storage_dir="/tmp/vectorstore",
            openrouter_base_url="https://example.com",
            openrouter_api_key="test-key",
            openrouter_model="test-llm",
        )

    @pytest.mark.anyio
    async def test_lifespan_requires_openrouter_settings(self):
        settings = Settings(
            embed_model_name="test-embed",
            openrouter_base_url=None,
            openrouter_api_key="test-key",
            openrouter_model="test-llm",
        )
        test_app = FastAPI()

        with patch("app.main.load_settings", return_value=settings):
            with pytest.raises(ValueError, match="OPENROUTER_BASE_URL"):
                async with lifespan(test_app):
                    pass

    @pytest.mark.anyio
    async def test_lifespan_wires_startup_dependencies(self, lifespan_settings):
        test_app = FastAPI()
        fake_embedder = Mock()
        fake_embedder.dim = 384
        fake_tokenizer = Mock()
        fake_vectorstore = Mock()
        fake_client = Mock()

        with (
            patch("app.main.load_settings", return_value=lifespan_settings),
            patch("app.main.Embedder", return_value=fake_embedder) as mock_embedder_cls,
            patch("app.main.AutoTokenizer.from_pretrained", return_value=fake_tokenizer) as mock_tokenizer_loader,
            patch("app.main.VectorStore", return_value=fake_vectorstore) as mock_vectorstore_cls,
            patch("app.main.OpenAI", return_value=fake_client) as mock_openai_cls,
        ):
            async with lifespan(test_app):
                assert test_app.state.config is lifespan_settings
                assert test_app.state.embedder is fake_embedder
                assert test_app.state.tokenizer is fake_tokenizer
                assert test_app.state.vectorstore is fake_vectorstore
                assert test_app.state.client is fake_client
                assert test_app.state.lock is not None

        mock_embedder_cls.assert_called_once_with("test-embed")
        mock_tokenizer_loader.assert_called_once_with("test-tokenizer")
        mock_vectorstore_cls.assert_called_once_with(
            dim=384,
            storage_dir="/tmp/vectorstore",
            embed_model_name="test-embed",
        )
        fake_vectorstore.load_or_create.assert_called_once()
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://example.com",
        )
        fake_vectorstore.save.assert_called_once()

    @pytest.mark.anyio
    async def test_lifespan_uses_embed_model_when_tokenizer_name_missing(self, lifespan_settings):
        settings = lifespan_settings.model_copy(update={"tokenizer_name": None})
        test_app = FastAPI()
        fake_embedder = Mock()
        fake_embedder.dim = 384

        with (
            patch("app.main.load_settings", return_value=settings),
            patch("app.main.Embedder", return_value=fake_embedder),
            patch("app.main.AutoTokenizer.from_pretrained") as mock_tokenizer_loader,
            patch("app.main.VectorStore", return_value=Mock()),
            patch("app.main.OpenAI", return_value=Mock()),
        ):
            async with lifespan(test_app):
                pass

        mock_tokenizer_loader.assert_called_once_with("test-embed")

    @pytest.mark.anyio
    async def test_lifespan_logs_vectorstore_save_error_on_shutdown(self, lifespan_settings):
        test_app = FastAPI()
        fake_embedder = Mock()
        fake_embedder.dim = 384
        fake_vectorstore = Mock()
        fake_vectorstore.save.side_effect = RuntimeError("save failed")

        with (
            patch("app.main.load_settings", return_value=lifespan_settings),
            patch("app.main.Embedder", return_value=fake_embedder),
            patch("app.main.AutoTokenizer.from_pretrained", return_value=Mock()),
            patch("app.main.VectorStore", return_value=fake_vectorstore),
            patch("app.main.OpenAI", return_value=Mock()),
            patch("app.main.logging.exception") as mock_log_exception,
        ):
            async with lifespan(test_app):
                pass

        fake_vectorstore.save.assert_called_once()
        mock_log_exception.assert_called_once()
