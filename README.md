# Lite RAG

Lightweight FastAPI service for indexing local `.txt` and `.md` documents, retrieving relevant chunks from a vector store, and answering questions with cited sources.

The source code is intentionally tight for now, covering the core components of a RAG backend: document ingestion, chunking, embeddings, vector search, prompt construction, structured LLM responses, and API error handling.

## Features

- Ingests `.txt` and `.md` files from `DATA_DIR`
- Supports optional subfolder ingest with `source_path`
- Keeps ingest paths inside `DATA_DIR`
- Stores embeddings in a local HNSW vector index
- Uses SentenceTransformers for local embeddings
- Supports cloud generation through OpenRouter or local generation through Ollama
- Returns cited source chunks with cosine similarity

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env
```

Edit `.env` with your model provider and local paths. For OpenRouter:

```env
LLM_PROVIDER=openrouter
LLM_API_KEY=your_key
LLM_MODEL=google/gemma-4-31b-it

EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DATA_DIR=data/raw
STORAGE_DIR=storage
```

For a local Ollama instance:

```env
LLM_PROVIDER=ollama
LLM_MODEL=gemma4

EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DATA_DIR=data/raw
STORAGE_DIR=storage
```

`LLM_BASE_URL` is optional. It defaults to the standard endpoint for the
selected provider, or you can set it to target a compatible custom endpoint.
Ollama must be running and have the configured model installed before queries
can generate answers.

Create `DATA_DIR` and add `.txt` or `.md` files before ingesting.

For a quick local demo, set `DATA_DIR=data/test` to use the included sample documents.

## Run

```bash
uvicorn app.main:app --reload
```

The first run may download the embedding model and tokenizer.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## API

### Ingest documents

Ingest all of `DATA_DIR`:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

Ingest a subfolder under `DATA_DIR`:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path": "test"}'
```

Force a full rebuild:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_rebuild": true}'
```

Document IDs are derived from case-sensitive paths relative to `DATA_DIR`.

### Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What do these documents say about onboarding?", "top_k": 5}'
```

Response shape:

```json
{
  "answer": "Answer text with citations like [1].",
  "used_retrieval": true,
  "sources": [
    {
      "document_name": "example.md",
      "document_id": "00000000-0000-0000-0000-000000000000",
      "chunk_id": "00000000-0000-0000-0000-000000000000-0",
      "snippet": "Source chunk text...",
      "cosine_similarity": 0.82
    }
  ]
}
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

GitHub Actions runs the same test suite on pushes to `main` and on pull requests.

## Current limitations

- No authentication or authorization
- No UI
- No PDF, DOCX, HTML, or image parsing
- Local single-service storage only
- Edited or deleted documents may require `force_rebuild=true`
- Public deployment would require authentication and additional hardening

## License

MIT License. See [LICENSE](LICENSE).
