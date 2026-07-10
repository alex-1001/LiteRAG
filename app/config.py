"""Configuration for app"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config: env vars with validation and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding (required for app)
    embed_model_name: str = Field(..., description="SentenceTransformer model name for embeddings.")

    # Vector store
    storage_dir: str = Field(
        default="./.vectorstore",
        description="Directory for vector index and metadata files.",
    )

    # Optional default data dir for ingest when request omits folder_path
    data_dir: Optional[str] = Field(default=None, description="Default folder path for document ingest.")

    # Chunking (ingest)
    tokenizer_name: Optional[str] = Field(default=None, description="HuggingFace AutoTokenizer model name for chunking. If not set, uses embed_model_name.")
    chunk_size: int = Field(default=200, ge=1, description="Token count per chunk.")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap in tokens between consecutive chunks.")

    # Retrieval (query)
    top_k: int = Field(default=5, ge=1, description="Default number of chunks to retrieve per query.")

    # OpenRouter / LLM (kept it optional so scripts/tests can load config without them; app validates in lifespan)
    openrouter_base_url: Optional[str] = Field(default=None, description="OpenRouter API base URL.")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key.")
    openrouter_model: Optional[str] = Field(default=None, description="Model name for chat completions.")


def load_settings() -> Settings:
    """Load and validate settings from environment and .env. Call at startup."""
    return Settings()
