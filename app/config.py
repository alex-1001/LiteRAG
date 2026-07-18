"""Configuration for app"""

from enum import StrEnum
from typing import Annotated, Optional

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    StringConstraints,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


ModelName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


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

    # Root folder for document ingest. Requests can only select paths inside this directory.
    data_dir: str = Field(..., description="Root folder that bounds document ingest.")

    # Chunking (ingest)
    tokenizer_name: Optional[str] = Field(default=None, description="HuggingFace AutoTokenizer model name for chunking. If not set, uses embed_model_name.")
    chunk_size: int = Field(default=200, ge=1, description="Token count per chunk.")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap in tokens between consecutive chunks.")

    # Retrieval (query)
    top_k: int = Field(default=5, ge=1, description="Default number of chunks to retrieve per query.")

    # LLM generation. Provider defaults are resolved when the chat model is built.
    llm_provider: LLMProvider = Field(
        ...,
        description="LLM provider used for answer generation.",
    )
    llm_model: ModelName = Field(
        ...,
        description="Provider-specific model identifier.",
    )
    llm_base_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description=(
            "Optional provider API base URL override. "
            "Uses the selected provider's default when omitted."
        ),
    )
    llm_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Provider API key. Required for OpenRouter but not local Ollama.",
    )

    @model_validator(mode="after")
    def validate_llm_configuration(self) -> "Settings":
        """Validate provider-specific configuration requirements."""
        if self.llm_provider is LLMProvider.OPENROUTER:
            if (
                self.llm_api_key is None
                or not self.llm_api_key.get_secret_value().strip()
            ):
                raise ValueError(
                    "LLM_API_KEY is required when LLM_PROVIDER=openrouter"
                )

        return self


def load_settings() -> Settings:
    """Load and validate settings from environment and .env. Call at startup."""
    return Settings()
