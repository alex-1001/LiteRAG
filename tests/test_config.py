import pytest
from pydantic import ValidationError

from app.config import Settings, load_settings


class TestSettings:
    """Test Settings model and load_settings()."""

    def test_load_settings_reads_env_and_returns_settings(self, monkeypatch):
        """load_settings() returns Settings with values from environment."""
        monkeypatch.setenv("EMBED_MODEL_NAME", "test-model")
        monkeypatch.delenv("TOKENIZER_NAME", raising=False)
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_MODEL", "test-llm")
        monkeypatch.setenv("STORAGE_DIR", "./.vectorstore")
        settings = load_settings()
        assert settings.embed_model_name == "test-model"
        assert settings.openrouter_base_url == "https://api.example.com"
        assert settings.openrouter_api_key == "test-key"
        assert settings.openrouter_model == "test-llm"
        assert settings.chunk_size == 200
        assert settings.chunk_overlap == 50
        assert settings.top_k == 5
        assert settings.storage_dir == "./.vectorstore"
        assert settings.tokenizer_name is None

    def test_settings_declares_expected_defaults(self):
        """Settings model declares the expected default values for optional fields (env/.env can override at runtime)."""
        # Test the declared defaults in the model, not runtime loading—avoids flakiness from .env/env.
        assert Settings.model_fields["chunk_size"].default == 200
        assert Settings.model_fields["chunk_overlap"].default == 50
        assert Settings.model_fields["top_k"].default == 5
        assert Settings.model_fields["storage_dir"].default == "./.vectorstore"
        assert Settings.model_fields["data_dir"].default is None
        assert Settings.model_fields["tokenizer_name"].default is None
        assert Settings.model_fields["openrouter_base_url"].default is None
        assert Settings.model_fields["openrouter_api_key"].default is None
        assert Settings.model_fields["openrouter_model"].default is None

    def test_settings_validation_rejects_invalid_chunk_size(self):
        """Invalid chunk_size (e.g. 0) fails validation when building Settings."""
        with pytest.raises(ValidationError):
            Settings(
                embed_model_name="m",
                openrouter_base_url="u",
                openrouter_api_key="k",
                openrouter_model="m",
                chunk_size=0,
                chunk_overlap=50,
            )

    def test_tokenizer_name_optional_defaults_to_none(self, monkeypatch):
        """When TOKENIZER_NAME is not set, tokenizer_name is None (embed_model_name used at runtime)."""
        monkeypatch.setenv("EMBED_MODEL_NAME", "test-embed")
        monkeypatch.delenv("TOKENIZER_NAME", raising=False)
        settings = load_settings()
        assert settings.tokenizer_name is None

    def test_tokenizer_name_read_when_set(self, monkeypatch):
        """When TOKENIZER_NAME is set, load_settings returns it."""
        monkeypatch.setenv("EMBED_MODEL_NAME", "test-embed")
        monkeypatch.setenv("TOKENIZER_NAME", "bert-base-uncased")
        settings = load_settings()
        assert settings.tokenizer_name == "bert-base-uncased"
