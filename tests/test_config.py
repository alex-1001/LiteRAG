import pytest
from pydantic import ValidationError

from app.config import LLMProvider, Settings, load_settings


def _valid_settings(**overrides) -> Settings:
    """Build isolated valid settings without reading the developer's .env file."""
    values = {
        "embed_model_name": "test-embed",
        "data_dir": "./data",
        "llm_provider": LLMProvider.OLLAMA,
        "llm_model": "test-llm",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _set_required_env(monkeypatch) -> None:
    """Set the fields load_settings() requires for unrelated env-loading tests."""
    monkeypatch.setenv("EMBED_MODEL_NAME", "test-embed")
    monkeypatch.setenv("DATA_DIR", "./data")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "test-llm")


class TestSettings:
    """Test Settings validation, defaults, and environment loading."""

    def test_load_settings_reads_neutral_llm_environment(self, monkeypatch):
        monkeypatch.setenv("EMBED_MODEL_NAME", "test-embed")
        monkeypatch.setenv("DATA_DIR", "./data")
        monkeypatch.setenv("STORAGE_DIR", "./.vectorstore")
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("LLM_MODEL", "provider/test-model")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("TOKENIZER_NAME", raising=False)

        settings = load_settings()

        assert settings.embed_model_name == "test-embed"
        assert settings.data_dir == "./data"
        assert settings.storage_dir == "./.vectorstore"
        assert settings.tokenizer_name is None
        assert settings.llm_provider is LLMProvider.OPENROUTER
        assert settings.llm_model == "provider/test-model"
        assert str(settings.llm_base_url) == "https://api.example.com/v1"
        assert settings.llm_api_key is not None
        assert settings.llm_api_key.get_secret_value() == "test-key"

    def test_settings_declares_expected_defaults_and_required_fields(self):
        assert Settings.model_fields["chunk_size"].default == 200
        assert Settings.model_fields["chunk_overlap"].default == 50
        assert Settings.model_fields["top_k"].default == 5
        assert Settings.model_fields["storage_dir"].default == "./.vectorstore"
        assert Settings.model_fields["tokenizer_name"].default is None
        assert Settings.model_fields["llm_base_url"].default is None
        assert Settings.model_fields["llm_api_key"].default is None

        assert Settings.model_fields["data_dir"].is_required()
        assert Settings.model_fields["embed_model_name"].is_required()
        assert Settings.model_fields["llm_provider"].is_required()
        assert Settings.model_fields["llm_model"].is_required()

    def test_settings_requires_data_dir(self):
        with pytest.raises(ValidationError):
            Settings(
                _env_file=None,
                embed_model_name="test-embed",
                llm_provider="ollama",
                llm_model="test-llm",
            )

    @pytest.mark.parametrize("missing_field", ["llm_provider", "llm_model"])
    def test_settings_requires_llm_provider_and_model(self, missing_field):
        values = {
            "embed_model_name": "test-embed",
            "data_dir": "./data",
            "llm_provider": "ollama",
            "llm_model": "test-llm",
        }
        del values[missing_field]

        with pytest.raises(ValidationError):
            Settings(_env_file=None, **values)

    def test_settings_strips_model_name(self):
        settings = _valid_settings(llm_model="  test-model  ")

        assert settings.llm_model == "test-model"

    def test_settings_rejects_whitespace_only_model_name(self):
        with pytest.raises(ValidationError):
            _valid_settings(llm_model="   ")

    def test_settings_rejects_unknown_provider(self):
        with pytest.raises(ValidationError):
            _valid_settings(llm_provider="unknown-provider")

    @pytest.mark.parametrize("api_key", [None, "", "   "])
    def test_openrouter_requires_nonempty_api_key(self, api_key):
        with pytest.raises(
            ValidationError,
            match="LLM_API_KEY is required",
        ):
            _valid_settings(
                llm_provider="openrouter",
                llm_api_key=api_key,
            )

    def test_openrouter_accepts_nonempty_api_key(self):
        settings = _valid_settings(
            llm_provider="openrouter",
            llm_api_key="secret",
        )

        assert settings.llm_provider is LLMProvider.OPENROUTER
        assert settings.llm_api_key is not None
        assert settings.llm_api_key.get_secret_value() == "secret"

    def test_ollama_does_not_require_api_key(self):
        settings = _valid_settings(
            llm_provider="ollama",
            llm_api_key=None,
        )

        assert settings.llm_provider is LLMProvider.OLLAMA
        assert settings.llm_api_key is None

    def test_settings_accepts_valid_base_url(self):
        settings = _valid_settings(
            llm_base_url="http://localhost:11434/v1",
        )

        assert str(settings.llm_base_url) == "http://localhost:11434/v1"

    def test_settings_rejects_invalid_base_url(self):
        with pytest.raises(ValidationError):
            _valid_settings(llm_base_url="not-a-url")

    def test_settings_rejects_invalid_chunk_size(self):
        with pytest.raises(ValidationError):
            _valid_settings(
                chunk_size=0,
                chunk_overlap=50,
            )

    def test_tokenizer_name_optional_defaults_to_none(self):
        settings = _valid_settings(tokenizer_name=None)

        assert settings.tokenizer_name is None

    def test_tokenizer_name_read_from_environment(self, monkeypatch):
        _set_required_env(monkeypatch)
        monkeypatch.setenv("TOKENIZER_NAME", "bert-base-uncased")

        settings = load_settings()

        assert settings.tokenizer_name == "bert-base-uncased"
