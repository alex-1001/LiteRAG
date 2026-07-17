from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from openai import APIError

from app.config import Settings
from app.llm import (
    OLLAMA_DEFAULT_BASE_URL,
    OLLAMA_PLACEHOLDER_API_KEY,
    OPENROUTER_DEFAULT_BASE_URL,
    ModelProviderError,
    OpenAICompatibleChatModel,
    build_chat_model,
)


def _response_with_content(content):
    """Build the smallest OpenAI-compatible response shape used by the adapter."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


@pytest.fixture
def mock_client():
    """Return a client configured with a successful chat completion response."""
    client = Mock()
    client.chat.completions.create.return_value = _response_with_content(
        "  reply text  "
    )
    return client


class TestOpenAICompatibleChatModelRequest:
    def test_complete_sends_bound_model_messages_and_default_options(
        self, mock_client
    ):
        messages = [{"role": "user", "content": "hello"}]
        chat_model = OpenAICompatibleChatModel(mock_client, "test-model")

        result = chat_model.complete(messages)

        request = mock_client.chat.completions.create.call_args.kwargs
        assert request["model"] == "test-model"
        assert request["messages"] == messages
        assert request["temperature"] == 0.0
        assert request["max_tokens"] == 1000
        assert "response_format" not in request
        assert result == "reply text"

    def test_complete_sends_custom_generation_options(self, mock_client):
        chat_model = OpenAICompatibleChatModel(mock_client, "test-model")

        chat_model.complete(
            [{"role": "user", "content": "hello"}],
            temperature=0.4,
            max_tokens=250,
        )

        request = mock_client.chat.completions.create.call_args.kwargs
        assert request["temperature"] == 0.4
        assert request["max_tokens"] == 250

    def test_complete_enables_json_response_format_only_when_requested(
        self, mock_client
    ):
        chat_model = OpenAICompatibleChatModel(mock_client, "test-model")

        chat_model.complete(
            [{"role": "user", "content": "return JSON"}],
            json_mode=True,
        )

        request = mock_client.chat.completions.create.call_args.kwargs
        assert request["response_format"] == {"type": "json_object"}


class TestOpenAICompatibleChatModelResponse:
    def test_complete_strips_surrounding_whitespace(self, mock_client):
        chat_model = OpenAICompatibleChatModel(mock_client, "test-model")

        result = chat_model.complete(
            [{"role": "user", "content": "hello"}]
        )

        assert result == "reply text"

    @pytest.mark.parametrize(
        ("response", "message"),
        [
            (SimpleNamespace(), "no choices"),
            (SimpleNamespace(choices=[]), "no choices"),
            (
                SimpleNamespace(choices=[SimpleNamespace()]),
                "no message",
            ),
        ],
    )
    def test_complete_rejects_missing_response_parts(
        self, response, message
    ):
        client = Mock()
        client.chat.completions.create.return_value = response
        chat_model = OpenAICompatibleChatModel(client, "test-model")

        with pytest.raises(ModelProviderError, match=message):
            chat_model.complete(
                [{"role": "user", "content": "hello"}]
            )

    @pytest.mark.parametrize("content", [None, 42, ["not", "text"]])
    def test_complete_rejects_non_text_content(self, content):
        client = Mock()
        client.chat.completions.create.return_value = _response_with_content(
            content
        )
        chat_model = OpenAICompatibleChatModel(client, "test-model")

        with pytest.raises(ModelProviderError, match="no text content"):
            chat_model.complete(
                [{"role": "user", "content": "hello"}]
            )

    @pytest.mark.parametrize("content", ["", "   ", "\n\t"])
    def test_complete_rejects_empty_or_whitespace_content(self, content):
        client = Mock()
        client.chat.completions.create.return_value = _response_with_content(
            content
        )
        chat_model = OpenAICompatibleChatModel(client, "test-model")

        with pytest.raises(ModelProviderError, match="only whitespace"):
            chat_model.complete(
                [{"role": "user", "content": "hello"}]
            )

    def test_complete_translates_api_error_and_preserves_cause(self):
        client = Mock()
        api_error = APIError(
            "provider unavailable",
            request=Mock(),
            body=None,
        )
        client.chat.completions.create.side_effect = api_error
        chat_model = OpenAICompatibleChatModel(client, "test-model")

        with pytest.raises(
            ModelProviderError,
            match="Model provider request failed",
        ) as exc_info:
            chat_model.complete(
                [{"role": "user", "content": "hello"}]
            )

        assert exc_info.value.__cause__ is api_error


def _factory_settings(**overrides) -> Settings:
    """Build settings without reading values from the developer's environment."""
    values = {
        "embed_model_name": "test-embed",
        "data_dir": "/test/data",
        "llm_provider": "ollama",
        "llm_model": "test-model",
        "llm_base_url": None,
        "llm_api_key": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class TestBuildChatModel:
    @pytest.mark.parametrize(
        ("provider_settings", "expected_base_url", "expected_api_key"),
        [
            (
                {
                    "llm_provider": "openrouter",
                    "llm_api_key": "openrouter-secret",
                },
                OPENROUTER_DEFAULT_BASE_URL,
                "openrouter-secret",
            ),
            (
                {"llm_provider": "ollama"},
                OLLAMA_DEFAULT_BASE_URL,
                OLLAMA_PLACEHOLDER_API_KEY,
            ),
        ],
    )
    def test_uses_provider_defaults(
        self,
        provider_settings,
        expected_base_url,
        expected_api_key,
    ):
        settings = _factory_settings(**provider_settings)
        fake_client = Mock()
        fake_chat_model = Mock()

        with (
            patch("app.llm.OpenAI", return_value=fake_client) as mock_openai,
            patch(
                "app.llm.OpenAICompatibleChatModel",
                return_value=fake_chat_model,
            ) as mock_adapter,
        ):
            result = build_chat_model(settings)

        mock_openai.assert_called_once_with(
            base_url=expected_base_url,
            api_key=expected_api_key,
        )
        mock_adapter.assert_called_once_with(
            client=fake_client,
            model="test-model",
        )
        assert result is fake_chat_model

    @pytest.mark.parametrize(
        ("provider_settings", "expected_api_key"),
        [
            (
                {
                    "llm_provider": "openrouter",
                    "llm_api_key": "openrouter-secret",
                },
                "openrouter-secret",
            ),
            (
                {"llm_provider": "ollama"},
                OLLAMA_PLACEHOLDER_API_KEY,
            ),
        ],
    )
    def test_honors_base_url_override(
        self,
        provider_settings,
        expected_api_key,
    ):
        settings = _factory_settings(
            **provider_settings,
            llm_base_url="http://custom-provider.test:9000/v1",
        )
        fake_client = Mock()

        with patch("app.llm.OpenAI", return_value=fake_client) as mock_openai:
            result = build_chat_model(settings)

        mock_openai.assert_called_once_with(
            base_url="http://custom-provider.test:9000/v1",
            api_key=expected_api_key,
        )
        assert isinstance(result, OpenAICompatibleChatModel)

    def test_rejects_openrouter_settings_without_api_key_defensively(self):
        valid_settings = _factory_settings(
            llm_provider="openrouter",
            llm_api_key="openrouter-secret",
        )
        invalid_settings = valid_settings.model_copy(
            update={"llm_api_key": None}
        )

        with pytest.raises(ValueError, match="OpenRouter API key is missing"):
            build_chat_model(invalid_settings)
