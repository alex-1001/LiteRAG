"""Interfaces and construction policy for model providers."""

from typing import Literal, Protocol, Sequence, TypedDict

from openai import APIError, OpenAI

from app.config import LLMProvider, Settings

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_PLACEHOLDER_API_KEY = "ollama"


class ModelProviderError(RuntimeError):
    """Base exception for model failing to return useful completion."""

    pass


class ChatMessage(TypedDict):
    """Message for chat client."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatModel(Protocol):
    """Application contract for chat completion backends."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a useful chat completion."""
        ...


class OpenAICompatibleChatModel:
    """Chat model backed by an OpenAI-compatible chat completions API."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        request = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            request["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**request)
        except APIError as e:
            raise ModelProviderError("Model provider request failed.") from e

        choices = getattr(response, "choices", None)
        if not choices:
            raise ModelProviderError("Model response contained no choices.")

        first_choice = choices[0]

        message = getattr(first_choice, "message", None)
        if message is None:
            raise ModelProviderError("Model response contained no message.")

        raw_content = getattr(message, "content", None)
        if not isinstance(raw_content, str):
            raise ModelProviderError("Model response contained no text content.")

        content = raw_content.strip()
        if not content:
            raise ModelProviderError("Model response contained only whitespace.")

        return content


def build_chat_model(settings: Settings) -> ChatModel:
    """Build the configured chat adapter and resolve provider defaults."""
    if settings.llm_provider is LLMProvider.OPENROUTER:
        base_url = (
            str(settings.llm_base_url)
            if settings.llm_base_url is not None
            else OPENROUTER_DEFAULT_BASE_URL
        )

        if settings.llm_api_key is None:
            raise ValueError("OpenRouter API key is missing")

        api_key = settings.llm_api_key.get_secret_value()

    elif settings.llm_provider is LLMProvider.OLLAMA:
        base_url = (
            str(settings.llm_base_url)
            if settings.llm_base_url is not None
            else OLLAMA_DEFAULT_BASE_URL
        )
        # The OpenAI client requires a value, but Ollama ignores it.
        api_key = OLLAMA_PLACEHOLDER_API_KEY

    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    client = OpenAI(base_url=base_url, api_key=api_key)

    return OpenAICompatibleChatModel(client=client, model=settings.llm_model)
