"""Interfaces for model providers"""
from typing import Literal, Protocol, Sequence, TypedDict
from openai import APIError, OpenAI

class ModelProviderError(RuntimeError):
    """Base exception for model failing to return useful completion."""
    pass

class ChatMessage(TypedDict):
    """Message for chat client"""
    role: Literal["system", "user", "assistant"]
    content: str
    
class ChatModel(Protocol):
    """Application contract for chat completion backends"""
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
    """Chat model backed by an OpenAI-compatible chat completions API"""
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