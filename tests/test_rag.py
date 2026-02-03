import json
import pytest
from unittest.mock import patch, Mock

from app.rag import (
    format_context,
    create_rag_messages,
    call_llm,
    answer_query,
    SYSTEM_PROMPT,
)
from openai import APIError
from tests.conftest import _make_chunk


class TestFormatContext:
    """Test suite for format_context()."""

    def test_empty_list_contains_begin_and_end_context(self):
        """Result contains BEGIN CONTEXT and END CONTEXT with no citation blocks."""
        result = format_context([])
        assert "BEGIN CONTEXT" in result
        assert "END CONTEXT" in result
        assert "[citation_id:" not in result

    def test_single_chunk_has_citation_block(self):
        """Single chunk produces one block with [citation_id: 1]:, text, and [/chunk]."""
        chunk = _make_chunk("chunk text here", chunk_id="0")
        result = format_context([chunk])
        assert "[citation_id: 1]:" in result
        assert "chunk text here" in result
        assert "[/chunk]" in result
        assert "BEGIN CONTEXT" in result
        assert "END CONTEXT" in result

    def test_multiple_chunks_ordered_with_citation_ids(self):
        """Multiple chunks get citation IDs 1, 2, ... N and appear in order."""
        chunks = [
            _make_chunk("first", chunk_id="0"),
            _make_chunk("second", chunk_id="1"),
            _make_chunk("third", chunk_id="2"),
        ]
        result = format_context(chunks)
        assert "[citation_id: 1]:" in result
        assert "[citation_id: 2]:" in result
        assert "[citation_id: 3]:" in result
        assert result.index("first") < result.index("second") < result.index("third")

    def test_chunk_text_preserved(self):
        """Newlines and special characters in chunk text appear unchanged."""
        chunk = _make_chunk("Line one\nLine two\n\tindented", chunk_id="0")
        result = format_context([chunk])
        assert "Line one\nLine two\n\tindented" in result


class TestCreateRagMessages:
    """Test suite for create_rag_messages()."""

    def test_returns_two_messages_system_and_user(self):
        """Returns list of two dicts with role and content for system and user."""
        messages = create_rag_messages("Q", "C")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "content" in messages[0]
        assert messages[1]["role"] == "user"
        assert "content" in messages[1]

    def test_system_message_equals_system_prompt(self):
        """System message content equals SYSTEM_PROMPT."""
        messages = create_rag_messages("Q", "C")
        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_user_message_contains_question_and_context(self):
        """User message is template with question and context substituted."""
        messages = create_rag_messages("my question?", "my context block")
        user_content = messages[1]["content"]
        assert "my question?" in user_content
        assert "my context block" in user_content


class TestCallLlm:
    """Test suite for call_llm()."""

    def test_client_provided_uses_it_and_returns_content(self):
        """When client is provided, OpenAI is not called; create() called with correct args; returns content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "  reply text  "
        mock_client.chat.completions.create.return_value = mock_response

        result = call_llm([{"role": "user", "content": "hi"}], client=mock_client, return_json=False)

        mock_client.chat.completions.create.assert_called_once()
        call_kw = mock_client.chat.completions.create.call_args[1]
        assert call_kw["messages"] == [{"role": "user", "content": "hi"}]
        assert call_kw["temperature"] == 0.0
        assert call_kw["max_tokens"] == 1000
        assert call_kw["response_format"] is None
        assert result == "reply text"

    def test_client_provided_return_json_sends_json_object_format(self):
        """When return_json=True, create() is called with response_format json_object."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value.choices = [Mock()]
        mock_client.chat.completions.create.return_value.choices[0].message.content = "{}"

        call_llm([], client=mock_client, return_json=True)

        call_kw = mock_client.chat.completions.create.call_args[1]
        assert call_kw["response_format"] == {"type": "json_object"}

    def test_client_none_uses_env_for_openai_and_model(self):
        """When client is None, OpenAI() gets env vars and create() gets model from env."""
        mock_openai_instance = Mock()
        mock_openai_instance.chat.completions.create.return_value.choices = [Mock()]
        mock_openai_instance.chat.completions.create.return_value.choices[0].message.content = "ok"

        with patch("app.rag.OpenAI", return_value=mock_openai_instance) as mock_openai_cls, patch(
            "app.rag.os.environ.get",
            side_effect=lambda k, d=None: {"OPENROUTER_BASE_URL": "https://x", "OPENROUTER_API_KEY": "k", "OPENROUTER_MODEL": "my-model"}.get(k, d),
        ):
            result = call_llm([{"role": "user", "content": "q"}], client=None)

        mock_openai_cls.assert_called_once_with(base_url="https://x", api_key="k")
        mock_openai_instance.chat.completions.create.assert_called_once()
        call_kw = mock_openai_instance.chat.completions.create.call_args[1]
        assert call_kw["model"] == "my-model"
        assert result == "ok"

    def test_returns_empty_string_when_content_empty(self):
        """When response content is empty or whitespace, returns empty string."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value.choices = [Mock()]
        mock_client.chat.completions.create.return_value.choices[0].message.content = "   "

        result = call_llm([], client=mock_client)
        assert result == ""

    def test_api_error_propagated(self):
        """When create() raises APIError, call_llm re-raises APIError."""
        mock_client = Mock()
        # APIError requires (message, request, body=None)
        mock_client.chat.completions.create.side_effect = APIError(
            "network error", request=Mock(), body=None
        )

        with pytest.raises(APIError, match="Error calling LLM"):
            call_llm([], client=mock_client)

        mock_client.chat.completions.create.assert_called_once()


class TestAnswerQuery:
    """Test suite for answer_query()."""

    def test_empty_query_raises_value_error(self):
        """Empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            answer_query("", [_make_chunk("x")])

    def test_empty_retrieved_chunks_returns_fixed_message_without_calling_llm(self):
        """Empty retrieved_chunks returns fixed message and [] without calling call_llm."""
        with patch("app.rag.call_llm") as mock_call_llm:
            answer, cited = answer_query("something", [])
        assert answer == "No relevant documents are associated with this query."
        assert cited == []
        mock_call_llm.assert_not_called()

    def test_happy_path_returns_answer_and_cited_sources(self):
        """Valid JSON from call_llm returns (answer, cited_sources) and call_llm called with messages from format_context."""
        chunks = [_make_chunk("a"), _make_chunk("b")]
        with patch("app.rag.call_llm") as mock_call_llm:
            mock_call_llm.return_value = json.dumps({"answer": "Yes.", "cited_sources": [1, 2]})
            answer, cited = answer_query("Q?", chunks)
        assert answer == "Yes."
        assert cited == [1, 2]
        mock_call_llm.assert_called_once()
        messages = mock_call_llm.call_args[0][0]
        assert len(messages) == 2
        user_content = messages[1]["content"]
        assert "[citation_id: 1]:" in user_content
        assert "[citation_id: 2]:" in user_content
        assert "a" in user_content
        assert "b" in user_content
        assert "Q?" in user_content

    def test_invalid_json_raises_value_error(self):
        """call_llm returning non-JSON raises ValueError."""
        with patch("app.rag.call_llm", return_value="not json at all"):
            with pytest.raises(ValueError, match="LLM returned invalid JSON"):
                answer_query("Q", [_make_chunk("x")])

    def test_missing_answer_or_cited_sources_raises_value_error(self):
        """JSON missing answer or cited_sources raises ValueError."""
        with patch("app.rag.call_llm", return_value=json.dumps({"answer": "x"})):
            with pytest.raises(ValueError, match="missing 'answer' or 'cited_sources'"):
                answer_query("Q", [_make_chunk("x")])
        with patch("app.rag.call_llm", return_value=json.dumps({"cited_sources": []})):
            with pytest.raises(ValueError, match="missing 'answer' or 'cited_sources'"):
                answer_query("Q", [_make_chunk("x")])

    def test_empty_answer_raises_value_error(self):
        """Empty answer string raises ValueError."""
        with patch("app.rag.call_llm", return_value=json.dumps({"answer": "", "cited_sources": []})):
            with pytest.raises(ValueError, match="LLM answer is empty"):
                answer_query("Q", [_make_chunk("x")])

    def test_answer_not_string_raises_type_error(self):
        """answer not a string raises TypeError."""
        with patch("app.rag.call_llm", return_value=json.dumps({"answer": 42, "cited_sources": []})):
            with pytest.raises(TypeError, match="LLM answer must be a string"):
                answer_query("Q", [_make_chunk("x")])

    def test_cited_sources_not_list_raises_type_error(self):
        """cited_sources not a list raises TypeError."""
        with patch("app.rag.call_llm", return_value=json.dumps({"answer": "ok", "cited_sources": "1,2"})):
            with pytest.raises(TypeError, match="LLM cited sources must be a list"):
                answer_query("Q", [_make_chunk("x")])

    def test_cited_sources_non_integer_raises_type_error(self):
        """cited_sources containing non-integer raises TypeError."""
        with patch("app.rag.call_llm", return_value=json.dumps({"answer": "ok", "cited_sources": [1, "2"]})):
            with pytest.raises(TypeError, match="LLM cited sources must be a list of integers"):
                answer_query("Q", [_make_chunk("a"), _make_chunk("b")])

    def test_cited_sources_out_of_range_raises_value_error(self):
        """Citation id <= 0 or > len(retrieved_chunks) raises ValueError."""
        chunks = [_make_chunk("a"), _make_chunk("b")]
        with patch("app.rag.call_llm", return_value=json.dumps({"answer": "ok", "cited_sources": [0]})):
            with pytest.raises(ValueError, match="within the range"):
                answer_query("Q", chunks)
        with patch("app.rag.call_llm", return_value=json.dumps({"answer": "ok", "cited_sources": [3]})):
            with pytest.raises(ValueError, match="within the range"):
                answer_query("Q", chunks)
