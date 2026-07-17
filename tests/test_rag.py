import json

import pytest
from pydantic import ValidationError

from app.llm import ModelProviderError
from app.rag import (
    InvalidModelResponse,
    SYSTEM_PROMPT,
    answer_query,
    create_rag_messages,
    format_context,
)
from tests.conftest import _make_chunk


class FakeChatModel:
    """Controllable test implementation of the application's ChatModel contract."""

    def __init__(
        self,
        response: str = "",
        *,
        error: Exception | None = None,
    ):
        self.response = response
        self.error = error
        self.calls = []

    def complete(self, messages, **options):
        self.calls.append({"messages": messages, "options": options})
        if self.error is not None:
            raise self.error
        return self.response


class TestFormatContext:
    """Test suite for format_context()."""

    def test_empty_list_contains_begin_and_end_context(self):
        result = format_context([])

        assert "BEGIN CONTEXT" in result
        assert "END CONTEXT" in result
        assert "[citation_id:" not in result

    def test_single_chunk_has_citation_block(self):
        chunk = _make_chunk("chunk text here", chunk_id="0")

        result = format_context([chunk])

        assert "[citation_id: 1]:" in result
        assert "chunk text here" in result
        assert "[/chunk]" in result
        assert "BEGIN CONTEXT" in result
        assert "END CONTEXT" in result

    def test_multiple_chunks_ordered_with_citation_ids(self):
        chunks = [
            _make_chunk("first", chunk_id="0"),
            _make_chunk("second", chunk_id="1"),
            _make_chunk("third", chunk_id="2"),
        ]

        result = format_context(chunks)

        assert "[citation_id: 1]:" in result
        assert "[citation_id: 2]:" in result
        assert "[citation_id: 3]:" in result
        assert result.index("first") < result.index("second") < result.index(
            "third"
        )

    def test_chunk_text_preserved(self):
        chunk = _make_chunk(
            "Line one\nLine two\n\tindented",
            chunk_id="0",
        )

        result = format_context([chunk])

        assert "Line one\nLine two\n\tindented" in result


class TestCreateRagMessages:
    """Test suite for create_rag_messages()."""

    def test_returns_two_messages_system_and_user(self):
        messages = create_rag_messages("Q", "C")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "content" in messages[0]
        assert messages[1]["role"] == "user"
        assert "content" in messages[1]

    def test_system_message_equals_system_prompt(self):
        messages = create_rag_messages("Q", "C")

        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_user_message_contains_question_and_context(self):
        messages = create_rag_messages("my question?", "my context block")

        user_content = messages[1]["content"]
        assert "my question?" in user_content
        assert "my context block" in user_content


class TestAnswerQuery:
    """Test answer_query() through the application-owned ChatModel boundary."""

    def test_empty_query_raises_without_calling_model(self):
        chat_model = FakeChatModel()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            answer_query("", [_make_chunk("x")], chat_model)

        assert chat_model.calls == []

    def test_empty_retrieval_returns_fixed_answer_without_calling_model(self):
        chat_model = FakeChatModel()

        answer, cited = answer_query("something", [], chat_model)

        assert answer == "No relevant documents are associated with this query."
        assert cited == []
        assert chat_model.calls == []

    def test_valid_response_returns_answer_and_citations(self):
        chunks = [_make_chunk("a"), _make_chunk("b")]
        chat_model = FakeChatModel(
            json.dumps(
                {
                    "answer": "Yes.",
                    "cited_sources": [1, 2],
                }
            )
        )

        answer, cited = answer_query("Q?", chunks, chat_model)

        assert answer == "Yes."
        assert cited == [1, 2]

    def test_sends_rag_messages_and_requests_json_mode(self):
        chunks = [_make_chunk("a"), _make_chunk("b")]
        chat_model = FakeChatModel(
            json.dumps(
                {
                    "answer": "Yes.",
                    "cited_sources": [1],
                }
            )
        )

        answer_query("Q?", chunks, chat_model)

        assert len(chat_model.calls) == 1
        call = chat_model.calls[0]
        assert call["options"] == {"json_mode": True}

        messages = call["messages"]
        assert len(messages) == 2
        user_content = messages[1]["content"]
        assert "[citation_id: 1]:" in user_content
        assert "[citation_id: 2]:" in user_content
        assert "a" in user_content
        assert "b" in user_content
        assert "Q?" in user_content

    def test_normalizes_answer_whitespace(self):
        chat_model = FakeChatModel(
            json.dumps(
                {
                    "answer": "  Answer with surrounding space.  ",
                    "cited_sources": [],
                }
            )
        )

        answer, cited = answer_query(
            "Q?",
            [_make_chunk("context")],
            chat_model,
        )

        assert answer == "Answer with surrounding space."
        assert cited == []

    @pytest.mark.parametrize(
        "response",
        [
            "not json at all",
            json.dumps({"answer": "missing citations"}),
            json.dumps({"cited_sources": []}),
            json.dumps(
                {
                    "answer": "extra field",
                    "cited_sources": [],
                    "extra": True,
                }
            ),
            json.dumps({"answer": "", "cited_sources": []}),
            json.dumps({"answer": "   ", "cited_sources": []}),
            json.dumps({"answer": 42, "cited_sources": []}),
            json.dumps({"answer": "ok", "cited_sources": "1,2"}),
            json.dumps({"answer": "ok", "cited_sources": ["1"]}),
            json.dumps({"answer": "ok", "cited_sources": [True]}),
            json.dumps({"answer": "ok", "cited_sources": [0]}),
            json.dumps({"answer": "ok", "cited_sources": [-1]}),
        ],
        ids=[
            "invalid-json",
            "missing-citations",
            "missing-answer",
            "extra-field",
            "empty-answer",
            "whitespace-answer",
            "non-string-answer",
            "citations-not-list",
            "string-citation",
            "boolean-citation",
            "zero-citation",
            "negative-citation",
        ],
    )
    def test_invalid_structured_response_raises_owned_error(
        self,
        response,
    ):
        chat_model = FakeChatModel(response)

        with pytest.raises(
            InvalidModelResponse,
            match="invalid RAG response shape",
        ) as exc_info:
            answer_query("Q", [_make_chunk("x")], chat_model)

        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_citation_above_retrieved_range_raises_owned_error(self):
        chat_model = FakeChatModel(
            json.dumps(
                {
                    "answer": "ok",
                    "cited_sources": [3],
                }
            )
        )
        chunks = [_make_chunk("a"), _make_chunk("b")]

        with pytest.raises(
            InvalidModelResponse,
            match="outside the retrieved chunk range",
        ):
            answer_query("Q", chunks, chat_model)

    def test_model_provider_error_propagates_unchanged(self):
        provider_error = ModelProviderError("provider failed")
        chat_model = FakeChatModel(error=provider_error)

        with pytest.raises(ModelProviderError) as exc_info:
            answer_query("Q", [_make_chunk("x")], chat_model)

        assert exc_info.value is provider_error
