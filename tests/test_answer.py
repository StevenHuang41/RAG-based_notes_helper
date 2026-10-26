from collections.abc import Iterator

import pytest

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag import answer as answer_module
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE, rag_answer


@pytest.fixture
def mock_dependencies(monkeypatch):
    class DummyLLM:
        def generate(self, prompt, line_width):
            return "generate dummy answer"

        def stream(self, prompt, line_width) -> Iterator:
            yield "stream"
            yield "dummy"
            yield "answer"

    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: DummyLLM(),
    )


@pytest.fixture
def mock_hits():
    hits = [
        {
            "text": "hits content",
            "source": "note.md",
            "chunk_id": 0,
            "score": 0.9,
        }
    ]
    return hits


def test_rag_answer_generate(monkeypatch, mock_dependencies, mock_hits):

    monkeypatch.setenv("STREAM", "false")

    result = rag_answer(query="q", hits=mock_hits)

    assert isinstance(result["answer"], str)
    assert result["answer"] == "generate dummy answer"
    assert result["citations"][0]["source"] == "note.md"


def test_rag_answer_returns_the_ordered_context_used_by_the_prompt(monkeypatch):
    hits = [
        {
            "source": "notes/first.md",
            "chunk_id": 4,
            "start_line": 8,
            "end_line": 9,
            "score": 0.95,
            "text": "Owner: Mina Chen.",
        },
        {
            "source": "notes/second.md",
            "chunk_id": 2,
            "start_line": 21,
            "end_line": 22,
            "score": 0.82,
            "text": "Launch date: 2026-07-14.",
        },
        {
            "source": "notes/third.md",
            "chunk_id": 7,
            "start_line": 3,
            "end_line": 3,
            "score": 0.51,
            "text": "This candidate is outside the answer limit.",
        },
    ]
    prompts = []

    class PromptLLM:
        def generate(self, prompt, line_width):
            prompts.append(prompt)
            return "Mina Chen; 2026-07-14."

    monkeypatch.setenv("LLM_MAX_CHUNKS", "2")
    monkeypatch.setenv("TOP_K", "2")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: PromptLLM(),
    )
    try:
        result = rag_answer("When does Mina's project launch?", hits=hits)
    finally:
        get_settings.cache_clear()

    expected_context = [
        {
            "source": "notes/first.md",
            "chunk_id": 4,
            "start_line": 8,
            "end_line": 9,
            "text": "Owner: Mina Chen.",
            "prompt_block": "file:notes/first.md\nOwner: Mina Chen.",
        },
        {
            "source": "notes/second.md",
            "chunk_id": 2,
            "start_line": 21,
            "end_line": 22,
            "text": "Launch date: 2026-07-14.",
            "prompt_block": (
                "file:notes/second.md\nLaunch date: 2026-07-14."
            ),
        },
    ]

    assert result["generation_context"] == expected_context
    prompt_content = prompts[0][1]["content"]
    assert (
        "Context: "
        "file:notes/first.md\nOwner: Mina Chen.\n\n"
        "file:notes/second.md\nLaunch date: 2026-07-14."
    ) in prompt_content
    assert "This candidate is outside the answer limit." not in prompt_content


def test_rag_answer_does_not_assume_the_user_owns_the_notes(
    monkeypatch,
    mock_hits,
):
    captured_prompts = []

    class PromptLLM:
        def generate(self, prompt, line_width):
            captured_prompts.append(prompt)
            return "I cannot find the answers in the notes."

    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: PromptLLM(),
    )

    rag_answer(query="Who am I?", hits=mock_hits)

    system_prompt = captured_prompts[0][0]["content"]
    assert "You are the owner of the notes" not in system_prompt
    assert "Who am I" in system_prompt


def test_rag_answer_stream(monkeypatch, mock_dependencies, mock_hits):

    monkeypatch.setenv("STREAM", "true")

    result = rag_answer(query="q", hits=mock_hits, stream=True)
    stream_answer = result["answer"]

    assert next(stream_answer) == "stream"
    assert next(stream_answer) == "dummy"
    assert next(stream_answer) == "answer"

    with pytest.raises(StopIteration):
        next(stream_answer)

    assert result["citations"][0]["source"] == "note.md"


def test_rag_answer_reports_provider_tokens_and_output_characters(
    monkeypatch,
    mock_hits,
):
    generation_samples = iter([4.0, 4.0125])
    monkeypatch.setattr(
        answer_module,
        "perf_counter",
        lambda: next(generation_samples),
    )

    class UsageLLM:
        provider_name = "fixture"
        model = "fixture-answer"
        last_usage = {
            "status": "available",
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
        }
        last_output = "raw answer"

        def generate(self, prompt, line_width):
            return "raw answer"

    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: UsageLLM(),
    )
    calls = []

    rag_answer(
        query="q",
        hits=mock_hits,
        on_generation=calls.append,
    )

    assert len(calls) == 1
    assert calls[0]["call_type"] == "answer"
    assert calls[0]["latency_ms"] == 12.5
    assert calls[0]["provider"] == "fixture"
    assert calls[0]["model"] == "fixture-answer"
    assert calls[0]["token_usage"] == {
        "status": "available",
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
    }
    assert calls[0]["characters"]["input"] > 0
    assert calls[0]["characters"]["output"] == 10


@pytest.mark.parametrize("stream", [False, True])
def test_rag_answer_empty_hits_abstains_without_calling_llm(
    monkeypatch,
    stream,
):
    def unexpected_llm_call():
        pytest.fail("LLM should not be called when retrieval returns no chunks")

    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        unexpected_llm_call,
    )

    result = rag_answer(query="q", hits=[], stream=stream)

    assert result == {
        "answer": ABSTENTION_MESSAGE,
        "citations": [],
    }
