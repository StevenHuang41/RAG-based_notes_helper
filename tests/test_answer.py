from collections.abc import Iterator

import pytest

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag import answer as answer_module
from rag_notes_helper.rag.answer import rag_answer
from rag_notes_helper.rag.retrieval import HybridSearchResults


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
    monkeypatch.setenv("FUSION_TOP_K", "2")
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


def test_rag_answer_labels_history_as_non_evidence(monkeypatch, mock_hits):
    prompts = []

    class PromptLLM:
        def generate(self, prompt, line_width):
            prompts.append(prompt)
            return "The project starts in April."

    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: PromptLLM(),
    )

    rag_answer(
        "When does it start?",
        hits=mock_hits,
        conversation_history=[("When does it start?", "In April.")],
    )

    prompt = prompts[0]
    assert "Conversation history (intent context only; not evidence)" in (
        prompt[1]["content"]
    )
    assert "When does it start?" in prompt[1]["content"]
    assert "In April." in prompt[1]["content"]
    assert "Use only the provided Context for factual claims" in prompt[0]["content"]


@pytest.mark.parametrize("stream", [False, True])
def test_rag_answer_keeps_newest_five_complete_turns_with_long_history(
    monkeypatch, mock_hits, stream
):
    prompts = []

    class PromptLLM:
        def generate(self, prompt, line_width):
            prompts.append(prompt)
            return "answer"

        def stream(self, prompt, line_width):
            prompts.append(prompt)
            yield "answer"

    class HistorySettings:
        line_width = 80
        llm = type("LLM", (), {
            "max_chunks": 5,
            "max_tokens": 20,

        })()

    monkeypatch.setattr(answer_module, "get_settings", lambda: HistorySettings())
    monkeypatch.setattr(answer_module, "get_llm", lambda: PromptLLM())
    history = [("excluded question", "excluded answer")] + [
        (f"question {i} " + "x" * 2000, f"answer {i} " + "y" * 2000)
        for i in range(5)
    ]
    result = rag_answer(
        "current question", hits=mock_hits,
        conversation_history=history, stream=stream,
    )
    if stream:
        assert "".join(result["answer"]) == "answer"
    prompt_content = prompts[0][1]["content"]
    assert "excluded question" not in prompt_content
    assert "excluded answer" not in prompt_content
    for user, assistant in history[-5:]:
        assert user in prompt_content
        assert assistant in prompt_content


def test_rag_answer_sends_only_top_reranked_evidence_and_preserves_scores(
    monkeypatch,
):
    reranked_hits = [
        {
            "source": "notes/launch.md",
            "chunk_id": 4,
            "start_line": 8,
            "end_line": 9,
            "score": 0.03,
            "reranker_score": 0.91,
            "text": "Owner: Mina Chen.",
        },
        {
            "source": "notes/launch.md",
            "chunk_id": 5,
            "start_line": 10,
            "end_line": 11,
            "score": 0.02,
            "reranker_score": 0.77,
            "text": "Launch date: 2026-07-14.",
        },
        {
            "source": "notes/chatter.md",
            "chunk_id": 1,
            "start_line": 1,
            "end_line": 1,
            "score": 0.01,
            "reranker_score": -0.25,
            "text": "Unrelated chatter that must not reach generation.",
        },
    ]
    hits = HybridSearchResults(
        vector_results=reranked_hits,
        bm25_results=reranked_hits,
        fused_results=reranked_hits,
        reranked_results=reranked_hits,
        generation_top_k=2,
        stage_latencies_ms={
            "vector": 0.1,
            "bm25": 0.2,
            "rrf": 0.0,
            "reranked": 0.3,
        },
        vector_top_k=3,
        bm25_top_k=3,
        fusion_top_k=3,
    )
    prompts = []

    class PromptLLM:
        def generate(self, prompt, line_width):
            prompts.append(prompt)
            context = prompt[1]["content"]
            assert "Owner: Mina Chen." in context
            assert "Launch date: 2026-07-14." in context
            assert "Unrelated chatter" not in context
            return "Mina Chen's project launches on 2026-07-14."

    monkeypatch.setenv("LLM_MAX_CHUNKS", "2")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: PromptLLM(),
    )
    try:
        result = rag_answer("Who owns the project and when does it launch?", hits=hits)
    finally:
        get_settings.cache_clear()

    assert result["answer"] == "Mina Chen's project launches on 2026-07-14."
    assert result["generation_context"] == [
        {
            "source": "notes/launch.md",
            "chunk_id": 4,
            "start_line": 8,
            "end_line": 9,
            "reranker_score": 0.91,
            "text": "Owner: Mina Chen.",
            "prompt_block": "file:notes/launch.md\nOwner: Mina Chen.",
        },
        {
            "source": "notes/launch.md",
            "chunk_id": 5,
            "start_line": 10,
            "end_line": 11,
            "reranker_score": 0.77,
            "text": "Launch date: 2026-07-14.",
            "prompt_block": "file:notes/launch.md\nLaunch date: 2026-07-14.",
        },
    ]
    assert len(prompts) == 1


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
        "answer": "I could not find any relevant information in the notes. Please try rephrasing your question.",
        "citations": [],
    }
