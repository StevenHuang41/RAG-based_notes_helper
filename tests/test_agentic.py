"""Public Agentic invariants after ADR-0008.

Recovery/empty-pool cases live in the two-phase Retrieval tests; accepted partial
answers, same-context revision, warning, and budget cases live in the two-phase
Answer tests. Legacy replace/expand/reopen assertions are deliberately retired.
"""

import json
from types import SimpleNamespace

import pytest

from rag_notes_helper import cli
from rag_notes_helper.rag import agentic
from rag_notes_helper.rag.llm.base import BaseLLM


class RawResponseLLM(BaseLLM):
    def __init__(self, response):
        super().__init__(model="test", api_key=None, line_width=20)
        self.response = response

    def _generate(self, prompt, **kwargs):
        return self.response

    def _stream(self, prompt, **kwargs):
        yield from ()


def test_generate_raw_preserves_json_without_display_wrapping():
    raw = '{"query":"a refined query with enough words to wrap across the terminal width"}'
    llm = RawResponseLLM(raw)
    assert llm.generate_raw([], line_width=20) == raw
    assert llm.generate([], line_width=20) != raw


@pytest.fixture
def runtime(monkeypatch):
    settings = SimpleNamespace(
        agentic_max_search_attempts=3, agentic_max_answer_attempts=3,
        llm=SimpleNamespace(max_chunks=5, context_window_tokens=50000, max_tokens=1024),
    )
    monkeypatch.setattr(agentic, "get_settings", lambda: settings)
    hit = {"source": "plan.md", "chunk_id": 0, "text": "The project starts in April.", "score": .9}
    monkeypatch.setattr(agentic, "search_notes", lambda query, **kwargs: [hit])
    answer_calls = []

    def answer(query, *, hits, **kwargs):
        answer_calls.append((query, hits, kwargs))
        return {"answer": hits[0]["text"], "citations": [], "generation_context": [
            {**h, "prompt_block": h["text"]} for h in hits
        ]}

    monkeypatch.setattr(agentic, "rag_answer", answer)
    return settings, hit, answer_calls


class Controller:
    provider_name = "fixture"
    model = "fixture"
    last_usage = {"status": "available", "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}

    def __init__(self):
        self.calls = []

    def generate_raw(self, messages, **options):
        self.calls.append((messages, options))
        payload = json.loads(messages[-1]["content"])
        return json.dumps({"coverage": "sufficient_evidence"} if "chunks_pool" in payload else {"action": "accept", "answer_coverage": "full"})


def test_ollama_uses_phase_specific_schema_and_disables_thinking(monkeypatch, runtime):
    controller = Controller()
    controller.provider_name = "ollama"
    monkeypatch.setattr(agentic, "get_llm", lambda: controller)
    result = cli.run_query(object(), object(), query="When does the project start?", agentic=True)
    assert result["review_passed"] is True
    assert len(controller.calls) == 2
    for messages, options in controller.calls:
        assert options == {"think": False, "format": json.loads(messages[-1]["content"])["action_schema"]}
    answer_schema = controller.calls[-1][1]["format"]
    assert "expand_evidence" not in json.dumps(answer_schema)
    assert '"search"' not in json.dumps(answer_schema)


def test_query_isolation_preserves_original_query_and_first_seen_order(monkeypatch, runtime):
    _, first, answer_calls = runtime
    second = {"source": "other.md", "chunk_id": 0, "text": "The budget is 20000.", "score": .8}
    queries = []

    def search(query, **kwargs):
        queries.append(query)
        return [first] if query == "first question" else [second]

    monkeypatch.setattr(agentic, "search_notes", search)
    monkeypatch.setattr(agentic, "get_llm", Controller)
    first_result = cli.run_query(object(), object(), query="first question", agentic=True)
    second_result = cli.run_query(object(), object(), query="second question", agentic=True)
    assert queries == ["first question", "second question"]
    assert [c["source"] for c in first_result["chunks_pool"]] == ["plan.md"]
    assert [c["source"] for c in second_result["chunks_pool"]] == ["other.md"]
    assert [call[0] for call in answer_calls] == queries
    assert first_result["answer_attempts"] == second_result["answer_attempts"] == 1


def test_history_is_intent_context_in_review_and_generation(monkeypatch, runtime):
    _, hit, answer_calls = runtime
    controller = Controller()
    monkeypatch.setattr(agentic, "get_llm", lambda: controller)
    history = [("When does it start?", "Earlier unsupported guess")]
    result = cli.run_query(object(), object(), query="What month?", agentic=True, conversation_history=history)
    assert result["review_passed"] is True
    payload = json.loads(controller.calls[-1][0][-1]["content"])
    assert payload["conversation_history"] == [{"user": history[0][0], "assistant": history[0][1]}]
    assert history[0][1] not in json.dumps(payload["generation_context"])
    assert "not evidence" in controller.calls[-1][0][0]["content"]
    assert answer_calls[0][2]["conversation_history"] == tuple(history)


def test_public_callbacks_record_real_phases_and_safe_progress(monkeypatch, runtime):
    controller = Controller()
    monkeypatch.setattr(agentic, "get_llm", lambda: controller)
    calls, progress, retrieval = [], [], []
    result = cli.run_query(
        object(), object(), query="When does it start?", agentic=True,
        on_generation=calls.append, on_progress_event=progress.append,
        on_retrieval=retrieval.append,
    )
    assert result["review_passed"] is True
    assert [c["phase"] for c in calls] == ["retrieval_review", "answer_review"]
    assert all(c["latency_ms"] >= 0 for c in calls)
    assert all(c["token_usage"]["total_tokens"] == 5 for c in calls)
    assert len(retrieval) == 1 and retrieval[0]["result_count"] == 1
    events = [event.event for event in progress]
    assert events.index("answer_started") < events.index("answer_review_started")
    assert progress[-1].event == "terminal" and progress[-1].outcome == "answered"
    assert all(not hasattr(event, "candidate_answer") for event in progress)


def test_review_ignores_unrelated_extra_fields_without_authorizing_search(monkeypatch, runtime):
    class ExtraController(Controller):
        def generate_raw(self, messages, **options):
            payload = json.loads(messages[-1]["content"])
            if "chunks_pool" in payload:
                return json.dumps({"coverage": "sufficient_evidence"})
            return json.dumps({"action": "accept", "query": "do not run", "evidence": [{"source": "invented", "chunk_id": 42}]})

    monkeypatch.setattr(agentic, "get_llm", ExtraController)
    result = cli.run_query(object(), object(), query="When?", agentic=True)
    assert result["review_passed"] is True
    assert result["search_queries"] == ["When?"]
    assert result["answer"] == runtime[1]["text"]


def test_answer_reviewer_keeps_newest_five_complete_turns_with_long_history(monkeypatch, runtime):
    settings, hit, answer_calls = runtime
    settings.llm.context_window_tokens = 7000
    controller = Controller()
    monkeypatch.setattr(agentic, "get_llm", lambda: controller)
    history = [("excluded question", "excluded answer")] + [
        (f"question {i} " + "x" * 7000, f"answer {i}") for i in range(5)
    ]
    result = cli.run_query(object(), object(), query="When?", agentic=True, conversation_history=history)
    assert result["review_passed"] is True
    payload = json.loads(controller.calls[-1][0][-1]["content"])
    assert payload["conversation_history"] == [
        {"user": user, "assistant": assistant} for user, assistant in history[-5:]
    ]
    assert payload["generation_context"][0]["text"] == hit["text"]
