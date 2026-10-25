import json

import pytest

from rag_notes_helper.rag import agentic as agentic_module
from rag_notes_helper.rag.agentic import (
    AgenticWorkflowError,
    agentic_answer,
)
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE
from rag_notes_helper.rag.llm.base import BaseLLM


def make_hit(source: str, chunk_id: int, text: str, score: float = 0.9) -> dict:
    return {
        "source": source,
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
    }


def action_json(action: dict) -> str:
    return json.dumps(action)


class FakeController:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts = []
        self.provider_name = "fixture"
        self.model = "fixture-controller"
        self.last_usage = {
            "status": "available",
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
        }

    def generate_raw(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class RawResponseLLM(BaseLLM):
    def __init__(self, response: str):
        super().__init__(model="test", api_key=None, line_width=20)
        self.response = response

    def _generate(self, prompt, **kws):
        return self.response

    def _stream(self, prompt, **kws):
        yield from ()


def test_generate_raw_preserves_json_without_display_wrapping():
    raw_json = (
        '{"action":"search","query":"a refined query with enough words '
        'to wrap across the normal terminal line width"}'
    )
    llm = RawResponseLLM(raw_json)

    assert llm.generate_raw([], line_width=20) == raw_json
    assert llm.generate([], line_width=20) != raw_json


def test_agentic_answer_uses_original_query_and_selected_evidence():
    hit_a = make_hit("plan.md", 0, "The project starts in April.")
    hit_b = make_hit("budget.md", 1, "The project budget is 20,000.")
    calls = []
    answer_calls = []
    controller = FakeController(
        [
            action_json(
                {
                    "action": "answer",
                    "evidence": [
                        {"source": "budget.md", "chunk_id": 1},
                    ],
                }
            )
        ]
    )

    def search(query):
        calls.append(query)
        return [hit_a, hit_b]

    def answer(query, *, hits, stream):
        answer_calls.append((query, hits, stream))
        return {"answer": "The budget is 20,000.", "citations": ["budget.md"]}

    result = agentic_answer(
        "What is the project budget?",
        rag=None,
        meta_store=None,
        search_fn=search,
        controller=controller,
        answer_fn=answer,
    )

    assert calls == ["What is the project budget?"]
    assert answer_calls == [
        ("What is the project budget?", [hit_b], False),
    ]
    assert result["answer"] == "The budget is 20,000."
    assert result["citations"] == ["budget.md"]


def test_agentic_reports_each_search_and_controller_latency(monkeypatch):
    hit = make_hit("note.md", 0, "Evidence.")
    controller = FakeController([
        action_json({"action": "search", "query": "follow-up"}),
        action_json({
            "action": "answer",
            "evidence": [{"source": "note.md", "chunk_id": 0}],
        }),
    ])
    samples = iter([0.0, 0.010, 1.0, 1.020, 2.0, 2.030, 3.0, 3.015])
    monkeypatch.setattr(agentic_module, "perf_counter", lambda: next(samples))
    retrieval_events = []
    generation_calls = []

    result = agentic_answer(
        "question",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=lambda *_args, **_kwargs: {
            "answer": "answer",
            "citations": [],
        },
        on_retrieval=retrieval_events.append,
        on_generation=generation_calls.append,
    )

    assert result["answer"] == "answer"
    assert [event["latency_ms"] for event in retrieval_events] == [10.0, 30.0]
    assert [call["call_type"] for call in generation_calls] == [
        "controller",
        "controller",
    ]
    assert [call["latency_ms"] for call in generation_calls] == [20.0, 15.0]
    assert [call["token_usage"]["total_tokens"] for call in generation_calls] == [
        5,
        5,
    ]


def test_agentic_answer_refines_search_and_deduplicates_evidence():
    first_hit = make_hit("plan.md", 0, "The project has two phases.")
    second_hit = make_hit("budget.md", 0, "The budget is 20,000.")
    calls = []
    answer_calls = []
    controller = FakeController(
        [
            action_json(
                {
                    "action": "search",
                    "query": "project budget",
                }
            ),
            action_json(
                {
                    "action": "answer",
                    "evidence": [
                        {"source": "plan.md", "chunk_id": 0},
                        {"source": "budget.md", "chunk_id": 0},
                    ],
                }
            ),
        ]
    )

    def search(query):
        calls.append(query)
        if len(calls) == 1:
            return [first_hit]
        return [first_hit, second_hit]

    def answer(query, *, hits, stream):
        answer_calls.append(hits)
        return {"answer": "Two phases; 20,000 budget.", "citations": []}

    result = agentic_answer(
        "Summarize the project plan and budget.",
        rag=None,
        meta_store=None,
        search_fn=search,
        controller=controller,
        answer_fn=answer,
    )

    assert calls == [
        "Summarize the project plan and budget.",
        "project budget",
    ]
    assert answer_calls == [[first_hit, second_hit]]
    assert result["answer"] == "Two phases; 20,000 budget."
    assert result["search_results"] == [
        [first_hit],
        [first_hit, second_hit],
    ]


def test_agentic_answer_reports_progress_for_each_search_and_decision():
    first_hit = make_hit("plan.md", 0, "The project has two phases.")
    second_hit = make_hit("budget.md", 0, "The budget is 20,000.")
    progress = []
    controller = FakeController(
        [
            action_json({"action": "search", "query": "project budget"}),
            action_json(
                {
                    "action": "answer",
                    "evidence": [
                        {"source": "plan.md", "chunk_id": 0},
                        {"source": "budget.md", "chunk_id": 0},
                    ],
                }
            ),
        ]
    )

    def search(query):
        return [first_hit] if query == "original question" else [second_hit]

    agentic_answer(
        "original question",
        rag=None,
        meta_store=None,
        search_fn=search,
        controller=controller,
        answer_fn=lambda query, *, hits, stream: {
            "answer": "Two phases; 20,000 budget.",
            "citations": [],
        },
        on_progress=lambda stage, search_count: progress.append(
            (stage, search_count)
        ),
    )

    assert progress == [
        ("searching", 1),
        ("reviewing", 1),
        ("searching", 2),
        ("reviewing", 2),
        ("generating", 2),
    ]


def test_agentic_empty_search_reports_abstention_progress():
    progress = []

    result = agentic_answer(
        "Question without note evidence",
        rag=None,
        meta_store=None,
        search_fn=lambda query: [],
        on_progress=lambda stage, search_count: progress.append(
            (stage, search_count)
        ),
    )

    assert result["answer"] == ABSTENTION_MESSAGE
    assert progress == [("searching", 1), ("abstaining", 1)]


def test_agentic_empty_initial_search_abstains_without_llm(monkeypatch):
    def unexpected_controller():
        pytest.fail("controller should not be created for empty retrieval")

    def unexpected_answer(*args, **kwargs):
        pytest.fail("answer generation should not run for empty retrieval")

    monkeypatch.setattr(
        "rag_notes_helper.rag.agentic.get_llm",
        unexpected_controller,
    )

    result = agentic_answer(
        "Question without note evidence",
        rag=None,
        meta_store=None,
        search_fn=lambda query: [],
        answer_fn=unexpected_answer,
    )

    assert result == {
        "answer": ABSTENTION_MESSAGE,
        "citations": [],
        "retrieved_evidence": [],
        "search_results": [[]],
    }


def test_agentic_invalid_json_is_retried_once():
    hit = make_hit("note.md", 0, "Relevant evidence.")
    controller = FakeController(
        [
            "not json",
            action_json(
                {
                    "action": "answer",
                    "evidence": [{"source": "note.md", "chunk_id": 0}],
                }
            ),
        ]
    )

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda query: [hit],
        controller=controller,
        answer_fn=lambda query, *, hits, stream: {
            "answer": "Grounded answer.",
            "citations": [],
        },
    )

    assert len(controller.prompts) == 2
    assert "validation_feedback" in controller.prompts[1][1]["content"]
    assert result["answer"] == "Grounded answer."


def test_agentic_fails_after_second_invalid_action_without_fallback():
    hit = make_hit("note.md", 0, "Relevant evidence.")
    controller = FakeController(["not json", "still not json"])

    with pytest.raises(
        AgenticWorkflowError,
        match="remained invalid after one retry",
    ):
        agentic_answer(
            "Question",
            rag=None,
            meta_store=None,
            search_fn=lambda query: [hit],
            controller=controller,
            answer_fn=lambda *args, **kwargs: pytest.fail(
                "Classic fallback or answer generation must not run"
            ),
        )

    assert len(controller.prompts) == 2


def test_agentic_retries_unknown_evidence_reference():
    hit = make_hit("note.md", 0, "Relevant evidence.")
    controller = FakeController(
        [
            action_json(
                {
                    "action": "answer",
                    "evidence": [{"source": "other.md", "chunk_id": 0}],
                }
            ),
            action_json(
                {
                    "action": "answer",
                    "evidence": [{"source": "note.md", "chunk_id": 0}],
                }
            ),
        ]
    )

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda query: [hit],
        controller=controller,
        answer_fn=lambda query, *, hits, stream: {
            "answer": "Grounded answer.",
            "citations": [],
        },
    )

    assert len(controller.prompts) == 2
    assert "not returned by search_notes" in controller.prompts[1][1]["content"]
    assert result["answer"] == "Grounded answer."


def test_agentic_searches_at_most_three_times_then_can_abstain():
    hit = make_hit("note.md", 0, "Relevant evidence.")
    calls = []
    controller = FakeController(
        [
            action_json({"action": "search", "query": "second search"}),
            action_json({"action": "search", "query": "third search"}),
            action_json({"action": "search", "query": "over the limit"}),
            action_json({"action": "abstain"}),
        ]
    )

    def search(query):
        calls.append(query)
        return [hit]

    result = agentic_answer(
        "original question",
        rag=None,
        meta_store=None,
        search_fn=search,
        controller=controller,
        answer_fn=lambda *args, **kwargs: pytest.fail(
            "abstention must not call final answer generation"
        ),
    )

    assert calls == ["original question", "second search", "third search"]
    assert len(controller.prompts) == 4
    last_prompt = controller.prompts[2][1]["content"]
    assert '"searches_remaining": 0' in last_prompt
    assert result == {
        "answer": ABSTENTION_MESSAGE,
        "citations": [],
        "retrieved_evidence": [hit],
        "search_results": [[hit], [hit], [hit]],
    }
