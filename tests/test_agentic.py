import json
from types import SimpleNamespace

import pytest

from rag_notes_helper.rag import agentic as agentic_module
from rag_notes_helper.rag import answer as answer_module
from rag_notes_helper.rag.agentic import (
    ANSWER_RETRY_EXHAUSTED_MESSAGE,
    agentic_answer,
)
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE
from rag_notes_helper.rag.llm.base import BaseLLM
from rag_notes_helper.rag.prompt_context import estimate_prompt_tokens


def make_hit(source: str, chunk_id: int, text: str, score: float = 0.9) -> dict:
    return {
        "source": source,
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
    }


def action_json(action: dict) -> str:
    return json.dumps(action)


def search_action(
    query: str,
    *,
    vector_min_score: float = 0.5,
    bm25_min_score_ratio: float = 0.25,
    evidence_mode: str = "extend",
) -> str:
    return action_json({
        "action": "search",
        "query": query,
        "vector_min_score": vector_min_score,
        "bm25_min_score_ratio": bm25_min_score_ratio,
        "evidence_mode": evidence_mode,
    })


def test_agentic_direct_06_generates_from_shared_top_k_before_answer_review(
    monkeypatch,
):
    hits = [
        make_hit(
            "delivery_timeline.md",
            8,
            "The public launch target remains 2026-06-01.",
        ),
        make_hit(
            "delivery_timeline.md",
            9,
            "The next council checkpoint is 2026-05-28. The coordinator "
            "will bring the marketing review outcome and the remaining "
            "recurring-event test evidence to that meeting.",
        ),
        make_hit("delivery_timeline.md", 10, "The draft pilot starts in April."),
    ]
    hits[0].update({"start_line": 8, "end_line": 8})
    hits[1].update({"start_line": 12, "end_line": 12})
    hits[2].update({"start_line": 13, "end_line": 13})
    query = (
        "In timeline version 3.2, what date is the next launch council "
        "checkpoint?"
    )
    expected_answer = "The next launch council checkpoint is 2026-05-28."
    events = []
    progress = []
    progress_events = []
    settings = SimpleNamespace(
        agentic_max_search_attempts=3,
        agentic_max_answer_attempts=3,
        llm=SimpleNamespace(max_chunks=2),
    )
    monkeypatch.setattr(agentic_module, "get_settings", lambda: settings)

    class RecordingController:
        provider_name = "fixture"
        model = "fixture-controller"

        def __init__(self):
            self.prompts = []

        def generate_raw(self, prompt):
            events.append("answer_review")
            self.prompts.append(prompt)
            return action_json({"action": "accept"})

    controller = RecordingController()
    answer_calls = []
    generation_context = [
        {
            "source": hit["source"],
            "chunk_id": hit["chunk_id"],
            "start_line": hit.get("start_line"),
            "end_line": hit.get("end_line"),
            "text": hit["text"],
            "prompt_block": f"file:{hit['source']}\n{hit['text']}",
        }
        for hit in hits[:2]
    ]

    def search(query):
        events.append("search")
        assert query == (
            "In timeline version 3.2, what date is the next launch council "
            "checkpoint?"
        )
        return hits

    def answer(query, *, hits, stream):
        events.append("answer")
        answer_calls.append((query, hits, stream))
        return {
            "answer": expected_answer,
            "citations": [],
            "generation_context": generation_context,
        }

    result = agentic_answer(
        query,
        rag=None,
        meta_store=None,
        search_fn=search,
        controller=controller,
        answer_fn=answer,
        on_progress=lambda stage, count: progress.append((stage, count)),
        on_progress_event=progress_events.append,
    )

    assert events == ["search", "answer", "answer_review"]
    assert answer_calls == [
        (query, hits[:2], False),
    ]
    review_payload = json.loads(controller.prompts[0][-1]["content"])
    assert review_payload["generation_context"] == generation_context
    assert review_payload["generation_context"][1]["text"] == (
        "The next council checkpoint is 2026-05-28. The coordinator "
        "will bring the marketing review outcome and the remaining "
        "recurring-event test evidence to that meeting."
    )
    assert review_payload["candidate_answer"] == expected_answer
    assert result["answer"] == expected_answer
    assert result["generation_context"] == generation_context
    assert result["answer_attempts"] == 1
    assert [event.event for event in progress_events] == [
        "workflow_started",
        "search_started",
        "search_completed",
        "answer_started",
        "answer_review_started",
        "answer_review_decided",
        "terminal",
    ]
    assert progress == [
        ("searching", 1),
        ("generating", 1),
        ("reviewing", 1),
    ]


def test_agentic_empty_initial_search_is_retrieval_empty_and_skips_models():
    def unexpected_provider_call(*_args, **_kwargs):
        pytest.fail("empty Retrieval must not call Answer or controller")

    class UnexpectedController:
        provider_name = "fixture"

        def generate_raw(self, _prompt):
            unexpected_provider_call()

    events = []
    progress = []
    progress_events = []
    result = agentic_answer(
        "Question without retrieved candidates",
        rag=None,
        meta_store=None,
        search_fn=lambda query: events.append(("search", query)) or [],
        controller=UnexpectedController(),
        answer_fn=unexpected_provider_call,
        on_progress=lambda stage, count: progress.append((stage, count)),
        on_progress_event=progress_events.append,
    )

    assert result["reason_code"] == "no_retrieval_results"
    assert result["response_origin"] == "deterministic"
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 0
    assert result["search_queries"] == ["Question without retrieved candidates"]
    assert result["search_results"] == [[]]
    assert result["generation_context"] == []
    assert events == [("search", "Question without retrieved candidates")]
    assert progress == [("searching", 1)]
    assert [event.event for event in progress_events] == [
        "workflow_started",
        "search_started",
        "search_completed",
        "terminal",
    ]


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
        if self.responses:
            return self.responses.pop(0)
        return action_json({"action": "accept"})


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


def test_agentic_ollama_answer_review_uses_answer_review_schema():
    hit = make_hit("plan.md", 0, "The project starts in April.")

    class RecordingOllamaController:
        provider_name = "ollama"

        def __init__(self):
            self.calls = []
            self.responses = [action_json({"action": "accept"})]

        def generate_raw(self, prompt, **options):
            self.calls.append((prompt, options))
            return self.responses.pop(0)

    controller = RecordingOllamaController()
    result = agentic_answer(
        "When does the project start?",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: {
            "answer": hits[0]["text"],
            "citations": [],
        },
    )

    assert result["answer"] == "The project starts in April."
    assert len(controller.calls) == 1
    review_schema = json.loads(
        controller.calls[0][0][1]["content"]
    )["action_schema"]
    assert controller.calls[0][1] == {
        "think": False,
        "format": review_schema,
    }


def test_agentic_answer_uses_original_query_and_ranked_generation_context():
    hit_a = make_hit("plan.md", 0, "The project starts in April.")
    hit_b = make_hit("budget.md", 1, "The project budget is 20,000.")
    calls = []
    answer_calls = []
    controller = FakeController([action_json({"action": "accept"})])

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
        ("What is the project budget?", [hit_a, hit_b], False),
    ]
    assert result["answer"] == "The budget is 20,000."
    assert result["citations"] == ["budget.md"]


def test_agentic_controller_receives_history_as_intent_context_only():
    hit = make_hit("plan.md", 0, "The project starts in April.")
    controller = FakeController([action_json({"action": "accept"})])
    history = [("When does it start?", "The project starts in April.")]

    agentic_answer(
        "What is the project start date?",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: {
            "answer": hits[0]["text"],
            "citations": [],
        },
        conversation_history=history,
    )

    review_prompt = controller.prompts[0]
    review_payload = json.loads(review_prompt[-1]["content"])
    assert review_payload["conversation_history"] == [{
        "user": "When does it start?",
        "assistant": "The project starts in April.",
    }]
    assert "never evidence" in review_prompt[0]["content"].lower()
    assert review_payload["generation_context"][0]["text"] == (
        "The project starts in April."
    )
    assert review_payload["candidate_pool"] == [{
        "source": "plan.md",
        "chunk_id": 0,
        "text": "The project starts in April.",
        "in_generation_context": True,
    }]


def test_agentic_default_answer_receives_history_as_non_evidence(monkeypatch):
    hit = make_hit("plan.md", 0, "The project starts in April.")
    controller = FakeController([action_json({"action": "accept"})])
    history = [("When does it start?", "The project starts in April.")]
    answer_calls = []

    def fake_rag_answer(
        query, *, hits, stream, conversation_history=(), **_kwargs
    ):
        answer_calls.append((query, hits, stream, conversation_history))
        return {"answer": "The project starts in April.", "citations": []}

    monkeypatch.setattr(agentic_module, "rag_answer", fake_rag_answer)

    agentic_answer(
        "When does it start?",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        conversation_history=history,
    )

    assert answer_calls == [
        ("When does it start?", [hit], False, tuple(history)),
    ]


def test_agentic_controller_trims_oldest_turn_against_full_prompt(monkeypatch):
    hit = make_hit("plan.md", 0, "The project starts in April.")
    controller = FakeController([action_json({"action": "accept"})])
    settings = SimpleNamespace(
        agentic_max_search_attempts=3,
        agentic_max_answer_attempts=3,
        llm=SimpleNamespace(
            context_window_tokens=14000,
            max_tokens=1024,
            max_chunks=5,
        ),
    )
    monkeypatch.setattr(agentic_module, "get_settings", lambda: settings)
    history = [
        ("old question " + "x" * 7000, "old answer " + "y" * 7000),
        ("recent question", "recent answer"),
    ]

    agentic_answer(
        "What is the project start date?",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: {
            "answer": hits[0]["text"],
            "citations": [],
        },
        conversation_history=history,
    )

    prompt = controller.prompts[0]
    payload = json.loads(prompt[-1]["content"])
    assert payload["conversation_history"] == [{
        "user": "recent question",
        "assistant": "recent answer",
    }]
    assert estimate_prompt_tokens(prompt) + settings.llm.max_tokens <= (
        settings.llm.context_window_tokens
    )


def test_agentic_calls_keep_search_and_evidence_state_isolated_per_query():
    def run_query(query: str, hit: dict) -> tuple[dict, FakeController, list]:
        controller = FakeController([action_json({"action": "accept"})])
        progress = []
        result = agentic_answer(
            query,
            rag=None,
            meta_store=None,
            search_fn=lambda _query: [hit],
            controller=controller,
            answer_fn=lambda _query, *, hits, stream: {
                "answer": hits[0]["text"],
                "citations": [],
            },
            on_progress=lambda stage, count: progress.append((stage, count)),
        )
        return result, controller, progress

    first_hit = make_hit("first.md", 0, "First query evidence.")
    second_hit = make_hit("second.md", 0, "Second query evidence.")

    first_result, _, first_progress = run_query("first question", first_hit)
    second_result, second_controller, second_progress = run_query(
        "second question",
        second_hit,
    )

    second_prompt = json.loads(second_controller.prompts[0][1]["content"])
    assert second_prompt["question"] == "second question"
    assert second_prompt["searches_used"] == 1
    assert second_prompt["generation_context"] == [
        {
            "source": "second.md",
            "chunk_id": 0,
            "text": "Second query evidence.",
            "score": 0.9,
        }
    ]
    assert second_prompt["candidate_pool"] == [{
        "source": "second.md",
        "chunk_id": 0,
        "text": "Second query evidence.",
        "in_generation_context": True,
    }]
    assert first_result["search_results"] == [[first_hit]]
    assert second_result["search_results"] == [[second_hit]]
    assert second_result["search_queries"] == ["second question"]
    assert first_progress == [
        ("searching", 1),
        ("generating", 1),
        ("reviewing", 1),
    ]
    assert second_progress == first_progress


def test_agentic_first_answer_preserves_the_retrieval_rank_order():
    lower_ranked = make_hit("summary.md", 0, "A general project summary.")
    higher_ranked = make_hit("ownership.md", 1, "Mina Chen owns the project.")
    controller = FakeController([action_json({"action": "accept"})])
    answer_calls = []

    def answer(query, *, hits, stream):
        answer_calls.append((query, hits, stream))
        return {"answer": "Mina Chen owns the project.", "citations": []}

    result = agentic_answer(
        "Who owns the project?",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [lower_ranked, higher_ranked],
        controller=controller,
        answer_fn=answer,
    )

    assert answer_calls == [
        ("Who owns the project?", [lower_ranked, higher_ranked], False),
    ]
    assert result["answer"] == "Mina Chen owns the project."


def test_agentic_answer_review_ignores_extra_action_fields():
    hit = make_hit("product_overview.md", 1, "Product owner: Lena Ortiz.")
    controller = FakeController([
        action_json({
            "action": "accept",
            "explanation": "Selected the matching product role.",
        })
    ])

    result = agentic_answer(
        "Who is the product owner?",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: {
            "answer": "Lena Ortiz is the product owner.",
            "citations": [
                {"source": selected["source"], "chunk_id": selected["chunk_id"]}
                for selected in hits
            ],
        },
    )

    assert result["answer"] == "Lena Ortiz is the product owner."
    assert result["citations"][0]["source"] == "product_overview.md"


def test_agentic_reports_each_search_and_controller_latency(monkeypatch):
    hit = make_hit("note.md", 0, "Evidence.")
    controller = FakeController([
        search_action("follow-up"),
        action_json({
            "action": "answer",
            "evidence": [{"source": "note.md", "chunk_id": 0}],
        }),
    ])
    samples = iter([
        0.0, 0.010, 1.0, 1.020, 2.0, 2.030, 3.0, 3.015, 4.0, 4.025,
    ])
    monkeypatch.setattr(agentic_module, "perf_counter", lambda: next(samples))
    retrieval_events = []
    generation_calls = []

    result = agentic_answer(
        "question",
        rag=None,
        meta_store=None,
        search_fn=lambda _query, **_kwargs: [hit],
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
        "controller",
    ]
    assert [call["latency_ms"] for call in generation_calls] == [20.0, 15.0, 25.0]
    assert [call["phase"] for call in generation_calls] == [
        "answer_review",
        "retrieval_review",
        "answer_review",
    ]
    assert [call["token_usage"]["total_tokens"] for call in generation_calls] == [
        5,
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
            search_action("project budget"),
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

    def search(query, **_kwargs):
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
    assert answer_calls == [[first_hit], [first_hit, second_hit]]
    assert result["answer"] == "Two phases; 20,000 budget."
    assert result["search_results"] == [
        [first_hit],
        [first_hit, second_hit],
    ]
    assert result["search_queries"] == [
        "Summarize the project plan and budget.",
        "project budget",
    ]


def test_agentic_answer_reports_progress_for_each_search_and_decision(monkeypatch):
    first_hit = make_hit("plan.md", 0, "The project has two phases.")
    second_hit = make_hit("budget.md", 0, "The budget is 20,000.")
    progress = []
    progress_events = []
    monkeypatch.setattr(
        agentic_module,
        "get_settings",
        lambda: SimpleNamespace(
            agentic_max_search_attempts=2,
            agentic_max_answer_attempts=2,
            llm=SimpleNamespace(max_chunks=5),
        ),
    )
    controller = FakeController(
        [
            search_action("project budget"),
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
    original_generate_raw = controller.generate_raw

    def generate_raw(prompt):
        assert progress_events[-1].event in {
            "controller_started",
            "answer_review_started",
        }
        return original_generate_raw(prompt)

    controller.generate_raw = generate_raw

    def search(query, **_kwargs):
        assert progress_events[-1].event == "search_started"
        assert progress_events[-1].search_query == query
        return [first_hit] if query == "original question" else [second_hit]

    def answer(query, *, hits, stream):
        assert progress_events[-1].event == "answer_started"
        return {
            "answer": "Two phases; 20,000 budget.",
            "citations": [],
        }

    agentic_answer(
        "original question",
        rag=None,
        meta_store=None,
        search_fn=search,
        controller=controller,
        answer_fn=answer,
        on_progress_event=progress_events.append,
        on_progress=lambda stage, search_count: progress.append(
            (stage, search_count)
        ),
    )

    assert progress == [
        ("searching", 1),
        ("generating", 1),
        ("reviewing", 1),
        ("searching", 2),
        ("reviewing", 2),
        ("generating", 2),
        ("reviewing", 2),
    ]
    assert [event.event for event in progress_events] == [
        "workflow_started",
        "search_started",
        "search_completed",
        "answer_started",
        "answer_review_started",
        "answer_review_decided",
        "search_started",
        "search_completed",
        "controller_started",
        "action_selected",
        "answer_started",
        "answer_review_started",
        "answer_review_decided",
        "terminal",
    ]
    assert [
        (
            event.search_attempt,
            event.search_query,
            event.search_action,
        )
        for event in progress_events
        if event.event == "search_started"
    ] == [
        (1, "original question", "initial"),
        (2, "project budget", "extend"),
    ]
    assert [
        (event.search_attempt, event.result_count)
        for event in progress_events
        if event.event == "search_completed"
    ] == [(1, 1), (2, 1)]
    assert [
        event.decision
        for event in progress_events
        if event.event in {"action_selected", "answer_review_decided"}
    ] == ["search", "answer", "accept"]
    assert all(
        event.max_search_attempts == 2
        and event.max_answer_attempts == 2
        for event in progress_events
    )
    assert progress_events[-1].outcome == "answered"


def test_agentic_uses_configured_total_attempt_budgets(monkeypatch):
    hit = make_hit("note.md", 0, "Grounded evidence.")
    settings = SimpleNamespace(
        agentic_max_search_attempts=2,
        agentic_max_answer_attempts=3,
        llm=SimpleNamespace(max_chunks=5),
    )
    monkeypatch.setattr(agentic_module, "get_settings", lambda: settings)
    search_queries = []
    controller = FakeController([
        search_action("follow-up"),
        search_action("must-not-run"),
        action_json({
            "action": "answer",
            "evidence": [{"source": "note.md", "chunk_id": 0}],
        }),
    ])

    result = agentic_answer(
        "original question",
        rag=None,
        meta_store=None,
        search_fn=lambda query, **_kwargs: search_queries.append(query) or [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: {
            "answer": "Grounded answer.",
            "citations": [],
        },
    )

    assert search_queries == ["original question", "follow-up"]
    assert result["search_attempts"] == 2
    assert result["answer_attempts"] == 2
    initial_controller_request = json.loads(controller.prompts[0][1]["content"])
    exhausted_search_request = json.loads(controller.prompts[1][1]["content"])
    retried_search_request = json.loads(controller.prompts[2][1]["content"])
    assert initial_controller_request["searches_remaining"] == 1
    assert initial_controller_request["answer_attempts_remaining"] == 2
    assert exhausted_search_request["searches_remaining"] == 0
    assert retried_search_request["searches_remaining"] == 0
    assert "validation_feedback" in retried_search_request


def test_answer_review_search_requires_a_later_configured_answer_attempt(
    monkeypatch,
):
    hit = make_hit("note.md", 0, "Grounded evidence.")
    settings = SimpleNamespace(
        agentic_max_search_attempts=3,
        agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=5),
    )
    monkeypatch.setattr(agentic_module, "get_settings", lambda: settings)
    controller = FakeController([
        search_action("look for more evidence"),
        action_json({"action": "accept"}),
    ])
    search_queries = []
    answer_calls = []

    result = agentic_answer(
        "original question",
        rag=None,
        meta_store=None,
        search_fn=lambda query, **_kwargs: search_queries.append(query) or [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: answer_calls.append(hits) or {
            "answer": "Grounded answer.",
            "citations": [],
        },
    )

    assert search_queries == ["original question"]
    assert len(answer_calls) == 1
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 1
    review_retry = json.loads(controller.prompts[-1][1]["content"])
    assert "validation_feedback" in review_retry
    assert "No Answer attempts remain" in review_retry["validation_feedback"]


def test_configured_answer_budget_counts_the_first_generation(monkeypatch):
    hit = make_hit("note.md", 0, "Grounded evidence.")
    settings = SimpleNamespace(
        agentic_max_search_attempts=3,
        agentic_max_answer_attempts=2,
        llm=SimpleNamespace(max_chunks=5),
    )
    monkeypatch.setattr(agentic_module, "get_settings", lambda: settings)
    controller = FakeController([
        action_json({"action": "revise", "feedback": "Improve the first answer."}),
        action_json({"action": "revise", "feedback": "Improve the second answer."}),
    ])
    answer_calls = []

    result = agentic_answer(
        "original question",
        rag=None,
        meta_store=None,
        search_fn=lambda _query, **_kwargs: [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream, revision_feedback=None: (
            answer_calls.append(revision_feedback)
            or {
                "answer": "Rejected candidate.",
                "citations": [],
            }
        ),
    )

    assert answer_calls == [None, "Improve the first answer."]
    assert result["answer_attempts"] == 2
    assert result["answer_retry_exhausted"] is True
    assert result["answer"] == ANSWER_RETRY_EXHAUSTED_MESSAGE


def test_agentic_invalid_json_is_retried_once():
    hit = make_hit("note.md", 0, "Relevant evidence.")
    controller = FakeController(
        [
            "not json",
            action_json({"action": "accept"}),
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


def test_agentic_degrades_after_second_invalid_action_without_fallback():
    hit = make_hit("note.md", 0, "Relevant evidence.")
    controller = FakeController(["not json", "still not json"])

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda query: [hit],
        controller=controller,
        answer_fn=lambda *_args, **_kwargs: {
            "answer": "Buffered candidate.",
            "citations": [],
        },
    )

    assert len(controller.prompts) == 2
    assert result["reason_code"] == "controller_failure"
    assert result["workflow_error"] == "controller_failure"
    assert result["retrieved_evidence"] == [hit]
    assert result["answer_attempts"] == 1
    assert result["answer"] != "Buffered candidate."


def test_agentic_controller_provider_failure_returns_degraded_outcome():
    hit = make_hit("note.md", 0, "Relevant evidence.")

    class FailedController:
        calls = 0

        def generate_raw(self, _prompt):
            self.calls += 1
            raise RuntimeError("provider unavailable")

    controller = FailedController()

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: {
            "answer": "Buffered candidate.",
            "citations": [],
        },
    )

    assert controller.calls == 1
    assert result["answer"].strip()
    assert result["reason_code"] == "controller_failure"
    assert result["workflow_error_type"] == "RuntimeError"
    assert result["retrieved_evidence"] == [hit]


def test_agentic_retries_unknown_evidence_reference():
    hit = make_hit("note.md", 0, "Relevant evidence.")
    controller = FakeController(
        [
            action_json({
                "action": "expand_evidence",
                "evidence": [{"source": "other.md", "chunk_id": 0}],
            }),
            action_json({"action": "accept"}),
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
            "generation_context": hits,
        },
    )

    assert len(controller.prompts) == 2
    assert "current candidate pool" in controller.prompts[1][1]["content"]
    assert result["answer"] == "Grounded answer."
    assert result["reason_code"] == "answered"
    assert result["response_origin"] == "answer_model"


def test_agentic_searches_at_most_three_times_then_can_abstain():
    hit = make_hit("note.md", 0, "Relevant evidence.")
    calls = []
    controller = FakeController(
        [
            search_action("second search"),
            search_action("third search"),
            search_action("over the limit"),
            action_json({"action": "abstain"}),
        ]
    )

    def search(query, **_kwargs):
        calls.append(query)
        return [hit]

    result = agentic_answer(
        "original question",
        rag=None,
        meta_store=None,
        search_fn=search,
        controller=controller,
        answer_fn=lambda *_args, **_kwargs: {
            "answer": "Rejected candidate.",
            "citations": [],
        },
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
        "search_queries": ["original question", "second search", "third search"],
        "search_attempts": 3,
        "reason_code": "insufficient_evidence",
        "response_origin": "controller",
        "answer_attempts": 1,
    }


def test_agentic_revises_answer_at_most_three_attempts_then_accepts():
    hit = make_hit("note.md", 0, "The owner is Mina.")
    controller = FakeController([
        action_json({"action": "revise", "feedback": "Name the owner."}),
        action_json({"action": "revise", "feedback": "Be direct."}),
        action_json({"action": "accept"}),
    ])
    answer_calls = []

    def answer(query, *, hits, stream, revision_feedback=None):
        answer_calls.append((query, hits, stream, revision_feedback))
        return {
            "answer": ["Someone owns it.", "Mina owns it.", "Mina."][
                len(answer_calls) - 1
            ],
            "citations": [],
        }

    result = agentic_answer(
        "Who owns it?",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=answer,
    )

    assert [call[3] for call in answer_calls] == [
        None,
        "Name the owner.",
        "Be direct.",
    ]
    assert result["answer"] == "Mina."
    assert result["answer_attempts"] == 3


def test_answer_review_can_expand_context_and_sees_the_exact_generation_limit(
    monkeypatch,
):
    hits = [
        make_hit("first.md", 0, "First candidate evidence."),
        make_hit("second.md", 1, "Second candidate evidence."),
        make_hit("third.md", 2, "Third candidate evidence."),
    ]
    settings = SimpleNamespace(
        agentic_max_search_attempts=3,
        agentic_max_answer_attempts=3,
        llm=SimpleNamespace(max_chunks=2),
        line_width=80,
    )
    monkeypatch.setattr(agentic_module, "get_settings", lambda: settings)
    monkeypatch.setattr(answer_module, "get_settings", lambda: settings)

    class AnswerModel:
        def __init__(self):
            self.responses = ["Rejected candidate.", "Approved candidate."]

        def generate(self, _prompt, **_kwargs):
            return self.responses.pop(0)

    answer_model = AnswerModel()
    monkeypatch.setattr(answer_module, "get_llm", lambda: answer_model)
    controller = FakeController([
        action_json({
            "action": "expand_evidence",
            "evidence": [{"source": "third.md", "chunk_id": 2}],
        }),
        action_json({"action": "accept"}),
    ])
    search_queries = []

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda query, **_kwargs: search_queries.append(query) or hits,
        controller=controller,
    )

    first_review = json.loads(controller.prompts[0][1]["content"])
    assert first_review["generation_context"] == [
        {
            "source": "first.md",
            "chunk_id": 0,
            "start_line": None,
            "end_line": None,
            "text": "First candidate evidence.",
            "prompt_block": "file:first.md\nFirst candidate evidence.",
        },
        {
            "source": "second.md",
            "chunk_id": 1,
            "start_line": None,
            "end_line": None,
            "text": "Second candidate evidence.",
            "prompt_block": "file:second.md\nSecond candidate evidence.",
        },
    ]
    assert first_review["candidate_pool"] == [
        {
            "source": hit["source"],
            "chunk_id": hit["chunk_id"],
            "text": hit["text"],
            "in_generation_context": hit["source"] != "third.md",
        }
        for hit in hits
    ]
    second_review = json.loads(controller.prompts[1][1]["content"])
    assert [item["source"] for item in second_review["generation_context"]] == [
        "third.md",
        "first.md",
    ]
    assert search_queries == ["Question"]
    assert result["answer"] == "Approved candidate."
    assert result["answer_attempts"] == 2


def test_answer_review_evidence_expansion_requires_an_answer_attempt(monkeypatch):
    hits = [
        make_hit("used.md", 0, "Current answer evidence."),
        make_hit("unused.md", 1, "Additional evidence."),
    ]
    settings = SimpleNamespace(
        agentic_max_search_attempts=3,
        agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=5),
    )
    monkeypatch.setattr(agentic_module, "get_settings", lambda: settings)
    controller = FakeController([
        action_json({
            "action": "expand_evidence",
            "evidence": [{"source": "unused.md", "chunk_id": 1}],
        }),
        action_json({
            "action": "expand_evidence",
            "evidence": [{"source": "unused.md", "chunk_id": 1}],
        }),
    ])
    answer_calls = []
    search_queries = []

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda query, **_kwargs: search_queries.append(query) or hits,
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: (
            answer_calls.append(hits)
            or {
                "answer": "Rejected candidate.",
                "citations": [],
                "generation_context": hits,
            }
        ),
    )

    retry_prompt = json.loads(controller.prompts[-1][1]["content"])
    assert len(answer_calls) == 1
    assert search_queries == ["Question"]
    assert "No Answer attempts remain" in retry_prompt["validation_feedback"]
    assert result["reason_code"] == "controller_failure"
    assert result["answer"] != "Rejected candidate."
    assert result["answer_attempts"] == 1


def test_answer_review_rejects_evidence_outside_the_candidate_pool():
    hit = make_hit("used.md", 0, "Current answer evidence.")
    controller = FakeController([
        action_json({
            "action": "expand_evidence",
            "evidence": [{"source": "missing.md", "chunk_id": 1}],
        }),
        action_json({"action": "accept"}),
    ])
    answer_calls = []

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda _query, **_kwargs: [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: (
            answer_calls.append(hits)
            or {
                "answer": "Grounded candidate.",
                "citations": [],
                "generation_context": hits,
            }
        ),
    )

    retry_prompt = json.loads(controller.prompts[-1][1]["content"])
    assert len(answer_calls) == 1
    assert "current candidate pool" in retry_prompt["validation_feedback"]
    assert result["answer"] == "Grounded candidate."


def test_agentic_returns_specific_insufficiency_after_three_answer_attempts():
    hit = make_hit("note.md", 0, "Partial evidence.")
    controller = FakeController([
        action_json({"action": "revise", "feedback": "Try again."}),
        action_json({"action": "revise", "feedback": "Still incomplete."}),
        action_json({"action": "revise", "feedback": "Still unsupported."}),
    ])
    answer_calls = []

    def answer(_query, *, hits, stream, revision_feedback=None):
        answer_calls.append(revision_feedback)
        return {
            "answer": "Unsupported candidate.",
            "citations": [{"source": hits[0]["source"]}],
            "generation_context": hits,
        }

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=answer,
    )

    assert len(answer_calls) == 3
    assert result["answer"] == ANSWER_RETRY_EXHAUSTED_MESSAGE
    assert result["citations"] == []
    assert result["answer_retry_exhausted"] is True
    assert result["reason_code"] == "answer_attempts_exhausted"
    assert result["response_origin"] == "deterministic"
    assert result["answer"] != "Unsupported candidate."
    assert result["answer_attempts"] == 3


def test_agentic_retries_invalid_answer_review_once():
    hit = make_hit("note.md", 0, "Grounded evidence.")
    controller = FakeController([
        "not valid review json",
        action_json({"action": "accept"}),
    ])

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: {
            "answer": "Grounded answer.",
            "citations": [],
        },
    )

    review_retry = json.loads(controller.prompts[-1][-1]["content"])
    assert "validation_feedback" in review_retry
    assert result["answer"] == "Grounded answer."


def test_agentic_answer_review_can_abstain():
    hit = make_hit("note.md", 0, "Partial evidence.")
    controller = FakeController([
        action_json({"action": "abstain"}),
    ])

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda _query: [hit],
        controller=controller,
        answer_fn=lambda _query, *, hits, stream: {
            "answer": "Unsupported candidate.",
            "citations": [],
        },
    )

    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["answer_attempts"] == 1


def test_agentic_preserves_each_search_ranking_for_retrieval_evaluation():
    first_ranked = [
        make_hit("first.md", 0, "First rank", score=0.9),
        make_hit("first.md", 1, "Second rank", score=0.8),
    ]
    second_ranked = [
        make_hit("second.md", 0, "Follow-up first", score=0.95),
        make_hit("first.md", 0, "Repeated chunk", score=0.7),
    ]
    search_results = iter([first_ranked, second_ranked])
    controller = FakeController(
        [
            search_action("follow-up"),
            action_json({
                "action": "answer",
                "evidence": [{"source": "first.md", "chunk_id": 0}],
            }),
        ]
    )

    result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda _query, **_kwargs: next(search_results),
        controller=controller,
        answer_fn=lambda *_args, **_kwargs: {
            "answer": "Grounded answer.",
            "citations": [],
        },
    )

    assert result["search_results"] == [first_ranked, second_ranked]
    assert result["search_queries"] == ["Question", "follow-up"]
