import json

import pytest

from rag_notes_helper.rag import agentic as agentic_module
from rag_notes_helper.rag.agentic import (
    ANSWER_RETRY_EXHAUSTED_MESSAGE,
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


def test_ollama_controller_disables_thinking_for_structured_actions():
    received_options = []

    class OllamaController:
        provider_name = "ollama"

        def generate_raw(self, prompt, **options):
            received_options.append(options)
            return action_json({"action": "abstain"})

    action = agentic_module._request_action(
        controller=OllamaController(),
        query="What is not documented?",
        hits=[],
        searches_used=0,
    )

    assert action.action == "abstain"
    assert received_options == [{"think": False}]


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


def test_agentic_calls_keep_search_and_evidence_state_isolated_per_query():
    def run_query(query: str, hit: dict) -> tuple[dict, FakeController, list]:
        controller = FakeController([
            action_json({
                "action": "answer",
                "evidence": [{"source": hit["source"], "chunk_id": hit["chunk_id"]}],
            })
        ])
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
    assert second_prompt["evidence"] == [
        {
            "source": "second.md",
            "chunk_id": 0,
            "score": 0.9,
            "text": "Second query evidence.",
        }
    ]
    assert first_result["search_results"] == [[first_hit]]
    assert second_result["search_results"] == [[second_hit]]
    assert second_result["search_queries"] == ["second question"]
    assert first_progress == [
        ("searching", 1),
        ("reviewing", 1),
        ("generating", 1),
    ]
    assert second_progress == first_progress


def test_agentic_reranks_selected_evidence_for_the_original_question():
    lower_ranked = make_hit("summary.md", 0, "A general project summary.")
    higher_ranked = make_hit("ownership.md", 1, "Mina Chen owns the project.")
    controller = FakeController([
        action_json({
            "action": "answer",
            "evidence": [
                {"source": lower_ranked["source"], "chunk_id": lower_ranked["chunk_id"]},
                {"source": higher_ranked["source"], "chunk_id": higher_ranked["chunk_id"]},
            ],
        })
    ])
    answer_calls = []

    class FakeCrossEncoder:
        def predict(self, pairs):
            assert pairs == [
                ("Who owns the project?", lower_ranked["text"]),
                ("Who owns the project?", higher_ranked["text"]),
            ]
            return [0.1, 0.95]

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
        cross_encoder=FakeCrossEncoder(),
    )

    assert answer_calls == [
        ("Who owns the project?", [higher_ranked | {"reranker_score": 0.95, "rerank_rank": 1}, lower_ranked | {"reranker_score": 0.1, "rerank_rank": 2}], False),
    ]
    assert result["answer"] == "Mina Chen owns the project."


def test_agentic_answer_ignores_extra_text_on_evidence_references():
    hit = make_hit("product_overview.md", 1, "Product owner: Lena Ortiz.")
    controller = FakeController([
        action_json({
            "action": "answer",
            "explanation": "Selected the matching product role.",
            "evidence": [{
                "source": hit["source"],
                "chunk_id": hit["chunk_id"],
                "text": hit["text"],
            }],
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
        "retrieval_review",
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
    assert answer_calls == [[first_hit, second_hit]]
    assert result["answer"] == "Two phases; 20,000 budget."
    assert result["search_results"] == [
        [first_hit],
        [first_hit, second_hit],
    ]
    assert result["search_queries"] == [
        "Summarize the project plan and budget.",
        "project budget",
    ]


def test_agentic_answer_reports_progress_for_each_search_and_decision():
    first_hit = make_hit("plan.md", 0, "The project has two phases.")
    second_hit = make_hit("budget.md", 0, "The budget is 20,000.")
    progress = []
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


def test_agentic_empty_search_can_abstain_after_controller_review():
    progress = []
    controller = FakeController([action_json({"action": "abstain"})])

    result = agentic_answer(
        "Question without note evidence",
        rag=None,
        meta_store=None,
        search_fn=lambda query: [],
        controller=controller,
        on_progress=lambda stage, search_count: progress.append(
            (stage, search_count)
        ),
    )

    assert result["answer"] == ABSTENTION_MESSAGE
    assert progress == [
        ("searching", 1),
        ("reviewing", 1),
        ("abstaining", 1),
    ]


def test_agentic_empty_initial_search_uses_controller_without_answer():
    def unexpected_answer(*args, **kwargs):
        pytest.fail("answer generation should not run for empty retrieval")
    controller = FakeController([action_json({"action": "abstain"})])

    result = agentic_answer(
        "Question without note evidence",
        rag=None,
        meta_store=None,
        search_fn=lambda query: [],
        controller=controller,
        answer_fn=unexpected_answer,
    )

    assert result == {
        "answer": ABSTENTION_MESSAGE,
        "citations": [],
        "retrieved_evidence": [],
        "search_results": [[]],
        "search_queries": ["Question without note evidence"],
        "search_attempts": 1,
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

    assert len(controller.prompts) == 3
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
        answer_fn=lambda *args, **kwargs: pytest.fail(
            "Classic fallback or answer generation must not run"
        ),
    )

    assert len(controller.prompts) == 2
    assert result["reason_code"] == "controller_failure"
    assert result["workflow_error"] == "controller_failure"
    assert result["retrieved_evidence"] == [hit]


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

    assert len(controller.prompts) == 3
    assert "not returned by search_notes" in controller.prompts[1][1]["content"]
    assert result["answer"] == "Grounded answer."


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
        "search_queries": ["original question", "second search", "third search"],
        "search_attempts": 3,
    }


def test_agentic_revises_answer_at_most_three_attempts_then_accepts():
    hit = make_hit("note.md", 0, "The owner is Mina.")
    controller = FakeController([
        action_json({
            "action": "answer",
            "evidence": [{"source": "note.md", "chunk_id": 0}],
        }),
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


def test_agentic_returns_specific_insufficiency_after_three_answer_attempts():
    hit = make_hit("note.md", 0, "Partial evidence.")
    controller = FakeController([
        action_json({
            "action": "answer",
            "evidence": [{"source": "note.md", "chunk_id": 0}],
        }),
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
        action_json({
            "action": "answer",
            "evidence": [{"source": "note.md", "chunk_id": 0}],
        }),
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
        action_json({
            "action": "answer",
            "evidence": [{"source": "note.md", "chunk_id": 0}],
        }),
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
