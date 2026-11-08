import json
import sys
from types import SimpleNamespace

import pytest

from rag_notes_helper import cli
from rag_notes_helper.rag import agentic as agentic_module
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE
from rag_notes_helper.rag.retrieval import HybridSearchResults


@pytest.mark.parametrize("query", ["", "   \t\n"])
def test_classic_blank_query_returns_structured_invalid_outcome(
    monkeypatch,
    query,
):
    monkeypatch.setattr(
        cli,
        "retrieve",
        lambda *_args, **_kwargs: pytest.fail("blank input must not run Retrieval"),
    )
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: pytest.fail("blank input must not call the Answer model"),
    )

    result = cli.run_query(object(), object(), query=query)

    assert result["answer"].strip()
    assert result["reason_code"] == "invalid_query"
    assert result["response_origin"] == "deterministic"
    assert result["citations"] == []
    assert result["generation_context"] == []
    assert result["retrieved_evidence"] == []
    assert result["search_queries"] == []
    assert result["retrieval_stages"] == []


def test_classic_empty_retrieval_returns_structured_no_evidence_outcome(
    monkeypatch,
):
    queries = []

    def retrieve_no_candidates(_rag, _meta_store, *, query, **_kwargs):
        queries.append(query)
        return []

    monkeypatch.setattr(cli, "retrieve", retrieve_no_candidates)
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: pytest.fail(
            "empty Retrieval must not call the Answer model"
        ),
    )

    result = cli.run_query(object(), object(), query="Who owns the launch?")

    assert queries == ["Who owns the launch?"]
    assert result["answer"] == (
        "I could not find sufficient supporting evidence in the notes."
    )
    assert result["reason_code"] == "no_evidence_after_searches"
    assert result["response_origin"] == "deterministic"
    assert result["citations"] == []
    assert result["generation_context"] == []
    assert result["retrieved_evidence"] == []
    assert result["search_queries"] == ["Who owns the launch?"]
    assert result["retrieval_stages"] == []


def test_classic_retrieval_failure_returns_structured_non_empty_outcome(
    monkeypatch,
):
    retrieval_events = []

    def failed_retrieval(*_args, **_kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(cli, "retrieve", failed_retrieval)
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: pytest.fail(
            "a Retrieval failure must not invoke the Answer model"
        ),
    )

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        on_retrieval=retrieval_events.append,
    )

    assert result["answer"] == agentic_module.RETRIEVAL_FAILURE_MESSAGE
    assert result["reason_code"] == "retrieval_failure"
    assert result["response_origin"] == "deterministic"
    assert result["workflow_error"] == "retrieval_failure"
    assert result["workflow_error_type"] == "RuntimeError"
    assert result["search_attempts"] == 1
    assert result["search_queries"] == ["Who owns the launch?"]
    assert result["search_results"] == []
    assert result["retrieved_evidence"] == []
    assert result["retrieval_stages"] == []
    assert retrieval_events[0]["search_index"] == 1
    assert retrieval_events[0]["result_count"] is None


def test_classic_grounded_answer_preserves_generation_evidence_and_metadata(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "Mina owns the launch.",
    }
    context = [{
        "source": "notes/launch.md",
        "chunk_id": 4,
        "text": "Mina owns the launch.",
        "prompt_block": "file:notes/launch.md\nMina owns the launch.",
    }]
    answer_calls = []

    monkeypatch.setattr(cli, "retrieve", lambda *_args, **_kwargs: [hit])

    def answer_once(query, *, hits, stream=False, **_kwargs):
        answer_calls.append((query, hits, stream))
        return {
            "answer": "Mina owns the launch.",
            "citations": [{"source": "notes/launch.md", "chunk_id": 4}],
            "generation_context": context,
        }

    monkeypatch.setattr(cli, "rag_answer", answer_once)

    result = cli.run_query(object(), object(), query="Who owns the launch?")

    assert answer_calls == [("Who owns the launch?", [hit], False)]
    assert result["answer"] == "Mina owns the launch."
    assert result["reason_code"] == "answered"
    assert result["response_origin"] == "answer_model"
    assert result["citations"] == [
        {"source": "notes/launch.md", "chunk_id": 4}
    ]
    assert result["generation_context"] == context
    assert result["retrieved_evidence"] == [hit]
    assert result["search_queries"] == ["Who owns the launch?"]


def test_classic_model_insufficient_evidence_is_distinct_from_empty_retrieval(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "The launch has an owner.",
    }
    monkeypatch.setattr(cli, "retrieve", lambda *_args, **_kwargs: [hit])
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": ABSTENTION_MESSAGE,
            "citations": [{"source": "notes/launch.md", "chunk_id": 4}],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(object(), object(), query="When is the launch?")

    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["reason_code"] == "insufficient_evidence"
    assert result["response_origin"] == "answer_model"


def test_classic_empty_model_text_returns_non_empty_deterministic_outcome(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "The launch has an owner.",
    }
    monkeypatch.setattr(cli, "retrieve", lambda *_args, **_kwargs: [hit])
    monkeypatch.setattr(
        cli,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "  ",
            "citations": [{"source": "notes/launch.md", "chunk_id": 4}],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(object(), object(), query="When is the launch?")

    assert result["answer"] == cli.NO_EVIDENCE_MESSAGE
    assert result["reason_code"] == "insufficient_evidence"
    assert result["response_origin"] == "deterministic"
    assert result["generation_context"] == [hit]


class _Controller:
    provider_name = "fixture"

    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = 0
        self.prompts = []

    def generate_raw(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def _hybrid_result(
    hits,
    *,
    vector_min_score,
    bm25_min_score_ratio,
    vector_pre_filter_count=2,
    vector_post_filter_count=None,
    fusion_top_k=3,
):
    vector_post_filter_count = (
        len(hits)
        if vector_post_filter_count is None
        else vector_post_filter_count
    )
    return HybridSearchResults(
        vector_results=list(hits),
        bm25_results=list(hits),
        fused_results=list(hits),
        reranked_results=list(hits),
        generation_top_k=1,
        stage_latencies_ms={
            "vector": 1.0,
            "bm25": 2.0,
            "rrf": 3.0,
            "reranked": 4.0,
        },
        vector_top_k=2,
        bm25_top_k=2,
        fusion_top_k=fusion_top_k,
        effective_thresholds={
            "vector_min_score": vector_min_score,
            "bm25_min_score_ratio": bm25_min_score_ratio,
        },
        configured_top_k={
            "vector": 2,
            "bm25": 2,
            "fusion": fusion_top_k,
            "rerank": fusion_top_k,
            "generation": 1,
        },
        filter_diagnostics={
            "vector": {
                "pre_filter_count": vector_pre_filter_count,
                "post_filter_count": vector_post_filter_count,
                "observed_top_k_score_bounds": {
                    "highest": 0.49,
                    "lowest": 0.41,
                },
                "highest_excluded_score": 0.49,
            },
            "bm25": {
                "pre_filter_count": 0,
                "post_filter_count": 0,
                "observed_top_k_score_bounds": None,
                "highest_excluded_score": None,
            },
        },
    )


def test_agentic_replace_discards_active_noise_but_preserves_search_audit(
    monkeypatch,
):
    noise = {
        "source": "notes/noise.md",
        "chunk_id": 1,
        "score": 0.7,
        "text": "An unrelated owner is mentioned.",
        "rank": 1,
        "vector_rank": 1,
        "vector_score": 0.7,
        "bm25_rank": None,
        "bm25_score": None,
        "fusion_score": 1 / 61,
    }
    owner = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.9,
        "text": "Mina owns the launch.",
        "rank": 1,
        "vector_rank": 1,
        "vector_score": 0.9,
        "bm25_rank": 1,
        "bm25_score": 2.0,
        "fusion_score": 2 / 61,
    }
    controller = _Controller([
        '{"action":"search","query":"launch owner",'
        '"vector_min_score":0.3,"bm25_min_score_ratio":0.1,'
        '"evidence_mode":"replace"}',
        '{"action":"answer","evidence":['
        '{"source":"notes/launch.md","chunk_id":4}]}',
        '{"action":"accept"}',
    ])
    searches = iter([
        _hybrid_result(
            [noise],
            vector_min_score=0.5,
            bm25_min_score_ratio=0.25,
        ),
        _hybrid_result(
            [owner],
            vector_min_score=0.3,
            bm25_min_score_ratio=0.1,
        ),
    ])

    class OriginalQuestionReranker:
        def __init__(self):
            self.queries = []

        def predict(self, pairs):
            self.queries.extend(query for query, _text in pairs)
            return [0.9] * len(pairs)

    reranker = OriginalQuestionReranker()
    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(
        agentic_module,
        "retrieve",
        lambda *_args, **_kwargs: next(searches),
    )
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda _query, *, hits, **_kwargs: {
            "answer": "Mina owns the launch.",
            "citations": hits,
            "generation_context": hits,
        },
    )

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        agentic=True,
        cross_encoder=reranker,
    )

    assert [hit["source"] for hit in result["retrieved_evidence"]] == [
        "notes/launch.md"
    ]
    assert result["search_results"] == [[noise], [owner]]
    assert len(result["retrieval_stages"]) == 2
    assert result["retrieval_stages"][0]["stages"]["rrf"]["results"] == [noise]
    rerank_stage = result["retrieval_stages"][1]["stages"]["reranked"]
    assert rerank_stage["provenance"] == {
        "evidence_mode": "replace",
        "candidate_count_before_rerank": 1,
        "active_search_ids": [2],
    }
    assert [hit["source"] for hit in rerank_stage["results"]] == [
        "notes/launch.md"
    ]
    assert set(reranker.queries) == {"Who owns the launch?"}


def test_agentic_extend_deduplicates_and_preserves_multi_search_provenance(
    monkeypatch,
):
    first = {
        "source": "notes/plan.md", "chunk_id": 1, "text": "Phase one.",
        "rank": 1, "score": 0.8,
        "vector_rank": 1, "vector_score": 0.8,
        "bm25_rank": None, "bm25_score": None,
        "fusion_score": 1 / 61,
    }
    shared_first = {
        "source": "notes/shared.md", "chunk_id": 2, "text": "Shared fact.",
        "rank": 2, "score": 0.7,
        "vector_rank": 2, "vector_score": 0.7,
        "bm25_rank": 1, "bm25_score": 3.0,
        "fusion_score": 1 / 62 + 1 / 61,
    }
    shared_second = {
        **shared_first,
        "rank": 1,
        "score": 0.95,
        "vector_rank": 1,
        "vector_score": 0.95,
        "bm25_rank": 2,
        "bm25_score": 4.0,
        "fusion_score": 1 / 61 + 1 / 62,
    }
    second = {
        "source": "notes/budget.md", "chunk_id": 3, "text": "Budget is 20k.",
        "rank": 2, "score": 0.85,
        "vector_rank": 2, "vector_score": 0.85,
        "bm25_rank": 1, "bm25_score": 5.0,
        "fusion_score": 1 / 62 + 1 / 61,
    }
    controller = _Controller([
        '{"action":"search","query":"project budget",'
        '"vector_min_score":0.2,"bm25_min_score_ratio":0.1,'
        '"evidence_mode":"extend"}',
        '{"action":"answer","evidence":['
        '{"source":"notes/shared.md","chunk_id":2},'
        '{"source":"notes/budget.md","chunk_id":3}]}',
        '{"action":"accept"}',
    ])
    searches = iter([
        _hybrid_result(
            [first, shared_first],
            vector_min_score=0.5,
            bm25_min_score_ratio=0.25,
            fusion_top_k=2,
        ),
        _hybrid_result(
            [shared_second, second],
            vector_min_score=0.2,
            bm25_min_score_ratio=0.1,
            fusion_top_k=2,
        ),
    ])

    class OriginalQuestionReranker:
        def __init__(self):
            self.calls = []

        def predict(self, pairs):
            self.calls.append(list(pairs))
            scores = {
                "Phase one.": 0.1,
                "Shared fact.": 0.9,
                "Budget is 20k.": 0.8,
            }
            return [scores[text] for _query, text in pairs]

    reranker = OriginalQuestionReranker()
    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(
        agentic_module,
        "retrieve",
        lambda *_args, **_kwargs: next(searches),
    )
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda _query, *, hits, **_kwargs: {
            "answer": "The shared fact and budget answer the question.",
            "citations": hits,
            "generation_context": hits,
        },
    )

    result = cli.run_query(
        object(),
        object(),
        query="Summarize the project and budget.",
        agentic=True,
        cross_encoder=reranker,
    )

    active = result["retrieved_evidence"]
    assert [(hit["source"], hit["chunk_id"]) for hit in active] == [
        ("notes/shared.md", 2),
        ("notes/budget.md", 3),
    ]
    shared = active[0]
    assert [item["search_id"] for item in shared["search_provenance"]] == [1, 2]
    assert shared["search_provenance"][0]["vector_rank"] == 2
    assert shared["search_provenance"][1]["vector_rank"] == 1
    assert shared["search_provenance"][0]["effective_thresholds"] == {
        "vector_min_score": 0.5,
        "bm25_min_score_ratio": 0.25,
    }
    assert shared["search_provenance"][1]["effective_thresholds"] == {
        "vector_min_score": 0.2,
        "bm25_min_score_ratio": 0.1,
    }
    second_rerank = result["retrieval_stages"][1]["stages"]["reranked"]
    assert second_rerank["top_k"] == 2
    assert second_rerank["returned_count"] == 2
    assert second_rerank["provenance"] == {
        "evidence_mode": "extend",
        "candidate_count_before_rerank": 3,
        "active_search_ids": [1, 2],
    }
    assert all(
        query == "Summarize the project and budget."
        for call in reranker.calls
        for query, _text in call
    )


def test_answer_review_can_reopen_retrieval_without_resetting_budgets(
    monkeypatch,
):
    first = {
        "source": "notes/plan.md",
        "chunk_id": 1,
        "score": 0.8,
        "text": "The project has a launch owner.",
    }
    second = {
        "source": "notes/owner.md",
        "chunk_id": 2,
        "score": 0.9,
        "text": "Mina owns the launch.",
    }
    controller = _Controller([
        '{"action":"answer","evidence":['
        '{"source":"notes/plan.md","chunk_id":1}]}',
        '{"action":"search","query":"launch owner Mina",'
        '"vector_min_score":0.3,"bm25_min_score_ratio":0.1,'
        '"evidence_mode":"extend"}',
        '{"action":"answer","evidence":['
        '{"source":"notes/plan.md","chunk_id":1},'
        '{"source":"notes/owner.md","chunk_id":2}]}',
        '{"action":"accept"}',
    ])
    searches = []
    answers = []

    def retrieve_after_review(_rag, _meta_store, *, query, **kwargs):
        searches.append((query, kwargs))
        return [first] if len(searches) == 1 else [second]

    def generate_candidate(_query, *, hits, **_kwargs):
        answers.append(list(hits))
        return {
            "answer": (
                "REJECTED CANDIDATE"
                if len(answers) == 1
                else "Mina owns the launch."
            ),
            "citations": hits,
            "generation_context": hits,
        }

    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(agentic_module, "retrieve", retrieve_after_review)
    monkeypatch.setattr(agentic_module, "rag_answer", generate_candidate)

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        agentic=True,
    )

    assert searches[1][0] == "launch owner Mina"
    assert searches[1][1]["vector_min_score"] == 0.3
    assert searches[1][1]["bm25_min_score_ratio"] == 0.1
    assert len(answers) == 2
    assert answers[0] == [first]
    assert answers[1] == [first, second]
    assert result["answer"] == "Mina owns the launch."
    assert result["answer"] != "REJECTED CANDIDATE"
    assert result["search_attempts"] == 2
    assert result["answer_attempts"] == 2
    assert result["search_queries"] == [
        "Who owns the launch?",
        "launch owner Mina",
    ]


def test_answer_review_can_request_clarification_without_exposing_candidate(
    monkeypatch,
):
    hit = {
        "source": "notes/decision.md",
        "chunk_id": 1,
        "score": 0.8,
        "text": "The team approved one of the proposals.",
    }
    controller = _Controller([
        '{"action":"answer","evidence":['
        '{"source":"notes/decision.md","chunk_id":1}]}',
        '{"action":"clarify","message":"Which proposal do you mean?"}',
    ])
    answer_calls = []

    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(
        agentic_module,
        "retrieve",
        lambda *_args, **_kwargs: [hit],
    )
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: answer_calls.append("candidate") or {
            "answer": "REJECTED CANDIDATE",
            "citations": [hit],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(
        object(), object(), query="What did they approve?", agentic=True
    )

    assert answer_calls == ["candidate"]
    assert result["answer"] == "Which proposal do you mean?"
    assert result["answer"] != "REJECTED CANDIDATE"
    assert result["citations"] == []
    assert result["reason_code"] == "needs_clarification"
    assert result["response_origin"] == "controller"
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 1


def test_agentic_mixed_cycle_stops_at_exact_search_and_answer_caps(
    monkeypatch,
):
    hits = [
        {
            "source": f"notes/search-{index}.md",
            "chunk_id": index,
            "score": 0.9,
            "text": f"Evidence from search {index}.",
        }
        for index in range(1, 4)
    ]
    controller = _Controller([
        '{"action":"answer","evidence":['
        '{"source":"notes/search-1.md","chunk_id":1}]}',
        '{"action":"search","query":"search two",'
        '"vector_min_score":0.4,"bm25_min_score_ratio":0.2,'
        '"evidence_mode":"replace"}',
        '{"action":"answer","evidence":['
        '{"source":"notes/search-2.md","chunk_id":2}]}',
        '{"action":"search","query":"search three",'
        '"vector_min_score":0.3,"bm25_min_score_ratio":0.1,'
        '"evidence_mode":"replace"}',
        '{"action":"answer","evidence":['
        '{"source":"notes/search-3.md","chunk_id":3}]}',
        '{"action":"accept"}',
    ])
    search_calls = []
    answer_calls = []

    def retrieve_three(*_args, query, **_kwargs):
        search_calls.append(query)
        return [hits[len(search_calls) - 1]]

    def answer_three(_query, *, hits, **_kwargs):
        answer_calls.append(list(hits))
        return {
            "answer": f"Candidate {len(answer_calls)}",
            "citations": hits,
            "generation_context": hits,
        }

    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(agentic_module, "retrieve", retrieve_three)
    monkeypatch.setattr(agentic_module, "rag_answer", answer_three)

    result = cli.run_query(
        object(), object(), query="Original question", agentic=True
    )

    assert search_calls == ["Original question", "search two", "search three"]
    assert len(answer_calls) == 3
    assert result["answer"] == "Candidate 3"
    assert result["search_attempts"] == 3
    assert result["answer_attempts"] == 3


def test_answer_review_search_at_search_cap_retries_without_fourth_search(
    monkeypatch,
):
    hit = {
        "source": "notes/evidence.md",
        "chunk_id": 1,
        "score": 0.9,
        "text": "Partial evidence.",
    }
    illegal_search = (
        '{"action":"search","query":"fourth search",'
        '"vector_min_score":0.1,"bm25_min_score_ratio":0.1,'
        '"evidence_mode":"extend"}'
    )
    controller = _Controller([
        '{"action":"search","query":"second search",'
        '"vector_min_score":0.4,"bm25_min_score_ratio":0.2,'
        '"evidence_mode":"replace"}',
        '{"action":"search","query":"third search",'
        '"vector_min_score":0.3,"bm25_min_score_ratio":0.1,'
        '"evidence_mode":"replace"}',
        '{"action":"answer","evidence":['
        '{"source":"notes/evidence.md","chunk_id":1}]}',
        illegal_search,
        '{"action":"abstain"}',
    ])
    search_calls = []

    def retrieve_three(*_args, query, **_kwargs):
        search_calls.append(query)
        return [hit]

    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(agentic_module, "retrieve", retrieve_three)
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "REJECTED CANDIDATE",
            "citations": [hit],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(object(), object(), query="Question", agentic=True)

    assert search_calls == ["Question", "second search", "third search"]
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["answer"] != "REJECTED CANDIDATE"
    assert result["search_attempts"] == 3
    assert result["answer_attempts"] == 2
    retry_prompt = json.loads(controller.prompts[-1][1]["content"])
    assert "validation_feedback" in retry_prompt
    assert retry_prompt["searches_remaining"] == 0


def test_answer_review_search_at_answer_cap_degrades_without_extra_work(
    monkeypatch,
):
    hit = {
        "source": "notes/evidence.md",
        "chunk_id": 1,
        "score": 0.9,
        "text": "Partial evidence.",
    }
    illegal_search = (
        '{"action":"search","query":"more evidence",'
        '"vector_min_score":0.1,"bm25_min_score_ratio":0.1,'
        '"evidence_mode":"extend"}'
    )
    controller = _Controller([
        '{"action":"answer","evidence":['
        '{"source":"notes/evidence.md","chunk_id":1}]}',
        '{"action":"revise","feedback":"Try again."}',
        '{"action":"revise","feedback":"Still incomplete."}',
        illegal_search,
        illegal_search,
    ])
    search_calls = []
    answer_calls = []

    def answer_three(*_args, **_kwargs):
        answer_calls.append("candidate")
        return {
            "answer": "REJECTED CANDIDATE",
            "citations": [hit],
            "generation_context": [hit],
        }

    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(
        agentic_module,
        "retrieve",
        lambda *_args, **_kwargs: search_calls.append("search") or [hit],
    )
    monkeypatch.setattr(agentic_module, "rag_answer", answer_three)

    result = cli.run_query(object(), object(), query="Question", agentic=True)

    assert search_calls == ["search"]
    assert len(answer_calls) == 3
    assert result["reason_code"] == "controller_failure"
    assert result["answer"] != "REJECTED CANDIDATE"
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 3


def test_agentic_empty_initial_retrieval_does_not_trigger_adaptive_search(
    monkeypatch,
):
    controller = _Controller([])
    searches = []

    def retrieve_adaptively(_rag, _meta_store, *, query, **kwargs):
        searches.append((query, kwargs))
        return _hybrid_result(
            [],
            vector_min_score=0.5,
            bm25_min_score_ratio=0.25,
            vector_post_filter_count=0,
        )

    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(agentic_module, "retrieve", retrieve_adaptively)
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: pytest.fail(
            "empty initial Retrieval must not invoke Answer generation"
        ),
    )

    result = cli.run_query(
        object(),
        object(),
        query="Who owns it?",
        agentic=True,
    )

    assert searches[0][0] == "Who owns it?"
    assert len(searches) == 1
    assert controller.calls == 0
    assert result["reason_code"] == "no_retrieval_results"
    assert result["response_origin"] == "deterministic"
    assert result["search_queries"] == ["Who owns it?"]
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 0
    assert result["generation_context"] == []
    assert result["retrieval_stages"][0]["evidence_mode"] == "replace"


@pytest.mark.parametrize(
    ("query", "clarification"),
    [
        ("asdf qwer zxcv", "Please clarify what information you want from the notes."),
        ("What did they decide?", "Which person or project does 'they' refer to?"),
    ],
)
def test_agentic_unclear_query_returns_normal_clarification(
    monkeypatch,
    query,
    clarification,
):
    hit = {
        "source": "notes/evidence.md",
        "chunk_id": 1,
        "score": 0.9,
        "text": "The notes contain a referenced decision.",
    }
    controller = _Controller([
        '{"action":"clarify","message":'
        + json.dumps(clarification)
        + "}",
    ])
    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(
        agentic_module,
        "retrieve",
        lambda *_args, **_kwargs: _hybrid_result(
            [hit],
            vector_min_score=0.5,
            bm25_min_score_ratio=0.25,
            vector_post_filter_count=1,
        ),
    )
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "Buffered candidate.",
            "citations": [hit],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(object(), object(), query=query, agentic=True)

    assert result["answer"] == clarification
    assert result["reason_code"] == "needs_clarification"
    assert result["response_origin"] == "controller"
    assert "workflow_error" not in result
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 1
    assert result["answer"] == clarification
    assert result["answer"] != "Buffered candidate."
    assert result["search_queries"] == [query]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("vector_min_score", 1.1),
        ("bm25_min_score_ratio", -0.1),
        ("vector_min_score", "NaN"),
    ],
)
def test_agentic_invalid_adaptive_threshold_retries_without_search(
    monkeypatch,
    field,
    invalid_value,
):
    hit = {
        "source": "notes/evidence.md",
        "chunk_id": 1,
        "score": 0.9,
        "text": "The answer is not stated in this partial note.",
    }
    invalid_action = {
        "action": "search",
        "query": "unsafe search",
        "vector_min_score": 0.2,
        "bm25_min_score_ratio": 0.1,
        "evidence_mode": "replace",
    }
    invalid_action[field] = invalid_value
    controller = _Controller([
        json.dumps(invalid_action),
        '{"action":"clarify","message":"Please clarify the subject."}',
    ])
    retrieval_calls = []

    def retrieve_once(*_args, **_kwargs):
        retrieval_calls.append("search")
        return _hybrid_result(
            [hit],
            vector_min_score=0.5,
            bm25_min_score_ratio=0.25,
            vector_post_filter_count=0,
        )

    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(agentic_module, "retrieve", retrieve_once)
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "Buffered candidate.",
            "citations": [hit],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(
        object(),
        object(),
        query="What did they decide?",
        agentic=True,
    )

    assert retrieval_calls == ["search"]
    assert controller.calls == 2
    assert result["reason_code"] == "needs_clarification"
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 1
    retry_input = controller.prompts[1][1]["content"]
    assert '"validation_feedback"' in retry_input


def test_agentic_can_raise_thresholds_without_changing_top_k(monkeypatch):
    noisy_hit = {
        "source": "notes/noise.md",
        "chunk_id": 1,
        "score": 0.3,
        "text": "Unrelated launch mention.",
    }
    controller = _Controller([
        '{"action":"search","query":"specific launch owner",'
        '"vector_min_score":0.8,"bm25_min_score_ratio":0.9,'
        '"evidence_mode":"replace","fusion_top_k":999}',
        '{"action":"abstain"}',
    ])
    searches = []

    def retrieve_with_narrowing(_rag, _meta_store, *, query, **kwargs):
        searches.append((query, kwargs))
        if len(searches) == 1:
            return _hybrid_result(
                [noisy_hit],
                vector_min_score=0.2,
                bm25_min_score_ratio=0.1,
            )
        return _hybrid_result(
            [],
            vector_min_score=kwargs["vector_min_score"],
            bm25_min_score_ratio=kwargs["bm25_min_score_ratio"],
            vector_post_filter_count=0,
        )

    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(agentic_module, "retrieve", retrieve_with_narrowing)
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "Buffered candidate.",
            "citations": [noisy_hit],
            "generation_context": [noisy_hit],
        },
    )

    result = cli.run_query(
        object(), object(), query="Who owns the launch?", agentic=True
    )

    assert searches[1][1]["vector_min_score"] == 0.8
    assert searches[1][1]["bm25_min_score_ratio"] == 0.9
    assert "fusion_top_k" not in searches[1][1]
    assert result["search_attempts"] == 2
    assert result["retrieval_stages"][1]["configured_top_k"] == {
        "vector": 2,
        "bm25": 2,
        "fusion": 3,
        "rerank": 3,
        "generation": 1,
    }


@pytest.mark.parametrize(
    ("responses", "failure_category"),
    [
        (["not json", "still not json"], "invalid_json"),
        (["{}", "{}"], "schema_invalid_action"),
    ],
)
def test_agentic_invalid_actions_return_degraded_outcome_with_search_trace(
    monkeypatch,
    responses,
    failure_category,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "Mina owns the launch.",
    }
    controller = _Controller(responses)
    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(
        agentic_module,
        "retrieve",
        lambda *_args, **_kwargs: [hit],
    )
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "UNREVIEWED CANDIDATE",
            "citations": [hit],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        agentic=True,
    )

    assert controller.calls == 2
    assert result["answer"].strip()
    assert result["reason_code"] == "controller_failure"
    assert result["response_origin"] == "deterministic"
    assert result["workflow_error"] == "controller_failure"
    assert result["controller_failure_category"] == failure_category
    assert result["citations"] == []
    assert result["retrieved_evidence"] == [hit]
    assert result["search_results"] == [[hit]]
    assert result["search_queries"] == ["Who owns the launch?"]
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 1
    assert result["answer"] != "UNREVIEWED CANDIDATE"
    assert "UNREVIEWED CANDIDATE" not in repr(result)
    assert all(response not in repr(result) for response in responses)


def test_agentic_invented_evidence_retries_then_returns_degraded_outcome(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "Mina owns the launch.",
    }
    invented_action = (
        '{"action":"expand_evidence","evidence":['
        '{"source":"notes/invented.md","chunk_id":99}]}'
    )
    controller = _Controller([invented_action, invented_action])
    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(
        agentic_module,
        "retrieve",
        lambda *_args, **_kwargs: [hit],
    )
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "UNREVIEWED CANDIDATE",
            "citations": [hit],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        agentic=True,
    )

    assert controller.calls == 2
    assert result["reason_code"] == "controller_failure"
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 1
    assert result["retrieved_evidence"] == [hit]
    assert result["controller_failure_category"] == "invalid_evidence_reference"
    assert result["answer"] != "UNREVIEWED CANDIDATE"
    assert "UNREVIEWED CANDIDATE" not in repr(result)
    assert invented_action not in repr(result)


def test_agentic_answer_review_rejects_search_when_no_answer_attempt_remains(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "Mina owns the launch.",
    }
    settings = SimpleNamespace(
        agentic_max_search_attempts=3,
        agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=5),
    )
    monkeypatch.setattr(agentic_module, "get_settings", lambda: settings)
    search_action = (
        '{"action":"search","query":"launch owner",'
        '"vector_min_score":0.5,"bm25_min_score_ratio":0.25,'
        '"evidence_mode":"extend"}'
    )
    controller = _Controller([search_action, search_action])
    retrieval_calls = []
    monkeypatch.setattr(
        agentic_module,
        "get_llm",
        lambda: controller,
    )
    monkeypatch.setattr(
        agentic_module,
        "retrieve",
        lambda *_args, **_kwargs: retrieval_calls.append("search") or [hit],
    )
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "UNREVIEWED CANDIDATE",
            "citations": [hit],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        agentic=True,
    )

    assert retrieval_calls == ["search"]
    assert result["search_attempts"] == 1
    assert result["answer_attempts"] == 1
    assert result["controller_failure_category"] == "out_of_budget_action"
    assert result["answer"] != "UNREVIEWED CANDIDATE"
    assert "UNREVIEWED CANDIDATE" not in repr(result)
    assert search_action not in repr(result)


def test_agentic_retrieval_exception_is_not_reported_as_empty_search(
    monkeypatch,
):
    controller_calls = []
    retrieval_events = []

    monkeypatch.setattr(
        agentic_module,
        "get_llm",
        lambda: controller_calls.append("called"),
    )

    def failed_retrieval(*_args, **_kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(agentic_module, "retrieve", failed_retrieval)

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        agentic=True,
        on_retrieval=retrieval_events.append,
    )

    assert controller_calls == []
    assert result["answer"].strip()
    assert result["reason_code"] == "retrieval_failure"
    assert result["workflow_error"] == "retrieval_failure"
    assert result["search_queries"] == ["Who owns the launch?"]
    assert result["search_results"] == []
    assert result["search_attempts"] == 1
    assert retrieval_events[0]["result_count"] is None


def test_agentic_rerank_failure_returns_degraded_outcome_with_search_trace(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "Mina owns the launch.",
        "rank": 1,
        "vector_rank": 1,
        "vector_score": 0.91,
        "bm25_rank": 1,
        "bm25_score": 2.0,
        "fusion_score": 0.03,
    }
    hybrid = _hybrid_result(
        [hit],
        vector_min_score=0.3,
        bm25_min_score_ratio=0.0,
    )

    monkeypatch.setattr(agentic_module, "retrieve", lambda *_args, **_kwargs: hybrid)

    def failed_rerank(*_args, **_kwargs):
        raise RuntimeError("cross-encoder unavailable")

    monkeypatch.setattr(
        agentic_module,
        "rerank_candidates",
        failed_rerank,
    )
    monkeypatch.setattr(
        agentic_module,
        "get_llm",
        lambda: pytest.fail("controller must not run after reranking fails"),
    )
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: pytest.fail(
            "Answer generation must not run after reranking fails"
        ),
    )

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        agentic=True,
    )

    assert result["answer"] == agentic_module.RETRIEVAL_FAILURE_MESSAGE
    assert result["reason_code"] == "retrieval_failure"
    assert result["response_origin"] == "deterministic"
    assert result["workflow_error"] == "retrieval_failure"
    assert result["workflow_error_type"] == "RuntimeError"
    assert result["search_attempts"] == 1
    assert result["search_queries"] == ["Who owns the launch?"]
    assert result["search_results"] == [[hit]]
    assert result["retrieved_evidence"] == [hit]
    assert len(result["retrieval_stages"]) == 1
    assert result["retrieval_stages"][0]["search_query"] == (
        "Who owns the launch?"
    )
    assert result["retrieval_stages"][0]["stages"]["vector"]["results"] == [hit]
    assert result["retrieval_stages"][0]["stages"]["bm25"]["results"] == [hit]
    assert result["retrieval_stages"][0]["stages"]["rrf"]["results"] == [hit]


def test_agentic_later_retrieval_failure_retains_completed_search(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "Mina owns the launch.",
    }
    controller = _Controller([
        '{"action":"search","query":"launch backup owner",'
        '"vector_min_score":0.4,"bm25_min_score_ratio":0.2,'
        '"evidence_mode":"replace"}',
    ])
    retrieval_calls = 0

    def retrieve_then_fail(*_args, **_kwargs):
        nonlocal retrieval_calls
        retrieval_calls += 1
        if retrieval_calls == 1:
            return [hit]
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(agentic_module, "retrieve", retrieve_then_fail)
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "Buffered candidate.",
            "citations": [hit],
            "generation_context": [hit],
        },
    )

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        agentic=True,
    )

    assert result["reason_code"] == "retrieval_failure"
    assert result["search_attempts"] == 2
    assert result["search_queries"] == [
        "Who owns the launch?",
        "launch backup owner",
    ]
    assert result["search_results"] == [[hit]]
    assert result["retrieved_evidence"] == [hit]


def test_agentic_answer_review_failure_hides_candidate_and_retains_hybrid_trace(
    monkeypatch,
):
    hit = {
        "source": "notes/launch.md",
        "chunk_id": 4,
        "score": 0.91,
        "text": "Mina owns the launch.",
        "rank": 1,
        "vector_rank": 1,
        "vector_score": 0.91,
        "bm25_rank": 1,
        "bm25_score": 2.0,
        "fusion_score": 0.03,
    }
    hybrid = HybridSearchResults(
        vector_results=[hit],
        bm25_results=[hit],
        fused_results=[hit],
        reranked_results=[hit],
        generation_top_k=1,
        stage_latencies_ms={
            "vector": 1.0,
            "bm25": 2.0,
            "rrf": 3.0,
            "reranked": 4.0,
        },
        vector_top_k=2,
        bm25_top_k=2,
        fusion_top_k=3,
    )
    controller = _Controller([
        '{"action":"answer","evidence":['
        '{"source":"notes/launch.md","chunk_id":4}]}',
        RuntimeError("controller unavailable"),
    ])
    monkeypatch.setattr(agentic_module, "get_llm", lambda: controller)
    monkeypatch.setattr(agentic_module, "retrieve", lambda *_args, **_kwargs: hybrid)
    monkeypatch.setattr(
        agentic_module,
        "rag_answer",
        lambda *_args, **_kwargs: {
            "answer": "UNREVIEWED CANDIDATE",
            "citations": [hit],
            "generation_context": [hit],
        },
    )

    class CrossEncoder:
        def predict(self, _pairs):
            return [0.95]

    result = cli.run_query(
        object(),
        object(),
        query="Who owns the launch?",
        agentic=True,
        cross_encoder=CrossEncoder(),
    )

    assert controller.calls == 2
    assert result["answer"] != "UNREVIEWED CANDIDATE"
    assert result["reason_code"] == "controller_failure"
    assert result["workflow_error_type"] == "RuntimeError"
    assert result["controller_failure_category"] == "provider_failure"
    assert result["citations"] == []
    assert result["answer_attempts"] == 1
    assert "controller unavailable" not in repr(result)
    assert "UNREVIEWED CANDIDATE" not in repr(result)
    assert result["generation_context"] == [hit]
    assert result["retrieved_evidence"][0]["source"] == "notes/launch.md"
    assert result["retrieval_stages"][0]["stages"]["vector"][
        "returned_count"
    ] == 1


def test_classic_repl_displays_invalid_input_outcome(monkeypatch, capsys):
    inputs = iter(["   ", ":q"])

    class MetaStore:
        def close(self):
            pass

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    cli.repl(object(), MetaStore())

    output = capsys.readouterr().out
    assert cli.INVALID_QUERY_MESSAGE in output


def test_classic_one_shot_displays_invalid_input_outcome(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        cli,
        "retrieve",
        lambda *_args, **_kwargs: pytest.fail("blank input must not run Retrieval"),
    )

    cli.run_onetime(object(), object(), query="   ")

    output = capsys.readouterr().out
    assert cli.INVALID_QUERY_MESSAGE in output


def test_cli_routes_submitted_whitespace_to_classic_one_shot(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["rag-app", "   "])
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: SimpleNamespace(model_dump_json=lambda: "{}"),
    )
    monkeypatch.setattr(cli, "load_or_build_index", lambda: "index")
    monkeypatch.setattr(cli, "MetaStore", lambda: "meta-store")
    monkeypatch.setattr(
        cli,
        "run_onetime",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    cli.main()

    assert calls == [((), {
        "query": "",
        "citations": False,
        "rag": "index",
        "meta_store": "meta-store",
        "agentic": False,
    })]
