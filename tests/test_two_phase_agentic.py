import json
from types import SimpleNamespace

import pytest

from rag_notes_helper import cli
from rag_notes_helper.rag import agentic


def test_query_reviews_large_complementary_pool_and_stops_after_second_search(monkeypatch):
    settings = SimpleNamespace(
        agentic_max_search_attempts=3, agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=1, max_tokens=64),
    )
    monkeypatch.setattr(agentic, "get_settings", lambda: settings)
    question = "Who owns launch and when?"
    owner = {"source": "owners.md", "chunk_id": 0, "text": "Lena owns launch. " + "x" * 26000, "score": .9}
    date = {"source": "dates.md", "chunk_id": 0, "text": "Launch is June 1. " + "y" * 26000, "score": .8}
    searches, reviews = [], []

    def search(query, **kwargs):
        searches.append(query)
        return [owner] if len(searches) == 1 else [owner, date]

    class Controller:
        provider_name = "fixture"
        model = "fixture"

        def generate_raw(self, messages):
            payload = json.loads(messages[-1]["content"])
            if "chunks_pool" not in payload:
                return json.dumps({"action": "accept"})
            reviews.append(payload)
            if len(reviews) == 1:
                return json.dumps({
                    "coverage": "partial_evidence", "missing_information": ["launch date"],
                    "next_search": {"query": "launch date", "vector_min_score": .2, "bm25_min_score_ratio": .1},
                })
            return json.dumps({"coverage": "sufficient_evidence", "missing_information": []})

    monkeypatch.setattr(agentic, "search_notes", search)
    monkeypatch.setattr(agentic, "get_llm", Controller)
    monkeypatch.setattr(agentic, "rerank_candidates", lambda query, candidates, **kwargs: candidates)
    monkeypatch.setattr(agentic, "rag_answer", lambda query, *, hits, **kwargs: {
        "answer": "Lena owns launch. The date is not in selected evidence.",
        "citations": [], "generation_context": hits,
    })

    result = cli.run_query(object(), object(), query=question, agentic=True)

    assert result["review_passed"] is True
    assert searches == [question, "launch date"]
    assert [review["question"] for review in reviews] == [question, question, question]
    assert len(reviews[2]["chunks_pool"]) == 1
    assert reviews[1]["last_search"]["query"] == "launch date"
    assert reviews[1]["chunks_pool"] == [
        {key: chunk[key] for key in ("source", "chunk_id", "text")}
        for chunk in (owner, date)
    ]
    assert [chunk["source"] for chunk in result["chunks_pool"]] == ["owners.md", "dates.md"]
    assert [entry["search_id"] for entry in result["chunks_pool"][0]["search_provenance"]] == [1, 2]
    assert result["retrieval_stop_reason"] == "sufficient_evidence"
    assert result["search_attempts"] == 2
    assert len(result["selected_pool"]) == 1


@pytest.mark.parametrize("pool_size", [1, 2, 3])
def test_query_reviews_pool_before_selecting_generation_context(monkeypatch, pool_size):
    settings = SimpleNamespace(
        agentic_max_search_attempts=3, agentic_max_answer_attempts=3,
        llm=SimpleNamespace(max_chunks=2, max_tokens=1024),
    )
    monkeypatch.setattr(agentic, "get_settings", lambda: settings)
    chunks = [
        {"source": "release.md", "chunk_id": i, "text": f"Fact {i}", "score": .9}
        for i in range(pool_size)
    ]
    events = []

    def search(query, **kwargs):
        events.append("search")
        assert query == "Who owns launch?"
        return chunks + [chunks[0]]

    monkeypatch.setattr(agentic, "search_notes", search)

    class Controller:
        provider_name = "fixture"
        model = "fixture"

        def generate_raw(self, messages):
            payload = json.loads(messages[-1]["content"])
            if "chunks_pool" in payload:
                events.append("retrieval_review")
                assert len(payload["chunks_pool"]) == (min(2, pool_size) if payload.get("phase") == "generation_context_review" else pool_size)
                assert payload["question"] == "Who owns launch?"
                return json.dumps({"coverage": "sufficient_evidence", "missing_information": []})
            events.append("answer_review")
            assert payload["candidate_answer"] == "Lena owns launch."
            return json.dumps({"action": "accept"})

    monkeypatch.setattr(agentic, "get_llm", Controller)
    reranks = []

    def rerank(query, candidates, **kwargs):
        reranks.append(query)
        return [{**c, "reranker_score": float(i)} for i, c in enumerate(reversed(candidates))]

    monkeypatch.setattr(agentic, "rerank_candidates", rerank)

    def answer(query, *, hits, stream, **kwargs):
        events.append("answer")
        assert stream is False
        return {
            "answer": "Lena owns launch.", "citations": [],
            "generation_context": [{**hit, "prompt_block": hit["text"]} for hit in hits],
        }

    monkeypatch.setattr(agentic, "rag_answer", answer)
    calls = []
    result = cli.run_query(object(), object(), query="Who owns launch?", agentic=True, on_generation=calls.append)
    assert events == ["search", "retrieval_review"] + (["retrieval_review"] if pool_size > 2 else []) + ["answer", "answer_review"]
    assert result["retrieval_coverage"] == {
        "assessed_question": "Who owns launch?", "coverage": "sufficient_evidence", "missing_information": [],
    }
    assert result["retrieval_stop_reason"] == "sufficient_evidence"
    assert len(result["chunks_pool"]) == pool_size
    assert [c["chunk_id"] for c in result["chunks_pool"]] == list(range(pool_size))
    assert all(c["search_provenance"][0]["search_id"] == 1 for c in result["chunks_pool"])
    assert len(result["selected_pool"]) == min(2, pool_size)
    assert result["final_selection"]["rerank_performed"] is (pool_size > 2)
    assert reranks == (["Who owns launch?"] if pool_size > 2 else [])
    assert [c["chunk_id"] for c in result["generation_context"]] == [c["chunk_id"] for c in result["selected_pool"]]
    assert result["retrieval_reviews"][0]["search_id"] == 1
    assert [c["phase"] for c in calls] == ["retrieval_review"] + (["generation_context_review"] if pool_size > 2 else []) + ["answer_review"]
    assert result["answer"] == "Lena owns launch."


@pytest.mark.parametrize("cap,final_coverage", [(1, "partial_evidence"), (3, "sufficient_evidence"), (3, "partial_evidence"), (5, "sufficient_evidence")])
def test_query_accumulates_rewritten_searches_with_independent_attempt_cap(monkeypatch, cap, final_coverage):
    settings = SimpleNamespace(
        agentic_max_search_attempts=cap, agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=10, max_tokens=1024),
    )
    monkeypatch.setattr(agentic, "get_settings", lambda: settings)
    seen = []
    hit = {"source": "release.md", "chunk_id": 1, "text": "Lena owns launch.", "score": .9}

    def search(query, **kwargs):
        seen.append((query, {k: v for k, v in kwargs.items() if k in {"vector_min_score", "bm25_min_score_ratio"}}))
        if len(seen) == 1:
            return [hit]
        if len(seen) == 2:
            return []
        return [hit, {"source": "timeline.md", "chunk_id": 2, "text": "Launch is June.", "score": .8}]

    monkeypatch.setattr(agentic, "search_notes", search)

    class Controller:
        provider_name = "fixture"
        model = "fixture"

        def generate_raw(self, messages):
            payload = json.loads(messages[-1]["content"])
            if "chunks_pool" not in payload:
                return json.dumps({"action": "accept"})
            assert payload["last_search"]["query"] == seen[-1][0]
            assert payload["question"] == "Who owns launch and when?"
            assert payload["searches_used"] == len(seen)
            coverage = final_coverage if len(seen) == cap else "partial_evidence"
            return json.dumps({
                "coverage": coverage, "missing_information": ["launch date"],
                "next_search": {"query": f"launch date {len(seen)}", "vector_min_score": .2, "bm25_min_score_ratio": .1},
            })

    monkeypatch.setattr(agentic, "get_llm", Controller)
    generated = []

    def answer(query, *, hits, **kwargs):
        generated.append(hits)
        return {"answer": "Lena owns launch; the date may be missing.", "citations": [], "generation_context": hits}

    monkeypatch.setattr(agentic, "rag_answer", answer)
    result = cli.run_query(object(), object(), query="Who owns launch and when?", agentic=True)
    assert len(seen) == cap
    assert result["search_attempts"] == cap
    assert result["answer_attempts"] == 1
    assert len(generated) == 1
    assert result["retrieval_stop_reason"] == ("sufficient_evidence" if final_coverage == "sufficient_evidence" else "search_budget_exhausted")
    assert result["retrieval_coverage"]["coverage"] == final_coverage
    assert len(result["retrieval_reviews"]) == cap
    assert seen[0] == ("Who owns launch and when?", {})
    if cap > 1:
        assert seen[1] == ("launch date 1", {"vector_min_score": .2, "bm25_min_score_ratio": .1})
    pool = result["chunks_pool"]
    assert [c["source"] for c in pool] == (["release.md"] if cap == 1 else ["release.md", "timeline.md"])
    assert [p["search_id"] for p in pool[0]["search_provenance"]] == ([1] if cap == 1 else [1, *range(3, cap + 1)])


@pytest.mark.parametrize("scenario", ["invalid_review", "provider_failure", "provider_context_failure", "invalid_then_valid", "repeated_plan"])
def test_query_handles_empty_pools_and_retrieval_review_degradation(monkeypatch, scenario):
    settings = SimpleNamespace(
        agentic_max_search_attempts=3, agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=1 if scenario == "provider_context_failure" else 5, max_tokens=64),
    )
    monkeypatch.setattr(agentic, "get_settings", lambda: settings)
    hit = {"source": "release.md", "chunk_id": 1, "text": "Lena owns launch.", "score": .9}
    empty_first = scenario in {"intelligible_empty", "unintelligible", "empty_then_found", "intelligibility_invalid", "intelligibility_failure"}
    searches = []

    def search(query, **kwargs):
        searches.append(query)
        if empty_first and (scenario != "empty_then_found" or len(searches) == 1):
            return []
        if scenario == "provider_context_failure":
            return [hit, {"source": "noise.md", "chunk_id": 0, "text": "x" * 20000, "score": .1}]
        return [hit]

    monkeypatch.setattr(agentic, "search_notes", search)
    if scenario == "provider_context_failure":
        monkeypatch.setattr(agentic, "rerank_candidates", lambda query, candidates, **kwargs: candidates)
    reviews = []
    checks = []

    class Controller:
        provider_name = "fixture"
        model = "fixture"

        def generate_raw(self, messages):
            payload = json.loads(messages[-1]["content"])
            if payload.get("phase") == "intelligibility":
                checks.append(payload["question"])
                if scenario == "unintelligible":
                    assert payload["conversation_history"] == [{"user": "x" * 60000, "assistant": "Earlier intent"}]
                if scenario == "intelligibility_invalid":
                    return "not JSON"
                if scenario == "intelligibility_failure":
                    raise RuntimeError("unavailable")
                return json.dumps({"intelligible": scenario != "unintelligible"})
            if "chunks_pool" in payload:
                reviews.append(payload)
                if scenario in {"provider_failure", "provider_context_failure"}:
                    raise RuntimeError("provider context length exceeded" if scenario == "provider_context_failure" else "provider unavailable")
                if scenario == "invalid_review" or (scenario == "invalid_then_valid" and len(reviews) == 1):
                    return "not json"
                if scenario == "repeated_plan":
                    return json.dumps({
                        "coverage": "partial_evidence", "missing_information": ["date"],
                        "next_search": {"query": "Who owns launch?", "vector_min_score": .3, "bm25_min_score_ratio": 0.0},
                    })
                return json.dumps({
                    "coverage": "sufficient_evidence" if payload["chunks_pool"] else "zero_evidence",
                    "missing_information": [],
                    "next_search": {"query": f"owner {len(searches)}", "vector_min_score": .1, "bm25_min_score_ratio": 0.0},
                })
            return json.dumps({"action": "accept"})

    monkeypatch.setattr(agentic, "get_llm", Controller)
    answers = []

    def answer(query, *, hits, **kwargs):
        answers.append(hits)
        return {"answer": "Lena owns launch.", "citations": [], "generation_context": hits}

    monkeypatch.setattr(agentic, "rag_answer", answer)
    history = [("x" * 60000, "Earlier intent")] if scenario == "unintelligible" else []
    result = cli.run_query(object(), object(), query="Who owns launch?", agentic=True, conversation_history=history)
    if scenario in {"intelligible_empty", "unintelligible", "intelligibility_invalid", "intelligibility_failure"}:
        assert not answers
        assert result["answer_attempts"] == 0
        assert result["reason_code"] == ("needs_clarification" if scenario == "unintelligible" else "no_retrieval_results")
        assert len(searches) == (3 if scenario == "intelligible_empty" else 1)
    elif scenario in {"invalid_review", "provider_failure", "provider_context_failure", "repeated_plan"}:
        assert not answers
        assert result["reason_code"] == "controller_failure"
        assert result["workflow_error"] == "controller_failure"
    else:
        assert len(answers) == 1
        assert result["answer"] == "Lena owns launch."
    assert len(checks) == (2 if scenario == "intelligibility_invalid" else 1 if empty_first else 0)
    if scenario.startswith("intelligibility_"):
        assert result["workflow_degraded"] is True
        assert result["retrieval_coverage"]["coverage"] == "unknown"
        assert not reviews
    if scenario in {"invalid_review", "provider_failure", "provider_context_failure", "repeated_plan"}:
        assert result["retrieval_coverage"]["coverage"] == "unknown"
        assert result["retrieval_stop_reason"] == "review_degraded"
        assert result["workflow_degraded"] is True
        assert result["retrieval_degradation"]["cause"] == {
            "invalid_review": "invalid_json", "provider_failure": "provider_failure", "provider_context_failure": "provider_failure", "repeated_plan": "schema_invalid_action",
        }[scenario]
        assert len(searches) == 1
        assert len(reviews) == {"invalid_review": 2, "provider_failure": 1, "provider_context_failure": 1, "repeated_plan": 2}[scenario]
    if scenario == "provider_context_failure":
        assert len(result["chunks_pool"]) == 2
        assert len(result["selected_pool"]) == 1
        assert reviews[0]["chunks_pool"][1]["text"] == "x" * 20000
    if scenario == "invalid_then_valid":
        assert len(reviews) == 2
        assert "validation_feedback" in reviews[1]
        assert result["retrieval_coverage"]["coverage"] == "sufficient_evidence"


@pytest.mark.parametrize("fault", ["first_search", "later_search", "final_rerank"])
def test_query_preserves_safe_trace_for_search_and_rerank_failures(monkeypatch, fault):
    settings = SimpleNamespace(
        agentic_max_search_attempts=3, agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=1, max_tokens=64),
    )
    monkeypatch.setattr(agentic, "get_settings", lambda: settings)
    hits = [{"source": "release.md", "chunk_id": i, "text": f"Fact {i}", "score": .9} for i in range(2)]
    calls = []

    def search(query, **kwargs):
        calls.append(query)
        if fault == "first_search" or (fault == "later_search" and len(calls) == 2):
            raise RuntimeError("SECRET provider error")
        return hits

    monkeypatch.setattr(agentic, "search_notes", search)

    class Controller:
        provider_name = "fixture"
        model = "fixture"

        def generate_raw(self, messages):
            return json.dumps({
                "coverage": "partial_evidence" if fault == "later_search" else "sufficient_evidence",
                "missing_information": [],
                "next_search": {"query": "missing date", "vector_min_score": .1, "bm25_min_score_ratio": 0.0},
            })

    monkeypatch.setattr(agentic, "get_llm", Controller)
    monkeypatch.setattr(agentic, "rag_answer", lambda *a, **k: pytest.fail("failed retrieval must not generate"))
    monkeypatch.setattr(agentic, "rerank_candidates", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("SECRET encoder error")))
    result = cli.run_query(object(), object(), query="Who owns launch?", agentic=True)
    assert result["reason_code"] == "retrieval_failure"
    assert result["workflow_error"] == "retrieval_failure"
    assert result["search_attempts"] == (2 if fault == "later_search" else 1)
    assert "SECRET" not in repr(result)
    assert len(result["search_results"]) == (0 if fault == "first_search" else 1)
    if fault != "first_search":
        assert len(result["chunks_pool"]) == 2
        assert len(result["retrieved_evidence"]) == 2


def test_query_retries_invalid_threshold_without_consuming_search_attempt(monkeypatch):
    settings = SimpleNamespace(
        agentic_max_search_attempts=2, agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=5, max_tokens=64),
    )
    monkeypatch.setattr(agentic, "get_settings", lambda: settings)
    hit = {"source": "release.md", "chunk_id": 1, "text": "Fact", "score": .9}
    searches = []
    monkeypatch.setattr(agentic, "search_notes", lambda query, **kwargs: searches.append(query) or [hit])
    reviews = []

    class Controller:
        provider_name = "fixture"
        model = "fixture"

        def generate_raw(self, messages):
            payload = json.loads(messages[-1]["content"])
            if "chunks_pool" not in payload:
                return '{"action":"accept"}'
            reviews.append(payload)
            return json.dumps({
                "coverage": "partial_evidence" if len(searches) == 1 else "sufficient_evidence",
                "missing_information": ["date"],
                "next_search": {"query": "date", "vector_min_score": 1.1 if len(reviews) == 1 else .1, "bm25_min_score_ratio": 0.0},
            })

    monkeypatch.setattr(agentic, "get_llm", Controller)
    monkeypatch.setattr(agentic, "rag_answer", lambda *a, **k: {"answer": "Fact", "citations": [], "generation_context": [hit]})
    result = cli.run_query(object(), object(), query="question", agentic=True)
    assert result["search_attempts"] == 2
    assert len(reviews) == 3
    assert "validation_feedback" in reviews[1]
    assert result["retrieval_stop_reason"] == "sufficient_evidence"
