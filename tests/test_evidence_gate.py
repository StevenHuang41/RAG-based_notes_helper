import json
from types import SimpleNamespace

from rag_notes_helper.rag import agentic


def test_zero_evidence_exhausts_search_without_answer(monkeypatch):
    monkeypatch.setattr(agentic, "get_settings", lambda: SimpleNamespace(
        agentic_max_search_attempts=2, agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=5),
    ))
    searches = []
    answers = []
    hit = {"source": "noise.md", "chunk_id": 1, "text": "Unrelated", "score": 0.5}

    def search(query, **kwargs):
        searches.append(query)
        return [hit]

    class Controller:
        def generate_raw(self, messages, **kwargs):
            return json.dumps({"coverage": "zero_evidence", "missing_information": ["owner"],
                               "next_search": {"query": "owner revised", "vector_min_score": 0.1,
                                               "bm25_min_score_ratio": 0.0}})

    result = agentic.agentic_answer("Who owns it?", rag=None, meta_store=None,
                                   search_fn=search, controller=Controller(),
                                   answer_fn=lambda *a, **k: answers.append(1))
    assert len(searches) == 2
    assert answers == []
    assert result["reason_code"] == "insufficient_evidence"
    assert result["response_origin"] == "controller"
    assert result["generation_context"] == []
    assert result["retrieval_coverage"]["coverage"] == "zero_evidence"
    assert result["retrieved_evidence"]


def test_selected_context_zero_evidence_blocks_generation(monkeypatch):
    monkeypatch.setattr(agentic, "get_settings", lambda: SimpleNamespace(
        agentic_max_search_attempts=1, agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=1),
    ))
    hits = [{"source": "notes.md", "chunk_id": i, "text": str(i), "score": 0.5} for i in range(2)]
    monkeypatch.setattr(agentic, "rerank_candidates", lambda q, candidates, **kwargs: candidates)
    assessments = []
    class Controller:
        def generate_raw(self, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            assessments.append(payload)
            return json.dumps({"coverage": "sufficient_evidence" if len(assessments) == 1 else "zero_evidence"})
    calls = []
    result = agentic.agentic_answer("question", rag=None, meta_store=None,
        search_fn=lambda q, **kwargs: hits, controller=Controller(),
        answer_fn=lambda *a, **kwargs: calls.append(1))
    assert len(assessments) == 2
    assert len(assessments[1]["chunks_pool"]) == 1
    assert result["retrieval_coverage"]["coverage"] == "sufficient_evidence"
    assert result["generation_context_coverage"]["coverage"] == "zero_evidence"
    assert result["reason_code"] == "insufficient_evidence"
    assert result["generation_context"] == []
    assert not calls


def test_retrieval_review_failure_never_generates(monkeypatch):
    monkeypatch.setattr(agentic, "get_settings", lambda: SimpleNamespace(
        agentic_max_search_attempts=1, agentic_max_answer_attempts=1,
        llm=SimpleNamespace(max_chunks=5),
    ))
    class Controller:
        def generate_raw(self, messages, **kwargs):
            raise RuntimeError("provider unavailable")
    calls = []
    hit = {"source": "notes.md", "chunk_id": 1, "text": "text", "score": .5}
    result = agentic.agentic_answer("question", rag=None, meta_store=None,
        search_fn=lambda q, **kwargs: [hit], controller=Controller(),
        answer_fn=lambda *a, **kwargs: calls.append(1))
    assert result["workflow_error"] == "controller_failure"
    assert result["controller_failure_category"] == "provider_failure"
    assert result["retrieval_coverage"]["coverage"] == "unknown"
    assert result["retrieved_evidence"]
    assert not calls
