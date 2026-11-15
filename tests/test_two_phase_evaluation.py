import json
from contextlib import contextmanager
from types import SimpleNamespace

from rag_notes_helper.eval import eval_runner


def test_two_phase_evaluation_keeps_answer_quality_separate_from_reliability(monkeypatch, tmp_path):
    names = ["degraded", "partial", "abstention", "empty", "warning", "no_context"]
    cases = [{
        "id": name, "category": "direct" if i < 3 else "unanswerable",
        "challenge_tags": [], "question": name, "reference_answer": "Lena owns launch.",
        "answerable": True, "expected_sources": ["release.md"],
        "expected_evidence": [{"source": "release.md", "start_line": 1, "end_line": 1, "text": "Lena owns launch."}],
    } for i, name in enumerate(names)]
    settings = SimpleNamespace(
        llm=SimpleNamespace(provider="fixture", model="answer", effective_eval_provider="fixture", effective_eval_model="judge", max_chunks=2),
        embed_model_name="fixture", cross_encoder_model="fixture", chunk_size=200, chunk_overlap=50,
        top_k=3, vector_top_k=3, bm25_top_k=3, fusion_top_k=3, min_retrieval_score=.3,
    )
    monkeypatch.setattr(eval_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(eval_runner, "load_qa_fixture", lambda: cases)

    @contextmanager
    def isolated():
        yield settings, object(), object()

    monkeypatch.setattr(eval_runner, "_isolated_fixture_index", isolated)
    hit = {"source": "release.md", "chunk_id": 0, "start_line": 1, "end_line": 1, "text": "Lena owns launch.", "score": .9,
           "search_provenance": [{"search_id": 1, "search_query": "owner"}, {"search_id": 2, "search_query": "launch owner"}]}

    def query(_rag, _meta, *, query, on_generation, **kwargs):
        deterministic = query in {"empty", "warning"}
        context = [] if query in {"empty", "no_context"} else [hit]
        result = {
            "answer": "No usable content." if deterministic else "Lena owns launch.",
            "reason_code": "no_retrieval_results" if query == "empty" else "answer_review_failure" if query == "warning" else "insufficient_evidence" if query == "abstention" else "answered",
            "response_origin": "deterministic" if deterministic else "answer_model",
            "review_passed": not deterministic, "answer_coverage": "partial" if query == "partial" else "none" if query == "abstention" else "full",
            "answer_missing_information": ["date"] if query == "partial" else [],
            "generation_context": context, "citations": [], "retrieved_evidence": [hit] if query != "empty" else [],
            "search_queries": [query], "chunks_pool": [hit] if query != "empty" else [], "selected_pool": context,
            "retrieval_coverage": {"assessed_question": query, "coverage": "partial_evidence", "missing_information": ["date"]},
            "retrieval_reviews": [{"search_id": 1, "coverage": "partial_evidence"}], "retrieval_stop_reason": "review_degraded" if query == "degraded" else "search_budget_exhausted",
            "retrieval_degradation": {"cause": "invalid_json", "search_id": 1} if query == "degraded" else None,
            "workflow_degraded": query == "degraded", "final_selection": {"rerank_performed": False, "latency_ms": None, "max_chunks": 2},
        }
        if query == "warning":
            result["workflow_error"] = "answer_review_failure"
            result["failure_category"] = "invalid_json"
        return result

    monkeypatch.setattr("rag_notes_helper.cli.run_query", query)
    batches = []

    def score_batches(requests, *, usage_recorder):
        batches.extend(requests)
        return [[{
            metric: None if row["question"] == "partial" and metric == "answer_correctness" else .75,
            "metric_reasons": {metric: "Judge failed."} if row["question"] == "partial" and metric == "answer_correctness" else {},
            "metric_latencies_ms": {metric: 1.0},
        } for row in request["rows"]] for request in requests for metric in request["metric_names"]]

    monkeypatch.setattr(eval_runner, "_run_ragas_batches_isolated", score_batches)
    run_dir = eval_runner.run_evaluation(agentic=True, output_root=tmp_path)
    results = json.loads((run_dir / "results.json").read_text())
    assert "context_window_tokens" not in results["run_metadata"]
    records = {c["case_id"]: c for c in results["cases"]}
    degraded = records["degraded"]
    assert degraded["chunks_pool"][0]["search_provenance"] == hit["search_provenance"]
    assert degraded["chunks_pool"][0]["evidence_lines_covered"] == 1
    assert degraded["retrieval_degradation"]["cause"] == "invalid_json"
    assert degraded["workflow_degraded"] is True
    assert degraded["generation_metrics"]["faithfulness"] == .75
    assert records["partial"]["judge_metrics"]["answer_correctness"]["status"] == "failed"
    assert records["partial"]["answer_missing_information"] == ["date"]
    assert records["no_context"]["judge_metrics"]["faithfulness"]["status"] == "unavailable"
    assert records["empty"]["judge_metrics"]["faithfulness"]["status"] == "unavailable"
    assert records["warning"]["judge_metrics"]["faithfulness"]["status"] == "unavailable"
    assert records["warning"]["failure_category"] == "invalid_json"
    summary = results["summary"]["workflows"]["agentic"]
    assert summary["ragas_metric_status_counts"]["faithfulness"] == {"eligible": 3, "scored": 3, "unavailable": 3, "failed": 0}
    assert summary["ragas_metric_status_counts"]["answer_correctness"] == {"eligible": 4, "scored": 3, "unavailable": 2, "failed": 1}
    assert summary["by_category"]["direct"]["ragas_metric_status_counts"]["answer_correctness"]["failed"] == 1
    assert results["workflow_reliability"]["successful_cases"] == 4
    assert results["workflow_degraded_count"] == 1
    assert results["workflow_error_count"] == 1
    assert results["workflow_outcome"] == "completed_with_errors"
    assert all(row["question"] not in {"empty", "warning"} for request in batches for row in request["rows"])

    def judge_failure(*args, **kwargs):
        raise RuntimeError("judge transport failed")

    monkeypatch.setattr(eval_runner, "_run_ragas_batches_isolated", judge_failure)
    failed_run = eval_runner.run_evaluation(agentic=True, output_root=tmp_path / "judge-failure")
    failed_results = json.loads((failed_run / "results.json").read_text())
    failed_records = {c["case_id"]: c for c in failed_results["cases"]}
    assert failed_records["degraded"]["judge_metrics"]["faithfulness"]["status"] == "failed"
    assert failed_records["empty"]["judge_metrics"]["faithfulness"]["status"] == "unavailable"
    assert failed_records["degraded"]["latency"]["judge_ms"] is None
    assert failed_results["summary"]["workflows"]["agentic"]["ragas_metric_status_counts"]["faithfulness"] == {
        "eligible": 3, "scored": 0, "unavailable": 3, "failed": 3,
    }
