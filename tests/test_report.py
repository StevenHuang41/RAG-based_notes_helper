import pytest

from rag_notes_helper.eval.report import write_readme


@pytest.mark.parametrize("workflow", ["classic", "agentic"])
def test_report_keeps_candidates_separate_from_final_answer_context(tmp_path, workflow):
    results = {
        "run_metadata": {"workflow": workflow, "top_k": 3},
        "summary": {"workflows": {workflow: {
            "final_context_metrics": {"recall": 1.0, "precision": 0.5, "ndcg": 0.8},
            "generation_metrics": {"faithfulness": 0.9},
            "retrieval_metrics": {"context_utilization": 0.7},
            "by_category": {"direct": {
                "case_count": 1,
                "final_context_metrics": {"recall": 1.0, "precision": 0.5, "ndcg": 0.8},
                "generation_metrics": {"faithfulness": 0.9},
                "retrieval_metrics": {"context_utilization": 0.7},
                "latency_ms": {"end_to_end_ms": 4.0},
            }},
        }}},
        "cases": [{
            "case_id": "context", "category": "direct", "question": "When?",
            "expected_sources": ["notes/launch.md"],
            "expected_evidence": [{"source": "notes/launch.md", "start_line": 2, "end_line": 2, "text": "date"}],
            "search_queries": ["When?", "follow up"],
            "search_results": [
                [{"source": "notes/noise.md", "chunk_id": 1, "start_line": 1, "end_line": 1, "score": 0.9, "text": "noise"}],
                [{"source": "notes/launch.md", "chunk_id": 2, "start_line": 2, "end_line": 2, "score": 0.8, "text": "date"}],
            ],
            "generation_context": [{"source": "notes/launch.md", "chunk_id": 2, "start_line": 2, "end_line": 2, "text": None, "prompt_block": None}],
            "context_prefix_metrics": [{"context_prefix_k": 1, "evidence_lines_covered": 1, "ground_truth_line_count": 1, "recall": 1.0, "precision": 0.5, "ndcg": 0.8}],
            "final_context_metrics": {"recall": 1.0, "precision": 0.5, "ndcg": 0.8},
            "generation_metrics": {"faithfulness": 0.9},
            "retrieval_metrics": {"context_utilization": 0.7},
        }],
    }

    path = tmp_path / "README.md"
    write_readme(path, results)
    report = path.read_text(encoding="utf-8")

    assert "| Search ID | Search query |" in report
    assert "| 2 | follow up |" in report
    assert "noise" in report
    assert "| 1 | 2 | notes/launch.md | 2–2 | Same as Search 2, rank 1 | 1/1 | 1.0000 | 0.5000 | 0.8000 |" in report
    assert "Retrieval top-k | Recall" not in report
    assert "Context-prefix metrics" not in report
    assert "Answerability outcomes" not in report


def test_report_uses_exact_token_sum_and_dash_for_unavailable_values(tmp_path):
    results = {
        "run_metadata": {"workflow": "classic"},
        "summary": {"workflows": {"classic": {"judge_usage_by_metric": {}}}},
        "cases": [{
            "case_id": "usage", "generation_usage": {"calls": [
                {"call_type": "answer", "provider": "fixture", "model": "v1", "latency_ms": 2,
                 "token_usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": None}},
                {"call_type": "controller", "provider": "fixture", "model": "v1", "latency_ms": 1,
                 "token_usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None}},
            ]},
        }],
    }
    path = tmp_path / "README.md"
    write_readme(path, results)
    report = path.read_text(encoding="utf-8")

    assert "| answer call 1 | 2 | fixture / v1 | 3 | 4 | 7 |" in report
    assert "| controller call 2 | 1 | fixture / v1 | — | — | — |" in report
    assert "Calculated input+output" not in report
    assert "Usage status" not in report
    assert "Input chars" not in report
