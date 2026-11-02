import pytest

from rag_notes_helper.eval.report import write_readme


@pytest.mark.parametrize("workflow", ["classic", "agentic"])
def test_report_keeps_candidates_separate_from_final_answer_context(tmp_path, workflow):
    results = {
        "run_metadata": {
            "workflow": workflow,
            "top_k": 3,
            "max_answer_chunks": 5,
        },
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
            "case_id": "context",
            **({"workflow": "agentic"} if workflow == "agentic" else {}),
            "category": "direct", "question": "When?",
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

    assert "| Search | Query |" in report
    assert "| 2 | follow up |" in report
    assert "| Generation@5 | 1.0000 | 0.5000 | 0.8000 |" in report
    if workflow == "agentic":
        assert "noise" not in report
        assert "| 1 | 2 | notes/launch.md | 2–2 | date | 1/1 |" in report
    else:
        assert "noise" in report
        assert "| 1 | 2 | notes/launch.md | 2–2 | date | 1/1 |" in report
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


def test_classic_query_detail_shows_stage_cutoffs_and_per_chunk_evidence(tmp_path):
    evidence = {
        "source": "notes/evidence.md",
        "chunk_id": 10,
        "start_line": 10,
        "end_line": 10,
        "text": "Reviewed evidence line.",
        "rank": 2,
        "stage_score": 0.8,
    }
    noise = {
        "source": "notes/noise.md",
        "chunk_id": 1,
        "start_line": 1,
        "end_line": 2,
        "text": "Retrieved only for ranking.",
        "rank": 1,
        "stage_score": 0.9,
    }
    stages = {
        "vector": {"top_k": 2, "metric_top_k": 2, "results": [evidence, noise]},
        "bm25": {"top_k": 2, "metric_top_k": 2, "results": [noise, evidence]},
        "rrf": {"top_k": 2, "metric_top_k": 2, "results": [noise, evidence]},
        "reranked": {"top_k": 2, "metric_top_k": 2, "results": [evidence, noise]},
    }
    stage_metrics = {
        "vector": {"recall": 0.5, "precision": 0.3333, "ndcg": 0.63},
        "bm25": {"recall": 0.5, "precision": 0.3333, "ndcg": 0.5},
        "rrf": {"recall": 0.5, "precision": 0.3333, "ndcg": 0.5},
        "reranked": {"recall": 0.5, "precision": 0.3333, "ndcg": 1.0},
    }
    results = {
        "run_metadata": {
            "workflow": "classic",
            "cross_encoder_model": "fixture/custom-cross-encoder",
            "max_answer_chunks": 1,
        },
        "summary": {"workflows": {"classic": {}}},
        "cases": [{
            "workflow": "classic",
            "case_id": "direct-06",
            "category": "direct",
            "question": "Which date is the checkpoint?",
            "reference_answer": "2026-05-28",
            "answer": "2026-05-28",
            "expected_evidence": [{
                "source": "notes/evidence.md",
                "start_line": 10,
                "end_line": 11,
                "text": "Reviewed evidence lines.",
            }],
            "search_queries": ["Which date is the checkpoint?"],
            "retrieval_stages": [{
                "search_id": 1,
                "search_query": "Which date is the checkpoint?",
                "generation_top_k": 1,
                "stages": stages,
            }],
            "retrieval_stage_metrics": stage_metrics,
            "generation_context": [{
                "source": "notes/evidence.md",
                "chunk_id": 10,
                "start_line": 10,
                "end_line": 10,
                "text": evidence["text"],
            }],
            "final_context_metrics": {"recall": 0.5, "precision": 1.0, "ndcg": 1.0},
            "context_prefix_metrics": [{
                "context_prefix_k": 1,
                "evidence_lines_covered": 1,
                "ground_truth_line_count": 2,
                "recall": 0.5,
                "precision": 1.0,
                "ndcg": 1.0,
            }],
            "generation_metrics": {},
            "retrieval_metrics": {},
            "latency": {
                "retrieval_stages_ms": {
                    "vector": 1.0, "bm25": 2.0, "rrf": 3.0, "reranked": 4.0,
                },
                "answer_generation_ms": 5.0,
                "rag_query_ms": 15.0,
            },
        }],
    }

    path = tmp_path / "README.md"
    write_readme(path, results)
    report = path.read_text(encoding="utf-8")

    assert "| Cross-encoder model | fixture/custom-cross-encoder |" in report
    assert "| Vector@2 | 0.5000 | 0.3333 | 0.6300 |" in report
    assert "| Fusion@2 | 0.5000 | 0.3333 | 0.5000 |" in report
    assert "| Rerank@2 | 0.5000 | 0.3333 | 1.0000 |" in report
    assert "| Generation@1 | 0.5000 | 1.0000 | 1.0000 |" in report
    assert "| Rank | Stage score | Chunk ID | Source path | Lines | Search | Evidence lines |" in report
    assert "| 2 | 0.8000 | 10 | notes/evidence.md | 10–10 | 1 | 1/2 |" in report
    assert "| 1 | 0.9000 | 1 | notes/noise.md | 1–2 | 1 | 0/2 |" in report
    assert "| Search | Query |" in report
    assert report.count("| Search | Query |") == 1
    assert "_Search query:" not in report
    assert "Pre-rerank and post-rerank comparison" not in report
    assert "| Context order | Chunk ID | Source path | Lines | Full context | Evidence lines |" in report
    assert "| 1 | 10 | notes/evidence.md | 10–10 | Reviewed evidence line. | 1/2 |" in report
    assert "Recall | Precision | Context-order NDCG" not in report
    assert "| Reranking | 4.0000 |" in report
    assert report.index(
        "**Retrieval stages and generation-context metrics**"
    ) < report.index("#### Latency")
    assert report.index("#### Expected evidence") < report.index("#### Search query")
    assert report.index("#### Search query") < report.index("#### Vector results")
    reranked_section = report.split("#### Reranked results", 1)[1].split(
        "#### Final generation context", 1
    )[0]
    assert "| 1 | 0.9000 | 1 | notes/noise.md | 1–2 | 1 | 0/2 |" in reranked_section
