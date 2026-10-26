from rag_notes_helper.eval.report import write_readme


def test_report_uses_final_context_metrics_and_concise_audit_tables(tmp_path):
    results = {
        "run_metadata": {"workflow": "agentic", "top_k": 5},
        "summary": {
            "case_count": 1,
            "workflows": {
                "agentic": {
                    "final_context_metrics": {
                        "recall": 0.5, "precision": 0.25, "ndcg": 0.4,
                    },
                    "generation_metrics": {
                        "faithfulness": 0.9, "answer_relevancy": None,
                        "answer_correctness": 0.8,
                    },
                    "retrieval_metrics": {"context_utilization": None},
                    "judge_usage_by_metric": {},
                    "latency_ms": {"end_to_end_ms": 12.0},
                    "by_category": {
                        "direct": {
                            "case_count": 1,
                            "final_context_metrics": {
                                "recall": 0.5, "precision": 0.25, "ndcg": 0.4,
                            },
                            "generation_metrics": {
                                "faithfulness": 0.9, "answer_relevancy": None,
                                "answer_correctness": 0.8,
                            },
                            "retrieval_metrics": {"context_utilization": None},
                            "latency_ms": {"end_to_end_ms": 12.0},
                        },
                    },
                },
            },
        },
        "cases": [{
            "case_id": "selected-context", "category": "direct",
            "question": "Which date?", "expected_evidence": [],
            "search_queries": ["Which date?"],
            "search_results": [[
                {"source": "notes/a.md", "chunk_id": 1, "start_line": 1,
                 "end_line": 1, "score": 0.9, "text": "candidate one"},
                {"source": "notes/b.md", "chunk_id": 2, "start_line": 3,
                 "end_line": 3, "score": 0.8, "text": "candidate two"},
            ]],
            "generation_context": [{
                "source": "notes/b.md", "chunk_id": 2, "start_line": 3,
                "end_line": 3, "text": None, "prompt_block": None,
            }],
            "context_prefix_metrics": [{
                "context_prefix_k": 1, "evidence_lines_covered": 1,
                "ground_truth_line_count": 2, "recall": 0.5,
                "precision": 0.25, "ndcg": 0.4,
            }],
            "final_context_metrics": {
                "recall": 0.5, "precision": 0.25, "ndcg": 0.4,
            },
            "generation_metrics": {
                "faithfulness": 0.9, "answer_relevancy": None,
                "answer_correctness": 0.8,
            },
            "retrieval_metrics": {"context_utilization": None},
            "judge_metrics": {},
            "latency": {"end_to_end_ms": 12.0},
        }],
    }

    report_path = tmp_path / "README.md"
    write_readme(report_path, results)
    report = report_path.read_text(encoding="utf-8")

    assert "Final context evidence metrics" in report
    assert "| recall | 0.5000 |" in report
    assert "| direct | 1 | 0.5000 | 0.2500 | 0.4000 | 0.9000 | — | 0.8000 | — | 12.0000 |" in report
    assert "Answerability outcomes" not in report
    assert "Eligible cases / selected cases" not in report
    assert "### Provider usage" not in report
    assert "| Search ID | Search query |" in report
    assert "| 1 | Which date? |" in report
    assert "| 1 | 2 | 2 | notes/b.md | 3–3 | 0.8000 | candidate two |" in report
    assert "| Final context order | Chunk ID | Source path | Lines | Context content | Evidence lines | Recall | Precision | Context-order NDCG |" in report
    assert "| 1 | 2 | notes/b.md | 3–3 | Same as Search 1, rank 2 | 1/2 | 0.5000 | 0.2500 | 0.4000 |" in report
    assert "Status/reason" not in report
    assert "| Metric | Score |" in report
    assert "Provider total tokens" not in report
    assert "| Process/call | Latency (ms) | Provider / model | Provider input tokens | Provider output tokens | Total tokens |" in report
