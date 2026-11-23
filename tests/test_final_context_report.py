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

    assert "### Retrieval metrics" in report
    assert "| Recall | 0.5000 |" in report
    assert "| direct | 1 | 0.5000 | 0.2500 | 0.4000 | 0.9000 | — | 0.8000 | — | 12.0000 |" in report
    assert "Answerability outcomes" not in report
    assert "Eligible cases / selected cases" not in report
    assert "### Provider usage" not in report
    assert "| Search | Query |" in report
    assert "| 1 | Which date? |" in report
    assert "candidate one" not in report
    assert report.count("candidate two") == 1
    assert "| Context order | Chunk ID | Source path | Lines | Full context | Evidence lines |" in report
    assert "| 1 | 2 | notes/b.md | 3–3 | candidate two | — |" in report
    assert "Status/reason" not in report
    assert "| Metric | Score | Applicability |" in report
    assert "Provider total tokens" not in report
    assert "| Process/call | Latency (ms) | Provider input tokens | Provider output tokens | Total tokens |" in report
    assert "### Latency\n" in report
    assert "### Mean latency" not in report
    assert "**Faithfulness** checks whether the answer is supported by the final Answer generation context" in report
    assert "**Answer Relevancy** checks whether the answer addresses the question" in report
    assert "**Answer Correctness** checks the answer against the expected answer" in report
    assert "**Context Utilization** checks whether the final Answer generation context is useful for the answer" in report


def test_overall_retrieval_metrics_keep_rows_when_values_are_unavailable(
    tmp_path,
):
    for workflow in ("classic", "agentic"):
        results = {
            "run_metadata": {"workflow": workflow},
            "summary": {
                "workflows": {
                    workflow: {
                        "final_context_metrics": {
                            "recall": None,
                            "precision": None,
                            "ndcg": None,
                        },
                    },
                },
            },
            "cases": [],
        }

        report_path = tmp_path / f"{workflow}.md"
        write_readme(report_path, results)
        report = report_path.read_text(encoding="utf-8")
        retrieval_section = report.split("### Retrieval metrics\n", 1)[1].split(
            "### Ragas diagnostics", 1
        )[0]

        assert "| Recall | — |" in retrieval_section
        assert "| Precision | — |" in retrieval_section
        assert "| NDCG | — |" in retrieval_section


def test_agentic_report_renders_outcomes_and_metric_applicability(tmp_path):
    evidence = {
        "source": "notes/delivery_timeline.md",
        "chunk_id": 9,
        "start_line": 12,
        "end_line": 12,
        "score": 0.9,
        "text": "The next council checkpoint is 2026-05-28.",
    }
    cases = [
        {
            "case_id": "direct-06",
            "workflow": "agentic",
            "category": "direct",
            "question": "What date is the checkpoint?",
            "answer": "The checkpoint is 2026-05-28.",
            "reason_code": "answered",
            "response_origin": "answer_model",
            "search_attempts": 1,
            "answer_attempts": 1,
            "search_results": [[evidence]],
            "retrieved_evidence": [evidence],
            "generation_context": [{
                "source": evidence["source"],
                "chunk_id": evidence["chunk_id"],
                "start_line": evidence["start_line"],
                "end_line": evidence["end_line"],
                "content_changed": False,
                "text": None,
                "prompt_block": None,
            }],
            "final_context_metrics": {
                "recall": 1.0,
                "precision": 1.0,
                "ndcg": 1.0,
            },
            "generation_metrics": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.9,
                "answer_correctness": 0.9,
            },
            "retrieval_metrics": {"context_utilization": 0.9},
            "metric_reasons": {"faithfulness": "Scored."},
            "generation_usage": {"calls": []},
            "latency": {},
        },
        {
            "case_id": "ambiguous-01",
            "workflow": "agentic",
            "category": "ambiguous",
            "question": "When will the public launch happen?",
            "answer": "Are you asking for the target or the approved date?",
            "reason_code": "needs_clarification",
            "response_origin": "controller",
            "search_attempts": 1,
            "answer_attempts": 1,
            "retrieved_evidence": [evidence],
            "search_results": [[evidence]],
            "generation_context": [],
            "final_context_metrics": {
                "recall": None,
                "precision": None,
                "ndcg": None,
            },
            "generation_metrics": {},
            "retrieval_metrics": {},
            "metric_reasons": {
                "answer_relevancy": "Skipped because the workflow requested clarification.",
                "faithfulness": "Skipped because the workflow requested clarification.",
                "answer_correctness": "Skipped because the workflow requested clarification.",
                "context_utilization": "Skipped because the workflow requested clarification.",
            },
            "generation_usage": {"calls": []},
            "latency": {},
        },
        {
            "case_id": "unanswerable-13",
            "workflow": "agentic",
            "category": "unanswerable",
            "question": "How many support requests were received?",
            "answer": "I cannot find that count in the notes.",
            "reason_code": "insufficient_evidence",
            "response_origin": "answer_model",
            "search_attempts": 1,
            "answer_attempts": 1,
            "retrieved_evidence": [evidence],
            "search_results": [[evidence]],
            "generation_context": [{
                "source": evidence["source"],
                "chunk_id": evidence["chunk_id"],
                "start_line": evidence["start_line"],
                "end_line": evidence["end_line"],
                "content_changed": False,
                "text": None,
                "prompt_block": None,
            }],
            "final_context_metrics": {
                "recall": 0.0,
                "precision": 0.0,
                "ndcg": 0.0,
            },
            "generation_metrics": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "answer_correctness": 0.9,
            },
            "retrieval_metrics": {"context_utilization": 0.7},
            "metric_reasons": {"answer_correctness": "Scored."},
            "generation_usage": {"calls": []},
            "latency": {},
        },
        {
            "case_id": "unanswerable-no-context",
            "workflow": "agentic",
            "category": "unanswerable",
            "question": "How many requests were received?",
            "answer": "The notes do not provide that count.",
            "reason_code": "insufficient_evidence",
            "response_origin": "answer_model",
            "search_attempts": 1,
            "answer_attempts": 1,
            "retrieved_evidence": [evidence],
            "search_results": [[evidence]],
            "generation_context": [],
            "final_context_metrics": {"recall": None, "precision": None, "ndcg": None},
            "generation_metrics": {},
            "retrieval_metrics": {},
            "metric_reasons": {
                "answer_relevancy": "Scored.",
                "faithfulness": "Unavailable because no final generation context was available.",
                "answer_correctness": "Scored.",
                "context_utilization": "Unavailable because no final generation context was available.",
            },
            "generation_usage": {"calls": []},
            "latency": {},
        },
        {
            "case_id": "retrieval-empty",
            "workflow": "agentic",
            "category": "direct",
            "question": "What was the date?",
            "answer": "No note chunks were retrieved for this query.",
            "reason_code": "no_retrieval_results",
            "response_origin": "deterministic",
            "search_attempts": 1,
            "answer_attempts": 0,
            "retrieved_evidence": [],
            "search_results": [[]],
            "generation_context": [],
            "final_context_metrics": {"recall": None, "precision": None, "ndcg": None},
            "generation_metrics": {},
            "retrieval_metrics": {},
            "metric_reasons": {
                "answer_relevancy": "Skipped because Retrieval returned no candidate chunks.",
                "faithfulness": "Skipped because Retrieval returned no candidate chunks.",
                "answer_correctness": "Skipped because Retrieval returned no candidate chunks.",
                "context_utilization": "Skipped because Retrieval returned no candidate chunks.",
            },
            "generation_usage": {"calls": []},
            "latency": {},
        },
        {
            "case_id": "answer-exhausted",
            "workflow": "agentic",
            "category": "unanswerable",
            "question": "Can these notes answer this?",
            "answer": "I cannot provide a sufficiently supported answer.",
            "reason_code": "answer_attempts_exhausted",
            "response_origin": "deterministic",
            "search_attempts": 2,
            "answer_attempts": 3,
            "answer_retry_exhausted": True,
            "retrieved_evidence": [evidence],
            "search_results": [[evidence], [evidence]],
            "generation_context": [],
            "final_context_metrics": {"recall": None, "precision": None, "ndcg": None},
            "generation_metrics": {},
            "retrieval_metrics": {},
            "metric_reasons": {
                "answer_relevancy": "Skipped because no accepted Answer candidate remained.",
                "faithfulness": "Skipped because no accepted Answer candidate remained.",
                "answer_correctness": "Skipped because no accepted Answer candidate remained.",
                "context_utilization": "Skipped because no accepted Answer candidate remained.",
            },
            "generation_usage": {"calls": []},
            "latency": {},
            "rejected_candidate_text": "REJECTED ANSWER MUST NEVER APPEAR",
            "raw_controller_output": "RAW PROVIDER OUTPUT MUST NEVER APPEAR",
        },
        {
            "case_id": "controller-error",
            "workflow": "agentic",
            "category": "direct",
            "question": "Who owns the release?",
            "answer": "I could not complete the bounded review.",
            "reason_code": "controller_failure",
            "response_origin": "deterministic",
            "controller_failure_category": "invalid_json",
            "workflow_error": "controller_failure",
            "search_attempts": 1,
            "answer_attempts": 1,
            "retrieved_evidence": [evidence],
            "search_results": [[evidence]],
            "generation_context": [],
            "final_context_metrics": {"recall": None, "precision": None, "ndcg": None},
            "generation_metrics": {},
            "retrieval_metrics": {},
            "metric_reasons": {
                "answer_relevancy": "Skipped because the workflow did not complete a final answer.",
                "faithfulness": "Skipped because the workflow did not complete a final answer.",
                "answer_correctness": "Skipped because the workflow did not complete a final answer.",
                "context_utilization": "Skipped because the workflow did not complete a final answer.",
            },
            "generation_usage": {"calls": []},
            "latency": {},
            "raw_controller_output": "RAW PROVIDER OUTPUT MUST NEVER APPEAR",
        },
    ]
    outcome_counts = {
        "bounded_outcomes": {
            "answered": 1,
            "answer_attempts_exhausted": 1,
            "insufficient_evidence": 2,
            "needs_clarification": 1,
            "no_retrieval_results": 1,
        },
        "insufficient_evidence_by_origin": {"answer_model": 2},
        "workflow_errors": {"controller_failure": 1},
    }
    results = {
        "run_metadata": {"workflow": "agentic", "max_answer_chunks": 1},
        "summary": {
            "case_count": len(cases),
            "workflows": {
                "agentic": {
                    "outcome_counts": outcome_counts,
                    "final_context_metrics": {
                        "recall": None,
                        "precision": None,
                        "ndcg": None,
                    },
                    "generation_metrics": {},
                    "retrieval_metrics": {},
                    "judge_usage_by_metric": {},
                    "latency_ms": {},
                },
            },
        },
        "workflow_error_count": 1,
        "workflow_reliability": {
            "successful_cases": 6,
            "selected_cases": 7,
            "ratio": 6 / 7,
        },
        "cases": cases,
    }

    report_path = tmp_path / "agentic-outcomes.md"
    write_readme(report_path, results)
    report = report_path.read_text(encoding="utf-8")

    assert "## Workflow outcomes" in report
    assert "| Bounded outcome | answered | — | 1 |" in report
    assert "| Bounded outcome | insufficient_evidence | answer_model | 2 |" in report
    assert "| Bounded outcome | needs_clarification | — | 1 |" in report
    assert "| Bounded outcome | no_retrieval_results | — | 1 |" in report
    assert "| Bounded outcome | answer_attempts_exhausted | — | 1 |" in report
    assert "| Workflow error | controller_failure | — | 1 |" in report
    assert "**Workflow reliability:** `6/7 (0.8571)`" in report

    assert "#### Workflow outcome" in report
    assert "| Terminal reason | answered |" in report
    assert "| Response origin | answer_model |" in report
    assert "| Search attempts | 1 |" in report
    assert "| Answer attempts | 1 |" in report
    assert "| Terminal reason | no_retrieval_results |" in report
    assert "| Answer attempts | 0 |" in report
    assert "| Controller failure category | invalid_json |" in report
    assert "Skipped because Retrieval returned no candidate chunks." in report
    assert "Unavailable because no final generation context was available." in report
    assert "REJECTED ANSWER MUST NEVER APPEAR" not in report
    assert "RAW PROVIDER OUTPUT MUST NEVER APPEAR" not in report
    assert report.count("### Retrieval metrics") == 1
    assert "| Recall | — |" in report
    assert "| Precision | — |" in report
    assert "| NDCG | — |" in report
    assert "| Generation@1 | — | — | — |" in report
