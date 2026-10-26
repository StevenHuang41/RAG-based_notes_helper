import re

import pytest

from rag_notes_helper.eval.report import write_readme


def _report_results(workflow):
    owner = {
        "source": "notes/owner.md",
        "chunk_id": 4,
        "start_line": 8,
        "end_line": 8,
        "score": 0.92,
        "text": "Owner: Mina Chen.",
        "stage": "retrieval",
        "method": "FAISS vector search",
    }
    launch = {
        "source": "notes/launch.md",
        "chunk_id": 2,
        "start_line": 12,
        "end_line": 13,
        "score": 0.88,
        "text": "Launch date: 2026-07-14. The schedule is approved.",
        "stage": "retrieval",
        "method": "FAISS vector search",
    }
    question = "When does Project Cedar launch?"
    is_agentic = workflow == "agentic"
    search_queries = (
        [question, "Project Cedar launch date confirmation"]
        if is_agentic
        else [question]
    )
    search_results = [[owner], [launch, owner]] if is_agentic else [[owner, launch]]
    generation_context = [
        {
            "source": launch["source"],
            "chunk_id": launch["chunk_id"],
            "start_line": launch["start_line"],
            "end_line": launch["end_line"],
            "content_changed": is_agentic,
            "text": "Launch date: 2026-07-14." if is_agentic else None,
            "prompt_block": (
                "file:notes/launch.md\nLaunch date: 2026-07-14."
                if is_agentic
                else None
            ),
        },
    ]
    if is_agentic:
        generation_context.append({
            "source": owner["source"],
            "chunk_id": owner["chunk_id"],
            "start_line": owner["start_line"],
            "end_line": owner["end_line"],
            "content_changed": False,
            "text": None,
            "prompt_block": None,
        })

    available_usage = {
        "status": "available",
        "input_tokens": 9,
        "output_tokens": 3,
        "total_tokens": 12,
    }
    partial_usage = {
        "status": "partial",
        "input_tokens": 5,
        "output_tokens": None,
        "total_tokens": None,
    }
    answerable_case = {
        "case_id": "direct-01",
        "category": "direct",
        "question": question,
        "reference_answer": "Project Cedar launches on 2026-07-14.",
        "answer": "Project Cedar launches on 2026-07-14.",
        "expected_answerable": True,
        "answerability_correct": True,
        "abstained": False,
        "expected_sources": [launch["source"]],
        "expected_evidence": [{
            "source": launch["source"],
            "start_line": launch["start_line"],
            "end_line": launch["end_line"],
            "text": "Launch date: 2026-07-14.",
        }],
        "expected_source_coverage": 0.2718,
        "expected_evidence_coverage": 0.1618,
        "retrieval_metrics": {
            "recall": 0.9,
            "precision": 0.7,
            "ndcg": 0.8,
            "context_utilization": 0.6123,
        },
        "generation_metrics": {
            "faithfulness": 0.91,
            "answer_relevancy": 0.82,
            "answer_correctness": 0.87,
        },
        "search_queries": search_queries,
        "search_results": search_results,
        "generation_context": generation_context,
        "judge_metrics": {
            "faithfulness": {"score": 0.91, "reason": "Claims match the context."},
            "answer_relevancy": {"score": 0.82},
            "answer_correctness": {"score": 0.87, "reason": "The response matches the review."},
            "context_utilization": {"score": 0.7345, "reason": "Diagnostic only."},
        },
        "generation_usage": {
            "call_count": 1,
            "token_usage": available_usage,
            "calls": [{
                "call_type": "answer",
                "provider": "fixture-answer",
                "model": "answer-v1",
                "latency_ms": 4.5,
                "token_usage": available_usage,
            }],
        },
        "judge_usage": {"call_count": 4, "token_usage": available_usage},
        "judge_usage_by_metric": {
            "context_utilization": {
                "call_count": 1,
                "token_usage": partial_usage,
                "calls": [{
                    "provider": "fixture-judge",
                    "model": "judge-v1",
                    "token_usage": partial_usage,
                }],
            },
        },
        "latency": {
            "retrieval_calls": [
                {"search_index": index, "latency_ms": 1.5}
                for index in range(1, len(search_results) + 1)
            ],
            "judge_metrics_ms": {"context_utilization": 2.0},
            "judge_ms": 2.0,
            "end_to_end_ms": 20.0,
        },
    }
    abstention_case = {
        "case_id": "unanswerable-01",
        "category": "unanswerable",
        "question": "What is the private launch code?",
        "reference_answer": "No launch code is documented.",
        "answer": "I cannot find the answers in the notes.",
        "expected_answerable": False,
        "answerability_correct": True,
        "abstained": True,
        "expected_sources": [],
        "expected_evidence": [],
        "search_queries": ["What is the private launch code?"],
        "search_results": [[]],
        "generation_context": [],
    }
    metric_summary = {
        "recall": 0.9,
        "precision": 0.7,
        "ndcg": 0.8,
        "context_utilization": 0.6123,
    }
    return {
        "run_metadata": {
            "workflow": workflow,
            "answer_provider": "fixture-answer",
            "answer_model": "answer-v1",
            "judge_provider": "fixture-judge",
            "judge_model": "judge-v1",
        },
        "summary": {
            "case_count": 2,
            "workflows": {
                workflow: {
                    "retrieval_metrics": metric_summary,
                    "generation_metrics": answerable_case["generation_metrics"],
                    "ragas_metric_eligible_case_counts": {
                        "faithfulness": 1,
                        "answer_relevancy": 1,
                        "answer_correctness": 1,
                        "context_utilization": 1,
                    },
                    "answerability_outcomes": {
                        "expected_answerable": {
                            "case_count": 1,
                            "answered_count": 1,
                            "abstained_count": 0,
                            "unavailable_count": 0,
                            "error_count": 0,
                            "unavailable_or_error_count": 0,
                            "correct_behavior_count": 1,
                        },
                        "expected_unanswerable": {
                            "case_count": 1,
                            "answered_count": 0,
                            "abstained_count": 1,
                            "unavailable_count": 0,
                            "error_count": 0,
                            "unavailable_or_error_count": 0,
                            "correct_behavior_count": 1,
                        },
                    },
                    "judge_usage_by_metric": {
                        "context_utilization": {
                            "call_count": 1,
                            "token_usage": partial_usage,
                        },
                    },
                    "generation_usage": {
                        "call_count": 1,
                        "token_usage": available_usage,
                    },
                    "by_category": {
                        "direct": {
                            "case_count": 1,
                            "generation_metrics": answerable_case[
                                "generation_metrics"
                            ],
                            "retrieval_metrics": metric_summary,
                            "ragas_metric_eligible_case_counts": {
                                "faithfulness": 1,
                                "answer_relevancy": 1,
                                "answer_correctness": 1,
                                "context_utilization": 1,
                            },
                            "latency_ms": {},
                            "generation_usage": {"token_usage": available_usage},
                            "judge_usage": {"token_usage": available_usage},
                        },
                        "unanswerable": {
                            "case_count": 1,
                            "generation_metrics": {
                                "faithfulness": None,
                                "answer_relevancy": None,
                                "answer_correctness": None,
                            },
                            "retrieval_metrics": {"context_utilization": None},
                            "ragas_metric_eligible_case_counts": {
                                "faithfulness": 0,
                                "answer_relevancy": 0,
                                "answer_correctness": 0,
                                "context_utilization": 0,
                            },
                            "latency_ms": {},
                            "generation_usage": {
                                "token_usage": {"status": "unavailable"}
                            },
                            "judge_usage": {
                                "token_usage": {"status": "unavailable"}
                            },
                        },
                    },
                },
            },
            "judge_usage": {
                "total": {"call_count": 4, "token_usage": available_usage},
            },
        },
        "category_counts": {"direct": 1, "unanswerable": 1},
        "cases": [answerable_case, abstention_case],
    }


@pytest.mark.parametrize("workflow", ["classic", "agentic"])
def test_report_distinguishes_evidence_context_diagnostics_and_execution_cost(
    tmp_path,
    workflow,
):
    report_path = tmp_path / "report.md"

    write_readme(report_path, _report_results(workflow))

    report = report_path.read_text(encoding="utf-8")
    assert report.index("## Overall metrics") < report.index(
        "## Evaluation process and metric guide"
    ) < report.index("## Query details")
    assert "[results.json](results.json)" in report
    assert "[run.log](run.log)" in report
    for metric in ("recall", "precision", "ndcg"):
        assert f"| {metric} |" in report
    assert "Recall measures evidence coverage" in report
    assert "Precision measures the share of retrieved lines that are relevant" in report
    assert "NDCG measures whether relevant evidence ranks early" in report
    assert "not calibrated probabilities" in report
    assert "Reference answer:" in report and "Generated answer:" in report
    assert "expected_source_coverage" not in report
    assert "expected_evidence_coverage" not in report
    judge_usage_table = report.split("#### Judge usage by metric", 1)[1].split(
        "## Metrics by category", 1
    )[0]
    assert "| context_utilization |" in judge_usage_table
    assert "| context_utilization |" in report
    assert "0.6123" in report and "0.7345" in report
    assert "| Faithfulness | 0.9100 | 1/2 |" in report
    assert (
        "| Context Utilization (context/retrieval diagnostic) | 0.6123 | 1/2 |"
        in report
    )
    assert "0.9100 (1/1 eligible)" in report
    assert "0.8200 (1/1 eligible)" in report
    assert "0.8700 (1/1 eligible)" in report
    assert "0.6123 (1/1 eligible)" in report
    assert "| Expected answerable | 1 | 1 | 0 | 0 | 1/1 |" in report
    assert "| Expected unanswerable | 1 | 0 | 1 | 0 | 1/1 |" in report
    assert "Answerability accuracy" not in report
    assert "N/A (0/1 eligible)" in report
    assert "Reviewed evidence text" in report
    assert "notes/launch.md" in report and "12–13" in report
    assert "Retrieved chunk text" in report and "Search query" in report
    assert "FAISS vector search" in report
    assert "No chunk returned" in report
    assert "No final-answer context was sent." in report
    assert "**Observed behavior:** `answered`" in report
    assert "**Observed behavior:** `abstained`" in report
    assert "**Answerability check:** true" in report
    assert "#### Citations" not in report

    query_details = report.split("## Query details", 1)[1]
    assert query_details.count("#### Execution audit: latency and token usage") == 2
    query_sections = re.split(r"(?m)^### ", query_details)[1:]
    assert len(query_sections) == 2
    for query_section in query_sections:
        assert "[results.json](results.json)" in query_section
        assert "[run.log](run.log)" in query_section
    assert "#### Token usage" not in query_details
    assert "#### Latency" not in query_details
    assert "| Process/call | Latency (ms) | Provider / model | " in report
    assert (
        "Provider total tokens | Calculated input+output tokens | "
        "Input chars | Output chars |"
    ) in report
    assert "fixture-answer / answer-v1" in report
    assert "fixture-judge / judge-v1" in report
    assert "Judge: context_utilization" in report
    assert "| partial | 5 | N/A | N/A | N/A | N/A | N/A |" in report

    if workflow == "agentic":
        retrieval = report.split("#### Retrieval", 1)[1].split(
            "#### Context passed to answer generation", 1
        )[0]
        assert retrieval.index("Project Cedar launch date confirmation") < retrieval.index(
            "Launch date: 2026-07-14."
        )
        context = report.split("#### Context passed to answer generation", 1)[1].split(
            "#### Ragas judge diagnostics", 1
        )[0]
        assert context.index("file:notes/launch.md<br>Launch date: 2026-07-14.") < context.index(
            "Same as Retrieval search 1, rank 1"
        )
    else:
        assert report.count("When does Project Cedar launch?") >= 2
        assert "Same as Retrieval search 1, rank 2" in report

    judge = report.split("#### Ragas judge diagnostics", 1)[1].split(
        "#### Execution audit: latency and token usage", 1
    )[0]
    for metric in (
        "Faithfulness",
        "Answer Relevancy",
        "Answer Correctness",
        "Context Utilization",
    ):
        assert judge.count(metric) == 1
    assert "Diagnostic" in judge
    assert "Skipped because expected answerability is unanswerable." in report


@pytest.mark.parametrize("workflow", ["classic", "agentic"])
def test_report_marks_an_empty_expected_answerability_group_as_na(
    tmp_path,
    workflow,
):
    results = _report_results(workflow)
    results["summary"]["workflows"][workflow]["answerability_outcomes"][
        "expected_unanswerable"
    ] = {
        "case_count": 0,
        "answered_count": None,
        "abstained_count": None,
        "unavailable_count": None,
        "error_count": None,
        "unavailable_or_error_count": None,
        "correct_behavior_count": None,
    }
    report_path = tmp_path / "empty-answerability-report.md"

    write_readme(report_path, results)

    report = report_path.read_text(encoding="utf-8")
    assert "| Expected unanswerable | 0 | N/A | N/A | N/A | N/A |" in report


def test_report_explains_when_ragas_evaluation_fails(tmp_path):
    results = _report_results("classic")
    case = results["cases"][0]
    case["ragas_error"] = "RuntimeError"
    case["generation_metrics"] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "answer_correctness": None,
    }
    case["retrieval_metrics"]["context_utilization"] = None
    case["metric_reasons"] = {}
    case["judge_metrics"] = {}
    report_path = tmp_path / "ragas-error-report.md"

    write_readme(report_path, results)

    report = report_path.read_text(encoding="utf-8")
    query_details = report.split("#### Ragas judge diagnostics", 1)[1].split(
        "#### Execution audit: latency and token usage", 1
    )[0]
    for metric in (
        "Faithfulness",
        "Answer Relevancy",
        "Answer Correctness",
        "Context Utilization (context/retrieval diagnostic)",
    ):
        assert (
            f"| {metric} | N/A | Ragas evaluation failed (RuntimeError). |"
            in query_details
        )


@pytest.mark.parametrize(
    ("status", "input_tokens", "expected"),
    [
        ("partial", 5, "| partial | 5 | N/A | N/A | N/A | N/A | N/A |"),
        (
            "unavailable",
            None,
            "| unavailable | N/A | N/A | N/A | N/A | N/A | N/A |",
        ),
    ],
)
def test_report_shows_missing_provider_usage_without_estimates(
    tmp_path,
    status,
    input_tokens,
    expected,
):
    token_usage = {
        "status": status,
        "input_tokens": input_tokens,
        "output_tokens": None,
        "total_tokens": None,
    }
    result = {
        "run_metadata": {"workflow": "classic"},
        "summary": {"workflows": {"classic": {}}},
        "cases": [{
            "case_id": "partial-usage",
            "generation_usage": {"calls": [{
                "call_type": "answer",
                "provider": "fixture",
                "model": "answer-v1",
                "token_usage": token_usage,
            }]},
        }],
    }
    report_path = tmp_path / "partial-report.md"

    write_readme(report_path, result)

    report = report_path.read_text(encoding="utf-8")
    assert expected in report
    assert "out=None" not in report
    assert "total=None" not in report


@pytest.mark.parametrize("workflow", ["classic", "agentic"])
def test_report_audits_search_registry_and_context_prefixes(tmp_path, workflow):
    results = _report_results(workflow)
    results["run_metadata"]["top_k"] = 3
    case = results["cases"][0]
    case["context_prefix_metrics"] = (
        [
            {
                "context_prefix_k": 1,
                "evidence_lines_covered": None,
                "ground_truth_line_count": 2,
                "recall": None,
                "precision": None,
                "ndcg": None,
                "line_provenance_available": False,
            },
            {
                "context_prefix_k": 2,
                "evidence_lines_covered": None,
                "ground_truth_line_count": 2,
                "recall": None,
                "precision": None,
                "ndcg": None,
                "line_provenance_available": False,
            },
        ]
        if workflow == "agentic"
        else [
            {
                "context_prefix_k": 1,
                "evidence_lines_covered": 1,
                "ground_truth_line_count": 2,
                "recall": 0.5,
                "precision": 0.5,
                "ndcg": 1.0,
                "line_provenance_available": True,
            },
        ]
    )
    report_path = tmp_path / f"{workflow}-context-prefix-report.md"

    write_readme(report_path, results)

    report = report_path.read_text(encoding="utf-8")
    assert "| Retrieval top-k | Recall | Precision | NDCG |" in report
    assert "| 3 | 0.9000 | 0.7000 | 0.8000 |" in report
    assert "Retrieval (1) uses the initial question" in report
    assert "Retrieval (2) identifies a controller-requested search" in report
    query_section = report.split("## Query details", 1)[1].split("## Artifacts", 1)[0]
    query_details = query_section.split("### direct-01", 1)[1]
    search_registry = query_details.split("#### Search registry", 1)[1].split(
        "#### Ranked retrieval results", 1
    )[0]
    ranked_results = query_details.split(
        "#### Ranked retrieval results", 1
    )[1].split("#### Context passed to answer generation", 1)[0]
    assert "| Search ID | Search query | Stage | Method |" in search_registry
    assert "| Retrieval (1) | When does Project Cedar launch? | retrieval | FAISS vector search |" in search_registry
    if workflow == "agentic":
        assert "| Retrieval (2) | Project Cedar launch date confirmation | retrieval | FAISS vector search |" in search_registry
        assert "When does Project Cedar launch?" not in ranked_results
        assert "Project Cedar launch date confirmation" not in ranked_results
    assert "| Search ID | Rank | Chunk ID | Source path | Lines | Score | Retrieved chunk text |" in ranked_results
    assert "| Retrieval (1) | 1 | 4 |" in ranked_results
    assert "Search query" not in ranked_results
    assert "FAISS vector search" not in ranked_results
    assert "Context prefix k" in query_details
    assert "Unique evidence lines / Recall" in query_details
    assert "| 1 | N/A | N/A | N/A |" in query_details if workflow == "agentic" else (
        "| 1 | 1/2 (0.5000) | 0.5000 | 1.0000 |" in query_details
    )
    assert "prefix rows assess evidence available at each context length" in report
    assert "one answer is generated from the complete final context" in report
    assert (
        "- **Expected sources:** [&quot;notes/launch.md&quot;]"
        "\n\n| Source path | Lines | Reviewed evidence text |"
    ) in report


@pytest.mark.parametrize("workflow", ["classic", "agentic"])
def test_report_separates_provider_totals_calculated_totals_and_characters(
    tmp_path,
    workflow,
):
    result = _report_results(workflow)
    workflow_summary = result["summary"]["workflows"][workflow]
    calculated_usage = {
        "status": "partial",
        "input_tokens": 7,
        "output_tokens": 5,
        "total_tokens": None,
    }
    answer_call = result["cases"][0]["generation_usage"]["calls"][0]
    answer_call["token_usage"] = calculated_usage
    answer_call["characters"] = {"input": 64, "output": 21}
    controller_call = {
        "call_type": "controller",
        "provider": "fixture-controller",
        "model": "controller-v1",
        "latency_ms": 2.5,
        "token_usage": {
            "status": "available",
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
        },
        "characters": {"input": 77, "output": 9},
    }
    generation_usage = {
        "call_count": 2,
        "token_usage": {
            "status": "partial",
            "input_tokens": 10,
            "output_tokens": 9,
            "total_tokens": 7,
        },
        "character_usage": {"input": 141, "output": 30},
        "calls": [controller_call, answer_call],
    }
    result["cases"][0]["generation_usage"] = generation_usage
    workflow_summary["generation_usage"] = generation_usage

    metric_names = (
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_utilization",
    )
    judge_call = {
        "call_type": "judge",
        "provider": "fixture-judge",
        "model": "judge-v1",
        "token_usage": {
            "status": "partial",
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": None,
        },
    }
    judge_usage_by_metric = {
        metric: {
            "call_count": 1,
            "token_usage": judge_call["token_usage"],
            "calls": [{**judge_call, "metric": metric}],
        }
        for metric in metric_names
    }
    result["cases"][0]["judge_usage_by_metric"] = judge_usage_by_metric
    workflow_summary["judge_usage_by_metric"] = judge_usage_by_metric
    judge_calls = [
        {**judge_call, "metric": metric}
        for metric in metric_names
    ]
    result["summary"]["judge_usage"]["total"] = {
        "call_count": len(judge_calls),
        "token_usage": {
            "status": "partial",
            "input_tokens": 8,
            "output_tokens": 4,
            "total_tokens": None,
        },
        "calls": judge_calls,
    }
    workflow_summary["by_category"] = {
        "direct": {
            "case_count": 1,
            "generation_usage": generation_usage,
            "judge_usage": result["summary"]["judge_usage"]["total"],
        },
    }

    report_path = tmp_path / "usage-report.md"
    write_readme(report_path, result)

    report = report_path.read_text(encoding="utf-8")
    assert "Provider total tokens" in report
    assert "Calculated input+output tokens" in report
    assert "Input chars" in report and "Output chars" in report
    assert "calculated input+output=12" in report
    assert "calculated input+output=19" in report
    assert "provider total=N/A" in report
    assert (
        "| controller call 1 | 2.5000 | fixture-controller / controller-v1 | "
        "available | 3 | 4 | 7 | N/A | 77 | 9 |"
    ) in report
    assert (
        "| answer call 2 | 4.5000 | fixture-answer / answer-v1 | "
        "partial | 7 | 5 | N/A | 12 | 64 | 21 |"
    ) in report
    assert "Judge: context_utilization" in report
    assert "Input chars | Output chars |" in report
    assert "N/A | N/A" in report
    judge_usage_table = report.split("#### Judge usage by metric", 1)[1].split(
        "## Metrics by category", 1
    )[0]
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_utilization",
    ):
        assert f"| {metric} |" in judge_usage_table
    category_table = report.split("## Metrics by category", 1)[1].split(
        "## Evaluation process and metric guide", 1
    )[0]
    assert "calculated input+output=19" in category_table


@pytest.mark.parametrize("workflow", ["classic", "agentic"])
def test_report_uses_clear_latency_names_and_definitions(tmp_path, workflow):
    result = _report_results(workflow)
    workflow_summary = result["summary"]["workflows"][workflow]
    workflow_summary["latency_ms"] = {
        "retrieval_ms": 1.5,
        "controller_ms": 2.5,
        "answer_generation_ms": 4.5,
        "generation_ms": 7.0,
        "judge_ms": 6.0,
        "rag_query_ms": 8.0,
        "end_to_end_ms": 14.0,
    }
    workflow_summary["by_category"] = {
        "direct": {
            "case_count": 1,
            "latency_ms": {"end_to_end_ms": 14.0},
            "generation_usage": {"token_usage": {"status": "unavailable"}},
            "judge_usage": {"token_usage": {"status": "unavailable"}},
        },
    }
    case = result["cases"][0]
    case["generation_usage"]["calls"].insert(0, {
        "call_type": "controller",
        "provider": "fixture-controller",
        "model": "controller-v1",
        "latency_ms": 2.5,
        "token_usage": {
            "status": "unavailable",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
    })
    case["latency"].update({
        "retrieval_calls": [{"search_index": 1, "latency_ms": 1.5}],
        "retrieval_ms": 1.5,
        "controller_ms": 2.5,
        "answer_generation_ms": 4.5,
        "generation_ms": 7.0,
        "judge_ms": 6.0,
        "rag_query_ms": 8.0,
        "end_to_end_ms": 14.0,
    })

    report_path = tmp_path / "latency-report.md"
    write_readme(report_path, result)

    report = report_path.read_text(encoding="utf-8")
    assert "Mean latency (ms)" in report
    assert "Latency (ms)" in report
    assert "RAG query latency" in report
    assert "Evaluation latency" in report
    assert "Mean evaluation latency (ms)" in report
    assert "Retrieval latency total" in report
    assert "Controller latency total" in report
    assert "Answer generation latency total" in report
    assert "RAG generation total" not in report
    assert "generation_ms" not in report
    assert "TTFT" not in report
    assert "RAG query latency runs from query processing through the terminal answer or abstention" in report
    assert "Evaluation latency adds all Ragas judge latency to RAG query latency" in report
