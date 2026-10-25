"""Human-readable Markdown rendering for evaluation run reports."""

from html import escape
import json
from pathlib import Path
from typing import Any


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        rendered = str(value)
    return escape(rendered).replace("|", "&#124;").replace("\r\n", "<br>").replace("\n", "<br>")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _header in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in rows
    )
    return lines


def _metric_rows(metrics: dict[str, Any]) -> list[list[Any]]:
    return [[name, value] for name, value in metrics.items()]


def _token_usage_summary(usage: dict[str, Any]) -> str:
    token_usage = usage.get("token_usage", {})
    return (
        f"{token_usage.get('status', 'unavailable')}: "
        f"in={token_usage.get('input_tokens', 'N/A')}, "
        f"out={token_usage.get('output_tokens', 'N/A')}, "
        f"total={token_usage.get('total_tokens', 'N/A')}"
    )


def _readme_report(results: dict[str, Any]) -> str:
    metadata = results.get("run_metadata", {})
    summary = results.get("summary", {})
    workflow = metadata.get("workflow", "unknown")
    timestamp = metadata.get("timestamp_local", "unknown")
    run_status = "failed" if results.get("error") else "completed"
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- **Status:** `{run_status}`",
        f"- **Workflow:** `{workflow}`",
        f"- **Run timestamp (Asia/Taipei):** `{timestamp}`",
        f"- **Fixture SHA-256:** `{metadata.get('fixture_sha256', 'unavailable')}`",
        "- **Artifacts:** [results.json](results.json) · [run.log](run.log)",
        "",
        "## Configuration",
        "",
    ]
    config_fields = (
        ("Answer provider", "answer_provider"),
        ("Answer model", "answer_model"),
        ("Judge provider", "judge_provider"),
        ("Judge model", "judge_model"),
        ("Embedding model", "embedding_model"),
        ("Chunk size", "chunk_size"),
        ("Chunk overlap", "chunk_overlap"),
        ("Top-k", "top_k"),
        ("Minimum retrieval score", "min_retrieval_score"),
        ("Maximum answer chunks", "max_answer_chunks"),
        ("Agentic maximum searches", "agentic_max_searches"),
    )
    lines.extend(_markdown_table(
        ["Setting", "Value"],
        [[label, metadata.get(key)] for label, key in config_fields],
    ))
    category_counts = results.get("category_counts", {})
    lines.extend([
        "",
        "## Fixture coverage",
        "",
        f"- **Cases:** {summary.get('case_count', len(results.get('cases', [])))}",
        f"- **Interpretation:** {_markdown_cell(summary.get('interpretation'))}",
        "",
    ])
    lines.extend(_markdown_table(
        ["Category", "Cases"],
        [[category, count] for category, count in sorted(category_counts.items())],
    ))

    workflow_summary = summary.get("workflows", {}).get(workflow, {})
    lines.extend(["", "## Overall metrics", ""])
    lines.extend(_markdown_table(
        ["Retrieval metric", "Score"],
        _metric_rows(workflow_summary.get("retrieval_metrics", {})),
    ))
    lines.extend(["", "### Generation metrics", ""])
    lines.extend(_markdown_table(
        ["Generation metric", "Score"],
        _metric_rows(workflow_summary.get("generation_metrics", {})),
    ))
    lines.extend(["", "### Answerability and evidence", ""])
    lines.extend(_markdown_table(
        ["Metric", "Score"],
        [
            ["Expected source coverage", workflow_summary.get("expected_source_coverage")],
            ["Expected evidence coverage", workflow_summary.get("expected_evidence_coverage")],
            ["Answerability accuracy", workflow_summary.get("answerability_accuracy")],
            ["Abstention rate", workflow_summary.get("abstention_rate")],
            ["Correct abstention rate", workflow_summary.get("correct_abstention_rate")],
        ],
    ))
    lines.extend(["", "### Mean latency (milliseconds)", ""])
    lines.extend(_markdown_table(
        ["Stage", "Latency"],
        _metric_rows(workflow_summary.get("latency_ms", {})),
    ))
    generation_usage = workflow_summary.get("generation_usage", {})
    lines.extend(["", "### Provider token usage", ""])
    usage_rows = []
    for label, usage in [
        ("Controller and answer", generation_usage),
        ("Judge total", summary.get("judge_usage", {}).get("total", {})),
    ]:
        token_usage = usage.get("token_usage", {})
        usage_rows.append([
            label,
            usage.get("call_count"),
            token_usage.get("status"),
            token_usage.get("input_tokens"),
            token_usage.get("output_tokens"),
            token_usage.get("total_tokens"),
        ])
    lines.extend(_markdown_table(
        ["Calls", "Count", "Status", "Input", "Output", "Total"],
        usage_rows,
    ))
    judge_by_metric = workflow_summary.get("judge_usage_by_metric", {})
    if judge_by_metric:
        lines.extend(["", "#### Judge usage by metric", ""])
        lines.extend(_markdown_table(
            ["Metric", "Calls", "Status", "Input", "Output", "Total"],
            [
                [
                    metric,
                    usage.get("call_count"),
                    usage.get("token_usage", {}).get("status"),
                    usage.get("token_usage", {}).get("input_tokens"),
                    usage.get("token_usage", {}).get("output_tokens"),
                    usage.get("token_usage", {}).get("total_tokens"),
                ]
                for metric, usage in judge_by_metric.items()
            ],
        ))

    category_summaries = workflow_summary.get("by_category", {})
    lines.extend(["", "## Metrics by category", ""])
    category_rows = []
    for category, metrics in sorted(category_summaries.items()):
        category_rows.append([
            category,
            metrics.get("case_count"),
            metrics.get("retrieval_metrics", {}).get("recall"),
            metrics.get("retrieval_metrics", {}).get("precision"),
            metrics.get("retrieval_metrics", {}).get("ndcg"),
            metrics.get("generation_metrics", {}).get("faithfulness"),
            metrics.get("generation_metrics", {}).get("answer_relevancy"),
            metrics.get("generation_metrics", {}).get("answer_correctness"),
            metrics.get("answerability_accuracy"),
            metrics.get("correct_abstention_rate"),
            metrics.get("latency_ms", {}).get("end_to_end_ms"),
            _token_usage_summary(metrics.get("generation_usage", {})),
            _token_usage_summary(metrics.get("judge_usage", {})),
        ])
    lines.extend(_markdown_table(
        [
            "Category", "Cases", "Recall", "Precision", "NDCG",
            "Faithfulness", "Answer relevancy", "Answer correctness",
            "Answerability accuracy", "Correct abstention", "Mean end-to-end ms",
            "Controller/answer tokens", "Judge tokens",
        ],
        category_rows,
    ))

    lines.extend(["", "## Query details", ""])
    for case in results.get("cases", []):
        lines.extend([
            f"### {case.get('case_id', 'unknown')}",
            "",
            f"- **Category:** `{case.get('category', 'unknown')}`",
            f"- **Question:** {_markdown_cell(case.get('question'))}",
            f"- **Reference answer:** {_markdown_cell(case.get('reference_answer'))}",
            f"- **Generated answer:** {_markdown_cell(case.get('answer'))}",
            f"- **Expected answerable:** {_markdown_cell(case.get('expected_answerable'))}",
            f"- **Predicted answerable:** {_markdown_cell(case.get('predicted_answerable'))}",
            f"- **Abstained:** {_markdown_cell(case.get('abstained'))}",
            "",
            "#### Retrieval",
            "",
        ])
        lines.extend(_markdown_table(
            ["Metric", "Score"],
            _metric_rows(case.get("retrieval_metrics", {}))
            + [
                ["Expected source coverage", case.get("expected_source_coverage")],
                ["Expected evidence coverage", case.get("expected_evidence_coverage")],
            ],
        ))
        lines.extend(["", "Retrieved chunks are ranked within each search; scores are FAISS similarity scores, not calibrated probabilities.", ""])
        chunk_rows = []
        for search_number, search_hits in enumerate(case.get("search_results", []), start=1):
            for rank, hit in enumerate(search_hits, start=1):
                chunk_rows.append([
                    search_number,
                    rank,
                    hit.get("chunk_id"),
                    hit.get("source"),
                    f"{hit.get('start_line', 'N/A')}–{hit.get('end_line', 'N/A')}",
                    hit.get("score"),
                    hit.get("text"),
                ])
        lines.extend(_markdown_table(
            ["Search", "Rank", "Chunk ID", "Source path", "Lines", "similarity_score (FAISS)", "Chunk text"],
            chunk_rows,
        ))
        lines.extend(["", "#### Citations", ""])
        citations = case.get("citations", [])
        lines.extend(_markdown_table(
            ["Source path", "Chunk ID", "Lines", "similarity_score (FAISS)"],
            [
                [
                    citation.get("source"),
                    citation.get("chunk_id"),
                    f"{citation.get('start_line', 'N/A')}–{citation.get('end_line', 'N/A')}",
                    citation.get("score"),
                ]
                for citation in citations
            ],
        ))
        lines.extend(["", "#### Generation", ""])
        lines.extend(_markdown_table(
            ["Metric", "Score"],
            _metric_rows(case.get("generation_metrics", {})),
        ))
        lines.extend(["", "#### LLM-as-judge", ""])
        judge_metrics = case.get("judge_metrics", {})
        lines.extend(_markdown_table(
            ["Metric", "Score", "Latency (ms)", "Token usage", "Rationale"],
            [
                [
                    metric,
                    detail.get("score"),
                    detail.get("latency_ms"),
                    detail.get("usage", {}).get("token_usage"),
                    detail.get("reason"),
                ]
                for metric, detail in judge_metrics.items()
            ],
        ))
        lines.extend(["", "#### Latency (milliseconds)", ""])
        lines.extend(_markdown_table(
            ["Stage", "Latency"],
            _metric_rows({
                key: value
                for key, value in case.get("latency", {}).items()
                if key != "retrieval_calls"
            }),
        ))
        search_latencies = case.get("latency", {}).get("retrieval_calls", [])
        if search_latencies:
            lines.extend(["", "Search latency details:", ""])
            lines.extend(_markdown_table(
                ["Search", "Latency (ms)"],
                [
                    [index, call.get("latency_ms")]
                    for index, call in enumerate(search_latencies, start=1)
                ],
            ))
        lines.extend(["", "#### Token usage", ""])
        generation_calls = case.get("generation_usage", {}).get("calls", [])
        lines.extend(_markdown_table(
            ["Call type", "Provider", "Model", "Status", "Input", "Output", "Total"],
            [
                [
                    call.get("call_type"),
                    call.get("provider"),
                    call.get("model"),
                    call.get("token_usage", {}).get("status"),
                    call.get("token_usage", {}).get("input_tokens"),
                    call.get("token_usage", {}).get("output_tokens"),
                    call.get("token_usage", {}).get("total_tokens"),
                ]
                for call in generation_calls
            ],
        ))
        lines.extend([""])

    lines.extend([
        "## Artifacts",
        "",
        "- [Machine-readable results](results.json)",
        "- [Evaluation process log](run.log)",
        "",
    ])
    return "\n".join(lines)


def write_readme(path: Path, results: dict[str, Any]) -> None:
    path.write_text(_readme_report(results), encoding="utf-8")
