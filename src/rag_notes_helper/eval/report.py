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


def _token_usage_cell(usage: dict[str, Any] | None) -> str:
    if not usage:
        return "N/A"
    return _token_usage_summary({"token_usage": usage})


def _answer_behavior(case: dict[str, Any]) -> str:
    if case.get("workflow_error"):
        return f"error ({case['workflow_error']})"
    if case.get("abstained") is True:
        return "abstained"
    if case.get("answer") is None:
        return "unavailable"
    return "answered"


def _execution_rows(case: dict[str, Any]) -> list[list[Any]]:
    latency = case.get("latency", {})
    rows: list[list[Any]] = []
    for index, call in enumerate(latency.get("retrieval_calls", []), start=1):
        rows.append([
            f"Retrieval search {call.get('search_index', index)}",
            call.get("latency_ms"),
            "N/A",
            "N/A",
        ])
    for index, call in enumerate(
        case.get("generation_usage", {}).get("calls", []),
        start=1,
    ):
        rows.append([
            f"{call.get('call_type', 'generation')} call {index}",
            call.get("latency_ms"),
            f"{call.get('provider', 'N/A')} / {call.get('model', 'N/A')}",
            _token_usage_cell(call.get("token_usage")),
        ])
    for metric, usage in case.get("judge_usage_by_metric", {}).items():
        calls = usage.get("calls", [])
        metric_latency = case.get("latency", {}).get("judge_metrics_ms", {}).get(metric)
        if not calls and metric_latency is None:
            continue
        providers = sorted({
            f"{call.get('provider', 'N/A')} / {call.get('model', 'N/A')}"
            for call in calls
        })
        rows.append([
            f"Judge: {metric}",
            metric_latency,
            ", ".join(providers) if providers else "N/A",
            _token_usage_cell(usage.get("token_usage")),
        ])

    for key, label, usage_key in (
        ("retrieval_ms", "Retrieval total", None),
        ("controller_ms", "Controller total", None),
        ("answer_generation_ms", "Answer generation total", None),
        ("generation_ms", "RAG generation total", "generation_usage"),
        ("judge_ms", "Judge total", "judge_usage"),
        ("rag_query_ms", "RAG query total", None),
        ("end_to_end_ms", "End-to-end total", None),
    ):
        if key in latency:
            usage = case.get(usage_key, {}) if usage_key else {}
            rows.append([
                label,
                latency.get(key),
                f"{usage.get('call_count')} calls" if usage_key else "—",
                _token_usage_cell(usage.get("token_usage")) if usage_key else "—",
            ])
    return rows


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
    visible_retrieval_metrics = {
        name: score
        for name, score in workflow_summary.get("retrieval_metrics", {}).items()
        if name in {"recall", "precision", "ndcg"}
    }
    lines.extend(_markdown_table(
        ["Retrieval metric", "Score"],
        _metric_rows(visible_retrieval_metrics),
    ))
    lines.extend(["", "### Ragas answer metrics", ""])
    lines.extend(_markdown_table(
        ["Ragas answer metric", "Score"],
        _metric_rows(workflow_summary.get("generation_metrics", {})),
    ))
    lines.extend(["", "### Answerability behavior", ""])
    lines.extend(_markdown_table(
        ["Metric", "Score"],
        [
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
                if metric in {"faithfulness", "answer_relevancy", "answer_correctness"}
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

    lines.extend([
        "",
        "## Evaluation process and metric guide",
        "",
        "1. The fixture supplies the reference answer, expected answerability, expected source paths, and reviewed evidence spans.",
        "2. The selected Classic or Agentic workflow runs retrieval. Each search returns ranked candidate chunks; Agentic RAG can search repeatedly and select a subset for final generation.",
        "3. The answer generator receives the ordered context chunks shown in each query's context section. Each chunk is formatted as `file:<source>\\n<chunk text>` and blocks are separated by a blank line. If the workflow abstains, no final-answer context is sent.",
        "4. Deterministic retrieval scores compare retrieved source lines with the reviewed evidence: Recall measures evidence coverage, Precision measures the share of retrieved lines that are relevant, and NDCG measures whether relevant evidence ranks early. For Agentic RAG, Recall and Precision use the union of search results, while NDCG is averaged across applicable searches.",
        "5. Deterministic answerability checks compare expected answerability with whether the workflow answered or abstained.",
        "6. Ragas scores run only for expected-answerable cases with an answer. Faithfulness checks whether answer claims are supported by the answer context; Answer Relevancy checks whether the response addresses the question; Answer Correctness compares the response with the reference answer. Answer Relevancy and Answer Correctness also use the configured local evaluation embeddings. These judge scores are diagnostics, not pass/fail guarantees.",
        "7. Context Utilization may still be computed as an advanced diagnostic in `results.json`; its score is omitted here. Its judge-call latency and provider-reported token usage remain visible in the query execution audit.",
        "8. Latency is measured per process. Token usage covers provider-reported LLM calls; local embedding work is not represented as token usage. Missing values are marked unavailable and are never estimated. Retrieval does not use LLM tokens.",
        "",
        "Retrieved similarity scores are ranking signals, not calibrated probabilities. A search rank is local to that search; the answer-context table shows the final order provided to answer generation.",
        "",
        "## Query details",
        "",
    ])
    for case in results.get("cases", []):
        behavior = _answer_behavior(case)
        lines.extend([
            f"### {case.get('case_id', 'unknown')}",
            "",
            f"- **Category:** `{case.get('category', 'unknown')}`",
            f"- **Question:** {_markdown_cell(case.get('question'))}",
            f"- **Reference answer:** {_markdown_cell(case.get('reference_answer'))}",
            f"- **Generated answer:** {_markdown_cell(case.get('answer'))}",
            f"- **Expected answerable:** {_markdown_cell(case.get('expected_answerable'))}",
            f"- **Observed behavior:** `{behavior}`",
            f"- **Answerability check:** {_markdown_cell(case.get('answerability_correct'))}",
            "",
            "#### Expected evidence",
            "",
        ])
        expected_sources = case.get("expected_sources", [])
        lines.append(f"- **Expected sources:** {_markdown_cell(expected_sources)}")
        evidence_rows = [
            [
                item.get("source"),
                f"{item.get('start_line', 'N/A')}–{item.get('end_line', 'N/A')}",
                item.get("text"),
            ]
            for item in case.get("expected_evidence", [])
        ]
        lines.extend(_markdown_table(
            ["Source path", "Lines", "Reviewed evidence text"],
            evidence_rows,
        ))
        lines.extend(["", "#### Retrieval", ""])
        lines.extend(_markdown_table(
            ["Line-grounded retrieval metric", "Score"],
            _metric_rows({
                name: score
                for name, score in case.get("retrieval_metrics", {}).items()
                if name in {"recall", "precision", "ndcg"}
            }),
        ))
        lines.extend([
            "",
            "Search ranks are independent. Current scores are FAISS similarity scores, not calibrated probabilities.",
            "",
        ])
        chunk_rows = []
        for search_number, search_hits in enumerate(case.get("search_results", []), start=1):
            search_query = case.get("search_queries", [])
            search_query = (
                search_query[search_number - 1]
                if search_number <= len(search_query)
                else case.get("question")
            )
            if not search_hits:
                chunk_rows.append([
                    search_number,
                    search_query,
                    "retrieval",
                    "FAISS vector search",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "No chunk returned",
                ])
            for rank, hit in enumerate(search_hits, start=1):
                stage = hit.get("stage", hit.get("retrieval_stage", "retrieval"))
                method = hit.get("method", hit.get("retrieval_method", "FAISS vector search"))
                chunk_rows.append([
                    search_number,
                    search_query,
                    stage,
                    method,
                    rank,
                    hit.get("chunk_id"),
                    hit.get("source"),
                    f"{hit.get('start_line', 'N/A')}–{hit.get('end_line', 'N/A')}",
                    hit.get("score"),
                    hit.get("text"),
                ])
        lines.extend(_markdown_table(
            ["Search", "Search query", "Stage", "Method", "Rank", "Chunk ID", "Source path", "Lines", "Score", "Retrieved chunk text"],
            chunk_rows,
        ))
        lines.extend(["", "#### Context passed to answer generation", ""])
        generation_context = case.get("generation_context", [])
        search_rank_by_reference = {}
        for search_number, search_hits in enumerate(case.get("search_results", []), start=1):
            for rank, hit in enumerate(search_hits, start=1):
                search_rank_by_reference.setdefault(
                    (hit.get("source"), hit.get("chunk_id")),
                    (search_number, rank, hit),
                )
        lines.extend(_markdown_table(
            ["Final context order", "Chunk ID", "Source path", "Lines", "Context content"],
            [
                [
                    order,
                    item.get("chunk_id"),
                    item.get("source"),
                    f"{item.get('start_line', 'N/A')}–{item.get('end_line', 'N/A')}",
                    (
                        f"Same as Retrieval search {search_rank[0]}, rank {search_rank[1]}"
                        if (search_rank := search_rank_by_reference.get(
                            (item.get("source"), item.get("chunk_id"))
                        )) and (
                            item.get("text") is None
                            or item.get("text") == search_rank[2].get("text")
                        )
                        else item.get("prompt_block", item.get("text"))
                    ),
                ]
                for order, item in enumerate(generation_context, start=1)
            ],
        ))
        if not generation_context:
            lines.append("No final-answer context was sent.")
        lines.extend(["", "#### Ragas judge diagnostics", ""])
        judge_metrics = case.get("judge_metrics", {})
        visible_judge_metrics = {
            name: judge_metrics.get(name, {})
            for name in ("faithfulness", "answer_relevancy", "answer_correctness")
        }
        lines.extend(_markdown_table(
            ["Metric", "Score", "Status/reason"],
            [
                [
                    metric,
                    detail.get("score", case.get("generation_metrics", {}).get(metric)),
                    detail.get("reason") or case.get("metric_reasons", {}).get(metric) or (
                        "scored" if detail.get("score", case.get("generation_metrics", {}).get(metric)) is not None
                        else "not scored"
                    ),
                ]
                for metric, detail in visible_judge_metrics.items()
            ],
        ))
        lines.extend(["", "#### Execution audit: latency and token usage", ""])
        lines.extend(_markdown_table(
            ["Process/call", "Latency (ms)", "Provider / model", "Token usage (status, in/out/total)"],
            _execution_rows(case),
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
