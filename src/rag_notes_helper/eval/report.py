"""Human-readable Markdown rendering for evaluation run reports."""

from html import escape
import json
from pathlib import Path
from typing import Any

from rag_notes_helper.eval.ragas_runner import (
    GENERATION_METRIC_NAMES,
)


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


def _metric_with_coverage(
    score: Any,
    eligible_case_count: int | None,
    case_count: int | None,
) -> Any:
    if eligible_case_count is None or case_count is None:
        return score
    return f"{_markdown_cell(score)} ({eligible_case_count}/{case_count} eligible)"


def _usage_values(usage: dict[str, Any] | None) -> list[Any]:
    if not usage:
        return ["N/A"] * 7

    token_usage = usage.get("token_usage", usage)
    input_tokens = token_usage.get("input_tokens")
    output_tokens = token_usage.get("output_tokens")
    provider_total = token_usage.get("total_tokens")
    calls = usage.get("calls")
    calculated_total = None
    if calls is not None:
        call_usages = [call.get("token_usage", {}) for call in calls]
        if call_usages and all(
            call_usage.get("total_tokens") is not None
            for call_usage in call_usages
        ):
            provider_total = sum(
                call_usage["total_tokens"]
                for call_usage in call_usages
            )
        else:
            provider_total = None
            if call_usages and all(
                call_usage.get("input_tokens") is not None
                and call_usage.get("output_tokens") is not None
                for call_usage in call_usages
            ):
                calculated_total = sum(
                    call_usage["input_tokens"] + call_usage["output_tokens"]
                    for call_usage in call_usages
                )
    elif usage.get("call_count", 1) != 1 and (
        token_usage.get("status") != "available"
    ):
        provider_total = None
    elif provider_total is None and usage.get("call_count", 1) == 1 and (
        input_tokens is not None and output_tokens is not None
    ):
        calculated_total = input_tokens + output_tokens

    character_usage = usage.get(
        "character_usage",
        usage.get("characters", {}),
    ) or {}
    return [
        token_usage.get("status", "unavailable"),
        input_tokens,
        output_tokens,
        provider_total,
        calculated_total,
        character_usage.get("input"),
        character_usage.get("output"),
    ]


def _usage_summary(usage: dict[str, Any]) -> str:
    (
        status,
        input_tokens,
        output_tokens,
        provider_total,
        calculated_total,
        input_characters,
        output_characters,
    ) = _usage_values(usage)
    return (
        f"{status}: input={input_tokens if input_tokens is not None else 'N/A'}, "
        f"output={output_tokens if output_tokens is not None else 'N/A'}, "
        f"provider total={provider_total if provider_total is not None else 'N/A'}, "
        "calculated input+output="
        f"{calculated_total if calculated_total is not None else 'N/A'}, "
        f"input chars={input_characters if input_characters is not None else 'N/A'}, "
        f"output chars={output_characters if output_characters is not None else 'N/A'}"
    )


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
            *(["N/A"] * 7),
        ])
    for index, call in enumerate(
        case.get("generation_usage", {}).get("calls", []),
        start=1,
    ):
        rows.append([
            f"{call.get('call_type', 'generation')} call {index}",
            call.get("latency_ms"),
            f"{call.get('provider', 'N/A')} / {call.get('model', 'N/A')}",
            *_usage_values(call),
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
            *_usage_values(usage),
        ])

    for key, label, usage_key in (
        ("retrieval_ms", "Retrieval latency total", None),
        ("controller_ms", "Controller latency total", None),
        ("answer_generation_ms", "Answer generation latency total", None),
        ("judge_ms", "Judge latency total", "judge_usage"),
        ("rag_query_ms", "RAG query latency", None),
        ("end_to_end_ms", "Evaluation latency", None),
    ):
        if key in latency:
            usage = case.get(usage_key, {}) if usage_key else {}
            rows.append([
                label,
                latency.get(key),
                f"{usage.get('call_count')} calls" if usage_key else "—",
                *(_usage_values(usage) if usage_key else (["—"] * 7)),
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
        ("Retrieval top-k", "top_k"),
        ("Minimum retrieval score", "min_retrieval_score"),
        ("Maximum answer chunks", "max_answer_chunks"),
        ("Agentic maximum searches", "agentic_max_searches"),
        ("Sample size", "sample_count"),
        ("Sampling seed", "seed"),
    )
    config_values = []
    for label, key in config_fields:
        value = metadata.get(key)
        if key == "sample_count" and value is None:
            value = "Full fixture"
        elif key == "seed" and value is None:
            value = "Not set"
        config_values.append([label, value])
    lines.extend(_markdown_table(
        ["Setting", "Value"],
        config_values,
    ))
    category_counts = results.get("category_counts", {})
    lines.extend([
        "",
        "## Fixture coverage",
        "",
        f"- **Cases:** {summary.get('case_count', len(results.get('cases', [])))}",
        f"- **Interpretation:** {_markdown_cell(summary.get('interpretation'))}",
        "- **Selected case IDs:** "
        + _markdown_cell(", ".join(metadata.get("selected_case_ids", []))),
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
    lines.extend(["", "### Ragas metrics and context diagnostics", ""])
    ragas_metric_rows = []
    ragas_eligible_counts = workflow_summary.get(
        "ragas_metric_eligible_case_counts",
        {},
    )
    selected_case_count = summary.get(
        "case_count",
        len(results.get("cases", [])),
    )
    for label, metric, source in (
        ("Faithfulness", "faithfulness", "generation_metrics"),
        ("Answer Relevancy", "answer_relevancy", "generation_metrics"),
        ("Answer Correctness", "answer_correctness", "generation_metrics"),
        (
            "Context Utilization (context/retrieval diagnostic)",
            "context_utilization",
            "retrieval_metrics",
        ),
    ):
        eligible_case_count = ragas_eligible_counts.get(metric)
        ragas_metric_rows.append([
            label,
            workflow_summary.get(source, {}).get(metric),
            (
                f"{eligible_case_count}/{selected_case_count}"
                if eligible_case_count is not None
                else None
            ),
        ])
    lines.extend(_markdown_table(
        ["Ragas metric", "Score", "Eligible cases / selected cases"],
        ragas_metric_rows,
    ))
    lines.extend(["", "### Answerability outcomes", ""])
    answerability_outcomes = workflow_summary.get("answerability_outcomes", {})
    answerability_rows = []
    for group_key, label in (
        ("expected_answerable", "Expected answerable"),
        ("expected_unanswerable", "Expected unanswerable"),
    ):
        outcome = answerability_outcomes.get(group_key, {})
        group_case_count = outcome.get("case_count")
        correct_count = outcome.get("correct_behavior_count")
        answerability_rows.append([
            label,
            group_case_count,
            outcome.get("answered_count"),
            outcome.get("abstained_count"),
            outcome.get("unavailable_or_error_count"),
            (
                f"{correct_count}/{group_case_count}"
                if group_case_count and correct_count is not None
                else None
            ),
        ])
    lines.extend(_markdown_table(
        [
            "Expected answerability", "Cases", "Answered", "Abstained",
            "Unavailable/error", "Correct behavior (count/cases)",
        ],
        answerability_rows,
    ))
    latency_labels = (
        ("retrieval_ms", "Retrieval"),
        ("controller_ms", "Controller"),
        ("answer_generation_ms", "Final answer generation"),
        ("judge_ms", "Ragas judge"),
        ("rag_query_ms", "RAG query latency"),
        ("end_to_end_ms", "Evaluation latency"),
    )
    mean_latency = workflow_summary.get("latency_ms", {})
    lines.extend(["", "### Mean latency (ms)", ""])
    lines.extend(_markdown_table(
        ["Stage", "Latency (ms)"],
        [[label, mean_latency.get(key)] for key, label in latency_labels],
    ))
    generation_usage = workflow_summary.get("generation_usage", {})
    lines.extend(["", "### Provider usage", ""])
    usage_rows = []
    for label, usage in [
        ("Controller and answer", generation_usage),
        ("Judge total", summary.get("judge_usage", {}).get("total", {})),
    ]:
        usage_rows.append([
            label,
            usage.get("call_count"),
            *_usage_values(usage),
        ])
    lines.extend(_markdown_table(
        [
            "Call group", "Calls", "Status", "Provider input tokens",
            "Provider output tokens", "Provider total tokens",
            "Calculated input+output tokens", "Input chars", "Output chars",
        ],
        usage_rows,
    ))
    judge_by_metric = workflow_summary.get("judge_usage_by_metric", {})
    lines.extend(["", "#### Judge usage by metric", ""])
    lines.extend(_markdown_table(
        [
            "Metric", "Calls", "Status", "Provider input tokens",
            "Provider output tokens", "Provider total tokens",
            "Calculated input+output tokens", "Input chars", "Output chars",
        ],
        [
            [metric, judge_by_metric.get(metric, {}).get("call_count", 0),
             *_usage_values(judge_by_metric.get(metric, {}))]
            for metric in (
                "faithfulness",
                "answer_relevancy",
                "answer_correctness",
                "context_utilization",
            )
        ],
    ))

    category_summaries = workflow_summary.get("by_category", {})
    lines.extend(["", "## Metrics by category", ""])
    category_rows = []
    for category, metrics in sorted(category_summaries.items()):
        metric_case_counts = metrics.get("ragas_metric_eligible_case_counts", {})
        category_case_count = metrics.get("case_count")
        category_rows.append([
            category,
            category_case_count,
            metrics.get("retrieval_metrics", {}).get("recall"),
            metrics.get("retrieval_metrics", {}).get("precision"),
            metrics.get("retrieval_metrics", {}).get("ndcg"),
            _metric_with_coverage(
                metrics.get("generation_metrics", {}).get("faithfulness"),
                metric_case_counts.get("faithfulness"),
                category_case_count,
            ),
            _metric_with_coverage(
                metrics.get("generation_metrics", {}).get("answer_relevancy"),
                metric_case_counts.get("answer_relevancy"),
                category_case_count,
            ),
            _metric_with_coverage(
                metrics.get("generation_metrics", {}).get("answer_correctness"),
                metric_case_counts.get("answer_correctness"),
                category_case_count,
            ),
            _metric_with_coverage(
                metrics.get("retrieval_metrics", {}).get("context_utilization"),
                metric_case_counts.get("context_utilization"),
                category_case_count,
            ),
            metrics.get("latency_ms", {}).get("end_to_end_ms"),
            _usage_summary(metrics.get("generation_usage", {})),
            _usage_summary(metrics.get("judge_usage", {})),
        ])
    lines.extend(_markdown_table(
        [
            "Category", "Cases", "Recall", "Precision", "NDCG",
            "Faithfulness (score, eligible/selected)",
            "Answer relevancy (score, eligible/selected)",
            "Answer correctness (score, eligible/selected)",
            "Context Utilization diagnostic (score, eligible/selected)",
            "Mean evaluation latency (ms)", "Controller/answer usage",
            "Judge usage",
        ],
        category_rows,
    ))

    lines.extend([
        "",
        "## Evaluation process and metric guide",
        "",
        "1. The fixture supplies the reference answer, expected answerability, expected source paths, and reviewed evidence spans.",
        "2. Retrieval (1) uses the initial question; Retrieval (2) identifies a controller-requested search, and later IDs identify subsequent controller-requested searches. Every search applies the configured Retrieval top-k as its maximum candidate count, though it may return fewer results. The Search registry records each query and its available stage/method metadata once; ranked result rows refer to that Retrieval (n) ID.",
        "3. Retrieved chunks are candidates, not necessarily model input. The Answer generation context table shows the ordered chunks passed to the final answer call; Agentic RAG may select, reorder, or transform those candidates. Each context block is formatted as `file:<source>\\n<chunk text>` and blocks are separated by a blank line. If the workflow abstains, no final-answer context is sent.",
        "4. Retrieval top-k Recall, Precision, and NDCG score retrieved candidate lists, not the Answer generation context. Recall measures evidence coverage: the share of reviewed evidence lines retrieved. Precision measures the share of retrieved lines that are relevant to the reviewed evidence. NDCG measures whether relevant evidence ranks early. For Agentic RAG, Recall and Precision use the union of that query's search results, while NDCG is averaged over applicable ranked search lists.",
        "5. Context prefix k counts the first k chunks in the exact Answer generation context order. Each row shows cumulative unique evidence-line coverage and Recall, Precision, and context-order NDCG. For interpretation, prefix rows assess evidence available at each context length while one answer is generated from the complete final context; they are not separate generated answers. Affected metrics show N/A when exact source-line provenance is unavailable, including transformed context without a line mapping.",
        "6. Answerability outcomes are counts grouped by the fixture's expected answerability. Errors and unavailable answers do not count as correct behavior; an empty group is shown as N/A.",
        "7. Ragas scores apply only to expected-answerable cases that produced an answer. The eligible-case count is shown beside each score; N/A means no selected case was eligible. Faithfulness checks whether answer claims are supported by the answer context; Answer Relevancy checks whether the response addresses the question; Answer Correctness compares the response with the reference answer. Answer Relevancy and Answer Correctness also use the configured local evaluation embeddings. These judge scores are diagnostics, not pass/fail guarantees.",
        "8. Context Utilization is a context/retrieval diagnostic, not an answer-quality metric. It is computed only when the case has retrieved context; its judge-call latency and provider-reported token usage remain visible in the query execution audit.",
        "9. Per-call latency is shown in milliseconds. RAG query latency runs from query processing through the terminal answer or abstention. Evaluation latency adds all Ragas judge latency to RAG query latency. Token fields are provider-reported. A calculated input+output total appears only when every included call has both counts and provider totals are missing. Partial or unavailable status stays visible. Recorded character counts are shown separately and never converted to tokens; retrieval and local embedding work are not provider token usage.",
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
            "- **Artifacts:** [results.json](results.json) · [run.log](run.log)",
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
        lines.extend([
            f"- **Expected sources:** {_markdown_cell(expected_sources)}",
            "",
        ])
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
        retrieval_metrics = case.get("retrieval_metrics", {})
        lines.extend(_markdown_table(
            ["Retrieval top-k", "Recall", "Precision", "NDCG"],
            [[
                metadata.get("top_k"),
                retrieval_metrics.get("recall"),
                retrieval_metrics.get("precision"),
                retrieval_metrics.get("ndcg"),
            ]],
        ))
        lines.extend(["", "#### Search registry", ""])
        search_results = case.get("search_results", [])
        if not isinstance(search_results, list):
            search_results = []
        search_queries = case.get("search_queries", [])
        if not isinstance(search_queries, list):
            search_queries = []
        search_registry_rows = []
        ranked_result_rows = []
        for search_number, search_hits in enumerate(search_results, start=1):
            if not isinstance(search_hits, list):
                search_hits = []
            search_id = f"Retrieval ({search_number})"
            search_query = (
                search_queries[search_number - 1]
                if search_number <= len(search_queries)
                else case.get("question")
            )
            stage_values = []
            method_values = []
            for hit in search_hits:
                stage_value = hit.get("stage", hit.get("retrieval_stage"))
                if stage_value is not None and stage_value not in stage_values:
                    stage_values.append(stage_value)
                method_value = hit.get("method", hit.get("retrieval_method"))
                if method_value is not None and method_value not in method_values:
                    method_values.append(method_value)
            stage = (
                stage_values[0]
                if len(stage_values) == 1
                else stage_values or None
            )
            method = (
                method_values[0]
                if len(method_values) == 1
                else method_values or None
            )
            search_registry_rows.append([search_id, search_query, stage, method])
            if not search_hits:
                ranked_result_rows.append([
                    search_id,
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "No chunk returned",
                ])
                continue
            for rank, hit in enumerate(search_hits, start=1):
                ranked_result_rows.append([
                    search_id,
                    rank,
                    hit.get("chunk_id"),
                    hit.get("source"),
                    f"{hit.get('start_line', 'N/A')}–{hit.get('end_line', 'N/A')}",
                    hit.get("score"),
                    hit.get("text"),
                ])
        lines.extend(_markdown_table(
            ["Search ID", "Search query", "Stage", "Method"],
            search_registry_rows,
        ))
        lines.extend(["", "#### Ranked retrieval results", ""])
        lines.extend(_markdown_table(
            ["Search ID", "Rank", "Chunk ID", "Source path", "Lines", "Score", "Retrieved chunk text"],
            ranked_result_rows,
        ))
        lines.extend([
            "",
            "Rank is local to its Retrieval search. Similarity scores rank candidates; they are not calibrated probabilities.",
            "",
        ])
        lines.extend(["", "#### Context passed to answer generation", ""])
        generation_context = case.get("generation_context", [])
        search_rank_by_reference = {}
        for search_number, search_hits in enumerate(case.get("search_results", []), start=1):
            for rank, hit in enumerate(search_hits, start=1):
                search_rank_by_reference.setdefault(
                    (hit.get("source"), hit.get("chunk_id")),
                    (search_number, rank, hit),
                )
        context_rows = []
        for order, item in enumerate(generation_context, start=1):
            search_rank = search_rank_by_reference.get(
                (item.get("source"), item.get("chunk_id"))
            )
            if search_rank:
                is_unchanged = (
                    item.get("content_changed") is False
                    if "content_changed" in item
                    else (
                        item.get("text") is None
                        or item.get("text") == search_rank[2].get("text")
                    )
                )
                content = (
                    f"Same as Retrieval ({search_rank[0]}), rank {search_rank[1]}"
                    if is_unchanged
                    else item.get("prompt_block", item.get("text"))
                )
            else:
                content = item.get("prompt_block", item.get("text"))
            context_rows.append([
                order,
                item.get("chunk_id"),
                item.get("source"),
                f"{item.get('start_line', 'N/A')}–{item.get('end_line', 'N/A')}",
                content,
            ])
        lines.extend(_markdown_table(
            ["Final context order", "Chunk ID", "Source path", "Lines", "Context content"],
            context_rows,
        ))
        if not generation_context:
            lines.append("No final-answer context was sent.")
        context_prefix_rows = []
        for prefix in case.get("context_prefix_metrics", []):
            covered = prefix.get("evidence_lines_covered")
            total_evidence_lines = prefix.get("ground_truth_line_count")
            recall = prefix.get("recall")
            if covered is None or total_evidence_lines is None:
                evidence_coverage = "N/A"
            else:
                recall_label = "N/A" if recall is None else f"{recall:.4f}"
                evidence_coverage = (
                    f"{covered}/{total_evidence_lines} ({recall_label})"
                )
            context_prefix_rows.append([
                prefix.get("context_prefix_k"),
                evidence_coverage,
                prefix.get("precision"),
                prefix.get("ndcg"),
            ])
        lines.extend(["", "#### Context-prefix metrics", ""])
        lines.extend(_markdown_table(
            ["Context prefix k", "Unique evidence lines / Recall", "Precision", "Context-order NDCG"],
            context_prefix_rows,
        ))
        lines.extend(["", "#### Ragas judge diagnostics", ""])
        judge_metrics = case.get("judge_metrics", {})
        visible_judge_metrics = (
            ("Faithfulness", "faithfulness"),
            ("Answer Relevancy", "answer_relevancy"),
            ("Answer Correctness", "answer_correctness"),
            ("Context Utilization (context/retrieval diagnostic)", "context_utilization"),
        )
        judge_rows = []
        for label, metric in visible_judge_metrics:
            detail = judge_metrics.get(metric, {})
            metric_scores = (
                case.get("generation_metrics", {})
                if metric in GENERATION_METRIC_NAMES
                else case.get("retrieval_metrics", {})
            )
            score = detail.get("score", metric_scores.get(metric))
            reason = detail.get("reason") or case.get(
                "metric_reasons", {}
            ).get(metric)
            if reason is None:
                if case.get("ragas_error"):
                    reason = f"Ragas evaluation failed ({case['ragas_error']})."
                elif score is not None:
                    reason = "Scored."
                elif case.get("expected_answerable") is not True:
                    reason = (
                        "Skipped because expected answerability is unanswerable."
                    )
                elif not case.get("answer"):
                    reason = "Skipped because no generated answer was available."
                else:
                    reason = "Not scored: no eligible result was returned."
            judge_rows.append([label, score, reason])
        lines.extend(_markdown_table(
            ["Metric", "Score", "Status/reason"],
            judge_rows,
        ))
        lines.extend(["", "#### Execution audit: latency and token usage", ""])
        lines.extend(_markdown_table(
            [
                "Process/call", "Latency (ms)", "Provider / model", "Usage status",
                "Provider input tokens", "Provider output tokens",
                "Provider total tokens", "Calculated input+output tokens",
                "Input chars", "Output chars",
            ],
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
