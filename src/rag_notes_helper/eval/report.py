"""Human-readable Markdown rendering for fixture evaluation reports."""

from html import escape
import json
from pathlib import Path
from typing import Any

from .retrieval_metrics import calculate_context_prefix_metrics


RAGAS_METRICS = (
    ("Faithfulness", "faithfulness", "generation_metrics"),
    ("Answer Relevancy", "answer_relevancy", "generation_metrics"),
    ("Answer Correctness", "answer_correctness", "generation_metrics"),
    ("Context Utilization", "context_utilization", "retrieval_metrics"),
)


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return escape(str(value)).replace("|", "&#124;").replace("\r\n", "<br>").replace("\n", "<br>")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(_markdown_cell(cell) for cell in row) + " |" for row in rows],
    ]


def _total_tokens(usage: dict[str, Any] | None) -> Any:
    """Use a provider total, else exact arithmetic on reported token fields."""
    if not usage:
        return None
    token_usage = usage.get("token_usage", usage) or {}
    calls = usage.get("calls")
    if calls:
        totals = [_total_tokens(call) for call in calls]
        return sum(totals) if all(total is not None for total in totals) else None
    if token_usage.get("total_tokens") is not None:
        return token_usage["total_tokens"]
    if token_usage.get("input_tokens") is not None and token_usage.get("output_tokens") is not None:
        return token_usage["input_tokens"] + token_usage["output_tokens"]
    return None


def _usage_columns(usage: dict[str, Any] | None) -> list[Any]:
    token_usage = (usage or {}).get("token_usage", usage or {}) or {}
    return [token_usage.get("input_tokens"), token_usage.get("output_tokens"), _total_tokens(usage)]


def _execution_rows(case: dict[str, Any]) -> list[list[Any]]:
    latency = case.get("latency", {}) or {}
    rows: list[list[Any]] = []
    for index, call in enumerate(latency.get("retrieval_calls", []), start=1):
        rows.append([f"Retrieval search {call.get('search_index', index)}", call.get("latency_ms"), "—", "—", "—", "—"])
    for index, call in enumerate((case.get("generation_usage", {}) or {}).get("calls", []), start=1):
        rows.append([f"{call.get('call_type', 'generation')} call {index}", call.get("latency_ms"), f"{call.get('provider', '—')} / {call.get('model', '—')}", *_usage_columns(call)])
    for metric, usage in (case.get("judge_usage_by_metric", {}) or {}).items():
        calls = usage.get("calls", [])
        metric_latency = (latency.get("judge_metrics_ms", {}) or {}).get(metric)
        if calls or metric_latency is not None:
            providers = sorted({f"{call.get('provider', '—')} / {call.get('model', '—')}" for call in calls})
            rows.append([f"Judge: {metric}", metric_latency, ", ".join(providers) or "—", *_usage_columns(usage)])
    for key, label in (("retrieval_ms", "Retrieval latency total"), ("controller_ms", "Controller latency total"), ("answer_generation_ms", "Answer generation latency total"), ("judge_ms", "Judge latency total"), ("rag_query_ms", "RAG query latency"), ("end_to_end_ms", "Evaluation latency")):
        if key in latency:
            rows.append([label, latency[key], "—", "—", "—", "—"])
    return rows


def _candidate_index(
    case: dict[str, Any],
) -> dict[tuple[Any, Any], dict[str, Any]]:
    candidates: dict[tuple[Any, Any], dict[str, Any]] = {}
    search_results = case.get("search_results", [])
    for search_hits in search_results if isinstance(search_results, list) else []:
        if not isinstance(search_hits, list):
            continue
        for hit in search_hits:
            if (
                isinstance(hit, dict)
                and hit.get("source") is not None
                and hit.get("chunk_id") is not None
            ):
                candidates[(hit["source"], hit["chunk_id"])] = hit
    retrieval_stages = case.get("retrieval_stages", [])
    for search in retrieval_stages if isinstance(retrieval_stages, list) else []:
        if not isinstance(search, dict):
            continue
        for stage in (search.get("stages", {}) or {}).values():
            if not isinstance(stage, dict):
                continue
            for hit in (stage.get("results", []) or []):
                if (
                    isinstance(hit, dict)
                    and hit.get("source") is not None
                    and hit.get("chunk_id") is not None
                ):
                    candidates[(hit["source"], hit["chunk_id"])] = hit
    return candidates


_RETRIEVAL_STAGE_LABELS = (
    ("vector", "Vector", "Vector results"),
    ("bm25", "BM25", "BM25 results"),
    ("rrf", "Fusion", "RRF results"),
    ("reranked", "Rerank", "Reranked results"),
)


def _retrieval_searches(case: dict[str, Any]) -> list[dict[str, Any]]:
    raw_searches = case.get("retrieval_stages", [])
    searches = (
        [search for search in raw_searches if isinstance(search, dict)]
        if isinstance(raw_searches, list)
        else []
    )
    if searches:
        return searches

    raw_results = case.get("search_results", [])
    raw_results = raw_results if isinstance(raw_results, list) else []
    queries = case.get("search_queries", [])
    queries = queries if isinstance(queries, list) else []
    search_count = max(len(raw_results), len(queries))
    return [
        {
            "search_id": search_id,
            "search_query": (
                queries[search_id - 1]
                if search_id <= len(queries)
                else case.get("question")
            ),
            "stages": {},
        }
        for search_id in range(1, search_count + 1)
    ]


def _evidence_lines(hit: dict[str, Any], case: dict[str, Any]) -> str | None:
    coverage = calculate_context_prefix_metrics(
        expected_evidence=case.get("expected_evidence", []) or [],
        context_chunks=[hit],
    )[0]
    total = coverage.get("ground_truth_line_count")
    covered = coverage.get("evidence_lines_covered")
    if not isinstance(total, int) or total == 0 or not isinstance(covered, int):
        return None
    return f"{covered}/{total}"


def _stage_cutoffs(case: dict[str, Any], stage_name: str) -> list[int]:
    cutoffs: list[int] = []
    for search in _retrieval_searches(case):
        stage = (search.get("stages", {}) or {}).get(stage_name)
        top_k = None
        if isinstance(stage, dict):
            top_k = stage.get("metric_top_k", stage.get("top_k"))
        if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k not in cutoffs:
            cutoffs.append(top_k)
    if cutoffs:
        return cutoffs

    metrics = (case.get("retrieval_stage_metrics", {}) or {}).get(stage_name, {}) or {}
    for cutoff in metrics.get("actual_cutoffs", []):
        top_k = cutoff.get("top_k") if isinstance(cutoff, dict) else None
        if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k not in cutoffs:
            cutoffs.append(top_k)
    return cutoffs


def _generation_cutoff(case: dict[str, Any], max_answer_chunks: Any) -> int | None:
    if type(max_answer_chunks) is int and max_answer_chunks >= 0:
        return max_answer_chunks
    for search in _retrieval_searches(case):
        cutoff = search.get("generation_top_k")
        if type(cutoff) is int and cutoff >= 0:
            return cutoff
    return None


def _quality_metric_rows(
    case: dict[str, Any],
    max_answer_chunks: Any,
) -> list[list[Any]]:
    stage_metrics = case.get("retrieval_stage_metrics", {}) or {}
    rows = []
    for stage_name, label, _ in _RETRIEVAL_STAGE_LABELS:
        metrics = stage_metrics.get(stage_name, {}) or {}
        cutoffs = _stage_cutoffs(case, stage_name)
        if not metrics and not cutoffs:
            for search in _retrieval_searches(case):
                stage = (search.get("stages", {}) or {}).get(stage_name)
                if isinstance(stage, dict):
                    metrics = stage.get("metrics", {}) or {}
                    break
        if not cutoffs and not any(
            metrics.get(metric) is not None
            for metric in ("recall", "precision", "ndcg")
        ):
            continue
        cutoff_label = ",".join(str(value) for value in cutoffs) if cutoffs else "—"
        rows.append([
            f"{label}@{cutoff_label}",
            metrics.get("recall"),
            metrics.get("precision"),
            metrics.get("ndcg"),
        ])
    generation_context = case.get("generation_context")
    if isinstance(generation_context, list):
        metrics = case.get("final_context_metrics", {}) or {}
        cutoff = _generation_cutoff(case, max_answer_chunks)
        rows.append([
            f"Generation@{cutoff if cutoff is not None else '—'}",
            metrics.get("recall"),
            metrics.get("precision"),
            metrics.get("ndcg"),
        ])
    return rows


def _render_quality_metrics(
    case: dict[str, Any],
    max_answer_chunks: Any,
) -> list[str]:
    return [
        "**Retrieval stages and generation-context metrics**",
        "",
        *_table(
            ["Stage", "Recall", "Precision", "NDCG"],
            _quality_metric_rows(case, max_answer_chunks),
        ),
    ]


def _render_search_queries(case: dict[str, Any]) -> list[str]:
    searches = _retrieval_searches(case)
    lines = ["#### Search query", ""]
    if searches:
        rows = [[search.get("search_id"), search.get("search_query")] for search in searches]
        lines.extend(_table(["Search", "Query"], rows))
    else:
        lines.append("No searches were recorded.")
    lines.append("")
    return lines


def _render_retrieval_stage(
    case: dict[str, Any],
    *,
    stage_name: str,
    heading: str,
) -> list[str]:
    rows = []
    for search in _retrieval_searches(case):
        stage = (search.get("stages", {}) or {}).get(stage_name)
        if isinstance(stage, dict):
            hits = stage.get("results", []) or []
        elif stage_name == "vector" and case.get("workflow") != "agentic":
            raw_search_results = case.get("search_results", [])
            search_id = search.get("search_id")
            if (
                isinstance(raw_search_results, list)
                and isinstance(search_id, int)
                and 1 <= search_id <= len(raw_search_results)
                and isinstance(raw_search_results[search_id - 1], list)
            ):
                hits = raw_search_results[search_id - 1]
            else:
                hits = []
        else:
            hits = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            score = hit.get("stage_score")
            if score is None:
                score = hit.get("score")
            rows.append([
                hit.get("rank"),
                score,
                hit.get("chunk_id"),
                hit.get("source"),
                f"{hit.get('start_line', '—')}–{hit.get('end_line', '—')}",
                search.get("search_id", "—"),
                _evidence_lines(hit, case),
            ])
    lines = [f"#### {heading}", ""]
    if rows:
        lines.extend(_table(
            ["Rank", "Stage score", "Chunk ID", "Source path", "Lines", "Search", "Evidence lines"],
            rows,
        ))
    else:
        lines.append("No candidates were returned.")
    lines.append("")
    return lines


def _render_expected_evidence(case: dict[str, Any]) -> list[str]:
    evidence_rows = [
        [
            item.get("source"),
            f"{item.get('start_line', '—')}–{item.get('end_line', '—')}",
            item.get("text"),
        ]
        for item in (case.get("expected_evidence", []) or [])
    ]
    return [
        "#### Expected evidence",
        "",
        *_table(["Source path", "Lines", "Reviewed evidence text"], evidence_rows),
        "",
    ]


def _render_generation_context(case: dict[str, Any]) -> list[str]:
    candidates = _candidate_index(case)
    rows = []
    for order, item in enumerate(case.get("generation_context", []) or [], start=1):
        candidate = candidates.get((item.get("source"), item.get("chunk_id")), {})
        if item.get("content_changed") is True:
            context_content = item.get("prompt_block", item.get("text"))
        else:
            context_content = item.get("text") or candidate.get("text")
            if context_content is None:
                context_content = item.get("prompt_block")
        rows.append([
            order,
            item.get("chunk_id"),
            item.get("source"),
            f"{item.get('start_line', '—')}–{item.get('end_line', '—')}",
            context_content,
            _evidence_lines(item, case),
        ])
    lines = ["#### Final generation context", "", "Context passed to answer generation.", ""]
    if rows:
        lines.extend(_table(
            ["Context order", "Chunk ID", "Source path", "Lines", "Full context", "Evidence lines"],
            rows,
        ))
    else:
        lines.append("No final-answer context was sent.")
    lines.append("")
    return lines


def _render_ragas_diagnostics(case: dict[str, Any]) -> list[str]:
    return [
        "",
        "**Ragas diagnostics**",
        "",
        *_table(
            ["Metric", "Score"],
            [[label, (case.get(source, {}) or {}).get(metric)] for label, metric, source in RAGAS_METRICS],
        ),
    ]


def _render_execution_audit(case: dict[str, Any]) -> list[str]:
    return [
        "",
        "#### Execution audit: latency and token usage",
        "",
        *_table(
            ["Process/call", "Latency (ms)", "Provider / model", "Provider input tokens", "Provider output tokens", "Total tokens"],
            _execution_rows(case),
        ),
        "",
    ]


def _query_header(case: dict[str, Any], max_answer_chunks: Any) -> list[str]:
    return [
        f"### {case.get('case_id', 'unknown')}",
        "",
        f"- **Category:** `{case.get('category', 'unknown')}`",
        "",
        "#### Question",
        "",
        _markdown_cell(case.get("question")),
        "",
        "#### Reference answer",
        "",
        _markdown_cell(case.get("reference_answer")),
        "",
        "#### Generated answer",
        "",
        _markdown_cell(case.get("answer")),
        "",
        "#### Quality metrics",
        "",
        *_render_quality_metrics(case, max_answer_chunks),
        *_render_ragas_diagnostics(case),
    ]


def _render_classic_query_details(
    case: dict[str, Any],
    max_answer_chunks: Any,
) -> list[str]:
    lines = _query_header(case, max_answer_chunks)
    latency = case.get("latency", {}) or {}
    stage_latency = latency.get("retrieval_stages_ms", {}) or {}
    latency_rows = [
        [label, stage_latency.get(key)]
        for key, label in (
            ("vector", "Vector search"),
            ("bm25", "BM25 search"),
            ("rrf", "Fusion"),
            ("reranked", "Reranking"),
        )
    ]
    latency_rows.extend([
        ["Answer generation", latency.get("answer_generation_ms")],
        ["Complete query", latency.get("rag_query_ms")],
    ])
    lines.extend(["", "#### Latency", ""])
    lines.extend(_table(["Stage", "Measured latency (ms)"], latency_rows))
    lines.extend([""])
    lines.extend(_render_expected_evidence(case))
    lines.extend(_render_search_queries(case))
    for stage_name, _, heading in _RETRIEVAL_STAGE_LABELS:
        lines.extend(_render_retrieval_stage(case, stage_name=stage_name, heading=heading))
    lines.extend(_render_generation_context(case))
    lines.extend(_render_execution_audit(case))
    return lines


def _render_agentic_query_details(
    case: dict[str, Any],
    max_answer_chunks: Any,
) -> list[str]:
    lines = _query_header(case, max_answer_chunks)
    latency = case.get("latency", {}) or {}
    lines.extend(["", "#### Latency", ""])
    per_search_rows = [
        [f"Search {call.get('search_index', index)}", "Retrieval search", call.get("latency_ms")]
        for index, call in enumerate(latency.get("retrieval_calls", []), start=1)
    ]
    for search in _retrieval_searches(case):
        for stage_name, label, _ in _RETRIEVAL_STAGE_LABELS:
            stage = (search.get("stages", {}) or {}).get(stage_name)
            if isinstance(stage, dict):
                per_search_rows.append([
                    f"Search {search.get('search_id', '—')}",
                    label,
                    stage.get("latency_ms"),
                ])
    lines.extend(_table(["Search", "Stage", "Measured latency (ms)"], per_search_rows))
    lines.extend(["", "**Measured totals**", ""])
    lines.extend(_table(
        ["Stage", "Latency (ms)"],
        [
            ["Retrieval searches", latency.get("retrieval_ms")],
            ["Controller latency", latency.get("controller_ms")],
            ["Answer generation", latency.get("answer_generation_ms")],
            ["Judge latency", latency.get("judge_ms")],
            ["RAG query latency", latency.get("rag_query_ms")],
            ["End-to-end evaluation latency", latency.get("end_to_end_ms")],
        ],
    ))
    lines.extend([""])
    lines.extend(_render_expected_evidence(case))
    lines.extend(_render_search_queries(case))
    for stage_name, _, heading in _RETRIEVAL_STAGE_LABELS:
        lines.extend(_render_retrieval_stage(case, stage_name=stage_name, heading=heading))
    lines.extend(_render_generation_context(case))
    lines.extend(_render_execution_audit(case))
    return lines


def _render_query_details(
    cases: list[dict[str, Any]],
    max_answer_chunks: Any,
) -> list[str]:
    """Render the per-query audit without mixing it into aggregate sections."""
    lines: list[str] = []
    for case in cases:
        if case.get("workflow") == "classic":
            lines.extend(_render_classic_query_details(case, max_answer_chunks))
            continue
        if case.get("workflow") == "agentic":
            lines.extend(_render_agentic_query_details(case, max_answer_chunks))
            continue
        lines.extend(_render_classic_query_details(case, max_answer_chunks))
    return lines


def _readme_report(results: dict[str, Any]) -> str:
    metadata = results.get("run_metadata", {}) or {}
    summary = results.get("summary", {}) or {}
    workflow = metadata.get("workflow", "unknown")
    workflow_summary = (summary.get("workflows", {}) or {}).get(workflow, {}) or {}
    lines = ["# RAG Evaluation Report", "", f"- **Status:** `{'failed' if results.get('error') else 'completed'}`", f"- **Workflow:** `{workflow}`", f"- **Run timestamp (Asia/Taipei):** `{metadata.get('timestamp_local', 'unknown')}`", "- **Artifacts:** [results.json](results.json) · [run.log](run.log)", "", "## Configuration", ""]
    lines.extend(_table(["Setting", "Value"], [[label, value] for label, value in (
        ("Answer provider", metadata.get("answer_provider")), ("Answer model", metadata.get("answer_model")),
        ("Judge provider", metadata.get("judge_provider")), ("Judge model", metadata.get("judge_model")),
        ("Embedding model", metadata.get("embedding_model")), ("Retrieval top-k", metadata.get("top_k")),
        ("Cross-encoder model", metadata.get("cross_encoder_model")),
        ("Maximum answer chunks", metadata.get("max_answer_chunks")),
        ("Sample size", metadata.get("sample_count") if metadata.get("sample_count") is not None else "Full fixture"),
        ("Sampling seed", metadata.get("seed") if metadata.get("seed") is not None else "Not set"),
    )]))
    lines.extend(["", "## Fixture coverage", "", f"- **Cases:** {summary.get('case_count', len(results.get('cases', [])))}", "- **Selected case IDs:** " + _markdown_cell(", ".join(metadata.get("selected_case_ids", []))), ""])
    lines.extend(["", "## Overall metrics", "", "### Final context evidence metrics", ""])
    lines.extend(_table(["Metric", "Score"], [[name, (workflow_summary.get("final_context_metrics", {}) or {}).get(name)] for name in ("recall", "precision", "ndcg")]))
    lines.extend(["", "### Ragas diagnostics", ""])
    lines.extend(_table(["Metric", "Score"], [[label, (workflow_summary.get(source, {}) or {}).get(metric)] for label, metric, source in RAGAS_METRICS]))
    lines.extend(["", "### Mean latency", ""])
    latency = workflow_summary.get("latency_ms", {}) or {}
    lines.extend(_table(["Stage", "Latency (ms)"], [[label, latency.get(key)] for key, label in (("retrieval_ms", "Retrieval"), ("controller_ms", "Controller"), ("answer_generation_ms", "Final answer generation"), ("judge_ms", "Ragas judge"), ("rag_query_ms", "RAG query latency"), ("end_to_end_ms", "Evaluation latency"))]))
    lines.extend(["", "### Judge usage by metric", ""])
    judge_usage = workflow_summary.get("judge_usage_by_metric", {}) or {}
    lines.extend(_table(["Metric", "Calls", "Input tokens", "Output tokens", "Total tokens"], [[metric, (judge_usage.get(metric, {}) or {}).get("call_count", 0), *_usage_columns(judge_usage.get(metric, {}))] for _, metric, _ in RAGAS_METRICS]))
    lines.extend(["", "## Metrics by category", ""])
    category_rows = []
    for category, metrics in sorted((workflow_summary.get("by_category", {}) or {}).items()):
        final = metrics.get("final_context_metrics", {}) or {}
        category_rows.append([category, metrics.get("case_count"), final.get("recall"), final.get("precision"), final.get("ndcg"), *[(metrics.get(source, {}) or {}).get(metric) for _, metric, source in RAGAS_METRICS], (metrics.get("latency_ms", {}) or {}).get("end_to_end_ms")])
    lines.extend(_table(["Category", "Cases", "Recall", "Precision", "NDCG", "Faithfulness", "Answer Relevancy", "Answer Correctness", "Context Utilization", "Mean evaluation latency (ms)"], category_rows))
    lines.extend(["", "## Evaluation process and metric guide", "", "1. **Final context evidence metrics** are line-grounded Recall, Precision, and Context-order NDCG for the complete ordered Answer generation context. Overall and category values average each completed query's final Context prefix row.", "2. Retrieved chunks are audit candidates. Retrieval top-k is the maximum candidates for one search; it does not mean every candidate reached answer generation.", "3. **Search ID** is the sequential Retrieval search number. Search ranks are local to their Search ID.", "4. **Faithfulness** checks whether the answer is supported by the final Answer generation context. **Answer Relevancy** checks whether the answer addresses the question. **Answer Correctness** checks the answer against the expected answer. **Context Utilization** checks whether the final Answer generation context is useful for the answer. A dash means the score was unavailable.", "5. RAG query latency ends at the terminal answer or abstention. Evaluation latency adds Ragas judge latency. Token totals use a provider-reported total when present; otherwise they are the exact reported input-plus-output sum, or a dash when neither is available.", "", "## Query details", ""])
    report_cases = []
    for case in results.get("cases", []):
        if "workflow" not in case and workflow == "agentic":
            case = {**case, "workflow": workflow}
        report_cases.append(case)
    lines.extend(_render_query_details(
        report_cases,
        metadata.get("max_answer_chunks"),
    ))
    lines.extend(["## Artifacts", "", "- [Machine-readable results](results.json)", "- [Evaluation process log](run.log)", ""])
    return "\n".join(lines)


def write_readme(path: Path, results: dict[str, Any]) -> None:
    """Write the report at the renderer's public artifact seam."""
    path.write_text(_readme_report(results), encoding="utf-8")
