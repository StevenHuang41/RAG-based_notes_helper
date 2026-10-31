"""Human-readable Markdown rendering for fixture evaluation reports."""

from html import escape
import json
from pathlib import Path
from typing import Any


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


def _context_content(item: dict[str, Any], reference: tuple[int, int, dict[str, Any]] | None) -> Any:
    if reference is None:
        return item.get("prompt_block", item.get("text"))
    search_id, rank, hit = reference
    unchanged = item.get("content_changed") is False if "content_changed" in item else (item.get("text") is None or item.get("text") == hit.get("text"))
    return f"Same as Search {search_id}, rank {rank}" if unchanged else item.get("prompt_block", item.get("text"))


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


def _classic_candidate_index(
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


def _render_classic_stage(
    case: dict[str, Any],
    *,
    stage_name: str,
    heading: str,
) -> list[str]:
    lines = ["", f"#### {heading}", ""]
    searches = case.get("retrieval_stages", [])
    if not isinstance(searches, list):
        searches = []
    if not searches:
        lines.extend(["No stage records are available.", ""])
        return lines
    for search in searches:
        stage = (search.get("stages", {}) or {}).get(stage_name)
        if not isinstance(stage, dict):
            continue
        search_id = search.get("search_id", "—")
        lines.append(
            f"_Search {search_id}: configured top-k {_markdown_cell(stage.get('top_k'))}; "
            f"returned {_markdown_cell(stage.get('returned_count'))}._"
        )
        query = search.get("search_query")
        if query is not None:
            lines.append(f"_Search query: {_markdown_cell(query)}_")
        lines.append("")
        rows = [
            [
                hit.get("rank"),
                hit.get("score"),
                hit.get("chunk_id"),
                hit.get("source"),
                f"{hit.get('start_line', '—')}–{hit.get('end_line', '—')}",
            ]
            for hit in stage.get("results", [])
            if isinstance(hit, dict)
        ]
        if rows:
            lines.extend(_table(
                ["Rank", "Stage score", "Chunk ID", "Source path", "Lines"],
                rows,
            ))
        else:
            lines.append("No candidates were returned.")
        metrics = stage.get("metrics", {}) or {}
        lines.extend(["", "Evidence metrics at this stage cutoff:", ""])
        lines.extend(_table(
            ["Recall", "Precision", "NDCG"],
            [[metrics.get("recall"), metrics.get("precision"), metrics.get("ndcg")]],
        ))
        lines.append("")
    return lines


def _render_classic_query_details(case: dict[str, Any]) -> list[str]:
    lines = [
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
        "**Final generation-context evidence**",
        "",
    ]
    final_metrics = case.get("final_context_metrics", {}) or {}
    lines.extend(_table(
        ["Metric", "Score"],
        [[name, final_metrics.get(key)] for key, name in (
            ("recall", "Recall"),
            ("precision", "Precision"),
            ("ndcg", "Context-order NDCG"),
        )],
    ))
    lines.extend(["", "**Ragas diagnostics**", ""])
    lines.extend(_table(
        ["Metric", "Score"],
        [[label, (case.get(source, {}) or {}).get(metric)] for label, metric, source in RAGAS_METRICS],
    ))
    lines.extend(["", "#### Latency", ""])
    latency = case.get("latency", {}) or {}
    stage_latency = latency.get("retrieval_stages_ms", {}) or {}
    latency_rows = [
        [label, stage_latency.get(key)]
        for key, label in (
            ("vector", "Vector search"),
            ("bm25", "BM25 search"),
            ("rrf", "RRF fusion"),
            ("reranked", "Reranking"),
        )
    ]
    latency_rows.extend([
        ["Answer generation", latency.get("answer_generation_ms")],
        ["Complete query", latency.get("rag_query_ms")],
    ])
    lines.extend(_table(
        ["Stage", "Measured latency (ms)"],
        latency_rows,
    ))
    for stage_name, heading in (
        ("vector", "Vector results"),
        ("bm25", "BM25 results"),
        ("rrf", "RRF results"),
        ("reranked", "Reranked results"),
    ):
        lines.extend(_render_classic_stage(
            case,
            stage_name=stage_name,
            heading=heading,
        ))
    comparisons = (case.get("rerank_comparison", {}) or {}).get("searches", [])
    if comparisons:
        lines.extend(["Pre-rerank and post-rerank comparison:", ""])
        comparison_rows = [
            [
                comparison.get("search_id"),
                comparison.get("candidate_pool_matches"),
                comparison.get("generation_top_k"),
                comparison.get("common_top_k"),
                (comparison.get("fused_metrics", {}) or {}).get("recall"),
                (comparison.get("fused_metrics", {}) or {}).get("precision"),
                (comparison.get("fused_metrics", {}) or {}).get("ndcg"),
                (comparison.get("reranked_metrics", {}) or {}).get("recall"),
                (comparison.get("reranked_metrics", {}) or {}).get("precision"),
                (comparison.get("reranked_metrics", {}) or {}).get("ndcg"),
            ]
            for comparison in comparisons
        ]
        lines.extend(_table(
            [
                "Search ID", "Same candidate pool", "Generation top-k",
                "Common cutoff", "RRF recall", "RRF precision", "RRF NDCG",
                "Reranked recall", "Reranked precision", "Reranked NDCG",
            ],
            comparison_rows,
        ))
        lines.append("")

    lines.extend(["#### Final generation context", "", "Context passed to answer generation.", ""])
    candidates = _classic_candidate_index(case)
    prefixes = {
        item.get("context_prefix_k"): item
        for item in case.get("context_prefix_metrics", [])
    }
    context_rows = []
    for order, item in enumerate(case.get("generation_context", []), start=1):
        prefix = prefixes.get(order, {})
        covered = prefix.get("evidence_lines_covered")
        total = prefix.get("ground_truth_line_count")
        candidate = candidates.get((item.get("source"), item.get("chunk_id")), {})
        if item.get("content_changed") is True:
            context_content = item.get("prompt_block", item.get("text"))
        else:
            context_content = item.get("text") or candidate.get("text")
            if context_content is None:
                context_content = item.get("prompt_block")
        context_rows.append([
            order,
            item.get("chunk_id"),
            item.get("source"),
            f"{item.get('start_line', '—')}–{item.get('end_line', '—')}",
            item.get("reranker_score", candidate.get("reranker_score")),
            context_content,
            f"{covered}/{total}" if covered is not None and total is not None else None,
            prefix.get("recall"),
            prefix.get("precision"),
            prefix.get("ndcg"),
        ])
    if context_rows:
        lines.extend(_table(
            [
                "Context order", "Chunk ID", "Source path", "Lines",
                "Reranker score", "Full context", "Evidence lines",
                "Recall", "Precision", "Context-order NDCG",
            ],
            context_rows,
        ))
    else:
        lines.extend(["No final-answer context was sent.", ""])
    lines.extend([
        "",
        "Retrieved chunk text appears only in the final generation-context table.",
        "",
        "#### Expected evidence",
        "",
        f"- **Expected sources:** {_markdown_cell(case.get('expected_sources', []))}",
        "",
    ])
    evidence_rows = [
        [
            item.get("source"),
            f"{item.get('start_line', '—')}–{item.get('end_line', '—')}",
            item.get("text"),
        ]
        for item in case.get("expected_evidence", [])
    ]
    lines.extend(_table(["Source path", "Lines", "Reviewed evidence text"], evidence_rows))
    lines.extend(["", "#### Execution audit: latency and token usage", ""])
    lines.extend(_table(
        ["Process/call", "Latency (ms)", "Provider / model", "Provider input tokens", "Provider output tokens", "Total tokens"],
        _execution_rows(case),
    ))
    lines.append("")
    return lines


def _render_query_details(cases: list[dict[str, Any]]) -> list[str]:
    """Render the per-query audit without mixing it into aggregate sections."""
    lines: list[str] = []
    for case in cases:
        if case.get("workflow") == "classic":
            lines.extend(_render_classic_query_details(case))
            continue
        lines.extend([f"### {case.get('case_id', 'unknown')}", "", f"- **Category:** `{case.get('category', 'unknown')}`", f"- **Question:** {_markdown_cell(case.get('question'))}", "", "#### Expected evidence", "", f"- **Expected sources:** {_markdown_cell(case.get('expected_sources', []))}", ""])
        lines.extend(_table(["Source path", "Lines", "Reviewed evidence text"], [[item.get("source"), f"{item.get('start_line', '—')}–{item.get('end_line', '—')}", item.get("text")] for item in case.get("expected_evidence", [])]))
        searches = case.get("search_results", []) if isinstance(case.get("search_results", []), list) else []
        queries = case.get("search_queries", []) if isinstance(case.get("search_queries", []), list) else []
        registry, ranked, retrieval_reference_by_chunk = [], [], {}
        for search_id, hits in enumerate(searches, start=1):
            hits = hits if isinstance(hits, list) else []
            registry.append([search_id, queries[search_id - 1] if search_id <= len(queries) else case.get("question")])
            if not hits:
                ranked.append([search_id, "—", "—", "—", "—", "—", "No chunk returned"])
            for rank, hit in enumerate(hits, start=1):
                retrieval_reference_by_chunk.setdefault((hit.get("source"), hit.get("chunk_id")), (search_id, rank, hit))
                ranked.append([search_id, rank, hit.get("chunk_id"), hit.get("source"), f"{hit.get('start_line', '—')}–{hit.get('end_line', '—')}", hit.get("score"), hit.get("text")])
        lines.extend(["", "#### Search registry", ""])
        lines.extend(_table(["Search ID", "Search query"], registry))
        lines.extend(["", "#### Ranked retrieval results", ""])
        lines.extend(_table(["Search ID", "Rank", "Chunk ID", "Source path", "Lines", "Score", "Retrieved chunk text"], ranked))
        lines.extend(["", "#### Context passed to answer generation", ""])
        prefixes = {item.get("context_prefix_k"): item for item in case.get("context_prefix_metrics", [])}
        context_rows = []
        for order, item in enumerate(case.get("generation_context", []), start=1):
            prefix = prefixes.get(order, {})
            covered, total = prefix.get("evidence_lines_covered"), prefix.get("ground_truth_line_count")
            context_rows.append([order, item.get("chunk_id"), item.get("source"), f"{item.get('start_line', '—')}–{item.get('end_line', '—')}", _context_content(item, retrieval_reference_by_chunk.get((item.get("source"), item.get("chunk_id")))), f"{covered}/{total}" if covered is not None and total is not None else None, prefix.get("recall"), prefix.get("precision"), prefix.get("ndcg")])
        lines.extend(_table(["Final context order", "Chunk ID", "Source path", "Lines", "Context content", "Evidence lines", "Recall", "Precision", "Context-order NDCG"], context_rows))
        if not context_rows:
            lines.extend(["", "No final-answer context was sent."])
        lines.extend(["", "#### Ragas diagnostics", ""])
        lines.extend(_table(["Metric", "Score"], [[label, (case.get(source, {}) or {}).get(metric)] for label, metric, source in RAGAS_METRICS]))
        lines.extend(["", "#### Execution audit: latency and token usage", ""])
        lines.extend(_table(["Process/call", "Latency (ms)", "Provider / model", "Provider input tokens", "Provider output tokens", "Total tokens"], _execution_rows(case)))
        lines.append("")
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
    lines.extend(_render_query_details(results.get("cases", [])))
    lines.extend(["## Artifacts", "", "- [Machine-readable results](results.json)", "- [Evaluation process log](run.log)", ""])
    return "\n".join(lines)


def write_readme(path: Path, results: dict[str, Any]) -> None:
    """Write the report at the renderer's public artifact seam."""
    path.write_text(_readme_report(results), encoding="utf-8")
