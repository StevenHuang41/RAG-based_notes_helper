# 06: Report Agentic search provenance and stage metrics

**What to build:** An Agentic evaluation report preserves the evidence and ranking from every search, then summarizes the evidence gathered and the final context used to answer or abstain.

**Blocked by:**
- 04: Orchestrate bounded Agentic RAG with LangGraph.
- 05: Report Classic retrieval stages, metrics, and latency.

**Status:** resolved

- [x] Preserve each Search ID, its search query, and its Vector, BM25, RRF, and reranked results with ranks and provenance.
- [x] Report each retrieval stage's actual cutoff, returned count, and line-anchored Recall, Precision, and NDCG for every Agentic query.
- [x] Aggregate each stage's Recall and Precision over the union of that stage's results across searches; report NDCG as the mean of applicable per-search rankings.
- [x] Compare pre-rerank and post-rerank metrics at a common cutoff for each search, while retaining each stage's actual-cutoff metrics.
- [x] Keep metrics for the exact final answer-generation context separate from metrics for all retrieved candidates.
- [x] Show the full text of chunks only in the final context section; keep per-search result tables to concise chunk provenance and ranking details.
- [x] Include per-search retrieval stages, retrieval-review controller, Answer attempts, aggregate RAG-query stages, and token-consuming controller/answer/judge calls where available, with measured values only.
- [x] Offline tests verify repeated searches, search-specific rank provenance, stage unions, mean per-search NDCG, abstention, and final-context metrics.

## Answer

Implemented in `f846723`. Agentic evaluation now normalizes every Search ID and its four retrieval stages, calculates each stage at its own cutoff, reports stage Recall/Precision over the union of cutoff-limited candidates, and averages applicable per-search NDCG. Pre-rerank and reranked results are compared per search only when candidate IDs match and at a common cutoff bounded by the final generation top-k. Exact final-context evidence metrics remain separate. Agentic Markdown tables show ranks and provenance without candidate text; complete context text appears only in the final generation-context section. Missing search, controller, answer, or eligible judge latency remains unavailable; total judge and end-to-end latency are unavailable when eligible judge timings are incomplete.

Validation: focused evaluation/report tests passed (9 tests); the full suite passed (183 tests) under Python 3.14 with the ticket 07 isolated MCP dependency environment on `PYTHONPATH`; Ruff and `git diff --check` passed. The full suite emitted one existing Google GenAI deprecation warning.

The 2026-11-03 report amendment moved judge latency out of the per-query RAG Latency table and into the Execution audit alongside judge token usage. It also removed repeated retrieval and end-to-end total rows from that audit.
