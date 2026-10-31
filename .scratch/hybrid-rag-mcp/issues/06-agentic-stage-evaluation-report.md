# 06: Report Agentic search provenance and stage metrics

**What to build:** An Agentic evaluation report preserves the evidence and ranking from every search, then summarizes the evidence gathered and the final context used to answer or abstain.

**Blocked by:**
- 04: Orchestrate bounded Agentic RAG with LangGraph.
- 05: Report Classic retrieval stages, metrics, and latency.

**Status:** ready-for-agent

- [ ] Preserve each Search ID, its search query, and its Vector, BM25, RRF, and reranked results with ranks and provenance.
- [ ] Report each retrieval stage's actual cutoff, returned count, and line-anchored Recall, Precision, and NDCG for every Agentic query.
- [ ] Aggregate each stage's Recall and Precision over the union of that stage's results across searches; report NDCG as the mean of applicable per-search rankings.
- [ ] Compare pre-rerank and post-rerank metrics at a common cutoff for each search, while retaining each stage's actual-cutoff metrics.
- [ ] Keep metrics for the exact final answer-generation context separate from metrics for all retrieved candidates.
- [ ] Show the full text of chunks only in the final context section; keep per-search result tables to concise chunk provenance and ranking details.
- [ ] Include per-search, controller, answer, judge, RAG-query, and end-to-end evaluation latency where available, with measured values only.
- [ ] Offline tests verify repeated searches, search-specific rank provenance, stage unions, mean per-search NDCG, abstention, and final-context metrics.
