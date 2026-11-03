# 03: Threshold-aware Hybrid retrieval foundation

**What to build:** Let one Hybrid Retrieval search accept validated Vector and BM25 filtering values while keeping every candidate limit under human configuration. Use an absolute normalized Vector minimum score and a query-relative BM25 minimum-score ratio, and return enough filter diagnostics to explain why a search returned many, few, or zero candidates. Apply configured starting thresholds to Classic RAG and the first Agentic Retrieval search without yet adding controller-selected adjustments.

**Blocked by:** 01: Structured non-empty Classic outcomes

**Status:** ready-for-agent

- [ ] A Retrieval search accepts a finite Vector minimum score in the inclusive range from zero to one.
- [ ] A Retrieval search accepts a finite BM25 minimum-score ratio in the inclusive range from zero to one.
- [ ] The BM25 effective cutoff is calculated from the highest positive score in that search; a search with no positive BM25 scores returns no BM25 candidates.
- [ ] NaN, infinity, missing required values, and out-of-range thresholds are rejected before Retrieval work executes.
- [ ] Vector, BM25, Fusion, Rerank, and Generation top-k values remain fixed by configuration and cannot be overridden through the threshold request.
- [ ] Vector and BM25 results are independently sorted before RRF, and RRF continues to fuse ranks rather than compare raw method scores.
- [ ] Each search returns its effective thresholds, configured top-k values, pre-filter and post-filter counts, observed top-k score bounds, and highest excluded score when available.
- [ ] Classic RAG and the first Agentic search use the configured starting thresholds.
- [ ] Structured query results preserve threshold diagnostics without changing Final generation context semantics.
- [ ] Focused tests cover Vector filtering, BM25 ratios under different raw score scales, fixed top-k behavior, RRF input ranks, and empty-score cases.
- [ ] Configuration examples and non-secret configuration display document both starting thresholds without modifying a user's environment file.

