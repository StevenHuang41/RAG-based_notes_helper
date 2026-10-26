# 02: Audit retrieval and ordered context prefixes

**What to build:** Reviewers can trace each retrieval search's ranked candidates and inspect how evidence accumulates in the exact ordered context supplied to answer generation.

**Blocked by:** 01: Complete Ragas and answerability summaries.

**Status:** ready-for-agent

- [ ] Record each search once in a search registry and identify it as Retrieval (1), Retrieval (2), and so on; explain that Retrieval (1) uses the initial question and later searches use controller requests.
- [ ] Show each search's available stage/method metadata in the registry and reference the search ID from ranked result rows containing rank, chunk ID, source path, lines, score, and full retrieved text.
- [ ] Show a compact Recall, Precision, and NDCG summary for the configured per-search retrieval top-k; for Agentic RAG, compute Recall and Precision over the union of a query's search results and NDCG as the mean over applicable ranked search lists.
- [ ] Preserve and render the exact ordered answer-generation context, separately from retrieved candidates.
- [ ] Store and render one cumulative context-prefix record per context chunk, with unique evidence-line coverage and Recall, Precision, and context-order NDCG. Show affected prefix metrics as N/A when exact source-line provenance is unavailable.
- [ ] Explain that retrieval top-k limits candidates per search, context-prefix k counts chunks in answer-generation order, and prefix rows assess evidence available to one answer generated from the complete final context.
- [ ] Cover Classic and Agentic behavior with offline tests for repeated searches, no results, retrieval results beyond the final context subset, overlapping evidence lines, and transformed context.
