# 05: Replace or extend Agentic evidence

**What to build:** Let every later Agentic Retrieval search explicitly replace the active candidate pool or extend it. Preserve all search audits, deduplicate extended candidates by source and chunk identity, and establish one active ranking by applying the configured Cross-encoder and Rerank top-k against the original user question. This must support both correcting an unhelpful first query and supplementing evidence for a multi-hop question.

**Blocked by:** 04: Agent-controlled adaptive searches and clarification

**Status:** ready-for-agent

- [ ] The first Retrieval search establishes a new active candidate pool.
- [ ] `replace` uses only the new search's fused candidates for future evidence selection while preserving every prior search result for audit and evaluation.
- [ ] `extend` combines the previous active fused candidates with the new search's fused candidates.
- [ ] Extended candidates are deduplicated by source and chunk identity before final reranking.
- [ ] A duplicate chunk retains provenance for every Search query, method rank, score, and threshold under which it appeared.
- [ ] The active candidate pool is reranked against the original user question rather than directly comparing raw scores from different rewritten queries.
- [ ] The configured Rerank top-k is applied after replacement or extension; the controller cannot enlarge it.
- [ ] The named Retrieval stages remain Vector, BM25, Fusion, and Rerank without adding a separate accumulated-rerank report stage.
- [ ] Structured Rerank provenance identifies the active evidence mode and candidate composition used by the controller.
- [ ] Retrieval stage metrics continue to include every executed search even when its candidates are later replaced.
- [ ] Tests cover one correction case using `replace`, one multi-hop case using `extend`, overlapping chunks, provenance preservation, original-question reranking, and fixed Rerank top-k.

