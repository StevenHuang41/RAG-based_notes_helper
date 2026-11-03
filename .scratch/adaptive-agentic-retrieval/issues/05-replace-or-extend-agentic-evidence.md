# 05: Replace or extend Agentic evidence

**What to build:** Let every later Agentic Retrieval search explicitly replace the active candidate pool or extend it. Preserve all search audits, deduplicate extended candidates by source and chunk identity, and establish one active ranking by applying the configured Cross-encoder and Rerank top-k against the original user question. This must support both correcting an unhelpful first query and supplementing evidence for a multi-hop question.

**Blocked by:** 04: Agent-controlled adaptive searches and clarification

**Status:** resolved

- [x] The first Retrieval search establishes a new active candidate pool.
- [x] `replace` uses only the new search's fused candidates for future evidence selection while preserving every prior search result for audit and evaluation.
- [x] `extend` combines the previous active fused candidates with the new search's fused candidates.
- [x] Extended candidates are deduplicated by source and chunk identity before final reranking.
- [x] A duplicate chunk retains provenance for every Search query, method rank, score, and threshold under which it appeared.
- [x] The active candidate pool is reranked against the original user question rather than directly comparing raw scores from different rewritten queries.
- [x] The configured Rerank top-k is applied after replacement or extension; the controller cannot enlarge it.
- [x] The named Retrieval stages remain Vector, BM25, Fusion, and Rerank without adding a separate accumulated-rerank report stage.
- [x] Structured Rerank provenance identifies the active evidence mode and candidate composition used by the controller.
- [x] Retrieval stage metrics continue to include every executed search even when its candidates are later replaced.
- [x] Tests cover one correction case using `replace`, one multi-hop case using `extend`, overlapping chunks, provenance preservation, original-question reranking, and fixed Rerank top-k.

## Comments

- Implemented in `be627c1` through the approved `run_query()` and `run_evaluation()` seams.
- `replace` and `extend` operate on per-search Fusion candidates, deduplicate by `(source, chunk_id)`, rerank the active pool against the original question, and apply the configured Rerank top-k.
- Runtime and evaluation records preserve prior Search stages plus per-chunk query, threshold, Vector/BM25 rank and score, and Fusion rank and score provenance. Rerank remains the existing named stage and records evidence mode, pre-rerank pool size, and contributing Search IDs.
- Verification: full offline pytest suite passed (231 test items); focused Agentic/query/Retrieval/evaluation/report tests passed (101); `py_compile` and `git diff --check` passed.
- Answer review still cannot reopen Retrieval in this slice; that state transition and budget behavior remain ticket 06 scope.
- Ruff was not run because the existing project virtual environment does not provide a Ruff executable. Live-provider behavior remains unverified.

