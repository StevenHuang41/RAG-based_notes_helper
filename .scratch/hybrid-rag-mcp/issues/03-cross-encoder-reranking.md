# 03: Rerank fused candidates before selecting generation context

**What to build:** Answers use the strongest evidence from the hybrid candidate set, reordered by a CrossEncoder before the configured number of chunks is sent to the answer model.

**Blocked by:** 02: Add shared hybrid retrieval with independent stage limits.

**Status:** resolved

- [x] Reorder the RRF candidate list with Sentence Transformers CrossEncoder using `cross-encoder/ms-marco-MiniLM-L6-v2` as the English baseline.
- [x] Treat CrossEncoder scores as ranking scores only; do not interpret or threshold them as calibrated answer confidence.
- [x] Configure `LLM_MAX_CHUNKS=5` as the default generation-context limit, independent of the Vector, BM25, and fusion candidate limits.
- [x] Send only the first `LLM_MAX_CHUNKS` reranked chunks to answer generation, preserving chunk text, source path, line range, and reranker score in the structured result.
- [x] Keep the Vector minimum score threshold exclusive to Vector search; do not apply it to fusion or CrossEncoder scores.
- [x] Offline tests verify rerank ordering, cutoff behavior, exact generation context membership and order, and answers grounded in selected evidence.

## Answer

Implemented in `542fe56`. Hybrid retrieval now lazily loads the English CrossEncoder, preserves Vector/BM25/RRF/reranked stage lists and timings, and passes only the configured top reranked evidence to answer generation. Agentic answers re-rank selected evidence against the original question; evaluation tests use an offline fake model.

Validation: full `pytest -q` passed (165 tests) using Python 3.14.4; `git diff --check` passed.
