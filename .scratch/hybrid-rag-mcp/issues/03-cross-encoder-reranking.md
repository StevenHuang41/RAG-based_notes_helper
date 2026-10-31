# 03: Rerank fused candidates before selecting generation context

**What to build:** Answers use the strongest evidence from the hybrid candidate set, reordered by a CrossEncoder before the configured number of chunks is sent to the answer model.

**Blocked by:** 02: Add shared hybrid retrieval with independent stage limits.

**Status:** ready-for-agent

- [ ] Reorder the RRF candidate list with Sentence Transformers CrossEncoder using `cross-encoder/ms-marco-MiniLM-L6-v2` as the English baseline.
- [ ] Treat CrossEncoder scores as ranking scores only; do not interpret or threshold them as calibrated answer confidence.
- [ ] Configure `LLM_MAX_CHUNKS=5` as the default generation-context limit, independent of the Vector, BM25, and fusion candidate limits.
- [ ] Send only the first `LLM_MAX_CHUNKS` reranked chunks to answer generation, preserving chunk text, source path, line range, and reranker score in the structured result.
- [ ] Keep the Vector minimum score threshold exclusive to Vector search; do not apply it to fusion or CrossEncoder scores.
- [ ] Offline tests verify rerank ordering, cutoff behavior, exact generation context membership and order, and answers grounded in selected evidence.
