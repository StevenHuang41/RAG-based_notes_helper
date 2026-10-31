# 05: Report Classic retrieval stages, metrics, and latency

**What to build:** A Classic evaluation run shows how each retrieval stage performed, what the answer model received, and how long each part of the query took.

**Blocked by:** 03: Rerank fused candidates before selecting generation context.

**Status:** ready-for-agent

- [ ] Record the Vector, BM25, RRF, and reranked candidate lists with their actual configured cutoff, returned count, rank, score, source, line range, and chunk identifier.
- [ ] Calculate line-anchored Recall, Precision, and NDCG for each stage at that stage's actual cutoff.
- [ ] Compare the pre-rerank and post-rerank rankings on the same fused candidate pool and common cutoff, including the final generation top-k.
- [ ] Calculate per-query overall retrieval evidence metrics from the exact final generation context; keep answer-quality diagnostics distinct from evidence metrics.
- [ ] Record measured latency for Vector search, BM25 search, fusion, reranking, answer generation, and the complete query; never estimate durations or provider token counts.
- [ ] Render each query in this order: question, reference answer, generated answer, quality metrics, latency table, Vector results, BM25 results, RRF results, reranked results, and final generation context.
- [ ] Keep retrieval-stage tables concise; show full chunk context only in the final generation-context table.
- [ ] Preserve the isolated evaluation corpus and reviewed QA ground truth; evaluation must not read or mutate runtime notes or the persistent runtime index.
- [ ] Use offline fake providers through the evaluation runner to verify structured metrics, fair reranker comparison, latency accounting, and rendered report behavior.
