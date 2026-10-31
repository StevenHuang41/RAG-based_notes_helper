# 05: Report Classic retrieval stages, metrics, and latency

**What to build:** A Classic evaluation run shows how each retrieval stage performed, what the answer model received, and how long each part of the query took.

**Blocked by:** 03: Rerank fused candidates before selecting generation context.

**Status:** resolved

- [x] Record the Vector, BM25, RRF, and reranked candidate lists with their actual configured cutoff, returned count, rank, score, source, line range, and chunk identifier.
- [x] Calculate line-anchored Recall, Precision, and NDCG for each stage at that stage's actual cutoff.
- [x] Compare the pre-rerank and post-rerank rankings on the same fused candidate pool and common cutoff, including the final generation top-k.
- [x] Calculate per-query overall retrieval evidence metrics from the exact final generation context; keep answer-quality diagnostics distinct from evidence metrics.
- [x] Record measured latency for Vector search, BM25 search, fusion, reranking, answer generation, and the complete query; never estimate durations or provider token counts.
- [x] Render each query in this order: question, reference answer, generated answer, quality metrics, latency table, Vector results, BM25 results, RRF results, reranked results, and final generation context.
- [x] Keep retrieval-stage tables concise; show full chunk context only in the final generation-context table.
- [x] Preserve the isolated evaluation corpus and reviewed QA ground truth; evaluation must not read or mutate runtime notes or the persistent runtime index.
- [x] Use offline fake providers through the evaluation runner to verify structured metrics, fair reranker comparison, latency accounting, and rendered report behavior.

## Answer

Implemented in `2cd8af8`. Classic evaluation records each retrieval stage's configured cutoff, returned candidates, rank, stage-specific score, source lines, and chunk ID; it scores each stage at its cutoff and compares RRF with reranked candidates only when their candidate pools match, using the common cutoff bounded by generation top-k. Final evidence metrics continue to use the exact ordered generation context. The report follows the requested query order, keeps candidate tables free of full chunk text, and shows measured stage, answer-generation, and query latency; missing measurements remain unavailable.

Validation: focused evaluation/report/metric tests passed (37 tests), full `pytest -q` passed (169 tests) with Python 3.14.4, `ruff check --ignore F841` passed for the changed Python files, and `git diff --check` passed. The full suite emitted existing Python 3.14 dependency deprecation warnings.
