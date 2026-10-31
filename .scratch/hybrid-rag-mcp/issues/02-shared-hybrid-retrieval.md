# 02: Add shared hybrid retrieval with independent stage limits

**What to build:** Classic and Agentic RAG searches combine semantic and lexical note matches, so either retrieval method can contribute evidence before the shared results are ranked for answer generation.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Each retrieval search returns separate Vector and BM25 result lists with their own ranks, scores, chunk provenance, and configured candidate limits.
- [x] Configure independent defaults of `VECTOR_TOP_K=30`, `BM25_TOP_K=20`, and `FUSION_TOP_K=15`.
- [x] Preserve the existing minimum relevance threshold for Vector results only; do not apply it to BM25 scores.
- [x] Fuse the two ranked lists with Reciprocal Rank Fusion and retain each candidate's Vector and BM25 ranks when present.
- [x] Classic RAG and every Agentic search call use the same hybrid retrieval service.
- [x] Remove validation that requires the Vector candidate limit to be no greater than the generation chunk limit.
- [x] Offline tests verify independent limits, RRF ordering, rank provenance, empty-result behavior, and both workflow entry points without live providers.

## Answer

Implemented in `b2d08b0`. The shared retrieval service returns Vector and BM25 rankings plus an RRF-ordered fused sequence. Candidate defaults are independent, and the Vector threshold remains exclusive to Vector matches. Focused offline regressions passed: `uv run --python 3.13 pytest tests/test_retrieval.py tests/test_agentic.py tests/test_cli_agentic.py tests/test_answer.py tests/test_eval_runner.py -q` (73 passed); `git diff --check` passed.
