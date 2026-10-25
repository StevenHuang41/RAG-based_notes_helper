# 04: Per-query latency and provider token usage

**What to build:** Capture per-stage latency and provider-reported token usage for retrieval, Classic/Agentic generation, and Ragas judge calls.

**Blocked by:** 03: Run one query mode per evaluation.

**Status:** resolved

- [x] Record each search/retrieval latency and the per-query aggregate.
- [x] Record each Agentic controller LLM call latency separately from vector-search latency; record final answer generation latency.
- [x] Record each Ragas metric judge latency and total judge latency per query.
- [x] Record end-to-end latency per query.
- [x] Persist every provider-reported usage record with call type (`controller`, `answer`, `judge`), provider/model, input/output/total tokens, and usage availability status. Judge calls also identify the Ragas metric that triggered them.
- [x] Aggregate usage per query, per judge metric, category, and run; keep controller/answer generation separate from judge usage.
- [x] Preserve `partial` and `unavailable` states without tokenizer estimates.
- [x] Include Ragas metric rationales only when returned by the metric API; do not store hidden reasoning or full prompts.
- [x] Add deterministic tests for stage timings, Agentic multiple calls, judge usage, aggregation, and missing/partial provider usage.

## Comments

- Resolved in commit `1c697c7`, merged to integration at `77b1a9c`. Focused post-merge coverage passed (40 tests); `git diff --check` passed.
