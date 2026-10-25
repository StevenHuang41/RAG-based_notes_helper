# 04: Per-query latency and provider token usage

**What to build:** Capture per-stage latency and provider-reported token usage for retrieval, Classic/Agentic generation, and Ragas judge calls.

**Blocked by:** 03: Run one query mode per evaluation.

**Status:** ready-for-agent

- [ ] Record each search/retrieval latency and the per-query aggregate.
- [ ] Record each Agentic controller LLM call latency separately from vector-search latency; record final answer generation latency.
- [ ] Record each Ragas metric judge latency and total judge latency per query.
- [ ] Record end-to-end latency per query.
- [ ] Persist every provider-reported usage record with call type (`controller`, `answer`, `judge`), provider/model, input/output/total tokens, and usage availability status.
- [ ] Aggregate usage per query, category, and run; keep controller/answer generation separate from judge usage.
- [ ] Preserve `partial` and `unavailable` states without tokenizer estimates.
- [ ] Include Ragas metric rationales only when returned by the metric API; do not store hidden reasoning or full prompts.
- [ ] Add deterministic tests for stage timings, Agentic multiple calls, judge usage, aggregation, and missing/partial provider usage.

