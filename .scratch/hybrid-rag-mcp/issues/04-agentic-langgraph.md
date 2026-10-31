# 04: Orchestrate bounded Agentic RAG with LangGraph

**What to build:** Agentic RAG uses a graph to decide whether to search again, answer from retrieved evidence, or abstain, while retaining the existing bounded and read-only behavior.

**Blocked by:** 03: Rerank fused candidates before selecting generation context.

**Status:** ready-for-agent

- [ ] Implement Agentic query state transitions with LangGraph and the shared hybrid retrieval and reranking flow.
- [ ] Preserve the maximum of three searches per query, including the initial search.
- [ ] Keep each query's graph state isolated; do not retain note content, evidence, or controller state across queries.
- [ ] Permit answers to reference only chunks returned by the current query's searches; abstain when evidence is insufficient.
- [ ] Preserve existing bounded action validation and failure behavior for invalid controller output or failed model calls.
- [ ] Do not add LangSmith, Phoenix, external tracing, or monitoring dependencies in this change.
- [ ] Offline tests with fake controller and answer providers verify search, answer, abstention, invalid-action, search-limit, and per-query isolation behavior.
- [ ] Update the architecture decision record to supersede the framework-light orchestration choice while preserving its historical rationale and safety constraints.
