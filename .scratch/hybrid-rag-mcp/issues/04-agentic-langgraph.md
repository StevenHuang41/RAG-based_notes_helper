# 04: Orchestrate bounded Agentic RAG with LangGraph

**What to build:** Agentic RAG uses a graph to decide whether to search again, answer from retrieved evidence, or abstain, while retaining the existing bounded and read-only behavior.

**Blocked by:** 03: Rerank fused candidates before selecting generation context.

**Status:** resolved

- [x] Implement Agentic query state transitions with LangGraph and the shared hybrid retrieval and reranking flow.
- [x] Preserve the maximum of three searches per query, including the initial search.
- [x] Keep each query's graph state isolated; do not retain note content, evidence, or controller state across queries.
- [x] Permit answers to reference only chunks returned by the current query's searches; abstain when evidence is insufficient.
- [x] Preserve existing bounded action validation and failure behavior for invalid controller output or failed model calls.
- [x] Do not add LangSmith, Phoenix, external tracing, or monitoring dependencies in this change.
- [x] Offline tests with fake controller and answer providers verify search, answer, abstention, invalid-action, search-limit, and per-query isolation behavior.
- [x] Update the architecture decision record to supersede the framework-light orchestration choice while preserving its historical rationale and safety constraints.

## Answer

Agentic RAG now runs through a reusable compiled LangGraph `StateGraph` without a checkpointer. Each `agentic_answer` invocation supplies fresh state; controller validation and its single validation retry, the three-search limit, evidence-reference checks, callback/usage reporting, and retrieval stage records remain intact. `langgraph>=1.1.3` is a direct runtime dependency and the existing 1.1.3 lock entry is now included in the editable project's direct dependency metadata. ADR-0005 records the new decision, and ADR-0001 retains its original rationale with a superseded status.

Verification passed offline: `tests/test_agentic.py` (17 passed), `tests/test_cli_agentic.py` (22 passed), the full suite (167 passed), `uv lock --check --offline --python /Users/Steven/.local/bin/python3.13`, and `git diff --check`. The pytest run used the repository virtualenv's Python 3.14 and emitted existing Pydantic V1 compatibility and Google GenAI deprecation warnings.
