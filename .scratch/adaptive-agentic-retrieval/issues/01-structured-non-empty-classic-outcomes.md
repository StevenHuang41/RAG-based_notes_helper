# 01: Structured non-empty Classic outcomes

**What to build:** Make the public query boundary return a structured, non-empty outcome for invalid input and every Classic RAG terminal path. Blank input must be identified without Retrieval search, an empty successful Retrieval search must produce a truthful deterministic response without Answer generation, and a non-empty Retrieval search must retain the existing single grounded Answer generation. Establish stable reason-code and response-origin fields that later Agentic and evaluation slices can reuse without changing Classic RAG into a reviewed or retrying workflow.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `run_query()` returns non-empty text plus stable reason-code and response-origin metadata for every Classic terminal outcome.
- [ ] Blank and whitespace-only input returns `invalid_query` without invoking Retrieval search or an LLM.
- [ ] A successful Classic Retrieval search with no candidates returns a deterministic `no_evidence_after_searches` response without invoking the Answer model.
- [ ] A Classic Retrieval search with candidates invokes the Answer model at most once and keeps the context-only grounding constraint.
- [ ] Classic RAG does not gain a controller, Answer review, Retrieval retry, or generation retry.
- [ ] Deterministic fallback wording states only that sufficient supporting evidence was not found; it does not claim that information is absent from the corpus.
- [ ] Existing citations and Answer generation context remain available for a normal generated answer.
- [ ] CLI one-shot and REPL behavior expose the structured outcome as non-empty user-visible text without changing workflow selection semantics.
- [ ] Focused tests use the public query boundary and assert observable calls and results rather than private helpers.

