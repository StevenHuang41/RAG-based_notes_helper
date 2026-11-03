# 01: Structured non-empty Classic outcomes

**What to build:** Make the public query boundary return a structured, non-empty outcome for invalid input and every Classic RAG terminal path. Blank input must be identified without Retrieval search, an empty successful Retrieval search must produce a truthful deterministic response without Answer generation, and a non-empty Retrieval search must retain the existing single grounded Answer generation. Establish stable reason-code and response-origin fields that later Agentic and evaluation slices can reuse without changing Classic RAG into a reviewed or retrying workflow.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `run_query()` returns non-empty text plus stable reason-code and response-origin metadata for every Classic terminal outcome.
- [x] Blank and whitespace-only input returns `invalid_query` without invoking Retrieval search or an LLM.
- [x] A successful Classic Retrieval search with no candidates returns a deterministic `no_evidence_after_searches` response without invoking the Answer model.
- [x] A Classic Retrieval search with candidates invokes the Answer model at most once and keeps the context-only grounding constraint.
- [x] Classic RAG does not gain a controller, Answer review, Retrieval retry, or generation retry.
- [x] Deterministic fallback wording states only that sufficient supporting evidence was not found; it does not claim that information is absent from the corpus.
- [x] Existing citations and Answer generation context remain available for a normal generated answer.
- [x] CLI one-shot and REPL behavior expose the structured outcome as non-empty user-visible text without changing workflow selection semantics.
- [x] Focused tests use the public query boundary and assert observable calls and results rather than private helpers.

## Comments

- Implementation: `9d1e440` (`feat(query): add structured Classic outcomes`).
- Verification: 47 focused tests passed across `test_query_outcomes.py`,
  `test_cli_agentic.py`, `test_answer.py`, and `test_retrieval.py` using the
  existing checkout environment with `PYTHONPATH` pointed at this worktree.
- Static checks: `py_compile` and `git diff --check` passed.
- Ruff was not executed because the existing checkout environment has no Ruff
  executable. A worktree-local environment could not be completed because
  sandboxed dependency download failed DNS resolution.

