# 02: Preserve Agentic traces in degraded outcomes

**What to build:** Make Agentic RAG terminate with a structured, non-empty degraded response when controller validation, controller generation, or Retrieval execution fails. Preserve every completed Retrieval search and generation audit record so a later failure cannot be reported as an earlier zero-result search. Keep operational failures distinct from normal abstention and never fall back silently to Classic RAG.

**Blocked by:** 01: Structured non-empty Classic outcomes

**Status:** resolved

- [x] Exhausted controller-action validation returns a non-empty `controller_failure` outcome with workflow-error metadata.
- [x] Controller-provider failure returns a non-empty `controller_failure` outcome without exposing an incomplete candidate answer.
- [x] Retrieval execution failure returns a non-empty `retrieval_failure` outcome and is distinguishable from a successful empty search.
- [x] A degraded outcome never counts as a normal or correct abstention and never invokes Classic RAG as a fallback.
- [x] Every Search query, candidate list, Retrieval stage record, latency measurement, controller call, and Answer call completed before failure remains in the structured result.
- [x] A controller failure after successful Hybrid retrieval retains real Vector, BM25, Fusion, and Rerank provenance.
- [x] Invalid controller output still receives exactly one validation retry before the degraded outcome is produced.
- [x] Validation feedback and its retry do not consume a Search or Answer attempt.
- [x] Evaluation case assembly consumes the structured degraded result instead of replacing its evidence and search trace with empty placeholders.
- [x] Tests cover invalid JSON, invalid evidence references, controller-provider failure, Retrieval failure, and failure after successful retrieval through public runtime and evaluation seams.

## Comments

- Implemented in `a8294a3` (`fix(agentic): preserve degraded workflow traces`).
- `run_query()` now returns deterministic, non-empty `controller_failure` or `retrieval_failure` outcomes and preserves completed Search, Retrieval-stage, Answer-attempt, and generation-context audit data without exposing an unreviewed candidate answer.
- `run_evaluation()` consumes structured degraded results directly, including partial evidence, queries, stage metrics, latency callbacks, reason code, response origin, and workflow-error metadata.
- Verification: 86 focused tests passed across Agentic runtime, CLI, query outcomes, stage reporting, and evaluation runner; the complete offline pytest suite passed. `py_compile` and `git diff --check` passed.
- Ruff remains unverified because the existing project environment has no Ruff executable; no network installation was attempted.
- Integration review fix (2026-11-04): Agentic active-pool rerank failures now return a degraded `retrieval_failure` while retaining the completed Search query, candidates, and Vector/BM25/RRF/Rerank stage record. The `run_query()` regression test passed. `_AgenticAuditTrace` is a typed write-ahead record distinct from LangGraph committed node state: it records work before fallible operations, so attempt counts and partial results remain available if a node raises before returning its state update. Focused query/Agentic/CLI/Retrieval tests passed (94); the full offline suite passed (239).
