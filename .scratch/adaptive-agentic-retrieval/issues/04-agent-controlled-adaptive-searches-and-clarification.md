# 04: Agent-controlled adaptive searches and clarification

**What to build:** Allow the Agentic controller to use observed Retrieval diagnostics when it rewrites a query and selects new Vector and BM25 thresholds for a later search. Add a validated clarification outcome for gibberish, missing referents, and other queries that cannot be safely rewritten. Keep the controller bounded, keep all top-k values under human control, and expose every effective query and threshold through runtime and evaluation results.

**Blocked by:** 02: Preserve Agentic traces in degraded outcomes; 03: Threshold-aware Hybrid retrieval foundation

**Status:** resolved

- [x] The validated Search action carries a non-blank rewritten query, Vector minimum score, BM25 minimum-score ratio, and evidence mode field needed by the next slice.
- [x] Later Search actions may raise or lower either threshold within the validated numeric bounds.
- [x] The controller receives the prior search's effective thresholds, filter counts, score diagnostics, returned candidates, and remaining Search and Answer budgets.
- [x] The controller cannot change any top-k value, the RRF constant, the embedding model, or the Cross-encoder model.
- [x] A meaningful empty first search can trigger at least one rewritten search when a safe rewrite exists.
- [x] The validated clarification action produces a non-empty `needs_clarification` outcome without inventing a missing referent or rewritten intent.
- [x] Gibberish and an underspecified meaningful query can clarify without being forced to exhaust Search attempts.
- [x] Clarification remains a normal bounded outcome rather than a workflow error and retains per-query isolation.
- [x] Each executed search records its Search ID, query, effective thresholds, diagnostics, candidate provenance, and latency.
- [x] Invalid thresholds or an illegal Search action receive one validation retry and do not consume a Search attempt before work executes.
- [x] `run_query()` tests cover threshold increases, threshold decreases, zero-result recovery, clarification, invalid threshold actions, and unchanged top-k values.
- [x] Evaluation results and the Search query registry expose each Search query with its Vector threshold and BM25 threshold ratio.

## Comments

- Implemented in `270e58c` using the approved `run_query()` and `run_evaluation()` seams with fake controller, Retrieval, Answer, and judge boundaries.
- Verification: full offline pytest suite passed; focused Agentic/query/CLI tests passed (64); focused evaluation/report/Retrieval/provider tests passed (64); `py_compile` and `git diff --check` passed.
- Search actions validate a rewritten query, finite `[0,1]` thresholds, and `replace`/`extend`; this slice records the evidence mode but intentionally leaves pool replacement, extension, deduplication, and original-question reranking to ticket 05.
- Ruff was not run because the existing project virtual environment does not provide a Ruff executable; no network installation was attempted.
- Live-provider controller JSON adherence and live evaluation behavior remain unverified.
