# 04: Agent-controlled adaptive searches and clarification

**What to build:** Allow the Agentic controller to use observed Retrieval diagnostics when it rewrites a query and selects new Vector and BM25 thresholds for a later search. Add a validated clarification outcome for gibberish, missing referents, and other queries that cannot be safely rewritten. Keep the controller bounded, keep all top-k values under human control, and expose every effective query and threshold through runtime and evaluation results.

**Blocked by:** 02: Preserve Agentic traces in degraded outcomes; 03: Threshold-aware Hybrid retrieval foundation

**Status:** ready-for-agent

- [ ] The validated Search action carries a non-blank rewritten query, Vector minimum score, BM25 minimum-score ratio, and evidence mode field needed by the next slice.
- [ ] Later Search actions may raise or lower either threshold within the validated numeric bounds.
- [ ] The controller receives the prior search's effective thresholds, filter counts, score diagnostics, returned candidates, and remaining Search and Answer budgets.
- [ ] The controller cannot change any top-k value, the RRF constant, the embedding model, or the Cross-encoder model.
- [ ] A meaningful empty first search can trigger at least one rewritten search when a safe rewrite exists.
- [ ] The validated clarification action produces a non-empty `needs_clarification` outcome without inventing a missing referent or rewritten intent.
- [ ] Gibberish and an underspecified meaningful query can clarify without being forced to exhaust Search attempts.
- [ ] Clarification remains a normal bounded outcome rather than a workflow error and retains per-query isolation.
- [ ] Each executed search records its Search ID, query, effective thresholds, diagnostics, candidate provenance, and latency.
- [ ] Invalid thresholds or an illegal Search action receive one validation retry and do not consume a Search attempt before work executes.
- [ ] `run_query()` tests cover threshold increases, threshold decreases, zero-result recovery, clarification, invalid threshold actions, and unchanged top-k values.
- [ ] Evaluation results and the Search query registry expose each Search query with its Vector threshold and BM25 threshold ratio.

