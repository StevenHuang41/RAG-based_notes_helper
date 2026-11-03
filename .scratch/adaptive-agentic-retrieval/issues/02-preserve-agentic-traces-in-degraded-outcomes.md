# 02: Preserve Agentic traces in degraded outcomes

**What to build:** Make Agentic RAG terminate with a structured, non-empty degraded response when controller validation, controller generation, or Retrieval execution fails. Preserve every completed Retrieval search and generation audit record so a later failure cannot be reported as an earlier zero-result search. Keep operational failures distinct from normal abstention and never fall back silently to Classic RAG.

**Blocked by:** 01: Structured non-empty Classic outcomes

**Status:** ready-for-agent

- [ ] Exhausted controller-action validation returns a non-empty `controller_failure` outcome with workflow-error metadata.
- [ ] Controller-provider failure returns a non-empty `controller_failure` outcome without exposing an incomplete candidate answer.
- [ ] Retrieval execution failure returns a non-empty `retrieval_failure` outcome and is distinguishable from a successful empty search.
- [ ] A degraded outcome never counts as a normal or correct abstention and never invokes Classic RAG as a fallback.
- [ ] Every Search query, candidate list, Retrieval stage record, latency measurement, controller call, and Answer call completed before failure remains in the structured result.
- [ ] A controller failure after successful Hybrid retrieval retains real Vector, BM25, Fusion, and Rerank provenance.
- [ ] Invalid controller output still receives exactly one validation retry before the degraded outcome is produced.
- [ ] Validation feedback and its retry do not consume a Search or Answer attempt.
- [ ] Evaluation case assembly consumes the structured degraded result instead of replacing its evidence and search trace with empty placeholders.
- [ ] Tests cover invalid JSON, invalid evidence references, controller-provider failure, Retrieval failure, and failure after successful retrieval through public runtime and evaluation seams.

