# 03: Record Agentic outcomes and Ragas eligibility in Evaluation results

**What to build:** Make each Evaluation result distinguish the Agentic terminal
outcome and response origin, preserve exact generation-context provenance, and
score only responses eligible for the applicable Ragas metrics.

**Blocked by:**

- 01: Send the initial Agentic Search directly to Answer generation.
- 02: Complete bounded Agentic Answer review and recovery.

**Status:** ready-for-agent

- [ ] Structured case results retain reason code, response origin, Search and
  Answer attempt counts, retrieved candidates, and the exact ordered Answer
  generation context as separate facts.
- [ ] Accepted Answer-model responses with a reference answer are eligible
  for applicable Ragas metrics, including an accepted model abstention for a
  valid unanswerable case.
- [ ] Controller clarification, controller-selected abstention, deterministic
  Retrieval-empty outcomes, exhausted rejected candidates, and workflow errors
  are skipped with explicit metric reasons.
- [ ] Context-dependent metrics remain unavailable when there is no final
  generation context; unavailable metrics are not reported as zero.
- [ ] Answerability and reliability summaries distinguish normal bounded
  outcomes from workflow errors and correctly account for Retrieval-empty and
  insufficient-evidence outcomes.
- [ ] The existing `run_evaluation()` seam verifies paired Classic and
  Agentic case IDs under the same seed, with no reads or mutations of runtime
  `data/` or `storage/`.
