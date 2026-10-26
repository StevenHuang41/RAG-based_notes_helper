# 02: Render an auditable evaluation report

**What to build:** Reviewers can move from aggregate evaluation results to a clear per-query audit of expected evidence, retrieved candidates, answer context, answer or abstention, judge diagnostics, and execution cost.

**Blocked by:** 01: Capture query and answer-context provenance.

**Status:** ready-for-agent

- [ ] Present aggregate results first, followed by a process and metric guide before query details.
- [ ] Show expected source paths and reviewed evidence spans alongside retrieval results grouped by search, with rank, source, line range, score, and chunk text.
- [ ] Distinguish retrieved candidates from the ordered context used for answer generation; show answerability and abstention behavior separately from answer quality.
- [ ] Explain deterministic line-based Recall, Precision, and NDCG separately from Ragas diagnostics; show Faithfulness, Answer Relevancy, and Answer Correctness once, without pass/fail thresholds.
- [ ] Omit the Context Utilization score from the human-readable report while retaining its machine-readable value and execution cost when available.
- [ ] Combine per-process latency and provider-reported token usage into one execution audit, including provider/model, usage status, input/output/total tokens, and totals; mark missing usage unavailable or partial and never estimate it.
- [ ] Link query details to the machine-readable results and process log.
- [ ] Add offline report-contract coverage for representative Classic and Agentic result records.

## Comments

- The current working tree contains the report implementation, but automated tests have not been run. Keep report contract coverage in this vertical slice.
