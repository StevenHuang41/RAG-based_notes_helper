# 02: Render an auditable evaluation report

**What to build:** Reviewers can move from aggregate evaluation results to a clear per-query audit of expected evidence, retrieved candidates, answer context, answer or abstention, judge diagnostics, and execution cost.

**Blocked by:** 01: Capture query and answer-context provenance.

**Status:** resolved

- [x] Present aggregate results first, followed by a process and metric guide before query details.
- [x] Show expected source paths and reviewed evidence spans alongside retrieval results grouped by search, with rank, source, line range, score, and chunk text.
- [x] Distinguish retrieved candidates from the ordered context used for answer generation; show answerability and abstention behavior separately from answer quality.
- [x] Explain deterministic line-based Recall, Precision, and NDCG separately from Ragas diagnostics; show Faithfulness, Answer Relevancy, and Answer Correctness once, without pass/fail thresholds.
- [x] Omit the Context Utilization score from the human-readable report while retaining its machine-readable value and execution cost when available.
- [x] Combine per-process latency and provider-reported token usage into one execution audit, including provider/model, usage status, input/output/total tokens, and totals; mark missing usage unavailable or partial and never estimate it.
- [x] Link query details to the machine-readable results and process log.
- [x] Add offline report-contract coverage for representative Classic and Agentic result records.

## Comments

- Implemented and validated on the integration branch. The full repository test suite passed (129 tests); no live providers were used.
- Post-checkpoint implementation and test changes remain unstaged and uncommitted because the Git staging approval was aborted.
