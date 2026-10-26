# 01: Complete Ragas and answerability summaries

**What to build:** Reviewers can interpret all supported Ragas scores against the selected fixture sample and inspect workflow answerability outcomes grouped by expected answerability.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Increment the structured evaluation result to schema version 4 and preserve the selected sample count and case IDs so aggregate denominators are auditable.
- [x] Show Faithfulness, Answer Relevancy, Answer Correctness, and Context Utilization in overall and category summaries; keep Context Utilization identified as a context/retrieval diagnostic.
- [x] Keep supported Ragas metrics visible with N/A when no selected case is eligible; show each metric per query with a score or an explicit not-scored reason.
- [x] Apply Ragas scoring only to expected-answerable cases that produced an answer.
- [x] Summarize answerability separately for expected-answerable and expected-unanswerable cases, including case, answered, abstained, unavailable/error, and correct-behavior counts. Treat unavailable/error outcomes as incorrect and show N/A for an empty group.
- [x] Explain the selected-sample denominator, Ragas eligibility, and the distinction between answerability behavior and answer quality in the report guide.
- [x] Cover Classic and Agentic structured results and rendered reports with synthetic fixtures and fake providers.

## Comments

- Implemented on the integration branch in commit `067c410`; focused evaluation-runner and report tests passed (23 passed).
- --eval-base, TTFT instrumentation, multiple-top-k generation sweeps, retrieval changes, hybrid search, and reranking are outside this feature.
