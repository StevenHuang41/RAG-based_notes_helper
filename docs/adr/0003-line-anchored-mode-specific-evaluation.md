# Use line-anchored evidence and mode-specific evaluation reports

Date: 2026-10-25
Status: Accepted

## Context

The synthetic QA benchmark must make retrieval quality inspectable at the source level. Chunk IDs change when chunking settings or source content changes, and text snippets alone do not identify their exact location. A single paired evaluation invocation also obscures which workflow a run is intended to measure, while a summary JSON and separate detail JSON duplicate report information.

## Decision

- Anchor reviewed fixture evidence and newly indexed chunks to 1-based, inclusive source line ranges.
- Compare retrieved evidence to ground truth by source path and line position. Merge overlapping lines before computing Recall and Precision; count off-source retrieved lines only in Precision's denominator.
- Compute NDCG from ranked chunks using newly covered relevant lines as graded relevance. For Agentic RAG, preserve each search's ranking as a separate list, compute NDCG per search, and report the mean over applicable searches. Compute query Recall and Precision over the union of chunks retrieved across its searches.
- Run only one workflow per invocation: Classic by default, Agentic when `--agentic` is supplied with `--eval`.
- Determine Ragas applicability from the produced outcome and available inputs rather than Expected answerability. Run Answer Relevancy and Answer Correctness for every non-empty normal or degraded response, including answers, abstentions, clarifications, insufficient-evidence responses, exhausted-answer responses, and workflow-error responses. Run Faithfulness and Context Utilization only when the exact Final generation context is non-empty; otherwise record an explicit unavailable reason rather than a synthetic score.
- Keep deterministic answerability, reason-code, response-origin, workflow-error, and evidence checks separate from Ragas diagnostics. Exclude workflow-error cases from normal semantic-response Ragas means while retaining their scores as per-case and error-only diagnostics.
- Report artifact-process status independently from workflow outcome. A successfully written evaluation containing query-level workflow errors is `completed` with workflow outcome `completed_with_errors`; report the error count and the ratio of normal bounded outcomes to selected cases.
- Write a local-time, second-resolution run directory containing a human-readable `README.md`, machine-readable `results.json`, and process `run.log`. Refuse to overwrite an existing run directory.
- Record provider-reported usage and measured latency per call, distinguishing controller, answer, and judge calls. Do not estimate missing token counts.

## Consequences

- Fixture source edits can invalidate line annotations; the validator must identify the affected case and source so annotations can be corrected.
- Existing persistent indexes may lack line metadata. They must remain queryable, but unknown legacy positions cannot be presented as known; fixture evaluation builds a fresh index.
- Agentic NDCG is summarized across multiple independently ranked search lists, while Recall and Precision describe the complete evidence gathered for the query.
- Every executed Search remains in Retrieval stage metrics even if later replaced or followed by a workflow failure. Final context evidence metrics describe only the exact ordered context supplied to the final Answer generation call.
- Reports are intentionally detailed and may be long as the fixture grows. JSON remains available for machine analysis.
- Ragas Answer Correctness uses reviewed reference answers and adds judge cost. Judging degraded responses does not turn them into successful workflow outcomes; Ragas metrics remain model-dependent diagnostics, not correctness guarantees.
