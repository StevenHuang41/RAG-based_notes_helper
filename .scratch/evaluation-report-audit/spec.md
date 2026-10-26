# Auditable RAG evaluation report

## Problem Statement

Reviewers can see many per-query columns and scores, but the report does not make clear which values are deterministic checks, which are model-based judge diagnostics, or how a retrieved chunk differs from the context actually sent to answer generation. Related evidence coverage values and repeated latency/token tables make the query section harder to audit. The separate Citations table can also imply that the answer visibly cited sources, although the application passes source metadata as generation context and asks the answer model not to include citations in its response.

## Solution

Keep aggregate results first, then explain the evaluation process and each metric group before showing query details. Give each query a complete audit trail: expected source evidence, ranked retrieval results for each search, the final ordered answer-generation context, generated answer or abstention, judge diagnostics, and one process-level latency/token table.

Use line-based Recall, Precision, and NDCG as the primary retrieval scores. Show expected source paths and reviewed evidence spans directly instead of adding source/evidence coverage scores. Present Faithfulness, Answer Relevancy, and Answer Correctness as distinct Ragas diagnostics, not pass/fail guarantees. Omit Context Utilization's score from the human-readable README while retaining its computed value in machine-readable results; its judge cost remains visible in execution telemetry.

## User Stories

1. As an evaluation reviewer, I want aggregate results before query details, so that I can understand the run before inspecting individual cases.
2. As an evaluation reviewer, I want a process guide between aggregate metrics and query details, so that I know which workflow, fixture evidence, and judge steps produced the results.
3. As an evaluation reviewer, I want deterministic retrieval metrics distinguished from Ragas judge diagnostics, so that I do not interpret unlike scores as equivalent evidence.
4. As an evaluation reviewer, I want Recall, Precision, and NDCG explained in plain language, so that I can tell whether a query missed evidence, retrieved irrelevant lines, or ranked evidence poorly.
5. As an evaluation reviewer, I want the report to explain how Agentic RAG aggregates Recall and Precision across searches and averages NDCG across applicable search lists, so that per-query ranking scores are not ambiguous.
6. As an evaluation reviewer, I want the expected source paths and reviewed evidence spans, including line ranges and text, so that I can compare retrieval directly with the fixture's ground truth.
7. As an evaluation reviewer, I do not want separate expected source/evidence coverage scores repeated beside line-based retrieval metrics, so that overlapping measurements do not obscure the evidence itself.
8. As an evaluation reviewer, I want every search query recorded beside its ranked results, so that I can understand why each Classic or Agentic retrieval list exists.
9. As an evaluation reviewer, I want retrieval results grouped by search and to show stage/method metadata when available, so that future hybrid retrieval or reranking can be inspected without conflating pipeline stages.
10. As an evaluation reviewer, I want every retrieved chunk's rank, source, line range, score, and full text, so that I can audit candidate quality and ranking decisions.
11. As an evaluation reviewer, I want scores identified as method-specific ranking signals rather than probabilities, so that I do not interpret similarity values as confidence.
12. As an evaluation reviewer, I want a distinct table showing the exact order of chunks passed to final answer generation, so that I can distinguish retrieval candidates from model input.
13. As an evaluation reviewer, I want unchanged generation-context chunks to refer to their retrieval rows by ID, so that the report avoids copying the same text twice.
14. As an evaluation reviewer, I want transformed or truncated generation context recorded with the actual text sent, so that IDs do not conceal what the model saw.
15. As an evaluation reviewer, I want the expected answerability, observed answer/abstention behavior, reference answer, and generated answer together, so that I can assess the outcome against the case expectation.
16. As an evaluation reviewer, I want answerability checks shown separately from answer quality scores, so that a correct abstention is not confused with a correct generated answer.
17. As an evaluation reviewer, I want Faithfulness, Answer Relevancy, and Answer Correctness explained as separate judge questions with their inputs, so that I understand what each score can and cannot establish.
18. As an evaluation reviewer, I want judge scores shown once per query with reasons only when available, so that duplicate score tables do not make one metric appear to be several measurements.
19. As an evaluation reviewer, I want judge scores described as model-dependent diagnostics without implicit pass/fail thresholds, so that a low score prompts investigation rather than an unsupported verdict.
20. As an evaluation reviewer, I want Context Utilization's score kept out of the README's primary metrics while its usage cost remains auditable, so that similar model-judged retrieval diagnostics do not dominate the main report.
21. As an evaluation reviewer, I want one query-level execution table with latency and token usage per process/call, so that I can relate runtime cost to retrieval, controller, answer, and judge work.
22. As an evaluation reviewer, I want provider/model, usage status, input/output/total tokens, and workflow/end-to-end latency totals, so that I can audit costs without reconstructing them from separate tables.
23. As an evaluation reviewer, I want missing token usage marked unavailable or partial and never estimated, so that the report does not present invented cost data.
24. As an evaluation reviewer, I want the report to clarify that local embedding work is not provider-reported LLM token usage, so that I understand the limits of the token totals.
25. As an evaluation reviewer, I want each query's report details linked to the machine-readable results and process log, so that I can inspect raw data when needed.
26. As a maintainer, I want the machine-readable result schema to preserve existing evaluation fields while adding search-query and generation-context provenance, so that automation can consume the richer audit record.
27. As a maintainer, I want the evaluation to continue using reviewed synthetic fixtures, so that report generation does not read personal notes or persistent user indexes.

## Implementation Decisions

- Keep the report renderer as the highest-level presentation seam: it consumes one evaluation result object and renders the complete Markdown report.
- Keep query result assembly responsible for recording each search query, ranked candidate list, final answer-generation context order, and any context text that differs from retrieved chunk text.
- Keep the answer-generation boundary responsible for returning the selected/truncated context in the order actually passed to the answer model.
- Preserve full retrieved chunk text in the retrieval audit. In the answer-generation-context table, refer to unchanged chunks by source, line range, and chunk ID; include actual context text and formatted block only when content differs.
- Preserve one ranked result list per search. Carry stage/method metadata through when a retrieval result provides it; identify the current FAISS vector search as the current retrieval method.
- Keep Recall, Precision, and NDCG as the primary deterministic line-grounded retrieval metrics. Keep answerability and abstention checks as deterministic behavior metrics.
- Keep Faithfulness, Answer Relevancy, and Answer Correctness as the visible Ragas answer diagnostics. Describe their inputs and applicability in the process guide. Do not introduce score thresholds.
- Keep Context Utilization computation and its result in machine-readable diagnostics when present, but omit its score from README metric tables. Include its measured judge latency and reported token usage in the execution audit.
- Replace separate citations, generation-score, judge-score, latency, and per-call token tables with the agreed context-provenance table, one judge-score table, and one combined per-query execution table.
- Remove source/evidence coverage scores from visible README tables while preserving expected source paths and reviewed evidence spans in each query audit and retaining underlying machine-readable fields.
- Increase the evaluation result schema version to 3 for the added search-query and generation-context provenance fields.
- Do not implement hybrid retrieval or reranking as part of this change; the report carries stage/method metadata when available so later pipeline stages can be represented.

## Testing Decisions

- Good tests assert externally observable Markdown and result-record behavior rather than private formatting helpers or implementation details.
- Test the Markdown report renderer, evaluation runner/result assembly, RAG answer-generation boundary, and Agentic workflow result assembly.
- Use the report writer as the primary seam. With representative classic and agentic result records, assert section order, metric visibility, evidence content, retrieval stages/searches, context ordering, and the combined execution audit.
- Use the evaluation result-record seam with offline fake RAG responses to assert that each search query is paired with its candidate list and that generation-context order, references, and changed text are recorded accurately.
- Use the answer-generation boundary with a fake LLM to assert that the returned context provenance matches the actual selected and maximum-chunk-limited prompt context.
- Cover no-result abstention, Agentic repeated searches, duplicated chunks across searches, missing judge reasons, unavailable/partial token usage, and transformed context text.
- Update the existing evaluation-runner report assertions to the new contract; reuse prior patterns from `test_eval_runner.py`, `test_agentic.py`, `test_answer.py`, and `test_ragas_runner.py`.
- Use fixture/fake providers only. No live provider credentials or network access are required for these tests.

## Out of Scope

- Changing retrieval algorithms, adding hybrid retrieval, or implementing reranking.
- Adding or recalibrating evaluation metrics, changing Ragas prompts, or introducing pass/fail score thresholds.
- Changing the synthetic QA fixture, its expected answers, or evidence annotations.
- Changing user-facing answer formatting or adding citations to generated answers.
- Changing evaluation index isolation, artifact naming, timestamp behavior, or log privacy rules.
- Reporting provider token estimates for local embedding work or calls that do not expose usage.

## Further Notes

- The implementation is present in the current working tree. This spec records the agreed behavior and test contract; automated tests have not yet been run.
- Results schema version 3 adds query search text and answer-generation-context provenance. Unchanged text is referenced rather than duplicated; changed context preserves the actual text and prompt block.
- Similarity scores are method-specific ranking signals and are not calibrated probabilities.
- The existing line-anchored and mode-specific evaluation decisions remain in force. Context Utilization remains available in JSON for diagnosis even though its score is omitted from the README.
