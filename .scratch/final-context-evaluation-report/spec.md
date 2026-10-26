# Final-context evaluation report

## Problem Statement

The evaluation report shows metrics for the configured Retrieval top-k candidates even when Agentic RAG selects fewer Retrieved chunks for the Answer generation context. This can report strong retrieval performance for evidence that the answer model did not receive.

The report also has repeated tables and fields that make review difficult. Eligibility counts, answerability outcomes, usage status, calculated-token columns, character counts, stage/method fields, and score reasons use space without helping a reviewer understand the completed query.

## Solution

Evaluate deterministic evidence quality against the final Answer generation context. Keep the configured Retrieval top-k candidates as a retrieval audit, but do not use those candidates for the primary Recall, Precision, and NDCG summaries.

For every query, render one Context prefix row for every chunk actually passed to answer generation. The final Context prefix row is the query's Final context evidence metrics. Overall and category metrics average those final rows.

Simplify report tables so they show only review-critical data. Keep per-metric Ragas judge usage and a concise per-query execution audit. Use one total-token field that takes the provider total when present and otherwise uses the exact input-plus-output sum.

## User Stories

1. As an evaluation reviewer, I want Recall, Precision, and NDCG to measure the Answer generation context, so that the scores describe evidence the answer model actually received.
2. As an evaluation reviewer, I want Retrieved chunks from Retrieval top-k kept in the report, so that I can audit candidates considered by the Agentic RAG controller.
3. As an evaluation reviewer, I want Retrieval top-k treated as a maximum candidate count for one Retrieval search, so that I do not assume every candidate enters Answer generation context.
4. As an evaluation reviewer, I want the report to show only the chunks selected for Answer generation context, so that I can see the actual model input.
5. As an evaluation reviewer, I want a query that retrieves five chunks and selects two to have two Context prefix rows, so that metrics follow the actual context size.
6. As an evaluation reviewer, I want the final Context prefix row to be the query's Final context evidence metrics, so that query, category, and overall results use one consistent scope.
7. As an evaluation reviewer, I want overall Final context evidence metrics to average the final Context prefix metrics of selected cases, so that aggregate values reflect completed query behavior.
8. As an evaluation reviewer, I want category Final context evidence metrics to average final Context prefix metrics only from cases in that category, so that category comparisons remain meaningful.
9. As an evaluation reviewer, I want a Context prefix row to calculate metrics cumulatively from the first k chunks in Answer generation context order, so that each row shows the evidence available after adding that chunk.
10. As an evaluation reviewer, I want the context table to show Evidence lines, Recall, Precision, and Context-order NDCG in each row, so that I can inspect cumulative evidence quality without a second metrics table.
11. As an evaluation reviewer, I want a prefix with no relevant Evidence span lines to display zero Recall, Precision, and NDCG when ground truth exists, so that a missed first chunk is visible as a score of zero.
12. As an evaluation reviewer, I want N/A only when a metric cannot be computed, such as missing exact source-line provenance or no fixture Evidence span, so that N/A does not hide a zero result.
13. As an evaluation reviewer, I want Context Utilization to receive the final Answer generation context, so that this Ragas context diagnostic assesses the evidence supplied to the answer model.
14. As an evaluation reviewer, I want Faithfulness and Context Utilization to use the same final Answer generation context, so that their diagnostics use the same evidence scope.
15. As an evaluation reviewer, I want the report to show Faithfulness, Answer Relevancy, Answer Correctness, and Context Utilization as score-only rows, so that Ragas diagnostics are easy to scan.
16. As an evaluation reviewer, I want an unavailable Ragas score to display a dash, so that the table remains stable when a metric cannot run.
17. As an evaluation reviewer, I want the report to omit eligible-case counts from the overall Ragas table, so that the table remains a score summary.
18. As an evaluation reviewer, I want the report to omit the Answerability outcomes table, so that it does not repeat a low-value aggregate view.
19. As an evaluation reviewer, I want the category table to show only category, case count, Final context evidence metrics, four Ragas scores, and mean evaluation latency, so that it is concise.
20. As an evaluation reviewer, I want Search registry to show a numeric Search ID and search query, so that I can identify what each Retrieval search asked.
21. As an evaluation reviewer, I want ranked retrieval rows to reuse the numeric Search ID, so that candidates can be traced to their Retrieval search.
22. As an evaluation reviewer, I want Search registry to omit stage and method when they provide no review value, so that the table is concise.
23. As an evaluation reviewer, I want the guide to define Search ID as the sequential Retrieval search number, so that I can interpret repeated Agentic RAG searches.
24. As an evaluation reviewer, I want the mean-latency heading to omit units when its table column already states milliseconds, so that the heading is concise.
25. As an evaluation reviewer, I want provider usage to be absent from the overall summary, so that controller, answer, and judge usage are not mixed without workflow context.
26. As an evaluation reviewer, I want Judge usage by metric to show calls, input tokens, output tokens, and total tokens, so that I can see the LLM cost of every Ragas metric.
27. As an evaluation reviewer, I want total tokens to use the provider total when available and exact input-plus-output otherwise, so that token usage remains visible when a provider omits its total.
28. As an evaluation reviewer, I want Judge usage by metric to omit status, duplicate calculated-total fields, and character fields, so that it contains only the cost data needed for comparison.
29. As an evaluation reviewer, I want the query execution audit to show process, latency, provider/model, input tokens, output tokens, and total tokens, so that I can compare Classic RAG and Agentic RAG calls directly.
30. As an evaluation reviewer, I want retrieval, controller, answer, judge, RAG query, and evaluation latency totals in the execution audit, so that I can locate time spent in the workflow.
31. As an evaluation reviewer, I want the execution audit to omit usage status, calculated-token columns, and character columns, so that it does not repeat token data.
32. As an evaluation reviewer, I want the guide to explain only terms that may confuse a reviewer, including Final context evidence metrics, Ragas metrics, Search ID, latency totals, and token-total behavior, so that the report is short and self-contained.
33. As an evaluation reviewer, I want the Expected sources bullet separated from its evidence table by a blank line, so that Markdown renders correctly.
34. As a maintainer, I want structured results to preserve candidate retrieval data, Answer generation context data, Context prefix metrics, and Final context evidence metrics, so that rendered values remain auditable.
35. As a maintainer, I want the evaluation to continue using the reviewed synthetic fixture and fake providers in tests, so that no personal notes or live provider access are required.

## Implementation Decisions

- Keep the evaluation runner as the source of structured evaluation records and the Markdown renderer as the report presentation boundary.
- Retain Retrieval top-k candidates and their ranked result lists for audit. They are not the primary metric scope.
- Define Final context evidence metrics as Recall, Precision, and NDCG calculated from the complete Answer generation context in its exact order.
- Use the final Context prefix record as the query-level Final context evidence metric record.
- Summarize Final context evidence metrics across queries and categories by averaging final Context prefix values. Do not average intermediate prefixes.
- Retain Context prefix records for every chunk sent to answer generation. Each record uses all chunks from order one through its own order.
- Score a context prefix with relevant ground truth available but no matched Evidence span lines as zero for Recall, Precision, and Context-order NDCG.
- Use N/A only when no ground truth is available or exact source-line provenance is unavailable for the prefix.
- Pass Answer generation context to Context Utilization, as already done for Faithfulness. Do not use all Retrieved chunks for this Ragas diagnostic.
- Show all four Ragas score rows in overall, category, and query summaries. Render an unavailable score as a dash.
- Remove eligible-case counts, Answerability outcomes, overall Provider usage, status columns, character columns, duplicated calculated-token columns, query-level judge reasons, Search registry stage, and Search registry method from the Markdown report.
- Use numeric Search IDs in Search registry and ranked result tables.
- Combine Context passed to answer generation and Context prefix metrics into one table with final context order, chunk provenance, Evidence lines, Recall, Precision, and Context-order NDCG.
- Keep the user-visible latency unit in table headings. The mean-latency heading does not repeat milliseconds.
- In token tables, use provider total tokens when present; otherwise show the exact sum of provider input and output tokens when both are present. Show a dash when neither total can be established.
- Keep RAG query latency as time through the terminal answer or abstention, and Evaluation latency as RAG query latency plus Ragas judge latency.
- Keep --eval-base outside this change. This fixture-backed --eval report describes the selected evaluation mode only.
- Increment the result schema version for the revised metric scope and report data.

## Testing Decisions

- Tests assert structured evaluation records and rendered Markdown. They do not assert private formatting helpers.
- Use run_evaluation with synthetic fixtures and fake providers as the primary integration seam.
- Assert a case with more Retrieved chunks than Answer generation context chunks: the ranked retrieval audit retains all candidates, while Final context evidence metrics use only selected context chunks.
- Assert Context prefix metrics for each cumulative context order, including zero metrics for an irrelevant first chunk and final-prefix values used by query, category, and overall summaries.
- Assert Context Utilization and Faithfulness receive the final Answer generation context rather than all Retrieved chunks.
- Assert Classic and Agentic evaluation records, including repeated Retrieval searches, no selected context, transformed context without line provenance, and no Evidence span.
- Use write_readme for focused Markdown contracts: concise headers, columns, dashes for unavailable scores, numeric Search IDs, one combined context table, blank Expected evidence spacing, and simplified judge/execution usage tables.
- Reuse existing evaluation-runner and report test fixtures. Use fake providers only.

## Out of Scope

- Changing Agentic RAG evidence-selection behavior or requiring all Retrieval top-k candidates to enter Answer generation context.
- Changing --eval-base.
- Adding TTFT or streaming evaluation.
- Running separate answer generations for each Context prefix.
- Running multiple Retrieval top-k sweeps.
- Changing retrieval algorithms, controller prompts, answer prompts, Ragas prompts, Ragas thresholds, fixture questions, fixture evidence spans, sampling, index isolation, or report artifact paths.

## Further Notes

- A Retrieved chunk is a candidate. An Answer generation context chunk is evidence that the answer model received.
- Final context evidence metrics measure retrieval plus Agentic RAG evidence selection. They are not pure candidate-retrieval metrics.
- Context prefix rows describe cumulative evidence. They do not describe multiple generated answers.
- Token totals shown as input plus output are exact arithmetic on provider-reported fields; they are not token estimates.
