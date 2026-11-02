# Evaluation report refinements

## Problem Statement

Reviewers cannot reliably tell which Ragas metrics were evaluated, what the answerability rates use as their denominator, or how evidence accumulates as chunks are added to the final answer context. The report also repeats ambiguous latency totals, leaves a provider token total as N/A when input and output are both known, and uses search and execution labels without enough explanation.

The current report already records the relevant fixture evidence, search results, answer context, judge usage, and query timing. Its presentation hides Context Utilization scores, does not give per-prefix context metrics, and makes several calculated or grouped values difficult to audit.

## Solution

Keep the report aggregate-first and make each section use terms that a reviewer can interpret without consulting implementation details. In the fixture-backed --eval report, show Faithfulness, Answer Relevancy, Answer Correctness, and Context Utilization wherever Ragas scores are summarized. Preserve Context Utilization's meaning as a context/retrieval diagnostic. Show N/A when a metric is supported but no selected case was eligible for scoring.

Show deterministic answerability outcomes as counts grouped by expected answerability. Keep the configured retrieval top-k results and their retrieval metrics distinct from the ordered context passed to generation. For each context prefix, show cumulative line-based coverage/Recall, Precision, and context-order NDCG.

Clarify process latency and provider usage, including calculated token totals when all input/output counts are present, separately recorded character counts when token counts are incomplete, and concise definitions of the terms used in the report.

## User Stories

1. As an evaluation reviewer, I want run coverage and aggregate results before query details, so that I can understand the run before inspecting individual cases.
2. As an evaluation reviewer, I want the report to state how many fixture cases were selected, so that I can interpret every aggregate denominator as the selected sample.
3. As an evaluation reviewer, I want the fixture-backed --eval report to list Faithfulness, Answer Relevancy, Answer Correctness, and Context Utilization, so that the displayed metrics match the metrics supported by this evaluation mode.
4. As an evaluation reviewer, I want Context Utilization identified as a context/retrieval diagnostic, so that I do not mistake it for answer quality.
5. As an evaluation reviewer, I want supported Ragas metrics to remain visible with N/A when no selected case was eligible, so that an unscored metric is distinguishable from an unsupported metric.
6. As an evaluation reviewer, I want all four Ragas scores in the overall and category summaries when applicable, so that I can compare the selected sample and fixture categories consistently.
7. As an evaluation reviewer, I want each query's judge diagnostics to include all four metric rows with a score or an explicit not-scored reason, so that missing eligibility is visible.
8. As an evaluation reviewer, I want judge call counts, latency, and token usage grouped by each of the four metrics, so that I can audit the cost of Context Utilization as well as the answer metrics.
9. As an evaluation reviewer, I want answerability summarized separately from answer quality, so that a correct abstention is not confused with a correct generated answer.
10. As an evaluation reviewer, I want answerability grouped by expected answerable and expected unanswerable cases, so that each group's outcome is evaluated against its own denominator.
11. As an evaluation reviewer, I want each answerability group to show the number of cases, answered cases, abstentions, unavailable or failed outcomes, and correct behaviors as a count over the group's cases, so that both successes and failure modes are visible.
12. As an evaluation reviewer, I want an empty expected-answerability group to show N/A rather than a zero score, so that the report does not imply that an absent case was evaluated.
13. As an evaluation reviewer, I want the report to distinguish retrieval top-k from context-prefix k, so that candidate retrieval limits are not confused with the number of chunks already passed to generation.
14. As an evaluation reviewer, I want each retrieval invocation identified as Retrieval (1), Retrieval (2), and so on, so that I can follow repeated Agentic searches without reading an ordinal as a fraction or total count.
15. As an evaluation reviewer, I want the report to explain that Retrieval (1) uses the initial query and later retrieval numbers identify searches requested by the controller, so that I can understand the order of the Agentic workflow.
16. As an evaluation reviewer, I want each search query and its available stage/method metadata shown once in a search registry, so that search provenance is clear without repeating those values on every chunk row.
17. As an evaluation reviewer, I want ranked retrieval results to reference their search ID and show rank, chunk ID, source path, lines, score, and full retrieved text, so that I can audit each result list.
18. As an evaluation reviewer, I want the retrieval section to show the Recall, Precision, and NDCG scores for the configured retrieval top-k, so that I can assess the candidate results returned by the run.
19. As an evaluation reviewer, I want the guide to explain that each retrieval invocation applies the configured top-k and that a search may return fewer results, so that result counts are not mistaken for the configured limit.
20. As an evaluation reviewer, I want Agentic retrieval Recall and Precision defined over the union of that query's search results and NDCG defined as the mean over its applicable ranked search lists, so that repeated searches have an auditable metric interpretation.
21. As an evaluation reviewer, I want a separate table showing the exact ordered context passed to answer generation, so that I can distinguish retrieved candidates from chunks the model actually received.
22. As an evaluation reviewer, I want a Context prefix k row to describe the first k chunks in the actual answer context order, so that prefix scores correspond to the model input sequence.
23. As an evaluation reviewer, I want each context prefix to show cumulative unique ground-truth evidence-line coverage and Recall together, for example 2/4 (0.5000), so that the amount and proportion of evidence covered are both clear without a redundant delta column.
24. As an evaluation reviewer, I want cumulative Precision for each context prefix, so that I can see how adding context changes the share of supplied source lines that support the fixture evidence.
25. As an evaluation reviewer, I want context-order NDCG for each context prefix, so that I can inspect whether supporting evidence appears early in the context supplied to generation.
26. As an evaluation reviewer, I want retrieval-level metrics and context-prefix metrics labeled by their distinct scopes, so that identical metric names do not imply that candidate rankings and model context are the same data.
27. As an evaluation reviewer, I want the guide to say that prefix metrics assess evidence available at each context length while one answer is generated from the complete final context, so that prefix rows are not interpreted as separate generated answers.
28. As an evaluation reviewer, I want expected sources and reviewed evidence spans displayed together with a blank line before the evidence table, so that the Markdown renders as intended.
29. As an evaluation reviewer, I want each query's Latency section to contain only RAG workflow stages, so that runtime query cost is not mixed with evaluation-only Ragas cost.
30. As an evaluation reviewer, I want latency units shown in the column heading rather than embedded in metric names, so that values are easy to scan.
31. As an evaluation reviewer, I want the Execution audit to show every token-consuming controller, answer, and Ragas judge call, so that provider usage and LLM latency can be audited without retrieval-only rows.
32. As an evaluation reviewer, I want RAG query latency defined from query processing start through the workflow's terminal answer or abstention, including retrieval and controller work, so that I can understand the user-facing workflow duration.
33. As an evaluation reviewer, I want Ragas judge latency shown with the token-consuming calls in the Execution audit, so that evaluation-only cost remains visible but separate from RAG latency.
34. As an evaluation reviewer, I want Controller, Answer, and Judge total rows to aggregate their preceding calls without claiming a provider/model for the aggregate, so that repeated Agentic work is easy to compare.
35. As an evaluation reviewer, I want the report not to present TTFT for the non-streaming evaluation path, so that a full generation duration is not misrepresented as first-token latency.
36. As an evaluation reviewer, I want provider-reported input, output, and total token fields kept distinct from calculated totals, so that token provenance is clear.
37. As an evaluation reviewer, I want an input-plus-output total calculated and labeled when both token counts are available but the provider total is absent, so that the report does not show N/A for an exact sum.
38. As an evaluation reviewer, I want token status to remain partial or unavailable when provider usage fields are missing, so that a calculated sum is not mistaken for complete provider reporting.
39. As an evaluation reviewer, I want character counts shown separately when recorded and token counts cannot be completed, so that available usage evidence is visible without treating characters as tokens.
40. As an evaluation reviewer, I want character usage marked N/A for judge calls when it is not recorded, so that the report does not estimate judge prompt sizes.
41. As an evaluation reviewer, I want token summaries by category to use the same reported-versus-calculated semantics as the overall and query tables, so that category totals do not contradict detailed usage.
42. As an evaluation reviewer, I want the guide to define every metric, status, search identifier, prefix identifier, and latency/token term used in the report, so that I can interpret it without implementation knowledge.
43. As an evaluation reviewer, I want the report to explain that similarity scores are ranking signals rather than calibrated probabilities, so that I do not treat them as confidence values.
44. As an evaluation reviewer, I want judge metrics described as model-dependent diagnostics rather than pass/fail guarantees, so that I use them to investigate answer quality rather than infer certainty.
45. As an evaluation reviewer, I want links to the machine-readable results and process log, so that I can inspect the complete record behind a rendered summary.
46. As a maintainer, I want the structured result to preserve the new answerability counts, token usage provenance, and ordered context-prefix metrics, so that the Markdown report can be audited against machine-readable data.
47. As a maintainer, I want the evaluation to continue using reviewed synthetic fixtures and fake providers in tests, so that report work does not read or mutate personal notes or require live provider access.

## Implementation Decisions

- Keep the Markdown report renderer as the presentation boundary and keep evaluation metric assembly in the evaluation runner.
- Store answerability counts by expected-answerability group and ordered context-prefix metrics in the machine-readable evaluation result as well as rendering them in Markdown.
- Increment the result schema version to 4 for the added structured summary and context-prefix metric data. Preserve existing raw provider usage fields.
- For fixture-backed --eval, show all four supported Ragas metrics: Faithfulness, Answer Relevancy, Answer Correctness, and Context Utilization. Keep Context Utilization classified as a Ragas context/retrieval diagnostic, consistent with the existing domain decision.
- Ragas scores apply only to eligible expected-answerable cases that produced an answer. Keep supported metric rows visible and use N/A when a run or category has no eligible score.
- Include Context Utilization in overall metrics, metrics by category, per-query diagnostics, and judge usage by metric.
- Replace rate-only answerability behavior with counts grouped by expected answerability. Include answered, abstained, unavailable/error, and correct-behavior counts over each group's cases; errors and unavailable outcomes do not count as correct. Use N/A for an empty group.
- Preserve the fixture sample count and selected case IDs so count denominators remain auditable.
- Treat the configured retrieval top-k as a per-search candidate limit. Show each retrieval invocation as Retrieval (n), with its query and available stage/method metadata in a search registry. Chunk result rows refer to the search ID and rank without repeating the query, stage, or method.
- Keep retrieval metrics scoped to retrieved candidate lists. For Agentic RAG, compute Recall and Precision over the union of a query's search results and NDCG as the mean over applicable search lists.
- For the answer-generation context, preserve exact generation order and report one cumulative context prefix for each k from 1 through the number of chunks passed to generation. Calculate line-grounded prefix metrics using the first k context chunks.
- In the context table, combine unique ground-truth line coverage and Recall in one display; show cumulative Precision and context-order NDCG in separately labeled columns. Do not add a separate marginal-evidence column.
- Keep context-prefix metrics distinct from retrieval metrics. Prefix metrics describe evidence available in the model input prefix; they do not represent a separate model answer for every prefix.
- If exact source-line provenance is unavailable for transformed context, mark affected prefix metrics N/A rather than claiming unsupported line coverage.
- Show a compact one-row retrieval metric summary for the configured top-k in the retrieval section. Show cumulative prefix metrics in the context table.
- Use human-readable latency labels with units in the heading. Keep each per-query Latency section limited to measured RAG workflow stages and the complete query.
- For Classic RAG, show Vector, BM25, Fusion, Reranking, Answer generation, and Complete query in execution order.
- For Agentic RAG, show each Search's Vector, BM25, Fusion, Reranking, and retrieval-review Controller latency; show each Answer generation and Answer-review Controller latency; then aggregate Vector, BM25, Fusion, Reranking, Controller, Answer generation, and Complete query.
- In the Execution audit, show only token-consuming controller, answer, and Ragas judge calls. Follow them with Controller, Answer, and Judge totals that aggregate latency and token usage without a provider/model value. Do not repeat retrieval, RAG-query, or end-to-end evaluation total rows there.
- Do not report TTFT for the non-streaming --eval path.
- Preserve provider-reported input/output/total token values. When provider total is missing but every included call has reported input and output values, show their exact sum as calculated. If any required input/output value is missing, show the total as unavailable rather than summing missing values as zero; keep provider status partial or unavailable.
- Show recorded character counts in separate character units where available. Never convert characters to token estimates. Judge character counts remain N/A when not recorded.
- Keep the process and metric guide concise while defining all report terms, metric scopes, eligibility, latency totals, token statuses, and the difference between retrieval top-k and context-prefix k.
- Add no score thresholds and make no changes to Ragas prompts or scoring algorithms.
- Keep --eval-base, TTFT instrumentation, multiple-top-k generation sweeps, retrieval algorithm changes, hybrid retrieval, reranking, fixture changes, answer formatting, and index isolation out of scope.

## Testing Decisions

- Tests assert externally observable result records and rendered Markdown, not private formatting helpers.
- Use run_evaluation with the existing synthetic fixture and fake providers as the primary integration seam. Assert that the structured result and README agree on answerability counts, metric applicability, per-search retrieval metrics, context-prefix metrics, latency, token/character usage, and sample denominators.
- Use write_readme for focused renderer contract cases that directly assert section order, visible columns and values, N/A states, Search ID explanations, and the blank line between Expected sources and its table.
- Reuse existing report-writer and evaluation-runner test patterns for both Classic and Agentic workflows.
- Cover answerable and unanswerable groups, an empty group, workflow errors/unavailable outcomes, a run/category with no eligible Ragas scores, no returned chunks, repeated Agentic retrieval searches, retrieval results exceeding the final context subset, overlapping source lines, duplicate evidence lines, and ordered context prefixes.
- Cover provider-reported totals, exact calculated totals, incomplete token inputs/outputs, unavailable token usage, recorded generation character counts, and unavailable judge character counts.
- Use fixture/fake providers only. Do not require network access, provider credentials, or the --eval-base user-index path.

## Out of Scope

- Changing the behavior or output format of --eval-base.
- Running separate answer generations for each context prefix or sweeping multiple retrieval top-k settings.
- Adding TTFT or changing the evaluation path to streaming.
- Changing retrieval, controller, or answer-generation behavior; bounded Agentic Answer review is governed separately by `.scratch/agentic-answer-review/spec.md` and ADR-0005.
- Adding hybrid retrieval, reranking, or new search algorithms.
- Changing fixture questions, reference answers, evidence annotations, or sampling behavior.
- Adding or recalibrating Ragas metrics, prompts, thresholds, or pass/fail gates.
- Changing the user's answer format, index isolation, report directory naming, or log privacy behavior.

## Further Notes

- This is a follow-up to the resolved auditable-report work. It supersedes that report specification's decision to hide the Context Utilization score while preserving the existing classification of Context Utilization as a retrieval/context diagnostic.
- Retrieval top-k and context-prefix k are different quantities. Retrieval top-k limits candidates returned by each search; context-prefix k counts the first k chunks actually provided to answer generation.
- A context-prefix row reports deterministic evidence metrics for that prefix. The workflow still generates one answer using the complete final context.
- Character counts describe prompt/output characters recorded by the application. They are not token counts and must not be converted into token estimates.
