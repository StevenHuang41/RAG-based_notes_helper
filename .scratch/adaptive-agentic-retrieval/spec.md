# Adaptive bounded Agentic retrieval and complete evaluation outcomes

Status: ready-for-agent

## Problem Statement

Notes Helper can complete an evaluation run while an Agentic RAG query has no
answer and its report incorrectly appears to show that retrieval returned no
candidates. In the observed run, Retrieval search returned candidates, the
controller failed to produce a valid action, and the evaluation error path
discarded the partial Retrieval search trace. The generated report therefore
made a controller failure look like a retrieval miss and still labeled the run
as completed without separately reporting workflow reliability.

The current bounded Agentic RAG can rewrite a query before another Retrieval
search, but it cannot adjust Vector or BM25 filtering for that search. It also
cannot reopen retrieval after Answer review begins. A candidate answer that is
irrelevant or insufficient can only be revised against the same selected
evidence or end in abstention, even when a focused Retrieval search could
recover missing evidence.

Classic RAG and Agentic RAG can also expose an empty answer or an operational
error instead of a stable, non-empty response that truthfully distinguishes an
invalid query, unclear intent, missing evidence, exhausted Answer attempts, a
controller failure, and a Retrieval search failure. Evaluation currently skips
Ragas diagnostics for expected-unanswerable and ambiguous cases before
considering whether the workflow returned a meaningful response, so the report
does not assess every non-empty semantic outcome against its Reference answer.

## Solution

Make every submitted query terminate in a structured, non-empty outcome while
preserving evidence grounding and honest failure semantics. Deterministic input
validation handles blank input. Classic RAG remains a fixed, single-search
workflow. Agentic RAG uses its existing LangGraph controller to decide whether
to answer, search again, revise an answer, clarify an unclear query, or abstain;
no separate Query router is introduced.

Allow each Agentic Retrieval search to carry a controller-selected Vector
minimum score and BM25 minimum-score ratio. The first search uses the configured
starting values. Later searches may raise or lower either threshold within
validated numeric bounds while the configured Retrieval top-k values remain
under human control. Give the controller enough pre-filter and post-filter
diagnostics to make that decision auditable rather than asking it to guess from
an empty candidate list.

Allow Answer review to reopen retrieval without resetting either budget.
Search attempts and Answer attempts remain separately capped at three,
including their initial attempts. A new search explicitly chooses whether its
candidate evidence replaces the previous active pool or extends it. Extended
evidence is deduplicated before Cross-encoder reranking against the original
question. Every executed search and its effective thresholds remain available
for evaluation even when its candidates are later replaced.

Evaluate every non-empty response with the Ragas diagnostics whose required
inputs exist. Answer Relevancy and Answer Correctness apply to normal answers,
abstentions, clarification responses, deterministic insufficient-evidence
responses, and degraded workflow-error responses. Faithfulness and Context
Utilization additionally require a non-empty Answer generation context; they
remain unavailable rather than receiving invented scores when no context
exists. Keep semantic-response aggregates separate from workflow-error
diagnostics and report evaluation execution status separately from workflow
outcome and reliability.

## User Stories

1. As a Notes Helper user, I want every submitted query to produce a non-empty response, so that retrieval or controller failures never look like a silent application failure.
2. As a Notes Helper user, I want an evidence-grounded answer when the notes support one, so that the response reflects the approved note corpus.
3. As a Notes Helper user, I want a truthful insufficient-evidence response when the notes do not support an answer, so that the application does not fill gaps with outside knowledge.
4. As a Notes Helper user, I want the system to say that it could not find sufficient supporting evidence rather than claim that information is absent from the corpus, so that a retrieval miss is not overstated as proof of absence.
5. As a Notes Helper user, I want blank or whitespace-only input identified as invalid input, so that it is not silently ignored or sent through retrieval.
6. As a Notes Helper user, I want an unclear or nonsensical query to receive a request for clarification, so that the agent does not invent an intended question.
7. As a Notes Helper user, I want an ambiguous but meaningful query to receive a clarification response when its missing referent cannot be safely inferred, so that the workflow does not guess which subject I meant.
8. As a Notes Helper user, I want a meaningful query with no supporting evidence distinguished from an invalid or unclear query, so that the response explains the actual limitation.
9. As a Classic RAG user, I want Classic RAG to remain a single-search, single-generation workflow, so that it remains a stable baseline for comparison.
10. As a Classic RAG user, I want an empty Retrieval search to return a deterministic response without calling the Answer model, so that an empty context cannot trigger unsupported generation.
11. As a Classic RAG user, I want a non-empty Retrieval search passed to one grounded Answer generation, so that Classic behavior remains predictable.
12. As a Classic RAG user, I want the Answer model to report insufficient evidence when its supplied context does not support the answer, so that a fixed workflow does not imply forced guessing.
13. As an Agentic RAG user, I want the first Retrieval search to use my original question and configured starting thresholds, so that every query begins from a reproducible baseline.
14. As an Agentic RAG user, I want the controller to rewrite a query when the current evidence is incomplete, so that later searches can target missing information.
15. As an Agentic RAG user, I want the controller to raise or lower the Vector minimum score for a later search, so that it can reduce noise or recover candidates without changing Vector top-k.
16. As an Agentic RAG user, I want the controller to raise or lower the BM25 minimum-score ratio for a later search, so that lexical retrieval can become narrower or broader without changing BM25 top-k.
17. As an Agentic RAG user, I want the controller to inspect threshold diagnostics before adjusting a threshold, so that its decision is based on observed score and filtering behavior.
18. As an Agentic RAG user, I want a meaningful zero-result query to receive at least one rewritten search when a safe rewrite exists, so that the workflow attempts bounded recovery before giving up.
19. As an Agentic RAG user, I want the controller to clarify immediately when no safe query rewrite exists, so that the agent does not manufacture intent merely to consume its Search budget.
20. As an Agentic RAG user, I want Answer review to request a new Retrieval search when the selected evidence cannot support a relevant answer, so that generation retries are not trapped behind inadequate evidence.
21. As an Agentic RAG user, I want Answer review to request a focused revision when the evidence is sufficient but the candidate answer is poor, so that unnecessary Retrieval searches are avoided.
22. As an Agentic RAG user, I want a later search to replace a clearly unhelpful candidate pool, so that stale evidence does not continue to influence the answer.
23. As an Agentic RAG user, I want a later search to extend a useful candidate pool for a multi-hop question, so that newly found evidence supplements rather than erases earlier evidence.
24. As an Agentic RAG user, I want duplicate chunks across rewritten searches deduplicated, so that repeated retrieval does not crowd out distinct evidence.
25. As an Agentic RAG user, I want an extended candidate pool reranked against my original question, so that candidates retrieved by different rewritten queries share one final relevance basis.
26. As an Agentic RAG user, I want every returned generated answer accepted by grounded Answer review, so that rejected candidate text is never exposed as the final answer.
27. As an Agentic RAG user, I want exhausted Answer attempts to produce a deterministic supported-response failure, so that the final rejected candidate is not returned.
28. As an Agentic RAG user, I want a controller failure to produce a non-empty degraded response, so that malformed actions do not leave the query unanswered.
29. As an Agentic RAG user, I want a Retrieval search failure distinguished from an empty successful search, so that an index or embedding error is not reported as missing note evidence.
30. As a privacy-conscious user, I want every query to retain per-query isolation, so that adaptive thresholds and evidence never persist into another query.
31. As a maintainer, I want Search attempts and Answer attempts counted separately, so that each resource budget remains understandable and bounded.
32. As a maintainer, I want the initial Retrieval search included in the three-Search limit, so that the limit means three attempts rather than three retries.
33. As a maintainer, I want the initial Answer generation included in the three-Answer limit, so that the generation limit has the same attempt semantics.
34. As a maintainer, I want an executed empty search to consume a Search attempt, so that work already performed is not omitted from the budget.
35. As a maintainer, I want a generated but rejected answer to consume an Answer attempt, so that review cannot create an unbounded generation loop.
36. As a maintainer, I want controller review and action-validation retries excluded from Search and Answer counts, so that formatting correction does not pretend retrieval or generation occurred.
37. As a maintainer, I want illegal thresholds and out-of-budget actions rejected before executing work, so that malformed controller output cannot bypass configured bounds.
38. As a maintainer, I want one validation retry for an invalid controller action, so that transient formatting failures remain recoverable without becoming an unbounded retry loop.
39. As a maintainer, I want the set of legal controller actions constrained by remaining budgets, so that the graph cannot search when no Answer attempt remains to use new evidence.
40. As a maintainer, I want stable outcome reason codes, so that runtime behavior and evaluation reports do not depend on user-facing wording.
41. As a maintainer, I want deterministic responses identified as deterministic output, so that they are not mistaken for evidence-generated model answers.
42. As an evaluation reviewer, I want every executed Retrieval search preserved after a later workflow failure, so that controller failure cannot erase retrieval provenance.
43. As an evaluation reviewer, I want each Search query displayed with its effective Vector and BM25 thresholds, so that I can audit what changed between searches.
44. As an evaluation reviewer, I want pre-filter and post-filter counts available in structured results, so that a zero-result search can be diagnosed as filtering, scoring, or corpus behavior.
45. As an evaluation reviewer, I want every configured Retrieval stage row present for every query, so that report structure remains stable across success, abstention, and failure cases.
46. As an evaluation reviewer, I want an executed stage to show its real metrics even when a later controller stage fails, so that unavailable values do not overwrite measured work.
47. As an evaluation reviewer, I want an unexecuted or genuinely unavailable stage to show a dash, so that missing data is not confused with a zero score.
48. As an evaluation reviewer, I want Retrieval stage metrics to include every actual Search attempt, so that replaced candidates remain part of the workflow audit.
49. As an evaluation reviewer, I want Generation metrics to use only the final Answer generation context, so that discarded candidates do not inflate evidence quality seen by the Answer model.
50. As an evaluation reviewer, I want Recall, Precision, and NDCG capitalized consistently, so that the report uses established metric names.
51. As an evaluation reviewer, I want Answer Relevancy and Answer Correctness evaluated for every non-empty response, so that abstentions, clarifications, and degraded responses are compared with the reviewed Reference answer.
52. As an evaluation reviewer, I want Faithfulness and Context Utilization evaluated only when an Answer generation context exists, so that an absent input is not assigned an invented score.
53. As an evaluation reviewer, I want an unavailable Faithfulness or Context Utilization score distinguished from a real zero, so that aggregate metrics preserve their semantics.
54. As an evaluation reviewer, I want workflow-error diagnostics separated from semantic-response Ragas aggregates, so that controller reliability is not hidden inside answer-quality means.
55. As an evaluation reviewer, I want workflow reliability shown as successful bounded outcomes over selected cases, so that a completed report does not imply that every workflow completed normally.
56. As an evaluation reviewer, I want evaluation execution status separated from workflow outcome, so that artifact-writing success and query-processing success remain distinct.
57. As an evaluation reviewer, I want a completed run with query failures labeled as completed with workflow errors, so that partial runtime failure is visible at the top of the report.
58. As an evaluation reviewer, I want every workflow-error response to retain its error category and diagnostic Ragas scores, so that operational and semantic consequences can both be inspected.
59. As an evaluation reviewer, I want fixed report rows and explicit reasons for unavailable metrics, so that the Markdown and structured result can be reconciled.
60. As a maintainer, I want evaluation changes to preserve isolated indexes and reviewed fixture ground truth, so that adaptive retrieval never reads or mutates the runtime corpus during evaluation.

## Implementation Decisions

- Keep Classic RAG as the default fixed workflow and Agentic RAG as the opt-in bounded LangGraph workflow. This project does not use LangChain for Classic RAG; describe it as a fixed pipeline rather than attributing it to a framework.
- Do not add a standalone Query router. Apply deterministic validation to blank or whitespace-only input, then let the Agentic controller classify the next action after the initial Retrieval search has supplied query-specific evidence and diagnostics.
- Add a validated clarification outcome for unclear, nonsensical, or underspecified queries. A clarification response is a normal bounded outcome, not a workflow error, and it must not infer a missing referent.
- Use stable reason codes at the structured result boundary. The required initial set is `invalid_query`, `needs_clarification`, `no_evidence_after_searches`, `insufficient_evidence`, `answer_attempts_exhausted`, `controller_failure`, and `retrieval_failure`.
- Produce non-empty user-facing text for every structured outcome. Deterministic fallback text must state only what the workflow observed; a successful empty search does not prove that information is absent from the corpus.
- Mark deterministic text with structured response-origin metadata. Keep normal bounded outcomes distinct from degraded workflow-error responses.
- Keep Classic RAG to one Retrieval search and at most one Answer generation. Do not add a controller, Answer review, Retrieval retry, or generation retry to Classic RAG.
- In Classic RAG, bypass the Answer model when the Retrieval search returns no candidates. When candidates exist, keep the existing context-only grounding constraint and permit the Answer model to return an insufficient-evidence response.
- Extend the Agentic Search action with a rewritten query, a Vector minimum score, a BM25 minimum-score ratio, and an evidence mode. The controller supplies explicit numeric thresholds for every later search; the first search uses configured starting values.
- Validate Vector minimum score and BM25 minimum-score ratio as finite values in the inclusive range from zero to one. Reject NaN, infinity, missing required fields, and out-of-range values before invoking retrieval.
- Keep Vector, BM25, Fusion, Rerank, and Generation top-k values under human configuration. The controller cannot change any top-k value, the RRF constant, the Cross-encoder model, or Generation context limits.
- Define the BM25 threshold as a ratio of the highest positive BM25 score observed for that Retrieval search. Apply the ratio within one query-specific ranked list; do not compare absolute BM25 scores across rewritten queries. A search with no positive BM25 score has no BM25 candidates.
- Return controller-facing threshold diagnostics for each Retrieval search: effective thresholds, configured top-k values, pre-filter and post-filter candidate counts, observed top-k score bounds, highest excluded score when available, candidate provenance, and remaining Search and Answer budgets.
- Record the same effective thresholds and complete filter diagnostics in the structured evaluation result. The human-readable Search query registry shows Search ID, query, Vector threshold, and BM25 threshold ratio without duplicating all diagnostics.
- Run Vector and BM25 candidate generation and sorting independently for each Search query, then apply RRF to the two query-specific ranked lists. RRF operates on ranks rather than comparing Vector and BM25 raw scores.
- Support `replace` and `extend` evidence modes. The first search establishes a new active pool. `replace` discards the previous active candidate pool for future evidence selection while preserving its audit record. `extend` combines the prior active fused candidates with the new fused candidates and deduplicates them by source and chunk identity.
- After replacement or extension, Cross-encoder rerank the active candidate pool against the original user question and apply the configured Rerank top-k. Preserve per-search provenance for a chunk that appeared under more than one query even though only one copy enters the active pool.
- Keep the named report stages Vector, BM25, Fusion, Rerank, and Generation. Do not introduce a separate accumulated-rerank display stage; represent the active candidate composition and evidence mode in structured Rerank provenance.
- Preserve every executed Search query, method result list, stage timing, effective threshold, and metric record even when a later search replaces its active candidates or a later controller call fails.
- Extend Answer review so it can accept, revise, request a new Retrieval search, clarify, or abstain. A new Retrieval search after Answer review consumes the next Search attempt and any subsequent generation consumes the next Answer attempt.
- Maintain separate Search and Answer counters, each capped at three attempts including the initial attempt. Switching between Retrieval and Answer phases never resets either counter.
- Count only actual Retrieval search invocations as Search attempts and actual Answer model invocations as Answer attempts. Controller calls, validation feedback, and rejected pre-execution actions do not consume either budget.
- Dynamically constrain legal actions by the remaining budgets. A search is illegal when no Search attempt remains or no Answer attempt remains to use its evidence. A revision is illegal when no Answer attempt remains.
- Give an invalid action one validation retry. If the retry remains invalid, terminate with a non-empty `controller_failure` outcome, retain partial trace, and do not silently fall back to Classic RAG or count the failure as a correct abstention.
- Treat Retrieval execution exceptions as `retrieval_failure`, not as empty search results. Preserve all trace data completed before the exception.
- Keep Ragas judges evaluation-only. They do not participate in runtime Answer review, Retrieval decisions, clarification, or fallback generation.
- Determine Ragas applicability from the produced outcome and available inputs rather than expected answerability alone. Run Answer Relevancy and Answer Correctness for every non-empty response, including abstentions, clarifications, insufficient-evidence responses, and degraded workflow-error responses.
- Run Faithfulness and Context Utilization only when the case has a non-empty Answer generation context. An absent context makes those diagnostics unavailable; do not substitute a high Faithfulness score or a zero Context Utilization score.
- Keep deterministic answerability, reason-code, response-origin, Retrieval evidence, and workflow reliability checks separate from model-based Ragas diagnostics.
- Separate normal semantic-response Ragas aggregates from workflow-error diagnostic scores. A workflow-error response remains judgeable per case but does not enter the normal semantic-response mean.
- Define evaluation run status independently from workflow outcome. A run can complete artifact generation while its workflow outcome is `completed_with_errors`. Report workflow reliability as normal bounded outcomes over selected cases and report the workflow-error count.
- Preserve Final context evidence metrics as Recall, Precision, and NDCG over the exact final Answer generation context. Retrieval stage metrics continue to describe all executed Retrieval searches, including searches whose candidates were later replaced.
- Render fixed Vector, BM25, Fusion, Rerank, and Generation metric rows for every query. Use measured values for executed stages and a dash only for stages that were not executed or could not be computed.
- Capitalize Recall, Precision, and NDCG in overall report tables. Keep all supported Ragas rows visible and attach explicit structured reasons to unavailable scores.
- Increment the machine-readable result schema version for new action parameters, search diagnostics, structured outcomes, workflow status, reliability, and revised Ragas applicability.
- Keep evaluation indexes isolated from the runtime note corpus and persistent index. Do not change reviewed questions, Reference answers, Expected answerability, or Evidence spans to improve scores.
- Reopen the accepted architectural decisions that prohibit Retrieval reopening after Answer review and restrict Ragas to expected-answerable generated answers. Update the governing ADRs before treating implementation as conformant.

## Testing Decisions

- Tests assert externally observable query results, structured evaluation records, and rendered reports rather than private LangGraph nodes, prompt wording, or formatting helpers.
- Use `run_query()` as the primary runtime seam for Classic and Agentic behavior. Drive it with fake controllers, fake Answer providers, and controlled Retrieval results to assert response text presence, reason codes, response origin, workflow errors, citations, search provenance, effective thresholds, evidence modes, and attempt counts.
- Use `run_evaluation()` as the primary evaluation seam. Exercise isolated fixture cases with fake providers and assert agreement between machine-readable results and the human-readable report for Ragas applicability, workflow reliability, status, stage rows, thresholds, partial traces, and Final context evidence metrics.
- Add focused Retrieval service tests only for behavior that is impractical to isolate through the two primary seams: Vector threshold filtering, BM25 maximum-score ratio filtering, finite numeric bounds, fixed top-k behavior, RRF input ranks, and pre-filter/post-filter diagnostics.
- Reuse existing query, Agentic workflow, retrieval, evaluation-runner, final-context, stage-report, and Markdown renderer test patterns. Extend their public-result assertions instead of coupling new tests to private graph functions.
- Cover deterministic invalid input without Retrieval or LLM calls, including blank and whitespace-only input.
- Cover a gibberish query that returns clarification without inventing a rewritten intent, and an ambiguous query that safely asks for the missing referent.
- Cover a meaningful empty first search followed by a controller-selected rewritten search with changed thresholds.
- Cover controller-selected threshold increases that reduce noisy candidates and threshold decreases that admit previously filtered candidates, while every top-k remains fixed.
- Cover BM25 ratio behavior under different raw score scales to prove that equal ratios are query-relative rather than absolute-score cutoffs.
- Cover controller access to pre-filter/post-filter counts and score bounds, including a zero-result search with a highest excluded score.
- Cover `replace` discarding the prior active pool without deleting its audit trace.
- Cover `extend` deduplicating overlapping chunks, preserving multi-search provenance, reranking the active pool against the original question, and enforcing the configured Rerank top-k.
- Cover a multi-hop case where the second search supplements evidence from the first search and a correction case where the second search replaces an unhelpful first result.
- Cover Answer review requesting Retrieval, then generation, without resetting either counter.
- Cover exactly three Search attempts and exactly three Answer attempts, including mixed Retrieval and Answer transitions, and assert that no fourth attempt executes.
- Cover invalid JSON, invalid threshold values, invented evidence references, and out-of-budget actions receiving one validation retry without consuming Search or Answer attempts.
- Cover exhausted validation, controller-provider failure, and Retrieval execution failure returning non-empty degraded responses with workflow errors and retained partial traces.
- Cover a successful empty Retrieval search separately from a Retrieval exception so their reason codes and workflow-error semantics cannot be conflated.
- Cover Classic empty retrieval bypassing generation, Classic non-empty retrieval making one grounded generation call, and Classic never entering controller, review, or retry behavior.
- Cover Ragas Answer Relevancy and Answer Correctness for answerable, ambiguous, unanswerable, clarification, deterministic fallback, and workflow-error responses.
- Cover Faithfulness and Context Utilization with a non-empty final context, plus explicit unavailable reasons rather than zero or fabricated scores when final context is empty.
- Cover semantic-response aggregates excluding workflow-error diagnostics while per-case workflow-error judge scores, workflow-error count, and workflow reliability remain visible.
- Cover an evaluation that writes all artifacts successfully but contains one workflow error; assert completed run status, `completed_with_errors` workflow outcome, and the correct reliability numerator and denominator.
- Cover a controller failure after successful retrieval and assert that Vector, BM25, Fusion, and Rerank rows retain real values while an unexecuted Generation row displays a dash.
- Cover stable stage rows when an earlier method returns zero candidates, title-cased metric labels, Search query threshold columns, and structured filter diagnostics.
- Use reviewed evaluation fixtures and fake providers for automated tests. Do not require live model credentials, network access, the runtime note corpus, or the persistent index.
- After offline checks pass, use a Classic and Agentic evaluation with the same sample and seed as a manual acceptance comparison. Treat live Ragas values as auditable observations, not stable unit-test expectations.

## Out of Scope

- Adding a standalone pre-retrieval Query router or a separate LLM classification call.
- Adding LangChain to Classic RAG or replacing the existing LangGraph Agentic orchestration framework.
- Allowing the controller to change Vector, BM25, Fusion, Rerank, or Generation top-k values, the RRF constant, the embedding model, or the Cross-encoder model.
- Using absolute BM25 raw-score thresholds across different queries.
- More than three Search attempts or three Answer attempts.
- Persistent conversation state, cross-query evidence, learned thresholds, or threshold state shared between queries.
- Runtime Ragas judging, Ragas-controlled routing, or using Ragas scores as runtime confidence thresholds.
- External web search, non-note evidence sources, write tools, or side effects outside the existing read-only Retrieval search boundary.
- Changing evaluation ground truth, fixture questions, Reference answers, Expected answerability, or Evidence spans to improve the resulting metrics.
- Treating Ragas diagnostics as correctness guarantees or assigning synthetic values to metrics whose required inputs are absent.
- Redesigning report artifacts, evaluation directory isolation, sampling semantics, or log privacy beyond the structured fields and presentation changes required by this feature.
- Claiming that an empty Retrieval result proves that information is absent from the allowed corpus.

## Further Notes

- This specification intentionally reopens ADR-0005's decision that Answer review cannot reopen retrieval. The ADR must be superseded or amended before implementation is considered architecturally complete.
- This specification intentionally reopens ADR-0003's decision to run Ragas answer diagnostics only for expected-answerable generated answers. The revised contract judges every non-empty outcome with the metrics whose inputs exist while retaining deterministic answerability checks.
- The observed Agentic evaluation incident was not a zero-result Retrieval search: retrieval recorded candidates, controller validation failed, and the evaluation error path discarded the candidate trace. Threshold adaptation is a new recovery capability, not the direct fix for that provenance defect.
- Vector similarity and BM25 scores remain ranking signals, not calibrated probabilities or answer-confidence values. Agent-selected thresholds control candidate filtering only; the controller and Answer review still determine evidence sufficiency.
- Rewritten-query scores are local to their Retrieval search. Cross-search extension combines candidate identities and provenance, then establishes a shared ordering by reranking against the original question; it does not directly compare raw BM25 scores from different queries.
- `run_query()` and `run_evaluation()` are the approved high-level acceptance seams. Focused lower-level tests exist only for threshold mathematics and diagnostics that cannot be observed precisely enough through those seams.
