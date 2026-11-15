# Separate Agentic evidence recovery from answer generation

Status: ready-for-agent

## Problem Statement

Users need bounded Agentic searches to collect useful evidence before generating an answer, retain partial findings across searches, and disclose missing information without presenting unsupported claims. The current implementation generates immediately after the initial non-empty Search and lets Answer review reopen Retrieval. Controller-selected evidence references can fail validation and suppress an otherwise useful outcome. Aggregate Ragas scores can also appear better when failed cases are excluded, obscuring the quality/reliability trade-off.

The approved design changes that behavior deliberately. Design approval does not imply implementation completion or improved live-provider quality.

## Solution

Use two phases. First, accumulate a deduplicated chunks pool across bounded searches and assess its Evidence coverage against the original question. Stop normally when evidence is sufficient or Search attempts are exhausted. Second, select a bounded generation context, generate an answer, and review it using only that context. Answer review can request revisions but cannot reopen Retrieval.

Empty final pools return deterministic search-limitation text. Non-empty pools reach generation even if their evidence is irrelevant. Accepted full answers, partial answers with explicit gaps, and abstentions remain distinguishable from warnings, review failures, exhausted revisions, and Retrieval degradation. Reports expose evidence selection and per-metric eligibility without hiding workflow failures.

## User Stories

1. As a user, I want Search attempts capped including the first Search, so that recovery has a predictable limit.
2. As a user, I want partial findings retained across searches, so that a rewritten query cannot discard earlier useful evidence.
3. As a user, I want repeated chunks deduplicated, so that repeated retrieval does not crowd the pool with copies.
4. As an evaluator, I want all Search provenance retained for repeated chunks, so that accumulation is auditable.
5. As a user, I want the accumulated pool assessed against my original question, so that rewritten queries do not redefine success.
6. As a user, I want missing information to guide later searches, so that recovery focuses on unanswered portions.
7. As an operator, I want bounded threshold adjustments independent from top-k, so that retrieval control has explicit limits.
8. As a user, I want an intelligible question with zero hits to receive bounded recovery, so that zero hits do not imply gibberish.
9. As a user, I want an unintelligible question to receive a rephrasing request when the pool is empty, so that unnecessary answer generation is avoided.
10. As a user, I want an empty final pool described as an unsuccessful search, so that the application does not claim to prove corpus absence.
11. As an operator, I want Retrieval review failures to preserve available evidence and degrade safely, so that invalid control output does not always prevent generation.
12. As an operator, I want reviewer input limits handled explicitly, so that truncated reviews cannot falsely claim complete pool assessment.
13. As a user, I want oversized pools reranked against my original question, so that generation receives a relevant bounded selection.
14. As an operator, I want final reranking skipped when the pool already fits the chunk limit, so that unnecessary reranking cost is avoided.
15. As a user, I want the answer model to see a non-empty selected pool even when relevance is poor, so that it can explain evidence limitations.
16. As a user, I want evidence-supported partial answers with disclosed gaps, so that useful information is not rejected merely for incompleteness.
17. As a user, I want unsupported claims and undisclosed gaps revised, so that final answers accurately reflect the supplied evidence.
18. As a user, I want revisions bounded separately from Search, so that generation cannot restart an unbounded retrieval cycle.
19. As a user, I want rejected candidates withheld after revision exhaustion, so that adding a disclaimer does not expose unsupported claims.
20. As a user, I want a clear warning for empty model output or failed verification, so that failures do not produce blank responses or internal error codes.
21. As an evaluator, I want before/after selection tables even when reranking is skipped, so that report layouts remain comparable.
22. As an evaluator, I want actual generation context audited separately from the pool, so that metrics describe the evidence the answer model received.
23. As an evaluator, I want metric applicability and sample counts reported, so that skipped failures cannot masquerade as quality improvements.
24. As an evaluator, I want accepted answers from degraded Retrieval runs scored when applicable, so that answer quality and controller reliability can be examined independently.
25. As a maintainer, I want Classic behavior preserved, so that the redesign remains confined to Agentic orchestration and associated evaluation.
26. As an operator, I want revised setting semantics documented without editing private configuration, so that rollout is explicit and reproducible.

## Implementation Decisions

### Architecture and compatibility

- ADR-0008 is the approved design authority. It amends the conflicting direct-first-generation, empty-result, selection, and post-generation recovery decisions in ADR-0005 and ADR-0007. Retain their remaining constraints, including read-only retrieval, query isolation, buffered candidates, and no persisted raw controller prompts or responses.
- Preserve Classic query behavior, shared Hybrid retrieval methods, model configuration, and runtime/evaluation corpus isolation. Agentic's final selection may now differ from Classic; report that difference rather than claiming identical generation contexts.
- Reuse existing query orchestration, structured outcome, callback, settings, and evaluation abstractions. Retire incompatible Agentic replace/reopen/expand action behavior from the new flow rather than retaining a second selectable legacy mode.

### Search and pool

- Use `AGENTIC_MAX_SEARCH_ATTEMPTS` with its existing default of three and existing supported configuration bounds. Each executed Search consumes one attempt, including the initial Search. Validation retries and coverage decisions do not consume Search attempts; failures never authorize an extra Search beyond the cap.
- Preserve original-query first Search and independently configured retrieval limits. `LLM_MAX_CHUNKS` controls generation selection, not the number of candidates returned by each Search.
- Retain every candidate returned by each completed Search in an accumulated per-query pool, deduplicated by `(source, chunk_id)`. Preserve original text, first-seen order, and provenance from every Search that returned the chunk. Do not replace earlier evidence or persist the pool across queries.
- Assess the full accumulated pool after each Search, including the final allowed Search, against the original question. Record `assessed_question`, coverage (`zero_evidence`, `partial_evidence`, `sufficient_evidence`), and missing information. No separate round coverage state is required; preserve per-Search audit records.
- Normal Retrieval termination has exactly two reasons: `sufficient_evidence` and `search_budget_exhausted`. A sufficient assessment takes precedence even on the final allowed Search. Store coverage independently from termination reason.
- While coverage is zero/partial and quota remains, require a legal next Search plan: rewrite query, adjust Vector minimum score and/or query-relative BM25 minimum-score ratio within `[0, 1]`. Focus rewrites on gaps in the original question. Do not let the controller change top-k, reranker model, or generation limits. Invalid/missing plans are degradation, not normal exhaustion; do not label them as proof that the corpus lacks an answer.

### Empty pools and Retrieval degradation

- Only when the accumulated pool is empty and intelligibility is not already established, assess the original question's intelligibility. A confirmed unintelligible question returns a deterministic rephrasing request without Answer generation. An intelligible question continues bounded recovery. Do not repeat a successful intelligibility assessment or run this gate merely because a later Search returns no new chunks while the pool is non-empty.
- An empty final pool returns deterministic text explaining that this search found no usable note content. It consumes no Answer attempt and does not enter generated-answer Ragas averages. Retain a distinct internal outcome from unintelligible input.
- Retrieval reviewers return coverage, missing information, and next-plan data rather than controller-selected evidence references. Invalid decisions receive one validation retry with feedback. Persistent invalidity, provider failure, or inability to fit the complete reviewer input plus reserved output into the model context budget triggers degraded Retrieval termination.
- Keep the pool intact on review context overflow; record `review_context_budget_exceeded`. Do not silently truncate reviewer input while claiming full assessment. Unconfirmed coverage is `unknown`, not zero. Other failures retain sanitized cause categories.
- Degraded Retrieval proceeds to final selection using the available pool. If that pool is empty, use the deterministic empty-pool outcome. Degradation is separately visible and cannot be mislabeled as either normal stop reason.
- Preserve explicit failure outcomes for actual Search or reranker exceptions; they must not be interpreted as successful empty searches or evidence of corpus absence.

### Final context selection

- If pool size exceeds `LLM_MAX_CHUNKS`, rerank the full deduplicated pool against the original question and select only the top configured number. Otherwise skip this final rerank and use first-seen pool order. This final pool-level rerank is distinct from existing within-Search ranking stages.
- Keep the before-selection pool immutable for audit. Record whether final reranking ran, scores where available, selected order, and selected chunk identities. Skipped reranking must not invent new scores.
- Respect the existing answer-input token budget and capture the exact ordered/truncated context actually supplied to generation. Pool sufficiency is not inherited by a reduced context. Reviewer support checks use this actual context.

### Generation and Answer review

- A non-empty selected pool reaches the answer model even when coverage is zero or partial. The model must answer supported portions and disclose missing information, or abstain when none can be supported. Do not use model prior knowledge as note evidence.
- Use the independent `AGENTIC_MAX_ANSWER_ATTEMPTS` setting, preserving its default of three and supported bounds. Count the initial generation and every revision; invalid review retries do not consume another Answer attempt.
- Review against the exact generation context for relevance, completeness relative to available evidence, disclosure of missing information, and factual support. Record `answer_coverage` (`full`, `partial`, `none`) independently from `review_passed`.
- Accept supported full answers, supported partial answers with explicit gaps, and appropriate abstentions. Missing corpus facts alone do not require impossible repeated completions. Unsupported factual claims or undisclosed gaps require same-context revision with actionable feedback.
- Answer review cannot Search, expand evidence, or replace the selected context. Buffer all candidates until accepted. On exhausted Answer attempts retain the internal `answer_attempts_exhausted` outcome and return a fixed warning, never a rejected candidate.
- Invalid review decisions receive one validation retry. Persistent invalidity or provider failure returns a fixed warning with a distinct sanitized review-failure cause. Empty answer-model output and generation provider failures also return a fixed warning with their own internal causes. Do not expose internal failure codes in user-facing text.
- A suitable warning is “本次未能產生通過驗證的答案，請稍後再試。” Preserve project response-language conventions when implementing the exact wording; the warning must not assert evidence absence.

### Evaluation and reporting

- Publish explicit machine-readable pool, selection, coverage, termination, review, and degradation data using the existing evaluation result contract. Evolve the schema/version as needed; Markdown and JSON must agree.
- Agentic query details show two aligned chunk metadata tables: accumulated pool after Retrieval, and selected pool after final rerank/top-k. Include order, chunk identity, source lines, Search provenance, relevant available scores, and expected-evidence line coverage using the report's conventions. Do not repeat raw chunk text in these tables.
- Show both tables even for empty pools or skipped reranking; skipped reranking is labeled explicitly. The second table represents selected chunks, not the full ranked oversized pool. Retain exact raw generation-context text in its existing separate audit section only.
- Preserve detailed per-Search audits. Attribute new Retrieval review latency to actual review calls, separate from Vector/BM25/Fusion/Rerank timings and post-generation Answer review. Measure the final pool-level rerank separately from per-Search reranking. Keep workflow latency independent from Ragas judge latency, and retain provider-reported usage without estimates.
- Compute final Recall, Precision, and NDCG from the exact answer-generation context against reviewed evidence spans. Keep their rows when unavailable. Do not substitute the full pool or pre-generation review prompt as final context.
- All accepted generated final answers, including partial answers and abstentions, enter per-metric Ragas applicability checks. Retrieval degradation alone must not exclude an otherwise accepted answer. Deterministic empty-pool/rephrasing/warning results and unaccepted candidates do not enter accepted-final-answer averages.
- Record `scored`, `unavailable`, or `failed` and a reason per metric; show evaluated-case counts so differing denominators are visible. Preserve the metric-specific requirements for reference answers and actual context; do not fabricate scores for empty contexts or judge failures.
- Report degradation/failure counts and workflow reliability independently of answer metrics. A run with degraded controller execution must not be described as fully reliable merely because its final answer passed review. Artifact completion remains independent from workflow outcome.
- Document the revised roles of the existing Search/Answer settings in the example configuration and maintained guides. Do not edit private environment configuration or benchmark ground truth to improve scores.

## Testing Decisions

- The user approved two existing public seams: `run_query()` results/callbacks and `run_evaluation()` JSON/Markdown artifacts. Prefer these over tests tied to private graph nodes or prompt-string layout. Verify observable behavior and explicit accepted examples, not a copy of implementation logic.
- At the query seam, use deterministic fake providers and controlled Retrieval results. Verify accumulation across searches, deduplication/provenance, first-seen order, original-question coverage, threshold/rewrite plans, no separate round state, sufficient stopping, exhaustion, and exact attempt caps including configurable limits.
- Cover intelligible empty recovery, unintelligible early exit, later empty Search with non-empty pool, empty final pool, invalid-plan retry/degradation, provider failures, reviewer context overflow, and retention of available evidence without false sufficiency.
- Cover pool sizes below, equal to, and above `LLM_MAX_CHUNKS`; verify that final reranking runs only above the limit and uses the original question. Capture the actual generation context through the public result/callback boundary.
- Cover accepted full/partial/abstaining outputs, missing-gap revisions, unsupported-claim revisions, one validation retry, exhausted revisions, empty model output, review/generation failures, and absence of any post-generation Retrieval. Verify rejected candidates are not exposed.
- At the evaluation seam, use controlled query outcomes and judge responses to assert both metadata tables, skip labels, actual-context metrics, distinct latency/usage phases, explicit applicability and counts, degraded-but-accepted scoring, and exclusion of deterministic warnings from accepted-answer averages.
- Prior art includes existing structured query-outcome tests, Agentic controller/callback tests, stage-report integration tests, final-context report tests, and evaluation runner tests. Replace expectations for deliberately superseded behavior rather than preserving conflicting legacy assertions.
- Run relevant focused tests during implementation and the full pytest suite once after integration. Use only configured lint/typechecking tools; report missing tooling and unverified behavior honestly.
- Live-provider acceptance is separate from deterministic tests: compare Classic and Agentic on identical reviewed cases, fixture, seed, models, and retrieval settings. Include the previously failing support-volume question and cases exercising recovery, empty retrieval, partial answers, and revision. Run workflows sequentially to reduce shared-provider timing interference. Record actual outputs, reliability, applicability denominators, latency, and usage; do not infer live quality from fake-provider tests or promise higher Ragas scores.

## Out of Scope

- General-purpose agent tools, external writes, cross-query evidence memory, or persistent controller state.
- Classic workflow redesign, new retrieval methods, changed embedding/reranker models, or controller-controlled top-k.
- Coverage-aware context packing, summary compression, batched pool review, or multi-generation answer synthesis. Final selection deliberately uses conditional original-query rerank and top-k.
- Post-generation evidence expansion or Retrieval reopening, a legacy-mode switch, or guaranteed perfect model judgments.
- Persisting raw controller prompts/responses, changing reviewed ground truth to inflate scores, or treating Ragas as a runtime acceptance gate.
- Ticket creation, code implementation, commits, pushes, merges into main, and destructive cleanup during this specification step.

## Further Notes

- This spec supersedes conflicting behavior from historical direct-generation/post-generation recovery plans for this change only. Retain those artifacts as history; do not modify or close their parent issues.
- Keep current unrelated and pre-existing working-tree changes, including the earlier Search latency-row fix, separate from new implementation work. A pre-existing fix is not evidence that this redesign is implemented.
- Next step: use `to-tickets` to propose independently verifiable vertical slices with genuine blocking edges. Publish tickets only after the user approves the breakdown.
