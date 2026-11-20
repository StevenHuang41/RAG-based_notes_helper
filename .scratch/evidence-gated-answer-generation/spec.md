# Evidence-gated answer generation

Status: ready-for-agent

## Problem Statement

The current answer workflow uses a fixed `ABSTENTION_MESSAGE` both as prompt text and as a classification signal. This couples answerability reporting to exact model wording. Agentic RAG can also send chunks to Answer generation after Retrieval review judges the accumulated chunks pool to have zero evidence, or after review fails and coverage is unknown. A user needs an honest outcome when a search finds no chunks or the controller finds no supporting evidence, without paying for an answer generated from unsupported context.

## Solution

Classic RAG returns a fixed, helpful response for an empty Retrieval search and otherwise generates directly from retrieved chunks. Agentic RAG returns that kind of response immediately when its first Retrieval search is empty. When chunks are present, it uses bounded Adaptive search and controller Evidence coverage assessments; it generates only when the exact Answer generation context has at least partial evidence. Generated answers use natural language, and Agentic Answer review checks them before they reach the user. A failed Retrieval review is reported as a workflow error, not as a lack of evidence.

## User Stories

1. As a notes user, I want an immediate response when Classic RAG finds no chunks, so that I know the search produced no results.
2. As a notes user, I want the same immediate behavior when the first Agentic RAG search finds no chunks, so that an empty result does not trigger unnecessary model calls.
3. As a notes user, I want the empty-result response to suggest adjusting my question, so that I can try another search.
4. As a notes user, I want that response to describe the search outcome, so that it does not claim my notes contain no answer.
5. As a Classic RAG user, I want retrieved chunks sent directly to Answer generation, so that the workflow remains a single-search baseline.
6. As a Classic RAG user, I want the Answer model to describe missing information naturally, so that its response is not constrained to a fixed sentence.
7. As an Agentic RAG user, I want the controller to assess the accumulated chunks pool against my original question, so that evidence from multiple searches can be considered together.
8. As an Agentic RAG user, I want another Adaptive search when evidence is missing and Search attempts remain, so that a revised search can recover relevant evidence.
9. As an Agentic RAG user, I want Search attempts bounded, so that the workflow eventually returns an outcome.
10. As an Agentic RAG user, I want zero-evidence chunks withheld from Answer generation after Search is exhausted, so that irrelevant chunks are not treated as a basis for an answer.
11. As an Agentic RAG user, I want a fixed no-related-information response when zero evidence remains, so that I receive a clear outcome.
12. As an Agentic RAG user, I want partial evidence passed to Answer generation, so that I can receive supported portions of an answer.
13. As an Agentic RAG user, I want sufficient evidence passed to Answer generation, so that the model can answer my question.
14. As an Agentic RAG user, I want the controller to assess the exact selected generation context, so that evidence discarded by selection is not assumed to reach the Answer model.
15. As an Agentic RAG user, I want partial answers to disclose missing information, so that I can tell what the notes support.
16. As an Agentic RAG user, I want Answer review to check generated answers against their actual context, so that unsupported claims or undisclosed gaps can be revised.
17. As an Agentic RAG user, I want revisions to remain bounded and use the same context, so that answer improvement does not restart Retrieval indefinitely.
18. As an Agentic RAG user, I want rejected answers withheld when review or revision fails, so that I do not mistake an unreviewed candidate for an accepted answer.
19. As an Agentic RAG user, I want a review failure reported as an error, so that a provider problem is not presented as missing evidence.
20. As an evaluation reviewer, I want deterministic empty and zero-evidence outcomes distinguished from workflow errors and generated answers, so that I can interpret reliability and answer quality correctly.
21. As an evaluation reviewer, I want Classic answerability marked unknown when a free-form response cannot be reliably classified, so that the report does not guess whether the model abstained.
22. As an evaluation reviewer, I want Agentic model abstention identified by Answer review rather than exact wording, so that paraphrases are classified consistently.
23. As an evaluation reviewer, I want retrieval traces, ordered generation context, and usage preserved, so that I can audit what each workflow actually did.
24. As a maintainer, I want the fixed answer-model abstention constant and text comparisons removed, so that prompt wording and evaluation classification are not coupled.
25. As a maintainer, I want the accepted architectural decisions reconciled with the new evidence gate, so that future changes follow the current behavior.

## Implementation Decisions

- Preserve the shared first Retrieval search and the existing independent Search and Answer attempt limits. A Search budget counts executed searches, including the first; controller calls and validation retries do not consume Search attempts.
- If the first Search yields zero chunks, both workflows return a deterministic response saying no related information was found in the search and suggesting a more suitable question. Do not claim the corpus lacks an answer or that the question is unintelligible. Do not call the Answer model; Agentic also skips controller review and additional searches.
- Classic sends nonempty results to the Answer model without a controller or retry budget. Prompts require grounding in Context and a natural explanation when information is missing; they do not prescribe an exact abstention sentence. Preserve Classic streaming and the existing safe handling of blank or failed responses.
- Agentic reviews the complete accumulated chunks pool against the original question after each nonempty-initial search. Preserve chunk deduplication, provenance, retrieval audits, configurable thresholds, and early stopping for `sufficient_evidence`. For `zero_evidence` or `partial_evidence`, use remaining Search attempts according to the controller's valid next-search plan. Keep coverage separate from the reason Search stopped.
- After Search finishes, `zero_evidence` returns a deterministic no-related-information outcome without an Answer attempt. Retain retrieved evidence and review trace for audit, but leave Answer generation context empty. Do not classify a model judgment as ground truth.
- For a pool judged `partial_evidence` or `sufficient_evidence`, retain existing final reranking and Generation top-k selection. Before generation, have the controller assess the exact ordered selected context against the original question when selection can have removed pool evidence; reuse the pool assessment when selected context is identical to the assessed pool. The final assessment does not reopen Search. If it is `zero_evidence`, return the deterministic no-related-information outcome; if partial or sufficient, pass that exact context to Answer. Record pool and selected-context assessments distinctly in the audit.
- Coverage review and selected-context review must use validated controller output and the existing bounded validation retry. Provider failure or exhausted invalid output produces a distinct workflow-error outcome with a safe user-facing message; preserve the error category and retrieved evidence, do not claim zero evidence, and do not call Answer. Unknown coverage is never accepted as evidence.
- Preserve independent bounded Answer review and revision using the exact same generation context. Accept supported partial answers that disclose gaps; do not return rejected candidates on failure or exhaustion. Keep accepted Answer-model abstention as an Answer review decision, including naturally worded responses.
- Remove `ABSTENTION_MESSAGE` and all behavior depending on exact answer-text equality. Use structured workflow outcomes and Answer review decisions for Agentic classification. For Classic generated text, report answerability and abstention as unknown rather than infer either from wording; keep accepted generated answers eligible for applicable quality metrics independently of that unknown classification.
- Preserve the distinction between retrieval-empty outcomes, controller-determined insufficient evidence, and workflow errors in reason codes, response origin, reliability accounting, and reports. Deterministic outcomes and errors do not enter generated-answer Ragas averages; accepted generated answers follow existing per-metric applicability rules. Preserve historic result compatibility.
- Update maintained workflow and evaluation documentation and reconcile the relevant accepted ADRs: the new initial-empty behavior, zero-evidence gate, selected-context assessment, and review-failure outcome amend the conflicting decisions in ADR-0008 and ADR-0009. Preserve their other applicable decisions, including full-pool review and no local context-window budget.
- Preserve existing user changes and evaluation isolation: evaluation uses the reviewed QA set and evaluation corpus with temporary indexes, never runtime notes or persistent storage.

## Testing Decisions

- Good tests assert externally observable query outcomes, calls at the workflow boundary, actual generation context, and evaluation records rather than internal graph node layout or exact prompt formatting.
- Use the existing Classic and Agentic query workflow entry points as the primary seam, with fake Retrieval and providers. Existing query outcome, two-phase Retrieval and Answer tests provide prior art; do not add a lower-level seam unless necessary.
- Cover first-search empty for both modes, including absence of controller, additional Search, and Answer calls for Agentic. Cover nonempty Classic direct generation and free-form unsupported answers.
- Cover zero coverage followed by successful Adaptive search, exhausted zero coverage, partial and sufficient coverage, and preservation of accumulated evidence and Search provenance. Verify no Answer call for the zero-evidence outcome.
- Cover a selected context whose coverage differs from the accumulated pool; assert only reviewed partial or sufficient selected evidence reaches Answer, in its original selected order. Cover invalid review after bounded retry and provider failure, including separate workflow-error reporting and no Answer call.
- Cover supported partial answers, missing-information disclosure, same-context Answer revisions, accepted natural-language abstentions, and exhausted or failed review without exposing a rejected answer.
- Use the existing evaluation runner as the second seam. Verify unknown Classic answerability, distinct Agentic outcomes and error reliability, metric eligibility and skip reasons, and truthful evidence/context and usage reporting. Existing two-phase evaluation and report tests provide prior art.
- Run focused tests with `uv run pytest`, followed by the complete suite because shared outcome and evaluation interfaces change. Distinguish deterministic fake-provider tests from any separately executed live-provider acceptance.

## Out of Scope

- Changing retrieval algorithms, note corpus contents, runtime index data, or benchmark ground truth.
- Adding a Classic controller, answerability classifier, or retry budget.
- Adding unbounded retries, cross-query memory, external observability services, or new model dependencies.
- Guaranteeing controller judgments, answer quality, or retrieval recall; no model assessment proves the corpus lacks an answer.
- Creating implementation tickets or implementing the feature as part of publishing this spec.

## Further Notes

This spec records the agreed direction, not an assertion that the current checkout implements it. In particular, the current Agentic retrieval-review failure fallback and initial empty-search behavior differ from the desired behavior. Existing local modifications must be inspected and preserved during implementation.
