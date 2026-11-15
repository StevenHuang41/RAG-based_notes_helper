# Remove the local context-window budget

Status: resolved

## Problem Statement

The application uses `LLM_CONTEXT_WINDOW_TOKENS` and a UTF-8-byte-based prompt estimate to trim conversation history, remove selected generation chunks, or reject controller review before calling the provider. This local setting does not increase or configure the provider's actual model capacity. When production models change, a fixed local estimate can unnecessarily withhold evidence or degrade review even when the deployment could accept the request.

Users want generation chunk selection controlled by `LLM_MAX_CHUNKS`, while retaining the existing bounded workflow, cross-search evidence accumulation, exact-context Answer review, and provider-failure handling.

## Solution

Remove `LLM_CONTEXT_WINDOW_TOKENS` and all of its local prompt-budget effects without introducing a replacement capacity setting or automatic model-capacity discovery. Submit the complete accumulated chunks pool to Retrieval review against the original question after each Search. Preserve early stopping and bounded adaptive Search. Select generation evidence using the existing conditional reranking and `LLM_MAX_CHUNKS`, then retain that evidence throughout Answer generation, review, and revisions.

Keep at most five complete conversation turns and the separate output cap `LLM_MAX_TOKENS`. Provider/deployment limits still apply; failures follow existing behavior. This change does not guarantee that every model accepts the full input or that a provider never truncates input.

## User Stories

1. As an operator, I want to change models without maintaining a separate local context-window estimate, so that a stale setting does not constrain the new model.
2. As a user, I want selected chunks submitted in full, so that a byte-based estimate does not remove evidence before generation.
3. As an operator, I want `LLM_MAX_CHUNKS` to control generation chunk count, so that evidence selection remains predictable.
4. As a user, I want findings accumulated across searches, so that later queries do not discard earlier partial evidence.
5. As a user, I want the entire accumulated pool assessed against my original question, so that complementary evidence across searches can establish sufficiency.
6. As a user, I want sufficient evidence to stop Search early, so that unnecessary retrieval does not consume the remaining budget.
7. As a user, I want missing information to guide bounded adaptive Search, so that recovery remains focused.
8. As an operator, I want original text, deduplication and Search provenance preserved, so that evidence remains auditable.
9. As a user, I want final pool reranking when the pool exceeds the generation limit, so that generation receives the existing ranked selection.
10. As an operator, I want final reranking skipped when the pool already fits, so that this change does not add model or ranking calls.
11. As a user, I want revisions and Answer review to use the same selected evidence, so that review assesses what generation actually received.
12. As a REPL user, I want my most recent five complete conversation turns retained without local budget trimming, so that follow-up intent remains available.
13. As an operator, I want the separate output-token limit preserved, so that removing the input estimate does not remove output control.
14. As a user, I want failed Retrieval review to preserve available evidence and use existing fallback, so that provider failure does not erase the pool.
15. As a user, I want Answer generation/review failures to return the existing safe warning, so that unreviewed candidates are not exposed.
16. As an evaluator, I want exact generation-context metrics and provider usage preserved, so that evidence and execution remain auditable.
17. As an evaluator, I want obsolete context-window metadata omitted from new results, so that a removed local estimate is not mistaken for model capacity.
18. As an operator, I want leftover private environment values harmlessly ignored, so that rollout does not require editing private configuration.
19. As a maintainer, I want historical evaluation artifacts retained and readable by existing supported readers, so that removing a field does not invalidate earlier observations.
20. As a maintainer, I want regression tests around existing public workflow seams, so that removal is verified without redesigning the application.

## Implementation Decisions

### Settings and prompt handling

- Remove the context-window setting and its cross-field validation. A leftover `LLM_CONTEXT_WINDOW_TOKENS` in the process environment or private environment file must have no effect and must not prevent startup, using the existing extra-field settings behavior.
- Preserve `LLM_MAX_CHUNKS`, `LLM_MAX_TOKENS`, model/provider configuration and independently configured Search/Answer attempt budgets.
- Remove the local prompt-token estimate, budget-fitting helper, obsolete imports, context-budget exception and budget-only call arguments where no remaining purpose exists. Do not remove history types or serialization still used by prompts.
- All applicable Answer and reviewer prompts retain the most recent five complete user/assistant pairs. Remove size-driven trimming while retaining this independent turn-count policy.
- Do not replace the setting with another input cap, model-capacity registry, tokenizer, provider token-count call, auto-truncation request option, or Ollama context option. Provider request parameters otherwise retain existing semantics.

### Retrieval and generation

- Preserve the current Hybrid retrieval pipeline, thresholds, per-stage limits, original-query first Search, deduplication and cumulative provenance. `LLM_MAX_CHUNKS` is not a per-Search retrieval limit.
- After each Search, assess the complete accumulated original chunks pool against the original question, including on the final permitted Search. Rewritten queries target missing information but never redefine overall success.
- Preserve sufficient-evidence early stopping, Search exhaustion, invalid-decision validation retry, empty-pool/intelligibility behavior and query isolation.
- Introduce no batch review, agent filtering, ledger, evidence extraction or summary compression. Earlier discussion of these alternatives is superseded by the user's explicit choice of the minimum removal change.
- Preserve final original-question reranking only when the pool exceeds `LLM_MAX_CHUNKS`, and select at most that number. Remove any additional budget-driven deletion from the selected generation context.
- Answer review and all revisions use the same ordered evidence, original text and provenance as the initial generation. Revision feedback may change; evidence selection does not.
- Preserve Classic retrieval, empty-result behavior and streaming semantics, with local budget-driven history trimming removed from the shared Answer path.

### Failures, evaluation and compatibility

- Remove runtime production of local `review_context_budget_exceeded` and `generation_context_budget_exceeded` causes once their checks are removed. Historical artifacts may still contain these causes; do not rewrite them or add a reader rejection for them.
- Retrieval reviewer provider failure retains the pool, records unknown coverage and existing degraded execution, stops further adaptive Search and continues to final selection. Preserve existing distinction between failure and normal Search exhaustion.
- Answer generation or review provider failure retains existing structured failure and safe-warning behavior. Preserve actual attempted generation-context audit metadata on failure and do not expose raw provider errors or rejected candidates.
- Do not introduce overflow-specific automatic retries, fallback model switching or silent application-side truncation. A provider capacity error follows its existing provider-failure path; do not claim actual model capacity or consumption was verified by this application.
- Remove `context_window_tokens` from newly produced evaluation settings metadata. Preserve actual-context Recall/Precision/NDCG, provider-reported token usage, latency accounting, applicability, workflow degradation and reliability reporting.
- Preserve historical report files. Retain existing reader/rendering tolerance for extra settings fields and old cause strings where such readers exist; no new historical-reader subsystem is required.
- Update example configuration and maintained documentation to describe the new behavior. Amend ADR-0008's local-budget rule through ADR-0009, retaining its remaining accepted decisions. Preserve historical specs/tickets as history and preserve the user's private environment file.
- The current checkout includes substantial uncommitted two-phase workflow changes. Treat its inspected implementation as the baseline and preserve those changes; do not restore an earlier committed workflow to simplify removal.

## Testing Decisions

- Prefer observable behavior through existing `run_query()` result/callback boundaries with deterministic fake providers and controlled Search results. Capture provider inputs to verify evidence and history actually submitted, rather than testing private graph-node structure or exact prompt formatting.
- Use the existing Settings seam for removal compatibility and `run_evaluation()` JSON/Markdown artifacts for evaluation behavior. These proposed seams were presented for user confirmation during spec creation; confirmation does not imply tests have been executed.
- Verify that a prompt exceeding the former default byte estimate reaches Retrieval review and that full accumulated evidence from different searches is present with the original question. Arrange complementary evidence and a sufficient controller response to verify early stopping after the second Search.
- Verify below/equal/above `LLM_MAX_CHUNKS` selection, unchanged conditional reranking, all selected chunks reaching generation without budget trimming, and exact evidence preserved through Answer review and a revision.
- Verify long history content retains the newest five complete turns and excludes older turns. Verify the configured output cap still reaches existing provider adapters through their established tests where affected.
- Simulate provider capacity rejection using existing failure seams: Retrieval review degrades while retaining the pool; Answer generation/review returns existing safe warnings and structured causes. These tests establish application fallback, not universal live-provider overflow behavior.
- Verify legacy environment values have no effect, no replacement setting is required, new evaluation metadata omits the removed field, and actual generation-context audit/metrics remain intact.
- Replace tests asserting deliberate budget trimming or local budget rejection with the new external behavior. Remove obsolete fixture settings; retain unrelated provider-failure and historical-rendering coverage.
- Prior art is the existing settings, Answer history, Agentic controller, two-phase Answer/retrieval, CLI outcome and evaluation/report suites.
- During implementation, run focused tests with `uv run pytest`; broaden to shared query/evaluation suites as warranted. Run configured lint/type checks when available and relevant. Record actual commands, failures and unverified live behavior. This specification step executes no runtime acceptance tests.

## Out of Scope

- Implementation, ticket publication, commits, pushes, merges or cleanup during this specification step.
- Batch review, evidence ledger, agent chunk filtering, coverage-aware context packing or multi-call answer synthesis.
- Automatic capacity discovery, a new input-budget setting, tokenizer dependencies or provider/runtime context provisioning.
- Changes to chunking, embeddings, retrieval algorithms, ranking models, Search/Answer attempt limits or model selection.
- Removing the five-turn history policy or output-token cap; changing existing provider error presentation or streaming behavior.
- Changing evaluation ground truth, runtime corpus/index artifacts or historical evaluation results. Evaluation remains isolated from runtime data/storage.
- Guarantees of full provider-side context retention, exact model judgment or improved answer scores.

## Further Notes

- ADR-0009 records the proposed architectural amendment; specification readiness is not implementation completion or live-provider verification.
- This spec supersedes conflicting local-budget requirements in the earlier two-phase design only for this change. Its complete-pool review and generation-selection architecture remain in effect.
- Follow the existing local issue-tracker convention for subsequent work. If tickets are requested, propose independently verifiable slices and obtain approval of the breakdown before publishing them.

## Comments

- Implemented on `integration/remove-local-context-window-budget` relative to isolated baseline `09e0a44`. All three implementation tickets are resolved. Full deterministic suite: 274 passed; Standards and Spec reviews: zero findings each. See the feature verification report for exact commands, baseline provenance, cleanup and live-provider limitations.
