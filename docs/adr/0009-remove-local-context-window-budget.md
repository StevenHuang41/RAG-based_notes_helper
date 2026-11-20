# Remove the local context-window budget

Status: Accepted — implemented locally; deterministic verification and live-provider limitations are recorded in the feature verification report.

Amended by [ADR-0011](0011-evidence-gated-answer-generation.md) for Retrieval-review failure handling and selected-context coverage assessment.

Remove `LLM_CONTEXT_WINDOW_TOKENS` and its runtime effects rather than replace it with model-capacity discovery. The current UTF-8-byte estimate can discard evidence or reject review before the provider is called. Provider capacity remains enforced by the deployment/provider; preserve existing provider-failure handling.

## Agreed design direction

- Remove the setting, its config validation, local prompt-budget checks, budget-driven history/chunk trimming, and its evaluation metadata field. Preserve the independent output limit `LLM_MAX_TOKENS`.
- Retain at most five complete user/assistant history turns, without budget-driven trimming. History clarifies intent and is never note evidence.
- Keep the original, deduplicated accumulated chunks pool with provenance across searches.
- Preserve the existing complete-pool review after each Search, assessed against the original question rather than only the rewritten Search query. Evidence from different searches can jointly establish sufficient coverage.
- Introduce no batch review, evidence ledger, or model-generated evidence extraction. These alternatives were considered but rejected for this change to avoid introducing an additional cross-batch evidence integration mechanism.
- Preserve early stopping for sufficient evidence and bounded adaptive Search for missing information. Retrieval review degradation continues through the existing fallback.
- Retain final pool reranking/selection and `LLM_MAX_CHUNKS` as the Answer generation chunk-count limit. Do not additionally drop selected chunks based on a local context budget. Answer review and revisions use the same exact generation evidence.
- Preserve existing provider-failure behavior: degraded Retrieval review may continue to final selection; failed Answer generation/review returns the existing safe warning.
- If Retrieval review fails, retain the complete raw pool, set coverage to `unknown`, stop adaptive Search and continue through existing degraded Retrieval fallback to final selection. Removing the local guard does not guarantee the deployment will accept the full input or process it without truncation.

## Relationship to existing decisions

This amends ADR-0008's local `review_context_budget_exceeded` behavior while preserving complete-pool review and the existing provider-failure fallback. ADR-0008's remaining decisions stay accepted. Evidence-pool coverage can still differ from final generation-context coverage after top-k selection.

## Implementation and acceptance scope

- Remove obsolete prompt-budget helpers, errors, imports and runtime branches where no other use remains. Keep token-usage reporting based on provider usage intact.
- Remove the setting from `.env.example` and maintained configuration documentation; preserve the private `.env`. Leftover environment values are ignored through the existing settings extra-field policy.
- Remove `context_window_tokens` from newly produced evaluation metadata. Preserve historical evaluation files and avoid making their extra fields a read-time error.
- Update affected tests and ADR documentation coherently. Verify that inputs exceeding the old local estimate reach the mocked provider, selected generation chunks are retained across revisions/review, the five-turn history limit remains, and provider-failure fallback remains intact.
- Run focused tests through `uv run pytest`, expanding coverage if shared-interface changes require it. Live-provider acceptance remains distinct from deterministic tests.
