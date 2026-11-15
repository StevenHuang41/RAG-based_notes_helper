# 03: Retire the context-window setting and integrate evaluation and documentation

**What to build:** Complete removal of the obsolete context-window configuration after both workflow paths stop using it. Startup tolerates leftover private environment values, new evaluation artifacts omit obsolete metadata, and maintained documentation describes complete-pool review, generation chunk selection and provider-failure fallback accurately.

**Blocked by:** 01 — Remove local budgets from Answer generation and review; 02 — Review the complete accumulated pool without a local budget.

Status: ready-for-agent

## Acceptance criteria

- [ ] Remove the `LLM_CONTEXT_WINDOW_TOKENS` settings field and cross-field validation after confirming no active runtime caller remains. Leave `LLM_MAX_CHUNKS`, `LLM_MAX_TOKENS`, model/provider and attempt-budget settings intact.
- [ ] Existing private environment-file and process-environment values for the removed setting have no effect and do not prevent startup, using existing settings extra-field behavior. Do not edit the user's private environment file.
- [ ] Remove unused byte-based estimation, budget-fitting helpers, budget-only arguments, errors, imports, fixture settings and obsolete branches. Preserve conversation serialization, five-turn policy and provider-reported usage accounting.
- [ ] New evaluation settings metadata omits `context_window_tokens`. Exact generation-context audit, Recall/Precision/NDCG, latency, token usage, applicability, degradation and reliability remain coherent in JSON and Markdown.
- [ ] Preserve historical artifacts and existing supported readers/renderers' tolerance for old extra settings fields and old budget-failure causes. Active workflows no longer generate local context-budget failure causes. Do not build a new historical-reader subsystem.
- [ ] Remove the setting from example configuration and update maintained configuration/workflow documentation. Finalize ADR-0009 consistently with the approved spec and annotate ADR-0008's superseded local-budget decision while retaining its other decisions and preserving historical specs/tickets.
- [ ] Verify integrated below/equal/above-generation-limit selection behavior, five-turn history with long messages, complete accumulated pool review, unchanged evidence through revision/review and existing provider-failure fallback.
- [ ] Use existing Settings tests for leftover-env compatibility and existing evaluation JSON/Markdown seams for metadata, audit and historical-rendering compatibility; prefer existing public query seams for integrated behavior.
- [ ] Run relevant settings, query, Answer, Agentic, CLI and evaluation/report tests through `uv run pytest`. Broaden testing based on changed shared interfaces or unresolved failures; run available relevant configured lint/type checks and report unavailable tooling honestly.
- [ ] Record exact verification commands, failures and unverified live behavior. Deterministic fake-provider tests must not be reported as live model acceptance or a guarantee against provider-side truncation.

## Integration constraints

- Both blockers must complete before removing the shared settings/helper contract. Preserve pre-existing uncommitted two-phase workflow changes and unrelated edits.
- Introduce no replacement input setting, batch review, ledger, tokenizer/model-capacity discovery or overflow-specific retries. Keep runtime/evaluation corpus isolation and reviewed ground truth unchanged.
- Preserve user data and historical evaluation results. Commits, pushes, merges and cleanup require their own authorization.

## Comments

- User approved this three-ticket breakdown and its blocking edges before publication. This ticket contracts the shared configuration only after both independently verified workflow migrations are complete.
