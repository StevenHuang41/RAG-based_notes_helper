# 02: Review the complete accumulated pool without a local budget

**What to build:** After each Search, send the complete accumulated original chunks pool to Retrieval review against the original question without local prompt-budget rejection. Preserve cross-search evidence accumulation, sufficient-evidence early stopping, bounded adaptive Search and existing degraded provider-failure fallback.

**Blocked by:** None (can start immediately).

Status: ready-for-agent

## Acceptance criteria

- [ ] Retrieval review no longer rejects input based on `LLM_CONTEXT_WINDOW_TOKENS` or a local prompt-size estimate. This also applies to any intelligibility request using the same budget-checking path.
- [ ] Each completed Search is followed by the existing full-pool review against the original question, including the final allowed Search. All accumulated chunks retain original text, deduplication, order and Search provenance.
- [ ] Demonstrate complementary evidence across two Searches: first review reports partial coverage and requests a focused rewrite; second review sees evidence from both Searches against the original question and reports sufficiency, stopping before the remaining Search budget is consumed.
- [ ] A Retrieval review prompt exceeding the former local default estimate reaches the fake provider with the complete pool. Do not introduce batching, filtering, a ledger, summary compression or a substitute input cap.
- [ ] Preserve independently configured retrieval-stage limits, filtering thresholds, original-query first Search, Search attempt counting, invalid-decision validation retries, query isolation and existing empty-pool/intelligibility outcomes.
- [ ] Simulated provider rejection records unknown coverage and existing degraded execution, retains the full pool, stops adaptive Search and proceeds to final selection. Do not mislabel degradation as normal exhaustion or sufficiency.
- [ ] Preserve conditional final reranking against the original question and `LLM_MAX_CHUNKS` generation selection. This setting does not become a per-Search limit or review batch limit.
- [ ] Update affected tests at existing query/result/callback seams to capture provider inputs and observable stopping/fallback behavior rather than private node structure. Remove expectations that local review overflow prevents the provider call.
- [ ] Run affected focused tests through `uv run pytest`; record actual commands, results and unverified live-provider behavior.

## Integration constraints

- Leave settings/metadata removal and unused shared helper cleanup to ticket 03. Preserve helpers still used by Answer until ticket 01 lands.
- This slice must be verifiable with the old Answer budget still present. Use test settings allowing downstream Answer generation, or existing controlled Answer seams, when isolating Retrieval changes.
- Coordinate edits in the shared controller module with ticket 01; shared files do not create a functional dependency between these independently testable behaviors.
- Preserve existing uncommitted work. Do not change output caps, provider provisioning, attempt budgets or user-facing failure presentation.

## Comments

- User approved this three-ticket breakdown and its blocking edges before publication. Provider limits may still reject or truncate input; removing the local check does not verify full model consumption.
