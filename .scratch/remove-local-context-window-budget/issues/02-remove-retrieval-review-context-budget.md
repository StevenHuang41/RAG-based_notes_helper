# 02: Review the complete accumulated pool without a local budget

**What to build:** After each Search, send the complete accumulated original chunks pool to Retrieval review against the original question without local prompt-budget rejection. Preserve cross-search evidence accumulation, sufficient-evidence early stopping, bounded adaptive Search and existing degraded provider-failure fallback.

**Blocked by:** None (can start immediately).

Status: resolved

## Acceptance criteria

- [x] Retrieval review no longer rejects input based on `LLM_CONTEXT_WINDOW_TOKENS` or a local prompt-size estimate. This also applies to any intelligibility request using the same budget-checking path.
- [x] Each completed Search is followed by the existing full-pool review against the original question, including the final allowed Search. All accumulated chunks retain original text, deduplication, order and Search provenance.
- [x] Demonstrate complementary evidence across two Searches: first review reports partial coverage and requests a focused rewrite; second review sees evidence from both Searches against the original question and reports sufficiency, stopping before the remaining Search budget is consumed.
- [x] A Retrieval review prompt exceeding the former local default estimate reaches the fake provider with the complete pool. Do not introduce batching, filtering, a ledger, summary compression or a substitute input cap.
- [x] Preserve independently configured retrieval-stage limits, filtering thresholds, original-query first Search, Search attempt counting, invalid-decision validation retries, query isolation and existing empty-pool/intelligibility outcomes.
- [x] Simulated provider rejection records unknown coverage and existing degraded execution, retains the full pool, stops adaptive Search and proceeds to final selection. Do not mislabel degradation as normal exhaustion or sufficiency.
- [x] Preserve conditional final reranking against the original question and `LLM_MAX_CHUNKS` generation selection. This setting does not become a per-Search limit or review batch limit.
- [x] Update affected tests at existing query/result/callback seams to capture provider inputs and observable stopping/fallback behavior rather than private node structure. Remove expectations that local review overflow prevents the provider call.
- [x] Run affected focused tests through `uv run pytest`; record actual commands, results and unverified live-provider behavior.

## Integration constraints

- Leave settings/metadata removal and unused shared helper cleanup to ticket 03. Preserve helpers still used by Answer until ticket 01 lands.
- This slice must be verifiable with the old Answer budget still present. Use test settings allowing downstream Answer generation, or existing controlled Answer seams, when isolating Retrieval changes.
- Coordinate edits in the shared controller module with ticket 01; shared files do not create a functional dependency between these independently testable behaviors.
- Preserve existing uncommitted work. Do not change output caps, provider provisioning, attempt budgets or user-facing failure presentation.

## Comments

- User approved this three-ticket breakdown and its blocking edges before publication. Provider limits may still reject or truncate input; removing the local check does not verify full model consumption.

- Implemented: remove the shared Retrieval/intelligibility local budget guard; preserve all existing review, ranking and fallback behavior.
- TDD red: `uv run --offline --no-sync pytest tests/test_two_phase_agentic.py::test_query_reviews_large_complementary_pool_and_stops_after_second_search` failed because the second complete-pool review never reached the provider (only one captured review).
- TDD green: the same command passed after guard removal (1 passed).
- Focused verification: `uv run --offline --no-sync pytest tests/test_two_phase_agentic.py` passed (22 tests).
- Related query verification: `uv run --offline --no-sync pytest tests/test_two_phase_agentic.py tests/test_agentic.py tests/test_cli_agentic.py tests/test_query_outcomes.py` passed (69 tests). Tests used `UV_CACHE_DIR=/private/tmp/rag-context-uv-cache` and the existing shared `UV_PROJECT_ENVIRONMENT`; no private environment file was copied.
- Added large cross-search complementary-evidence coverage with original-question assessment, full text, deduplication/provenance and round-two sufficient stopping. Replaced local overflow expectations with provider-reported capacity failure and verified unknown coverage, preserved pool and bounded final selection. Long-history intelligibility review also reaches the provider.
- `git diff --check` passed. No configured typechecker/linter was found; none was installed. Pytest reported one third-party LangChain deprecation warning. No live-provider capacity or truncation behavior was tested.
