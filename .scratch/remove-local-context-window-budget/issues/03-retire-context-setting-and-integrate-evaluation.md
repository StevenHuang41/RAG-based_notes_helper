# 03: Retire the context-window setting and integrate evaluation and documentation

**What to build:** Complete removal of the obsolete context-window configuration after both workflow paths stop using it. Startup tolerates leftover private environment values, new evaluation artifacts omit obsolete metadata, and maintained documentation describes complete-pool review, generation chunk selection and provider-failure fallback accurately.

**Blocked by:** 01 — Remove local budgets from Answer generation and review; 02 — Review the complete accumulated pool without a local budget.

Status: resolved

## Acceptance criteria

- [x] Remove the `LLM_CONTEXT_WINDOW_TOKENS` settings field and cross-field validation after confirming no active runtime caller remains. Leave `LLM_MAX_CHUNKS`, `LLM_MAX_TOKENS`, model/provider and attempt-budget settings intact.
- [x] Existing private environment-file and process-environment values for the removed setting have no effect and do not prevent startup, using existing settings extra-field behavior. Do not edit the user's private environment file.
- [x] Remove unused byte-based estimation, budget-fitting helpers, budget-only arguments, errors, imports, fixture settings and obsolete branches. Preserve conversation serialization, five-turn policy and provider-reported usage accounting.
- [x] New evaluation settings metadata omits `context_window_tokens`. Exact generation-context audit, Recall/Precision/NDCG, latency, token usage, applicability, degradation and reliability remain coherent in JSON and Markdown.
- [x] Preserve historical artifacts and existing supported readers/renderers' tolerance for old extra settings fields and old budget-failure causes. Active workflows no longer generate local context-budget failure causes. Do not build a new historical-reader subsystem.
- [x] Remove the setting from example configuration and update maintained configuration/workflow documentation. Finalize ADR-0009 consistently with the approved spec and annotate ADR-0008's superseded local-budget decision while retaining its other decisions and preserving historical specs/tickets.
- [x] Verify integrated below/equal/above-generation-limit selection behavior, five-turn history with long messages, complete accumulated pool review, unchanged evidence through revision/review and existing provider-failure fallback.
- [x] Use existing Settings tests for leftover-env compatibility and existing evaluation JSON/Markdown seams for metadata, audit and historical-rendering compatibility; prefer existing public query seams for integrated behavior.
- [x] Run relevant settings, query, Answer, Agentic, CLI and evaluation/report tests through `uv run pytest`. Broaden testing based on changed shared interfaces or unresolved failures; run available relevant configured lint/type checks and report unavailable tooling honestly.
- [x] Record exact verification commands, failures and unverified live behavior. Deterministic fake-provider tests must not be reported as live model acceptance or a guarantee against provider-side truncation.

## Integration constraints

- Both blockers must complete before removing the shared settings/helper contract. Preserve pre-existing uncommitted two-phase workflow changes and unrelated edits.
- Introduce no replacement input setting, batch review, ledger, tokenizer/model-capacity discovery or overflow-specific retries. Keep runtime/evaluation corpus isolation and reviewed ground truth unchanged.
- Preserve user data and historical evaluation results. Commits, pushes, merges and cleanup require their own authorization.

## Comments

- User approved this three-ticket breakdown and its blocking edges before publication. This ticket contracts the shared configuration only after both independently verified workflow migrations are complete.

- Implemented removal of the setting/validator, obsolete byte-estimation/history-fitting helpers, remaining budget-only import and controller cause, stale test settings and new evaluation metadata field. The five-turn history serializer and provider-reported usage remain intact.
- TDD Settings red: `uv run --offline --no-sync pytest tests/test_eval_settings.py::test_removed_context_window_env_is_ignored` failed for both `1` and `not-a-token-count` (old cross-field/integer validation). Green after removing the field/validator: 2 passed.
- TDD evaluation red: `uv run --offline --no-sync pytest tests/test_two_phase_evaluation.py` failed because new run metadata still included `context_window_tokens`. Green after removing the writer field: 1 passed.
- Added startup compatibility coverage for invalid leftover values in a temporary environment file and retained historical report rendering with both old metadata and an old local-budget degradation cause.
- Focused integration verification: `uv run --offline --no-sync pytest tests/test_eval_settings.py tests/test_two_phase_evaluation.py tests/test_two_phase_report.py tests/test_answer.py tests/test_agentic.py tests/test_two_phase_agentic.py tests/test_two_phase_answers.py tests/test_cli_agentic.py tests/test_query_outcomes.py tests/test_eval_runner.py` passed (154 tests).
- All test commands used `UV_CACHE_DIR=/private/tmp/rag-context-uv-cache` and the existing shared `UV_PROJECT_ENVIRONMENT`. No private environment file was copied or edited. One third-party LangChain deprecation warning remained.
- `git diff --check` passed. Source search found no active setting, estimator/fitting helper, fixed-context budget argument or local-budget cause references. Historical documentation/test references are deliberate compatibility evidence.
- Example configuration and README now describe complete-pool review, chunk selection and provider-failure handling; ADR-0009 is accepted and ADR-0008 records the amendment. No new runtime capability configuration or review mechanism was added.
- No configured typechecker/linter is available; none was installed. Live-provider capacity/truncation and answer-quality behavior remain unverified. The parent integration pass will run the full suite once and record final review/verification separately.
