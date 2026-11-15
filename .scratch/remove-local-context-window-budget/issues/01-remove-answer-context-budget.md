# 01: Remove local budgets from Answer generation and review

**What to build:** Submit all selected Answer generation evidence without local byte-estimate trimming or rejection. Preserve that exact ordered evidence through Answer review and revisions, the newest five complete conversation turns, and existing output limits and provider-failure behavior. Apply the shared Answer history change to Classic as well as Agentic without changing their retrieval or streaming contracts.

**Blocked by:** None (can start immediately).

Status: resolved

## Acceptance criteria

- [x] Answer generation and Answer review no longer use `LLM_CONTEXT_WINDOW_TOKENS` to trim history, drop chunks or reject requests. Prompts exceeding the former default byte estimate reach the provider boundary.
- [x] Final selected chunks up to `LLM_MAX_CHUNKS` retain their complete text, order, identity and provenance in initial generation, Answer review and every revision. Feedback may change; the evidence does not.
- [x] Retain the newest five complete user/assistant turns regardless of their text length; older turns remain excluded. History remains intent context rather than note evidence.
- [x] Preserve the separate `LLM_MAX_TOKENS` output limit and existing provider request semantics. Introduce no replacement input budget or provider auto-truncation option.
- [x] Provider failure during Agentic Answer generation retains attempted generation-context audit information and returns the existing structured failure and safe warning. Answer review failure also preserves its existing safe-warning behavior without exposing rejected candidates or raw errors.
- [x] Preserve Classic empty-result and streaming behavior, and independent Answer attempt limits and review validation retries.
- [x] Replace affected trimming/rejection assertions with observable new behavior using controlled retrieval and fake providers through existing query/result/callback seams. Cover long history, generation above the old estimate, same-evidence revision/review and both generation/review provider failures.
- [x] Run the affected focused tests through `uv run pytest`; report actual commands and outcomes and distinguish deterministic verification from live-provider behavior.

## Integration constraints

- Retain the settings field and any shared helpers still required by Retrieval review until ticket 03. Retire budget-only Answer call arguments and exception handling when no remaining caller needs them; coordinate overlapping controller-module edits with ticket 02.
- This ticket must remain independently verifiable with existing Retrieval budget behavior still present. Use compatible fake settings for scenarios that reach Retrieval review, so an unchanged Retrieval guard does not prevent testing this slice.
- Preserve the existing uncommitted two-phase implementation and unrelated changes. Do not create batch review, evidence extraction, a ledger or new model-capacity discovery.

## Comments

- User approved this three-ticket breakdown and its blocking edges before publication. Specification readiness does not establish implementation or test completion.

- Implemented Answer generation/review removal at the approved existing seams. The settings field and Retrieval-only estimate remain for later tickets.
- TDD evidence: full-selected-context test failed with `2 != 3`, then passed after removing chunk-budget deletion; long-review-history test failed because history was omitted, then passed after removing reviewer budget fitting/rejection; Classic long-history tests failed in both streaming modes, then both passed after retaining the newest five complete turns.
- Focused verification: `UV_CACHE_DIR=/private/tmp/rag-context-uv-cache UV_PROJECT_ENVIRONMENT=/Users/Steven/github_repos/my/RAG-based_notes_helper/.venv uv run --offline --no-sync pytest tests/test_answer.py tests/test_agentic.py tests/test_two_phase_answers.py tests/test_query_outcomes.py tests/test_cli_agentic.py tests/test_openrouter_provider.py` — 77 passed, one third-party LangChain pending-deprecation warning.
- `git diff --check` passed. No configured typechecker/linter is available; typechecking was not run. No live-provider acceptance was run, and no provider-side capacity/retention guarantee is claimed.
