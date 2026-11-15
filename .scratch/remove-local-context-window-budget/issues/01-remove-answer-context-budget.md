# 01: Remove local budgets from Answer generation and review

**What to build:** Submit all selected Answer generation evidence without local byte-estimate trimming or rejection. Preserve that exact ordered evidence through Answer review and revisions, the newest five complete conversation turns, and existing output limits and provider-failure behavior. Apply the shared Answer history change to Classic as well as Agentic without changing their retrieval or streaming contracts.

**Blocked by:** None (can start immediately).

Status: ready-for-agent

## Acceptance criteria

- [ ] Answer generation and Answer review no longer use `LLM_CONTEXT_WINDOW_TOKENS` to trim history, drop chunks or reject requests. Prompts exceeding the former default byte estimate reach the provider boundary.
- [ ] Final selected chunks up to `LLM_MAX_CHUNKS` retain their complete text, order, identity and provenance in initial generation, Answer review and every revision. Feedback may change; the evidence does not.
- [ ] Retain the newest five complete user/assistant turns regardless of their text length; older turns remain excluded. History remains intent context rather than note evidence.
- [ ] Preserve the separate `LLM_MAX_TOKENS` output limit and existing provider request semantics. Introduce no replacement input budget or provider auto-truncation option.
- [ ] Provider failure during Agentic Answer generation retains attempted generation-context audit information and returns the existing structured failure and safe warning. Answer review failure also preserves its existing safe-warning behavior without exposing rejected candidates or raw errors.
- [ ] Preserve Classic empty-result and streaming behavior, and independent Answer attempt limits and review validation retries.
- [ ] Replace affected trimming/rejection assertions with observable new behavior using controlled retrieval and fake providers through existing query/result/callback seams. Cover long history, generation above the old estimate, same-evidence revision/review and both generation/review provider failures.
- [ ] Run the affected focused tests through `uv run pytest`; report actual commands and outcomes and distinguish deterministic verification from live-provider behavior.

## Integration constraints

- Retain the settings field and any shared helpers still required by Retrieval review until ticket 03. Retire budget-only Answer call arguments and exception handling when no remaining caller needs them; coordinate overlapping controller-module edits with ticket 02.
- This ticket must remain independently verifiable with existing Retrieval budget behavior still present. Use compatible fake settings for scenarios that reach Retrieval review, so an unchanged Retrieval guard does not prevent testing this slice.
- Preserve the existing uncommitted two-phase implementation and unrelated changes. Do not create batch review, evidence extraction, a ledger or new model-capacity discovery.

## Comments

- User approved this three-ticket breakdown and its blocking edges before publication. Specification readiness does not establish implementation or test completion.
