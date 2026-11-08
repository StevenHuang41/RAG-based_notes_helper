# 03: Configure Agentic Retrieval and Answer budgets

**What to build:** Let operators configure separate total-attempt limits for Agentic Retrieval and Answer generation, with effective values visible in evaluation provenance.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `AGENTIC_MAX_SEARCH_ATTEMPTS` and `AGENTIC_MAX_ANSWER_ATTEMPTS` default to 3 and accept values from 1 through 5.
- [x] Each limit includes its initial attempt; controller calls and validation retries consume neither budget.
- [x] A Search requested during Answer review consumes the next Search attempt only when both a Search attempt and a later Answer attempt remain.
- [x] Effective limits are documented in `.env.example` and README and recorded in evaluation metadata; the user's `.env` is not changed.
- [x] Budget behavior is covered through the approved workflow seam.

## Comments

- Implemented in `e36f6d3` and merged into integration at `87ef7e3`.
- Settings, Agentic budget, CLI, and evaluation focused tests passed; the complete offline suite passed with 266 tests.
- `.env` was not modified.
