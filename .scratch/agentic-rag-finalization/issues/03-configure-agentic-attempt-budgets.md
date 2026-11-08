# 03: Configure Agentic Retrieval and Answer budgets

**What to build:** Let operators configure separate total-attempt limits for Agentic Retrieval and Answer generation, with effective values visible in evaluation provenance.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `AGENTIC_MAX_SEARCH_ATTEMPTS` and `AGENTIC_MAX_ANSWER_ATTEMPTS` default to 3 and accept values from 1 through 5.
- [ ] Each limit includes its initial attempt; controller calls and validation retries consume neither budget.
- [ ] A Search requested during Answer review consumes the next Search attempt only when both a Search attempt and a later Answer attempt remain.
- [ ] Effective limits are documented in `.env.example` and README and recorded in evaluation metadata; the user's `.env` is not changed.
- [ ] Budget behavior is covered through the approved workflow seam.
