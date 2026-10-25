# 03: Add multi-hop and temporal cases

**What to build:** Let evaluators check whether Notes Helper combines evidence across project notes and selects the valid fact when dates, versions, or approval states differ.

**Blocked by:** 01: Establish the realistic fixture foundation

**Status:** resolved

- [ ] Add eleven reviewed multi-hop cases, bringing the multi-hop category to fifteen cases.
- [ ] Require each case to use evidence from multiple notes or to resolve a fact using its date, version, or approval status.
- [ ] Include source-attribution and temporal/version challenge tags where applicable.
- [ ] Give each case a unique ID, reviewed reference answer, answerability label, expected sources, and exact line-anchored evidence for every supporting fact.
- [ ] Confirm fixture validation accepts the cases and the evaluation runner processes them using the active chunk settings.

## Comments

- Implemented as `e774382` and merged as `7d4722c`. Added eleven reviewed multi-hop/temporal cases, bringing the category to fifteen. Full suite passed (120 tests); `git diff --check` was clean.
