# 05: Add unanswerable cases with missing-information notes

**What to build:** Let evaluators check that Notes Helper abstains when the fixture corpus does not contain information required to answer a question.

**Blocked by:** 01: Establish the realistic fixture foundation

**Status:** resolved

- [ ] Add nine reviewed unanswerable cases, bringing the unanswerable category to thirteen cases.
- [ ] For each case, explain what information is missing from the notes.
- [ ] Mark the cases unanswerable, omit supporting evidence, and use a reviewed abstention reference rather than inventing a factual answer.
- [ ] Confirm fixture validation accepts cases without supporting evidence and the evaluation runner checks expected abstention.
- [ ] Confirm the completed benchmark contains fifty reviewed cases across the agreed category counts: ten direct, fifteen multi-hop, twelve ambiguous, and thirteen unanswerable.
- [ ] Confirm the evaluation runner processes the completed fixture using the active chunk settings and remains isolated from user notes and persistent user storage.

## Comments

- Implemented as `0567838` and merged as `0e78ee5`. Added nine reviewed unanswerable cases, bringing the fixture to fifty with the agreed category counts. Full suite and focused fixture/evaluation tests passed; `git diff --check` was clean.
