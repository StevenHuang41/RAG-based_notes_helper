# 01: Establish the realistic fixture foundation

**What to build:** Make the Aurora Planner benchmark a more realistic, self-contained project corpus while keeping the existing seed questions usable through the evaluation workflow.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [ ] Expand the current seven-note corpus to exactly eight related notes by adding one note. Make at least seven notes longer than 200 characters and include one short note below 200 characters.
- [ ] Represent realistic project documents with overlapping topics, repeated facts, dates, versions, and proposed, approved, superseded, or current status distinctions.
- [ ] Include both stale facts that can be resolved from a newer or approved source and facts that remain genuinely ambiguous or absent.
- [ ] Preserve all 16 existing QA seed IDs and the four existing case categories, updating reference answers and line-anchored evidence to match the revised notes.
- [ ] Support overlapping challenge tags and explicit explanations for ambiguous and unanswerable cases.
- [ ] Keep each evidence annotation tied to its source and 1-based inclusive line range with exact text.
- [ ] Confirm the reviewed fixture loads and the evaluation runner processes it using the active chunk settings, isolated from user notes and persistent user storage.
- [ ] Use offline fake generation and judge providers in automated evaluation tests; do not require live credentials.

## Comments

- Implemented and merged as `d2de3fc` (merge commit `5d1440c`). Eight-note corpus, updated seed evidence, and evaluation metadata validation are complete. Full suite passed (119 tests); Classic and Agentic evaluation workflows passed with `CHUNK_SIZE=200` and `CHUNK_OVERLAP=50` using fake providers.
