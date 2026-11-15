# 04: Accept supported partial answers and bound same-context revisions

**What to build:** Users receive reviewed full answers, partial answers with disclosed gaps, or abstentions. Unsupported or inadequately qualified answers are revised using the same evidence, with safe warnings when generation or review cannot finish.

**Blocked by:** 01 — Establish the two-phase Agentic success path.

**Status:** resolved

- [x] Non-empty selected pools reach generation even when evidence is judged irrelevant. Generated factual claims remain restricted to supplied note evidence.
- [x] Review the exact ordered/truncated generation context for relevance, supported completeness, missing-information disclosure, and factual support. Do not inherit pool sufficiency after top-k/token truncation.
- [x] Record `answer_coverage` (`full`, `partial`, `none`) separately from `review_passed`. Accept supported partial answers with explicit gaps and appropriate abstentions rather than requiring unavailable facts.
- [x] Unsupported claims or undisclosed gaps trigger actionable same-context revision. Search, evidence expansion, and context replacement are not available Answer review actions.
- [x] `AGENTIC_MAX_ANSWER_ATTEMPTS` preserves existing defaults/bounds and counts initial generation plus revisions independently of Search quota. Invalid review decisions receive one validation retry without consuming another Answer attempt.
- [x] Exhausted rejected candidates produce the internal `answer_attempts_exhausted` outcome and a fixed warning; candidates remain buffered and are never published as partial answers merely by adding a disclaimer.
- [x] Empty generation, generation provider failure, persistent invalid review, and review provider failure produce non-empty fixed warnings with distinct sanitized causes. User-facing text does not expose internal failure codes or imply evidence absence.
- [x] Public results/callbacks retain review acceptance, answer coverage, attempts, exact context, phase latency/usage, and terminal cause. No raw controller prompts/responses are persisted.
- [x] Deterministic public-seam tests cover accepted full/partial/abstaining answers, missing-gap and unsupported-claim revisions, invalid-decision retry, revision exhaustion, empty generation/provider failures, review failure, and no post-generation Retrieval or rejected-candidate leakage.

## Comments

Implemented and patch-integrated. Runtime, Classic, provider, and CLI focused checks: 100 passed, one existing warning. Accepted partial/abstaining outputs, fixed-context revision, exhaustion/warnings, structured output, history/isolation, and real input-budget fitting verified. Removed obsolete legacy graph helpers after migration; tests remained green. No commit created pending authorization.
