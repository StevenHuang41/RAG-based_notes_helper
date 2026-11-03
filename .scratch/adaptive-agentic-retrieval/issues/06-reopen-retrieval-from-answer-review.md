# 06: Reopen retrieval from Answer review

**What to build:** Extend bounded Answer review so an insufficient or irrelevant candidate answer can request another adaptive Retrieval search rather than being limited to revision against the same evidence. Maintain independent three-attempt Search and Answer budgets across the complete LangGraph cycle, dynamically restrict actions that cannot be completed with the remaining budgets, and preserve buffering so rejected candidate answers are never shown.

**Blocked by:** 05: Replace or extend Agentic evidence

**Status:** resolved

- [x] Answer review can accept, request a focused revision, request an adaptive Retrieval search, clarify, or abstain.
- [x] A Search requested after Answer review uses the validated query, thresholds, and evidence mode from the review action.
- [x] Search attempts and Answer attempts are independently capped at three, with each initial attempt included.
- [x] Switching from Answer review back to Retrieval and then to generation never resets either counter.
- [x] Only an actual Retrieval invocation consumes a Search attempt; an empty successful search still consumes it.
- [x] Only an actual Answer generation consumes an Answer attempt; a candidate rejected by review still consumes it.
- [x] Controller calls, validation feedback, and actions rejected before work do not consume Search or Answer attempts.
- [x] Search is unavailable when no Search attempt remains or no Answer attempt remains to use new evidence; revision is unavailable when no Answer attempt remains.
- [x] An illegal out-of-budget action receives one validation retry, then terminates as `controller_failure` if still invalid.
- [x] Exhausted Answer attempts return a deterministic `answer_attempts_exhausted` response without exposing the rejected candidate.
- [x] Every accepted generated answer has passed grounded Answer review against the current selected evidence.
- [x] Tests cover mixed Search and Answer sequences, exact budget boundaries, all terminal actions, rejected candidate buffering, and guaranteed graph termination.
- [x] The governing Agentic RAG ADR is superseded or amended to permit bounded Retrieval reopening while preserving read-only operation and per-query isolation.

## Comments

- Implemented in `bdb1188` through the public `run_query()` seam, with existing lower-level tests retained for exact graph and exhaustion contracts.
- Answer review now accepts `accept`, `revise`, `search`, `clarify`, and `abstain`. Review-requested searches reuse the validated rewritten query, thresholds, and evidence mode, then return to controller evidence selection without resetting either counter.
- Search and Answer work retain independent hard caps of three. Controller calls and validation retries do not consume either budget; an illegal Search action receives one retry and repeated invalid output becomes `controller_failure` without additional Retrieval or generation.
- A third rejected candidate cannot trigger a fourth generation: revision routes to deterministic `answer_attempts_exhausted`, clears citations, and never exposes candidate text.
- ADR-0005 now permits bounded Retrieval reopening while retaining read-only operation, per-query isolation, grounded review, and buffered candidates.
- Verification before resolution: full offline pytest suite passed (236 test items); focused query/Agentic/CLI/Retrieval/evaluation/report intersection passed (121); `py_compile` and `git diff --check` passed.
- Live-provider JSON adherence and behavior remain unverified; all verification used offline fakes.

