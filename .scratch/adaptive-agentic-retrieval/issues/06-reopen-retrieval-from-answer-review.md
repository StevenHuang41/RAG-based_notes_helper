# 06: Reopen retrieval from Answer review

**What to build:** Extend bounded Answer review so an insufficient or irrelevant candidate answer can request another adaptive Retrieval search rather than being limited to revision against the same evidence. Maintain independent three-attempt Search and Answer budgets across the complete LangGraph cycle, dynamically restrict actions that cannot be completed with the remaining budgets, and preserve buffering so rejected candidate answers are never shown.

**Blocked by:** 05: Replace or extend Agentic evidence

**Status:** ready-for-agent

- [ ] Answer review can accept, request a focused revision, request an adaptive Retrieval search, clarify, or abstain.
- [ ] A Search requested after Answer review uses the validated query, thresholds, and evidence mode from the review action.
- [ ] Search attempts and Answer attempts are independently capped at three, with each initial attempt included.
- [ ] Switching from Answer review back to Retrieval and then to generation never resets either counter.
- [ ] Only an actual Retrieval invocation consumes a Search attempt; an empty successful search still consumes it.
- [ ] Only an actual Answer generation consumes an Answer attempt; a candidate rejected by review still consumes it.
- [ ] Controller calls, validation feedback, and actions rejected before work do not consume Search or Answer attempts.
- [ ] Search is unavailable when no Search attempt remains or no Answer attempt remains to use new evidence; revision is unavailable when no Answer attempt remains.
- [ ] An illegal out-of-budget action receives one validation retry, then terminates as `controller_failure` if still invalid.
- [ ] Exhausted Answer attempts return a deterministic `answer_attempts_exhausted` response without exposing the rejected candidate.
- [ ] Every accepted generated answer has passed grounded Answer review against the current selected evidence.
- [ ] Tests cover mixed Search and Answer sequences, exact budget boundaries, all terminal actions, rejected candidate buffering, and guaranteed graph termination.
- [ ] The governing Agentic RAG ADR is superseded or amended to permit bounded Retrieval reopening while preserving read-only operation and per-query isolation.

