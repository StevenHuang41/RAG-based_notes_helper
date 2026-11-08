# 02: Complete bounded Agentic Answer review and recovery

**What to build:** Let post-generation Answer review improve or safely end a
candidate by revising it with the same evidence, expanding from current
candidates, requesting a bounded Search, clarifying an ambiguous query, or
abstaining.

**Blocked by:** 01: Send the initial Agentic Search directly to Answer generation.

**Status:** ready-for-agent

- [ ] `revise` reuses the same generation context and includes focused feedback
  that identifies the required correction.
- [ ] Evidence expansion selects only known candidates from the active pool
  that are absent from the current generation context, and prioritizes them
  for the next Answer attempt.
- [ ] Adaptive Search is requested only when needed evidence is absent from
  the current candidate pool. It preserves validated query rewriting,
  filtering thresholds, and `replace` or `extend` evidence mode.
- [ ] A later Search runs only when both a Search attempt and a later Answer
  attempt remain. Search and Answer budgets stay independent; controller calls
  and validation retries consume neither budget.
- [ ] Clarification is used for query ambiguity or material underspecification.
  Missing or irrelevant evidence leads to adaptive Search when useful and
  within budget, or abstention; it does not by itself lead to clarification.
- [ ] Terminal outcomes distinguish `needs_clarification`,
  `insufficient_evidence`, `answer_attempts_exhausted`, `controller_failure`,
  and `retrieval_failure`, while accepted generated answers retain
  `response_origin=answer_model`.
- [ ] Controller failures retain safe workflow trace and a sanitized failure
  category without persisting prompts or model outputs. A failed review never
  returns its buffered candidate as a fallback.
- [ ] Existing `agentic_answer()` behavior tests cover `ambiguous-01`,
  `unanswerable-13`, recovery actions, budget exhaustion, and a controller-
  invalid replay.
