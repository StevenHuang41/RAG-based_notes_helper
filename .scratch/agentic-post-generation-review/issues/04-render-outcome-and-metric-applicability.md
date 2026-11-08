# 04: Render Agentic outcomes and metric applicability in Evaluation reports

**What to build:** Make the human-readable Evaluation report show which
Agentic outcomes completed, which Ragas metrics were scored or skipped, and
what evidence reached Answer generation.

**Blocked by:** 03: Record Agentic outcomes and Ragas eligibility in Evaluation results.

**Status:** ready-for-agent

- [ ] Per-case report details show terminal reason, response origin, attempt
  counts, and metric applicability without exposing rejected candidate text or
  raw controller output.
- [ ] Skipped and unavailable Ragas metrics include their recorded reason;
  context-dependent metrics are unavailable when the final generation context
  is empty.
- [ ] Keep Retrieval-stage candidate metrics distinct from final-context
  Recall, Precision, and NDCG. Preserve one overall Retrieval metrics table
  based on final generation context, with unavailable rows visible.
- [ ] Aggregate reporting keeps normal bounded outcomes and workflow errors
  distinguishable, including workflow reliability and token/latency accounting.
- [ ] Report behavior is covered through the existing report-rendering and
  `run_evaluation()` test seams for `direct-06`, `ambiguous-01`, and
  `unanswerable-13`.
