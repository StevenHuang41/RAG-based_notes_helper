# 04: Render Agentic outcomes and metric applicability in Evaluation reports

**What to build:** Make the human-readable Evaluation report show which
Agentic outcomes completed, which Ragas metrics were scored or skipped, and
what evidence reached Answer generation.

**Blocked by:** 03: Record Agentic outcomes and Ragas eligibility in Evaluation results.

**Status:** resolved

- [x] Per-case report details show terminal reason, response origin, attempt
  counts, and metric applicability without exposing rejected candidate text or
  raw controller output.
- [x] Skipped and unavailable Ragas metrics include their recorded reason;
  context-dependent metrics are unavailable when the final generation context
  is empty.
- [x] Keep Retrieval-stage candidate metrics distinct from final-context
  Recall, Precision, and NDCG. Preserve one overall Retrieval metrics table
  based on final generation context, with unavailable rows visible.
- [x] Aggregate reporting keeps normal bounded outcomes and workflow errors
  distinguishable, including workflow reliability and token/latency accounting.
- [x] Report behavior is covered through the existing report-rendering and
  `run_evaluation()` test seams for `direct-06`, `ambiguous-01`, and
  `unanswerable-13`.

## Comments

- Implemented in `4ebb248` (`feat(eval): render Agentic outcomes and applicability`).
- Focused: `/Users/Steven/github_repos/my/RAG-based_notes_helper/.venv/bin/pytest -q -o addopts='' tests/test_final_context_report.py tests/test_report.py tests/test_agentic_stage_report.py` passed (12 tests).
- Full: `/Users/Steven/github_repos/my/RAG-based_notes_helper/.venv/bin/pytest -q -o addopts='' --tb=short` passed (273 tests).
- `python -m py_compile src/rag_notes_helper/eval/report.py tests/test_final_context_report.py` and `git diff --check` passed.
- Per-case report details show terminal outcome, response origin, attempt counts, controller failure category, and Ragas applicability reasons. The aggregate `Workflow outcomes` table separates bounded outcomes from workflow errors and shows insufficient-evidence counts by origin. Existing latency/token accounting and final-context Retrieval metrics remain intact.
