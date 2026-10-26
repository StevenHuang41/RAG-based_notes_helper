# 01: Record final-context evaluation metrics

**What to build:** A fixture-backed evaluation run records deterministic evidence quality from the exact final Answer generation context, so its structured results describe the evidence the answer model actually received.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [ ] Store Final context evidence metrics from the final Context prefix for each query, then use them in query, category, and overall structured summaries.
- [ ] Keep ranked Retrieval top-k candidates as an audit only; do not use them as the primary Recall, Precision, and NDCG scope.
- [ ] Score each Context prefix cumulatively and record zero, rather than N/A, when fixture ground truth exists but the prefix has no relevant Evidence span lines.
- [ ] Run Context Utilization with the final Answer generation context.
- [ ] Increment the structured-result schema version and add offline fake-provider coverage through run_evaluation for Classic and Agentic records, including repeated searches, empty context, transformed context, and absent Evidence spans.

## Comments

- This is a follow-up to resolved evaluation-report-refinements issues. It changes deterministic metric scope from Retrieval top-k candidates to the final Answer generation context.

## Answer

- Implemented in `35dcd40` and merged into the integration branch. Schema v5 records final-context evidence metrics, preserves ranked retrieval data for audit, applies zero scores to grounded misses, and sends final generation context to Context Utilization.
- Validation: `30 passed` with `tests/test_eval_runner.py tests/test_retrieval_metrics.py -q` on Python 3.13, plus `git diff --check` after merge.
