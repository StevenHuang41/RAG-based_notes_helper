# 07: Integrate and verify the two-phase Agentic workflow

**What to build:** Deliver the complete approved Agentic workflow with coherent operator documentation, passing integration checks, reviewed implementation, and evidence of actual live-provider behavior alongside the unchanged Classic baseline.

**Blocked by:** 05 — Evaluate accepted answers independently from workflow reliability; 06 — Report before/after pool selection and phase audits.

**Status:** resolved

- [x] Verify the integrated query and evaluation contracts agree across all predecessor tickets and satisfy the parent spec/ADR-0008. Retire incompatible legacy Agentic replace/reopen/expand behavior and update contradictory tests/docs/glossary language without deleting historical tracker artifacts.
- [x] Update example configuration and maintained operating/architecture guides for independent Search and Answer total-attempt budgets, accumulation, conditional final rerank, warnings, and evaluation applicability. Preserve private environment configuration.
- [x] Preserve unrelated pre-existing working-tree changes and account for the earlier Search latency-row fix without treating it as evidence of this redesign's completion.
- [x] Run appropriate focused checks during integration and the full pytest suite after integration. Run configured lint/typechecking checks if available; explicitly report unavailable tooling and remaining failures.
- [x] Perform the integration code review against the approved spec and repository standards, fix findings, and rerun checks justified by those fixes. Do not claim integration complete while required findings remain unresolved.
- [x] Execute Classic and Agentic live evaluations sequentially using the same reviewed QA IDs, fixture, seed, answer/judge models, and shared retrieval settings. Include the previously failing support-volume question and cases exercising evidence recovery, empty retrieval, supported partial answers, and revision; distinguish any scripted-only coverage from observed live behavior.
- [x] Record commands, actual outputs, artifact locations, configuration, case IDs, outcomes, degradation/failures, metric applicability denominators, latency, and usage. A completed artifact run or higher aggregate Ragas score alone is not a pass criterion.
- [x] Confirm accepted answers do not expose unsupported rejected candidates, Search/Answer caps hold, empty/gibberish/warning outcomes remain distinct, and Classic regressions are absent. Disclose any unmet live acceptance criteria rather than marking the work complete.
- [x] Keep deterministic tests, live-provider results, implementation status, and remaining limitations distinct in the final verification report. No remote changes, merge into main, destructive cleanup, or commit is authorized by this ticket alone.

## Comments

Integration and verification completed locally; see the feature verification report for commands, review fixes, artifacts, and explicit live limitations. Final full suite: 264 passed, one existing warning. The matched live comparison contains five accepted Agentic answers, three revision-exhausted warnings, and five Retrieval degradations; it does not establish superior quality. Final-runtime support-volume smoke returns an accepted abstention after three Searches without degradation. Accepted partial answers and empty-final-pool behavior are deterministically tested, not claimed as observed live successes. Runtime/benchmark hashes unchanged. Four generated worktrees archived and cleaned; original detached worktree retained. No commit/push/main merge.
