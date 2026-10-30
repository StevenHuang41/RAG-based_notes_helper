# 01: Relocate fixture-backed evaluation assets

**What to build:** Make Classic and Agentic fixture-backed evaluation run from the repository's explicit evaluation corpus and reviewed QA set, while keeping the index isolated from user data and persistent storage. A completed run writes the established audit artifacts to the new generated-results location, and a missing or incomplete benchmark fails clearly instead of using another data source.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] The checked-in evaluation corpus and reviewed QA set have one version-controlled home under the evaluation-assets boundary; pytest retains no duplicate benchmark fixture copy.
- [x] Both evaluation modes load the relocated corpus and reviewed QA set, build an isolated temporary index, and preserve corpus-relative evidence identifiers and line-anchored validation.
- [x] The default generated report location uses the new results directory, retains timestamped run directories, collision refusal, `README.md`, `results.json`, and `run.log`, and is ignored by Git.
- [x] Missing evaluation assets fail with an actionable error and never fall back to application data or persistent storage.
- [x] Offline fake-provider coverage at the evaluation-runner seam verifies the relocated inputs, isolated execution, result artifacts, and failure boundary for both workflow modes.

## Answer

Implemented by `383dc8e` and integrated through `b18cf4d`. The corpus and
reviewed QA set now have one evaluation-assets home; the runner validates the
assets, uses the relocated corpus for its isolated index, and defaults report
artifacts to the new results root. Historical root-level `eval/` artifacts
remain ignored and untouched. The full offline suite passed: 148 tests (with
two dependency warnings).
