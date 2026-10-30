# 03: Document evaluation-assets boundary

**What to build:** Publish a coherent, durable explanation of the new evaluation-assets boundary after the runtime migration and candidate-authoring removal are complete. Documentation distinguishes the evaluation corpus, reviewed QA set, generated results, pytest code, user data, and persistent storage, while preserving the historical architecture record.

**Blocked by:** 01: Relocate fixture-backed evaluation assets; 02: Remove candidate benchmark authoring.

**Status:** resolved

- [x] User-facing documentation accurately describes the fixed evaluation inputs, isolated evaluation behavior, supported evaluation commands, new report location, and generated-result Git policy.
- [x] The domain glossary uses stable terms for the evaluation corpus, reviewed QA set, and evaluation result, with their final locations.
- [x] A new accepted ADR supersedes the former repository-local evaluation layout assertions without rewriting the older ADR, and explains the trade-off behind separating pytest code, benchmark inputs, and generated outputs.
- [x] Repository guidance has no stale references to the former fixture location, report location, candidate-draft lifecycle, or candidate generator.
- [x] Documentation checks and the relevant offline test suite pass after the final terminology and references are synchronized.

## Answer

Merged documentation commit `8d08002` into the integration branch. README and ADR-0004 describe the final evaluation-assets boundary; the glossary already contained the agreed terms. Targeted offline evaluation and QA-fixture tests passed (38 passed), and `git diff --check` passed after integration.
