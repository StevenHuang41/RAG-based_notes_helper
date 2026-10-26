# 03: Clarify latency and provider-usage accounting

**What to build:** Reviewers can audit per-call and aggregate latency and provider usage without ambiguous totals or token estimates.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Show retrieval, controller, answer-generation, and judge-call latency with units in the column headings, including per-query calls and aggregate means.
- [x] Define RAG query latency as query processing through the workflow's terminal answer or abstention, and evaluation latency as RAG query latency plus all Ragas judge latency.
- [x] Remove the ambiguous combined controller-plus-answer generation total from the Markdown report and do not report TTFT for the non-streaming evaluation path.
- [x] Keep provider-reported input, output, and total token fields distinct; show an exact calculated input-plus-output total when the provider total is absent and both inputs are available for every included call.
- [x] Preserve partial or unavailable usage status when required input/output counts are missing; never sum missing counts as zero.
- [x] Show recorded character counts separately from token counts, never convert characters to token estimates, and show judge character counts as N/A when they were not recorded.
- [x] Apply the same reported-versus-calculated semantics to category totals and judge usage grouped by all four Ragas metrics.
- [x] Define latency, evaluation latency, token provenance/status, and character counts in the report guide; cover available, calculated, partial, and unavailable usage with offline fake-provider tests for Classic and Agentic reports.

## Comments

- Implemented on the integration branch in commit `3666c8f` and merged in `ce07d3e`; focused evaluation-runner and report tests passed (27 passed).
- Code-review follow-up in commit `bcdcdd9` defines provider token statuses and calculated-total semantics; 6 focused tests passed.
