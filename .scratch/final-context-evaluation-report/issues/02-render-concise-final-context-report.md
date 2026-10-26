# 02: Render concise final-context report

**What to build:** Evaluation reviewers can distinguish ranked Retrieval top-k candidates from the Answer generation context, inspect final-context evidence metrics, Ragas diagnostics, and execution cost in one concise Markdown report.

**Blocked by:** 01: Record final-context evaluation metrics.

**Status:** ready-for-agent

- [ ] Render Context prefix rows and their final-context evidence metrics beside the ordered Answer generation context, while preserving ranked retrieval candidates as an audit.
- [ ] Present overall and category summaries from the structured final-context metrics, with all four Ragas scores and a dash for unavailable scores.
- [ ] Use numeric Search IDs and simplify the Search registry, judge usage, latency, and execution-audit tables to review-critical columns.
- [ ] Show provider total tokens when reported, otherwise the exact input-plus-output sum, and a dash when neither is available.
- [ ] Keep the guide concise, explain final-context evidence metrics, Ragas diagnostics, Search IDs, latency, and token totals, and preserve Markdown spacing around expected evidence.
- [ ] Add offline write_readme contract coverage for Classic and Agentic reports, including unavailable scores and tokens.
