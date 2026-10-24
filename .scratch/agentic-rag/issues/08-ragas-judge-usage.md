# 08: Separate Ragas judge usage

**What to build:** Report the cost of Ragas evaluation calls separately from Classic and Agentic generation usage, so benchmark results do not attribute judge tokens to either RAG workflow.

**Blocked by:** 07: Workflow token and character usage.

Status: ready-for-agent

- [ ] Evaluation artifacts record provider-reported input, output, and total token usage for Ragas judge calls when available.
- [ ] Judge usage is shown in its own per-run and summary fields, separate from per-question Classic and Agentic usage.
- [ ] When the judge provider does not expose usage, the report marks it unavailable rather than estimating it from text length.
- [ ] Judge input and output character counts, if included, remain separate from token fields.
- [ ] Automated tests verify judge usage is not merged into either workflow's totals and unavailable values remain explicit.
