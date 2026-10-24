# 07: Workflow token and character usage

**What to build:** Show the generation cost of Classic and Agentic RAG in evaluation results, including Agentic controller calls, so maintainers can compare quality gains with provider-reported token usage.

**Blocked by:** 06: Paired Classic and Agentic evaluation.

**Status:** ready-for-agent

- [ ] The supported generation providers expose normalized input, output, and total token usage when the provider reports it, without changing existing answer text or streaming behavior.
- [ ] Evaluation results record usage per generation call and aggregate it per question and workflow, including Agentic controller and final-answer calls.
- [ ] Missing provider usage is explicitly unavailable; it is not estimated from character counts.
- [ ] Input and output character counts are reported in separate fields.
- [ ] Usage and character counts are included in the run's per-case results and summary.
- [ ] Automated tests cover provider usage normalization, unavailable usage, separate character counts, and per-workflow aggregation using fake provider responses.
