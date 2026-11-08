# 02: Align the Classic and Agentic answer baseline

**What to build:** Keep Classic and Agentic comparisons on the same first Retrieval and answer-generation baseline, while allowing either workflow to return a natural, organized, evidence-grounded answer.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Both workflows use the original query, the same starting Retrieval configuration, answer model, shared answer prompt, and Generation top-k.
- [ ] Classic remains one Retrieval search followed by one grounded Answer generation, without controller or answer-review calls.
- [ ] The shared prompt permits natural organization and formatting while requiring every factual claim to use supplied note evidence.
- [ ] The public answer-generation behavior demonstrates the same baseline for both modes.
