# 06: Paired Classic and Agentic evaluation

**What to build:** Extend the fixed notes benchmark so maintainers can compare Classic RAG and Agentic RAG on the exact same questions and corpus, with results that make per-case differences and aggregate tradeoffs clear.

**Blocked by:** 03: Agentic RAG one-shot workflow; 05: Synthetic notes corpus and Classic evaluation baseline.

**Status:** ready-for-agent

- [ ] --eval runs every QA case through both workflows against the same isolated indexed fixture corpus.
- [ ] Each case records the workflow's answer or abstention, citations, retrieved evidence, and evaluation results.
- [ ] The report presents per-case and aggregate metrics for both workflows, including faithfulness, answer relevancy, context precision, context recall, evidence/source coverage, and answerability behavior.
- [ ] The report makes the multi-hop cases and correct abstentions inspectable without claiming statistical significance from the 16-case seed.
- [ ] The legacy base.json smoke data remains separate from the paired notes RAG comparison.
- [ ] Automated tests verify paired execution and artifacts with controlled model responses.
