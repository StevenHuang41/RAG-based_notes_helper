# 03: Agentic RAG one-shot workflow

**What to build:** Let a user opt into Agentic RAG for a one-shot query. The workflow searches with the original question, lets the controller inspect accumulated note evidence and choose a refined search, selected-evidence answer, or abstention, then produces a cited answer or a clear workflow error.

**Blocked by:** 02: Classic RAG empty-evidence abstention.

Status: ready-for-agent

- [ ] Classic remains the default; --agentic selects Agentic RAG for a one-shot query.
- [ ] The first search uses the user's original question, and one query performs no more than three searches total.
- [ ] The controller action is strict JSON validated with Pydantic; answer references are checked against chunks retrieved for the current query.
- [ ] Valid search actions trigger another read-only note search; valid answer actions pass only selected, deduplicated evidence to final answer generation and citations.
- [ ] The controller can abstain when accumulated evidence is insufficient; empty initial retrieval uses the deterministic no-evidence behavior from ticket 02.
- [ ] Invalid JSON, schema, evidence references, or out-of-budget actions get one retry; a second invalid action returns a clear workflow error without falling back to Classic.
- [ ] Agentic uses the configured existing LLM provider and keeps state scoped to one query.
- [ ] Automated tests cover the one-shot CLI behavior, search bound, evidence selection, citations, abstention, and invalid-action handling.
