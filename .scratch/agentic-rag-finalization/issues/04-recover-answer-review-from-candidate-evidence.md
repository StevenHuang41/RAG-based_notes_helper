# 04: Recover an answer using candidate evidence

**What to build:** Let Answer review improve a candidate by selecting additional evidence already retrieved for the current query, or request another bounded Retrieval search when that candidate pool cannot support the answer.

**Blocked by:** 01: Constrain Ollama Agentic action output; 03: Configure Agentic Retrieval and Answer budgets

**Status:** ready-for-agent

- [ ] Answer review evaluates the exact ordered context supplied to the candidate's Answer generation call.
- [ ] Review can select additional evidence from the current candidate pool and generate a new candidate from the resulting context.
- [ ] A same-evidence revision includes focused feedback; a new Retrieval search is used only when the current candidate pool is inadequate.
- [ ] Search and Answer budgets remain independent, every returned generated answer passes review, and rejected candidates are never exposed.
- [ ] Insufficient evidence or exhausted budgets return the existing deterministic bounded outcome.
