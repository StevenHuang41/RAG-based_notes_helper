# 04: Recover an answer using candidate evidence

**What to build:** Let Answer review improve a candidate by selecting additional evidence already retrieved for the current query, or request another bounded Retrieval search when that candidate pool cannot support the answer.

**Blocked by:** 01: Constrain Ollama Agentic action output; 03: Configure Agentic Retrieval and Answer budgets

**Status:** resolved

- [x] Answer review evaluates the exact ordered context supplied to the candidate's Answer generation call.
- [x] Review can select additional evidence from the current candidate pool and generate a new candidate from the resulting context.
- [x] A same-evidence revision includes focused feedback; a new Retrieval search is used only when the current candidate pool is inadequate.
- [x] Search and Answer budgets remain independent, every returned generated answer passes review, and rejected candidates are never exposed.
- [x] Insufficient evidence or exhausted budgets return the existing deterministic bounded outcome.

## Comments

- Implemented in `980633d` and merged into integration at `e601b00`.
- Focused Agentic and query-outcome tests passed (57 tests); the complete offline suite passed with 266 tests.
