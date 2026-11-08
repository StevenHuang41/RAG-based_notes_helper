# 06: Report bounded Agentic REPL progress

**What to build:** Show users the observable Agentic workflow as it runs, including its decisions and configured attempt limits, without exposing hidden model reasoning or unaccepted answers.

**Blocked by:** 03: Configure Agentic Retrieval and Answer budgets; 04: Recover an answer using candidate evidence

**Status:** resolved

- [x] Progress identifies clarification, read-only search actions, Search attempt/query/result count, Answer attempt, answer-review decision, and terminal outcome.
- [x] Long-running model calls visibly remain in progress.
- [x] Displayed attempt limits reflect the effective configuration.
- [x] Progress never displays chain-of-thought, raw prompts, or rejected candidate answers.
- [x] Progress behavior is covered through the approved REPL input/output seam.

## Comments

- Implemented in `319d129`.
- Focused REPL and Agentic tests passed (54 tests); the complete offline suite passed with 266 tests.
- Progress events expose only observable stages, safe search metadata, decisions, counts, and outcomes; no live provider behavior is required for this UI contract.
