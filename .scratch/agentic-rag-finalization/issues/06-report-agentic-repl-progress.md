# 06: Report bounded Agentic REPL progress

**What to build:** Show users the observable Agentic workflow as it runs, including its decisions and configured attempt limits, without exposing hidden model reasoning or unaccepted answers.

**Blocked by:** 03: Configure Agentic Retrieval and Answer budgets; 04: Recover an answer using candidate evidence

**Status:** ready-for-agent

- [ ] Progress identifies clarification, read-only search actions, Search attempt/query/result count, Answer attempt, answer-review decision, and terminal outcome.
- [ ] Long-running model calls visibly remain in progress.
- [ ] Displayed attempt limits reflect the effective configuration.
- [ ] Progress never displays chain-of-thought, raw prompts, or rejected candidate answers.
- [ ] Progress behavior is covered through the approved REPL input/output seam.
