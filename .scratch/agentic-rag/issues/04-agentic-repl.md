# 04: Agentic RAG in the REPL

**What to build:** Let users combine the REPL interaction style with Agentic RAG, so every question in that session uses the workflow selected at startup while Classic remains the default.

**Blocked by:** 03: Agentic RAG one-shot workflow.

Status: ready-for-agent

- [ ] --repl and --agentic can be combined to start an Agentic REPL.
- [ ] --repl alone continues to start a Classic REPL.
- [ ] A question supplied when starting the REPL is answered first using the selected workflow, then the REPL begins.
- [ ] The selected workflow remains fixed for the session; each entered question starts with fresh Agentic state.
- [ ] Automated tests verify the CLI combinations and the query-before-REPL behavior.
