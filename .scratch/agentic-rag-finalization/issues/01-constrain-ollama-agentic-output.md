# 01: Constrain Ollama Agentic action output

**What to build:** Let Ollama-backed Agentic RAG receive parseable controller and answer-review decisions that match the action each workflow phase expects, while preserving the bounded failure behavior when output remains invalid.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Controller and answer-review requests use their corresponding structured output schemas.
- [x] If the configured Ollama model rejects a schema, a JSON-format fallback still uses strict application validation.
- [x] The existing single validation retry remains; invalid output never bypasses validation or exposes a rejected candidate.
- [x] Provider request behavior and safe invalid-output outcomes are covered through the approved public seams.

## Comments

- Implemented in `3d97cfb` and merged into integration at `887501c`.
- Focused Ollama payload, Agentic workflow, and query-outcome tests passed; the complete offline suite passed with 266 tests.
- Live Ollama schema acceptance was not exercised; provider request behavior was verified through offline fakes.
