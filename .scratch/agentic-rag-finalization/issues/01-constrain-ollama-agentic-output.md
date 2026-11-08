# 01: Constrain Ollama Agentic action output

**What to build:** Let Ollama-backed Agentic RAG receive parseable controller and answer-review decisions that match the action each workflow phase expects, while preserving the bounded failure behavior when output remains invalid.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Controller and answer-review requests use their corresponding structured output schemas.
- [ ] If the configured Ollama model rejects a schema, a JSON-format fallback still uses strict application validation.
- [ ] The existing single validation retry remains; invalid output never bypasses validation or exposes a rejected candidate.
- [ ] Provider request behavior and safe invalid-output outcomes are covered through the approved public seams.
