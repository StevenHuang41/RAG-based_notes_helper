# 05: Add bounded conversation context to Agentic REPL

**What to build:** Let Agentic REPL follow-up questions use recent turns to resolve intent and references, while requiring every answer to retrieve its own note evidence.

**Blocked by:** 01: Constrain Ollama Agentic action output

**Status:** ready-for-agent

- [ ] REPL keeps at most five user/assistant pairs and bounds the context sent to the configured model.
- [ ] History is trimmed against a configured total context-window budget, with the answer output-token budget reserved separately from `LLM_MAX_TOKENS`.
- [ ] The context-window setting is documented in `.env.example` and README; the actual `.env` is unchanged.
- [ ] History is explicitly labeled as conversational context and is never treated as factual evidence.
- [ ] Each REPL query performs its own Retrieval search; one-shot queries and evaluation cases begin with empty history.
- [ ] Conversation history is cleared when the REPL session ends.
- [ ] Follow-up behavior is covered through the approved REPL input/output seam.
