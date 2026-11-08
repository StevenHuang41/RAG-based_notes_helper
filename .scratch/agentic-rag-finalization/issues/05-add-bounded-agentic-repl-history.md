# 05: Add bounded conversation context to Agentic REPL

**What to build:** Let Agentic REPL follow-up questions use recent turns to resolve intent and references, while requiring every answer to retrieve its own note evidence.

**Blocked by:** 01: Constrain Ollama Agentic action output

**Status:** resolved

- [x] REPL keeps at most five user/assistant pairs and bounds the context sent to the configured model.
- [x] History is trimmed against a configured total context-window budget, with the answer output-token budget reserved separately from `LLM_MAX_TOKENS`.
- [x] The context-window setting is documented in `.env.example` and README; the actual `.env` is unchanged.
- [x] History is explicitly labeled as conversational context and is never treated as factual evidence.
- [x] Each REPL query performs its own Retrieval search; one-shot queries and evaluation cases begin with empty history.
- [x] Conversation history is cleared when the REPL session ends.
- [x] Follow-up behavior is covered through the approved REPL input/output seam.

## Comments

- Implemented in `4842d63` and merged into integration at `33c09f8`.
- Focused CLI, Agentic, answer, and settings tests passed; the complete offline suite passed with 266 tests.
- `.env` was not modified.
