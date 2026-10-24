# 02: Agentic terminal progress

**What to build:** Show concise progress stages while Agentic RAG searches notes, reviews evidence, generates an answer, or abstains, while preserving clean answer output.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Agentic one-shot queries and Agentic REPL queries report search count and workflow stages.
- [x] Progress is written to stderr only when stderr is attached to a terminal.
- [x] Progress messages do not include note text, retrieved chunks, or evidence content.
- [x] Answer text and streamed answer content remain on stdout.
- [x] Insufficient-evidence abstention has a clear progress stage.
- [x] Classic RAG output remains unchanged.
- [x] Automated tests capture stdout and stderr for terminal and non-terminal behavior and verify no note content appears in progress.

## Comments

- Resolved in commit 7ccf67c. Agentic stage progress is terminal-only on stderr; the focused progress and CLI suite passed (51 tests).
