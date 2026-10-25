# 02: Agentic terminal progress

**What to build:** Show concise progress stages while Agentic RAG searches notes, reviews evidence, generates an answer, or abstains, while preserving clean answer output.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Agentic one-shot queries and Agentic REPL queries report search count and workflow stages.
- [ ] Progress is written to stderr only when stderr is attached to a terminal.
- [ ] Progress messages do not include note text, retrieved chunks, or evidence content.
- [ ] Answer text and streamed answer content remain on stdout.
- [ ] Insufficient-evidence abstention has a clear progress stage.
- [ ] Classic RAG output remains unchanged.
- [ ] Automated tests capture stdout and stderr for terminal and non-terminal behavior and verify no note content appears in progress.
