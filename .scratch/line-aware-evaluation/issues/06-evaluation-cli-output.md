# 06: Clarify evaluation CLI progress and index output

**What to build:** Make evaluation terminal output identify the selected workflow and synthetic fixture index, and remove duplicate index-success text.

**Blocked by:** 03: Run one query mode per evaluation; 05: Timestamped evaluation artifacts and README report.

**Status:** ready-for-agent

- [ ] Show concise evaluation progress for the selected workflow and fixture-index lifecycle when stderr is a terminal.
- [ ] Do not display the user-index message `Index not found. Building index from notes ...` during fixture evaluation.
- [ ] Print the successful index-build message once.
- [ ] Keep query text, note content, and retrieved chunks out of progress output.
- [ ] Add CLI output coverage for both modes and the one-time build message.

