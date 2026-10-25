# 06: Clarify evaluation CLI progress and index output

**What to build:** Make evaluation terminal output identify the selected workflow and synthetic fixture index, and remove duplicate index-success text.

**Blocked by:** 03: Run one query mode per evaluation; 05: Timestamped evaluation artifacts and README report.

**Status:** ready-for-agent

- [x] Show concise evaluation progress for the selected workflow and fixture-index lifecycle when stderr is a terminal.
- [x] Do not display the user-index message `Index not found. Building index from notes ...` during fixture evaluation.
- [x] Print the successful index-build message once.
- [x] Keep query text, note content, and retrieved chunks out of progress output.
- [x] Add CLI output coverage for both modes and the one-time build message.

## Comments

- Resolved in commit `c60ed88`, merged to integration at `88027f3`. Focused CLI/evaluation/index coverage passed (22 tests); `git diff --check` passed.

