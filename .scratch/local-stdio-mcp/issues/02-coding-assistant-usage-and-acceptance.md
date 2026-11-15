# 02: Document coding assistant usage and verify Codex app

**What to build:** A user can follow the documentation to connect a local coding assistant to note search, request evidence including a chosen `top_k`, and understand how the assistant uses the returned chunks. Demonstrate this workflow in Codex app.

**Blocked by:** 01: Provide local stdio MCP search

**Status:** resolved

- [x] README provides prerequisites, reproducible local setup and configuration for Codex app and Claude Code, prompt examples including `top_k`, index refresh behavior, and troubleshooting. Clearly distinguish verified client behavior from untested examples.
- [x] README explains the returned chunk and provenance data and that returned content enters the assistant's context, with subsequent handling determined by the client and model provider.
- [x] The system workflow is represented in Mermaid and includes the local coding assistant, MCP server, retrieval, and returned evidence; the repository tree remains text. Check the diagram renders or uses compatible syntax.
- [x] Actually connect the server from Codex app and observe a successful `search_notes` invocation returning chunks and provenance; exercise a prompt specifying `top_k` and inspect the generated answer against the returned evidence.
- [x] Record the actual commands or configuration, observed tool call and answer, validation outcome, and any blockers. Do not report protocol tests or simulated calls as a real Codex app acceptance pass.
- [x] Claude Code testing is optional; do not claim it passed unless actually performed. An unavailable Codex app or failed real connection leaves the required acceptance unverified and is reported explicitly.

## Comments

README and Mermaid changes are integrated through 1459572. Codex app end-to-end acceptance remains open: computer-use access to the Codex app was denied. A real stdio MCP client call succeeded with OMP_NUM_THREADS=1; this is not a Codex app acceptance pass.

## Acceptance result

Codex app session 01a1de8c-26d2-7f01-b2fe-bd20bec85590 invoked rag_notes_integration_check.search_notes with query "What is Notes Helper?" and top_k=3. The tool returned three chunks from notes_helper.md (lines 1-41, 127-154, and 89-137) with chunk IDs and ranking provenance; the displayed answer is supported by those chunks. The local test used OMP_NUM_THREADS=1. Claude Code was not tested. README verification status was updated in integration commit 393e0d9.
