# Local stdio MCP note search

Status: ready-for-agent

## Problem Statement

A user wants a local coding assistant to search their notes through MCP and use retrieved evidence to answer. The existing remote HTTP server requires OAuth infrastructure and does not provide a simple local stdio connection.

## Solution

Provide a local stdio MCP server for coding assistants on the same machine. It exposes the existing read-only note search, returns ranked chunks and provenance, and lets the assistant generate the answer. Document setup and demonstrate a real connection from Codex app.

## User Stories

1. As a notes user, I want a local coding assistant to search my notes, so that it can use my own evidence.
2. As a Codex app user, I want to connect to the server locally, so that I can ask questions about notes.
3. As a user of another stdio MCP client, I want a standard MCP server, so that I can connect my preferred assistant.
4. As a developer, I want setup instructions, so that I can reproduce the connection.
5. As a user, I want the index refreshed at server startup, so that searches reflect notes at that point.
6. As a user, I want to know when to restart, so that later note changes are reflected.
7. As a user, I want only the approved note corpus searched, so that search remains scoped.
8. As a user, I want a read-only search tool, so that the tool does not modify notes.
9. As a user, I want to ask for a number of chunks in my prompt, so that my assistant can pass a per-search limit.
10. As a user, I want a sensible default and a bounded maximum, so that a result is useful and manageable.
11. As a user, I want the applied limit in the response, so that I can see when a request was capped.
12. As a user, I want chunk text, source and line range, so that my assistant can ground its answer.
13. As a user, I want ranking provenance, so that I can inspect retrieval results.
14. As a user, I want empty search results distinguished from errors, so that I can understand what happened.
15. As a user, I want invalid inputs rejected, so that the tool contract remains clear.
16. As a user, I want a Mermaid system workflow, so that I can understand the architecture.
17. As a maintainer, I want a real Codex app smoke test, so that I can verify client interoperability.
18. As a user, I want to know that returned evidence enters my assistant's context, so that I understand the data flow.

## Implementation Decisions

- Replace the existing remote Streamable HTTP and OAuth MCP interface with a local stdio MCP server; remove obsolete remote authentication, settings, documentation and tests where no longer used. Preserve unrelated application behavior.
- Reuse the existing shared Hybrid retrieval and runtime index lifecycle. Refresh the index from the Allowed note corpus at server startup; subsequent changes require a restart to guarantee they are reflected.
- Expose one read-only MCP search tool, `search_notes`, with only a nonempty query and optional positive integer `top_k`. Do not provide filesystem, write, or answer-generation tools. Retain existing query and source validation.
- Default `top_k` to 5, cap requests at 20, and return the applied limit. The MCP limit is independent of `LLM_MAX_CHUNKS`; prompting for a limit affects only the tool call and does not edit configuration. Returning fewer results than the applied limit is valid when retrieval yields fewer candidates.
- Return ranked chunk text, relative source, chunk ID, line bounds, and ranking provenance. An empty result is a normal response; retrieval, index, or invalid metadata failures are explicit errors, not empty results.
- Keep stdout suitable for MCP protocol communication, with diagnostics on stderr. Close runtime resources on shutdown.
- Update the existing architectural decision to show it has been superseded and record the local transport choice and trade-off. Remove outdated remote-server claims.
- README documents prerequisites, setup, local client configuration, prompt examples including `top_k`, index refresh, returned evidence entering the client context, troubleshooting, and actual verification status. Include Codex app and Claude Code examples without claiming untested compatibility. Replace the system workflow ASCII diagram with Mermaid; retain the repository tree as text.
- An assistant may use or ignore tool results; a prompt alone does not guarantee a tool call or evidence-grounded answer. The Codex acceptance demonstration must observe the actual call and answer.

## Testing Decisions

- Good tests assert observable behavior and contract, not private implementation details. Prefer existing MCP client and injected search seam to additional abstractions.
- Exercise MCP tool discovery and calls, input validation, default and explicit limits, cap and applied limit, structured provenance, empty results, error propagation, and rejection of unsafe source metadata. Use the existing MCP client contract tests as prior art.
- Exercise a real stdio subprocess at the MCP protocol boundary, including startup and shutdown, to catch protocol-output contamination and launch/configuration problems. Use isolated fixtures or test doubles where needed.
- Verify runtime index initialization and shared retrieval wiring using the existing runtime integration-test pattern; protect evaluation corpus and runtime data isolation.
- Run focused tests and appropriate broader checks after removing shared authentication code. Inspect the README Mermaid rendering or compatible syntax.
- Required acceptance: actually launch the server through Codex app, observe a successful `search_notes` invocation with chunks and provenance, and inspect the resulting answer against those chunks. Exercise a prompt specifying `top_k`. Report actual outcome and any environment blockers; do not substitute fake-provider tests for this acceptance.

## Out of Scope

- Remote HTTP access, OAuth, public endpoints, tunnel setup, multi-user access, arbitrary file access, and write operations.
- Server-side answer generation or changing `LLM_MAX_CHUNKS` through a prompt.
- Background file watching, instant index refresh, packaging a plugin, and guaranteed support for every MCP client.
- A redesign of retrieval algorithms or a claim that MCP itself improves retrieval quality.

## Further Notes

The server runs with the local user's permissions; stdio avoids a network listener but does not isolate it from processes or clients authorized to launch it. The client and its model provider determine how returned content is handled. An actual Claude Code smoke test is optional and must not be claimed unless performed.
