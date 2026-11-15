# 01: Provide local stdio MCP search

**What to build:** A local coding assistant can launch a standard stdio MCP server and search the Allowed note corpus using the existing Hybrid retrieval pipeline, receiving ranked evidence with provenance. Replace the remote HTTP and OAuth interface with this local interface and provide enough setup guidance to exercise it independently.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The server starts over stdio without HTTP, OAuth, a network listener, or credentials for remote access; stdout remains valid MCP protocol output and diagnostics go to stderr.
- [x] Startup refreshes the runtime index from the Allowed note corpus; resources close on shutdown, and runtime index and evaluation corpus isolation is preserved.
- [x] The only exposed tool is read-only `search_notes`, accepting a nonempty query and optional positive integer `top_k`; unexpected arguments and invalid inputs are rejected.
- [x] The default `top_k` is 5 and the maximum is 20, independently of `LLM_MAX_CHUNKS`. The response reports the applied limit; requests above 20 are capped. A prompt-specified limit applies to the individual tool call and does not change configuration.
- [x] Responses contain ranked chunk text, relative source, chunk ID, line range, and ranking provenance. Empty search results are distinguishable from retrieval, index, and invalid metadata errors; unsafe source metadata is rejected.
- [x] The effect of upstream retrieval stage limits on available results is verified and documented: a limit is a maximum, not a guarantee that the requested number of chunks exists or reaches the final ranking. Existing retrieval-stage settings remain distinct from the MCP limit.
- [x] Obsolete remote authentication, settings, documentation, and tests are removed where no longer used. The existing ADR is marked superseded and the local stdio choice and trade-off are recorded.
- [x] A minimal documented local launch and client configuration lets a user independently exercise the new interface.
- [x] MCP client contract tests cover discovery, inputs, limits, provenance, empty results, errors, and source validation; a real stdio subprocess test checks protocol communication, startup, and shutdown. Runtime initialization and shared retrieval wiring are verified using existing test seams.
- [x] Focused tests and appropriate broader checks are run; results and any unverified behavior are reported.

## Comments

Implemented on integration/local-stdio-mcp through 1459572. MCP contract, isolated subprocess, runtime retrieval and complete pytest suite passed. A real local MCP client retrieved three sample chunks with provenance using OMP_NUM_THREADS=1; Codex app acceptance remains in ticket 02.
