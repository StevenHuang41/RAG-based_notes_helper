# 07: Expose hybrid note search through an authenticated MCP server

**What to build:** A connected model client can request final evidence chunks from the same local hybrid retrieval pipeline used by the application, without receiving filesystem access or asking the MCP server to generate an answer.

**Blocked by:**
- 01: Enforce the allowed runtime note corpus.
- 03: Rerank fused candidates before selecting generation context.

**Status:** resolved

- [x] Expose one read-only `search_notes` tool over Streamable HTTP; it performs Vector search, BM25 search, RRF fusion, CrossEncoder reranking, and final evidence selection.
- [x] Accept a query and optional `top_k`; default to `LLM_MAX_CHUNKS` and clamp requests to the configured maximum.
- [x] Return selected chunk text, relative source path, chunk identifier, line range, and ranking provenance; do not call an answer-generation model.
- [x] Do not accept arbitrary filesystem paths or expose tools for reading, writing, or deleting files; constrain all results to the allowed runtime corpus.
- [x] Protect search access with OAuth 2.1-compatible bearer-token validation and a read-only search permission. Make the issuer configurable; defer selection of a provider vendor.
- [x] Keep source notes and the index on the local machine. Document that returned evidence is transmitted to the connected model provider for answer generation.
- [x] Document stable HTTPS/tunnel setup for remote clients and the current client-plan boundary: Anthropic lists custom remote MCP connectors on Free, Pro, Max, Ultra, Team, and Enterprise (Free is limited to one connector); OpenAI lists full MCP for Business and Enterprise/Edu, and read/fetch MCP for Pro (Plus is not listed). Hosted-client smoke testing is not required.
- [x] Offline tests verify the public tool contract, actual Streamable HTTP authorization, top-k bounds, path boundary, result provenance, and runtime wiring to shared retrieval; no live provider or external monitoring service is required.

## Answer

Implemented as `rag-mcp-server`, a separate MCP resource server with exactly one `search_notes` tool. It refreshes the local runtime index at startup, calls the existing hybrid retrieval pipeline directly, and returns only ranked evidence and provenance. OIDC discovery plus issuer JWKS verifies asymmetric JWT signature, exact issuer, expiry, resource audience, and token claims; the SDK enforces the configured read scope at the HTTP transport boundary. The server has no static-token fallback or filesystem tools.

README documents the evidence handoff to hosted clients, required issuer/resource settings, loopback default, host allowlist, HTTPS reverse proxy/tunnel setup, and current plan boundaries with official source links. ADR-0006 records the external integration boundary and token strategy.

Verification: `tests/test_mcp_server.py` passes offline, including ASGI/Streamable HTTP checks for missing and invalid bearer tokens (401), missing read scope (403), valid-scope tool calls, and protected-resource metadata. The local wheel metadata includes the `rag-mcp-server` console script. No hosted-client smoke test is required.
