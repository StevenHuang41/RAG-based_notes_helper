# 07: Expose hybrid note search through an authenticated MCP server

**What to build:** A connected model client can request final evidence chunks from the same local hybrid retrieval pipeline used by the application, without receiving filesystem access or asking the MCP server to generate an answer.

**Blocked by:**
- 01: Enforce the allowed runtime note corpus.
- 03: Rerank fused candidates before selecting generation context.

**Status:** ready-for-agent

- [ ] Expose one read-only `search_notes` tool over Streamable HTTP; it performs Vector search, BM25 search, RRF fusion, CrossEncoder reranking, and final evidence selection.
- [ ] Accept a query and optional `top_k`; default to `LLM_MAX_CHUNKS` and clamp requests to the configured maximum.
- [ ] Return selected chunk text, relative source path, chunk identifier, line range, and ranking provenance; do not call an answer-generation model.
- [ ] Do not accept arbitrary filesystem paths or expose tools for reading, writing, or deleting files; constrain all results to the allowed runtime corpus.
- [ ] Protect search access with OAuth 2.1-compatible bearer-token validation and a read-only search permission. Make the issuer configurable; defer selection of a provider vendor.
- [ ] Keep source notes and the index on the local machine. Document that returned evidence is transmitted to the connected ChatGPT or Claude model for its answer generation.
- [ ] Document stable HTTPS/tunnel setup for remote clients and the current client-plan boundary: Claude Free can be used for a remote connector smoke test; ChatGPT Plus website integration is plan-gated and is not a required acceptance check.
- [ ] Offline tests verify the tool contract, authorization, top-k bounds, path boundary, result provenance, and absence of answer generation; no live provider or external monitoring service is required.
