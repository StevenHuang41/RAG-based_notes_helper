# 02: Classic RAG empty-evidence abstention

**What to build:** When Classic RAG finds no note chunks, give the user a clear abstention immediately instead of asking the LLM to answer without evidence.

**Blocked by:** None (can start immediately).

Status: ready-for-agent

- [ ] An empty retrieval result produces a deterministic, user-visible abstention.
- [ ] The LLM is not called and no citations are shown when retrieval returns no chunks.
- [ ] Existing Classic answers and citations continue to work when retrieval returns evidence.
- [ ] Automated tests verify both the empty-evidence behavior and the normal evidence-backed path.
