# 01: Capture query and answer-context provenance

**What to build:** Evaluation results let a reviewer trace each Classic or Agentic search query to its ranked candidates and reconstruct the exact ordered context used to generate the answer.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Preserve each search query with its corresponding ranked candidates and available stage/method metadata.
- [ ] Record the final answer-generation context in the exact selected and truncated order passed to the model.
- [ ] Refer to retrieval records for unchanged context; retain actual text and prompt block when context is transformed.
- [ ] Preserve existing result fields while recording search and answer-context provenance in schema version 3.
- [ ] Keep fixture evaluation isolated from user notes and the persistent user index.
- [ ] Add offline fake-provider coverage for Classic and Agentic results, including repeated searches, duplicate chunks, and transformed or truncated context.

## Comments

- The current working tree contains this behavior's implementation, but automated tests have not been run. Keep provenance contract coverage in this vertical slice.
