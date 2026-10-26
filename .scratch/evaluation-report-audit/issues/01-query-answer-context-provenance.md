# 01: Capture query and answer-context provenance

**What to build:** Evaluation results let a reviewer trace each Classic or Agentic search query to its ranked candidates and reconstruct the exact ordered context used to generate the answer.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Preserve each search query with its corresponding ranked candidates and available stage/method metadata.
- [x] Record the final answer-generation context in the exact selected and truncated order passed to the model.
- [x] Refer to retrieval records for unchanged context; retain actual text and prompt block when context is transformed.
- [x] Preserve existing result fields while recording search and answer-context provenance in schema version 3.
- [x] Keep fixture evaluation isolated from user notes and the persistent user index.
- [x] Add offline fake-provider coverage for Classic and Agentic results, including repeated searches, duplicate chunks, and transformed or truncated context.

## Comments

- Implemented and validated on the integration branch. The full repository test suite passed (129 tests); no live providers were used.
- Post-checkpoint implementation and test changes remain unstaged and uncommitted because the Git staging approval was aborted.
