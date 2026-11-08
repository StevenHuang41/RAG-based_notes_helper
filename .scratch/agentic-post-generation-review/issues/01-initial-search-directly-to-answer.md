# 01: Send the initial Agentic Search directly to Answer generation

**What to build:** Let an Agentic query with retrieved candidates produce its
first buffered Answer from the shared Generation top-k, then review that
candidate before returning it. Give an empty initial candidate pool its own
deterministic Retrieval-empty outcome.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Deterministic query validation and the original-query first Search use
  the shared Classic/Agentic Retrieval and ranking baseline.
- [x] For a non-empty initial ranked candidate pool, the first Answer
  generation receives the same ordered Generation top-k as Classic RAG, with
  no controller sufficiency decision before generation.
- [x] The candidate remains buffered until post-generation Answer review
  accepts it; the reviewer receives the exact ordered context used for that
  candidate.
- [x] If the initial ranked candidate pool is empty, return
  `no_retrieval_results` without a controller call, Answer-model call, or
  Answer attempt. Do not report it as clarification.
- [x] Progress reports the actual Search, Answer generation, and Answer review
  phases and does not show a pre-generation controller phase or candidate
  text.
- [x] Behavior is covered through the existing `agentic_answer()` seam for
  `direct-06` and empty initial Retrieval.

## Comments

- Implemented in `3620825`.
- `tests/test_agentic.py`, `tests/test_query_outcomes.py`, and `tests/test_cli_agentic.py` passed (84 tests); `git diff --check` passed.
- No-hit Search returns `no_retrieval_results`, records zero Answer attempts, and skips controller and Answer generation.
