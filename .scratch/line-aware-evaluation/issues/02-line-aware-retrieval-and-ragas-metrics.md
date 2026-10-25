# 02: Line-aware retrieval metrics and answer correctness

**What to build:** Compute source-matched retrieval Recall, Precision, and NDCG, and add reference-based Ragas Answer Correctness to the existing answer diagnostics.

**Blocked by:** 01: Line-anchored fixture evidence and chunks.

**Status:** resolved

- [x] Compute Recall and Precision over unique `(source path, line number)` positions using inclusive intervals.
- [x] Match ground-truth lines only against retrieved lines from the same source path; count retrieved lines from unrelated files in the Precision denominator only.
- [x] Merge overlapping spans from the same source before counting lines, so overlap never inflates a numerator or denominator.
- [x] Return not-applicable values for empty denominators; return zero Recall for non-empty ground truth with no retrieved evidence.
- [x] Compute NDCG@configured `top_k` with newly covered ground-truth line count as chunk relevance; repeated coverage adds no gain.
- [x] For Agentic evaluation, retain each search's ranked list, compute NDCG per applicable list, and report the mean; compute Recall/Precision from the deduplicated union across searches.
- [x] Keep Ragas Context Utilization as a retrieval diagnostic and add Ragas Answer Correctness using the reviewed reference answer alongside Faithfulness and Answer Relevancy.
- [x] Keep answerable-case Ragas scoring limited to expected-answerable cases; use deterministic answerability/abstention metrics for unanswerable and ambiguous cases.
- [x] Add hand-calculated interval and ranking coverage for same-file overlap, multiple sources, wrong-source retrieval, one-line spans, empty results, and ranking changes.
- [x] Verify reference answer is passed only to Answer Correctness, not to the reference-free Ragas metrics.

## Comments

- Resolved in commit `41086ce`, merged to integration at `2061a7b`. Focused post-merge coverage passed (24 tests); the implementer also reported the full suite passed (112 tests). `git diff --check` passed.
