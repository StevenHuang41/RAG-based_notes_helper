# 02: Line-aware retrieval metrics and answer correctness

**What to build:** Compute source-matched retrieval Recall, Precision, and NDCG, and add reference-based Ragas Answer Correctness to the existing answer diagnostics.

**Blocked by:** 01: Line-anchored fixture evidence and chunks.

**Status:** ready-for-agent

- [ ] Compute Recall and Precision over unique `(source path, line number)` positions using inclusive intervals.
- [ ] Match ground-truth lines only against retrieved lines from the same source path; count retrieved lines from unrelated files in the Precision denominator only.
- [ ] Merge overlapping spans from the same source before counting lines, so overlap never inflates a numerator or denominator.
- [ ] Return not-applicable values for empty denominators; return zero Recall for non-empty ground truth with no retrieved evidence.
- [ ] Compute NDCG@configured `top_k` with newly covered ground-truth line count as chunk relevance; repeated coverage adds no gain.
- [ ] For Agentic evaluation, retain each search's ranked list, compute NDCG per applicable list, and report the mean; compute Recall/Precision from the deduplicated union across searches.
- [ ] Keep Ragas Context Utilization as a retrieval diagnostic and add Ragas Answer Correctness using the reviewed reference answer alongside Faithfulness and Answer Relevancy.
- [ ] Keep answerable-case Ragas scoring limited to expected-answerable cases; use deterministic answerability/abstention metrics for unanswerable and ambiguous cases.
- [ ] Add hand-calculated interval and ranking coverage for same-file overlap, multiple sources, wrong-source retrieval, one-line spans, empty results, and ranking changes.
- [ ] Verify reference answer is passed only to Answer Correctness, not to the reference-free Ragas metrics.

