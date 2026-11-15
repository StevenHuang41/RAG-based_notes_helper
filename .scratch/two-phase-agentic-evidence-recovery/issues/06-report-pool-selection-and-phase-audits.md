# 06: Report before/after pool selection and phase audits

**What to build:** Agentic query reports let readers compare the accumulated pool with the selected generation pool and audit the actual workflow cost without duplicating raw chunk context in metadata tables.

**Blocked by:** 03 — Handle empty pools and degraded Retrieval review; 04 — Accept supported partial answers and bound same-context revisions.

**Status:** resolved

- [x] Render two aligned metadata tables for every Agentic query: accumulated chunks pool after Retrieval, and selected pool after final rerank/top-k. Include order, identity, source lines, Search provenance, relevant available scores, and evidence-line coverage using existing report conventions.
- [x] Show both tables for skipped reranking and empty pools; explicitly label skipped final rerank. At/below the chunk limit, preserve pool order and do not invent final-rerank scores.
- [x] For oversized pools, the second table shows the selected top-k in final rank order, not every discarded ranked candidate. Preserve the complete before-selection pool for comparison.
- [x] Do not repeat raw chunk text in the new tables. Retain the separate exact generation-context audit, including actual token truncation, and preserve detailed per-Search audits.
- [x] Show assessed original question, pool coverage, missing information, normal stop reason or degradation, and answer coverage/review status without implying that pool sufficiency guarantees selected-context sufficiency.
- [x] Attribute Retrieval review latency to actual controller calls, separately from Search method/ranking stages and post-generation Answer review. Final pool rerank is distinguishable from per-Search reranking, including when skipped.
- [x] Preserve provider-reported usage and distinguish workflow cost from Ragas judge cost. Missing measurements stay unavailable; do not synthesize controller latency rows for calls that did not occur.
- [x] Keep JSON/Markdown artifact representations consistent. Coordinate shared result contracts with ticket 05 without making score eligibility a prerequisite for independently verifying pool tables and runtime phase audits.
- [x] Deterministic evaluation-artifact tests cover multiple Searches, deduplication, rerank run/skipped/empty cases, evidence lines, metadata-only tables, separate actual context, phase ordering/measurement, and retention of Classic reporting behavior.

## Comments

Implemented and patch-integrated. Implementer focused checks: 17 passed, one existing dependency warning. Two metadata-only pool tables, real review-call latency, final pool rerank cost, and degradation/denominator rendering verified. Integration/full/live acceptance remains ticket 07. No commit created pending authorization.
