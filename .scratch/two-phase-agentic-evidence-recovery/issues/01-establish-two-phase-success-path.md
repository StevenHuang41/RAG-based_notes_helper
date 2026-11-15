# 01: Establish the two-phase Agentic success path

**What to build:** A query whose first Search supplies sufficient evidence completes Retrieval review, accumulated-pool selection, answer generation, and same-context Answer review. Establish the public result/callback contract used by later recovery and reporting work. Follow the parent spec and ADR-0008; this ticket proves the sufficient-first-search slice, not the entire redesign.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Preserve original-query first Search, independently configured retrieval limits, read-only behavior, and per-query isolation. Classic behavior remains unchanged.
- [x] Build a pool deduplicated by `(source, chunk_id)`, retaining original text, first-seen order, and Search provenance. Return auditable snapshots rather than mutating the before-selection pool.
- [x] Retrieval review assesses the complete pool against the original question, recording assessed question, coverage, and missing information without asking the controller to select evidence references.
- [x] A sufficient assessment ends Retrieval with `sufficient_evidence` before any Answer generation. No separate round coverage state is introduced.
- [x] When pool size exceeds `LLM_MAX_CHUNKS`, final-rerank against the original question and select top-k. At or below the limit, skip this final rerank and preserve first-seen order; do not confuse final pool reranking with within-Search ranking.
- [x] Record pool identity/provenance, selection order, rerank execution/scores where available, coverage, and stop reason in public query results/callbacks. Preserve exact ordered/truncated Answer generation context.
- [x] Generate from the selected non-empty context and buffer the candidate until Answer review accepts it. The new flow does not permit post-generation Search, evidence expansion, or context replacement.
- [x] Deterministic tests through `run_query()` results/callbacks verify sufficient-first-search behavior, deduplication, all three pool-size relationships to the chunk limit, original-query final reranking, and no rejected text exposure. Preserve relevant Classic regression coverage.
- [x] Remove/update conflicting expectations only for the behavior intentionally migrated by this slice. Later tickets complete adaptive, empty, degraded, and revision paths; do not claim them implemented here.

## Comments

Implemented and patch-integrated on the integration branch. Focused public-seam checks: 13 passed, 22 deselected; whitespace checks passed. Verified the sufficient-first-search slice and three pool-size cases. Adaptive, empty/degraded, and revision paths remain assigned to subsequent tickets. No commit created while local-commit authorization remains pending.
