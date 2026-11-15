# 02: Recover evidence with bounded adaptive searches

**What to build:** A query with zero or partial evidence searches again using focused rewrites and threshold adjustments, preserving findings until the accumulated pool is sufficient or Search quota is exhausted, then continues to selection and generation.

**Blocked by:** 01 — Establish the two-phase Agentic success path.

**Status:** resolved

- [x] `AGENTIC_MAX_SEARCH_ATTEMPTS` retains its existing default and supported bounds. Count each executed Search, including the first; controller decisions and validation retries do not consume Search attempts or authorize extra searches.
- [x] Each completed Search appends new candidates without replacing prior chunks. Repeated identities append provenance without duplicating chunks or overwriting original text.
- [x] After every Search, including the last allowed Search, assess the accumulated pool against the original question. Preserve per-Search audits while maintaining only overall coverage.
- [x] Zero/partial coverage with remaining quota produces a legal next plan focused on missing information. Allow rewritten queries and Vector minimum score/BM25 minimum-score ratio within `[0, 1]`, without controller changes to top-k, models, or generation limits.
- [x] Normal termination reasons are exactly `sufficient_evidence` and `search_budget_exhausted`, with sufficient taking precedence on the final allowed Search. Record zero/partial/sufficient coverage separately from termination.
- [x] Exhausted non-empty pools still proceed to context selection and generation even when coverage remains zero or partial. Do not claim that exhausted searches prove corpus absence.
- [x] Public query results/callbacks expose effective search plans, attempts, missing information, provenance, and normal termination reason for downstream evaluation.
- [x] Deterministic public-seam tests cover cross-search complementary evidence, repeated chunks, later searches returning no new chunks, legal threshold changes, configured attempt limits, and the absence of post-generation Retrieval.
- [x] Search failure cannot bypass the attempt cap. Controller-invalid/degraded paths are completed by ticket 03 and must not be falsely recorded as normal exhaustion.

## Comments

Implemented and patch-integrated. Focused new workflow and Classic regression checks: 17 passed, 22 deselected; diff checks passed. Verified configurable 1/3/5 Search caps, complementary accumulated evidence, duplicate provenance, last-search sufficient precedence, and retention after later empty Search. No local commit created pending authorization.
