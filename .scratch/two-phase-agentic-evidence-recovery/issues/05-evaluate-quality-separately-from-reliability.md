# 05: Evaluate accepted answers independently from workflow reliability

**What to build:** An evaluation run scores accepted generated answers when each metric is applicable while exposing controller degradation, failure counts, and differing score denominators in JSON and Markdown.

**Blocked by:** 03 — Handle empty pools and degraded Retrieval review; 04 — Accept supported partial answers and bound same-context revisions.

**Status:** resolved

- [x] Evaluation consumes public query results without losing pool, coverage, termination, review, or degradation metadata. Evolve the result schema/version as required and keep JSON/Markdown consistent.
- [x] Accepted full answers, supported partial answers, and accepted abstentions enter per-metric applicability checks. Retrieval degradation alone does not exclude an otherwise accepted generated answer.
- [x] Deterministic rephrasing/empty-pool/warning results and rejected candidates are excluded from accepted-final-answer Ragas averages, with explicit reasons.
- [x] Each metric records `scored`, `unavailable`, or `failed`, an explanatory reason when needed, and aggregate applicable/scored/failed counts sufficient to understand the denominator. Preserve reference/context requirements; never fabricate unavailable scores.
- [x] Report degradation/failure counts and workflow reliability independently from answer metrics. Accepted answers after controller degradation do not make the run appear fully reliable. Artifact completion remains separate from workflow outcome.
- [x] Final Recall, Precision, and NDCG use the exact generation context and reviewed evidence spans; retain their rows when unavailable and do not substitute the accumulated pool.
- [x] Evaluation remains isolated from runtime notes/indexes and retains the reviewed benchmark unchanged. Judge failures remain distinguishable from runtime workflow failures.
- [x] Deterministic tests through `run_evaluation()` artifacts verify degraded-but-accepted scoring, exclusions, accepted partial/abstaining answers, unavailable/failed metrics, category/overall denominators, reliability, and JSON/Markdown agreement. Use existing final-context and evaluation-runner test patterns.

## Comments

Implemented and patch-integrated. Implementer focused checks: 32 passed, one existing dependency warning. Schema 11 / summary 4; accepted degraded answers scored independently of reliability; explicit metric status and counts verified. Integration/full/live acceptance remains ticket 07. No commit created pending authorization.
