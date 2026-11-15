# 03: Handle empty pools and degraded Retrieval review

**What to build:** Empty searches distinguish an unintelligible original question from an intelligible question lacking results, while review failures preserve available evidence and produce an explicit degraded outcome rather than silently aborting useful generation.

**Blocked by:** 02 — Recover evidence with bounded adaptive searches.

**Status:** resolved

- [x] Check original-question intelligibility only when the accumulated pool is empty and intelligibility has not already been established. A later empty Search with a non-empty pool does not trigger this gate.
- [x] Confirmed unintelligibility returns a deterministic rephrasing request without Answer generation. Confirmed intelligibility allows bounded recovery and is not checked repeatedly.
- [x] An empty final pool returns deterministic text describing unsuccessful search without asserting corpus absence. It consumes no Answer attempt and remains distinct from unintelligible input and provider failures.
- [x] Invalid Retrieval review/intelligibility decisions receive one validation retry with feedback. Missing/invalid next plans do not masquerade as normal Search budget exhaustion.
- [x] Persistent invalid decisions and review provider failures record sanitized degradation causes. Unconfirmed coverage is `unknown`, never fabricated zero or sufficient evidence.
- [x] Check complete reviewer input plus reserved output against the configured model context budget. Overflow records `review_context_budget_exceeded`; do not truncate the reviewed pool silently or discard stored chunks.
- [x] Degraded Retrieval ends recovery and proceeds to final selection/generation with an available non-empty pool; an empty pool uses the deterministic empty-pool outcome. Preserve degradation independently from the final answer outcome.
- [x] Actual Search/reranker exceptions retain explicit failure causes and available safe traces, without being converted into successful empty results. Persist no raw controller prompts/responses.
- [x] Public results/callbacks include attempt counts, confirmed/unconfirmed coverage, intelligibility outcome where applicable, degradation cause, and preserved pool/provenance.
- [x] Deterministic public-seam tests cover intelligible empty recovery, unintelligible early return, final empty pool, invalid-then-valid decisions, persistent invalidity, provider failure, review context overflow, Search/reranker failures, and no rejected or raw controller text exposure.

## Comments

Implemented and patch-integrated. New workflow and retained Classic query checks: 31 passed, one existing dependency warning. Verified empty/intelligibility branches, invalid-plan retry, distinct-plan validation, provider/context-budget degradation, and retained failure traces. Superseded replace/reopen assertions were migrated to accumulation and same-context coverage rather than left as failing legacy contracts. No commit created pending local-commit authorization.
