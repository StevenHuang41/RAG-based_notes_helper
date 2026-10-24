# 05: Ragas test-set candidate authoring

**What to build:** Provide a one-time development workflow that generates QA candidates from the synthetic notes corpus and lets maintainers review and add approved cases to the fixed benchmark.

**Blocked by:** 04: Reference-free Ragas evaluation.

**Status:** resolved

- [x] A development-only Ragas workflow generates candidate questions, reference answers, and reference contexts from the version-controlled synthetic notes documents.
- [x] Candidate generation uses the effective Ragas provider/model configuration and the existing local evaluation embedding model; it does not require the Notes Helper FAISS index.
- [x] Candidate generation reads only synthetic fixture documents and never reads or modifies the user's notes.
- [x] Generated cases remain drafts until a maintainer verifies the query and reference answer and adds expected source paths and evidence snippets.
- [x] Approved candidates augment the existing reviewed benchmark; the current direct, multi-hop, unanswerable, and ambiguous seed cases remain.
- [x] The QA fixture validator accepts additional approved cases with unique IDs and the required ground-truth fields.
- [x] The normal evaluation command reads only the approved, fixed fixture and never regenerates QA.
- [x] Each evaluation run still builds an isolated Notes Helper index from the synthetic notes fixture and records a fixture digest.
- [x] Automated tests validate candidate-to-fixture mapping and fixture loading using offline sample output, without live generation calls.

## Comments

- Resolved in commit 222d6af. Candidate generation is draft-only and fixture-based; the QA validator preserves original seed IDs while allowing reviewed additions. The full suite passed (91 tests); no live generation was run.
