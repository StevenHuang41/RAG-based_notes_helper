# 05: Ragas test-set candidate authoring

**What to build:** Provide a one-time development workflow that generates QA candidates from the synthetic notes corpus and lets maintainers review and add approved cases to the fixed benchmark.

**Blocked by:** 04: Reference-free Ragas evaluation.

**Status:** ready-for-agent

- [ ] A development-only Ragas workflow generates candidate questions, reference answers, and reference contexts from the version-controlled synthetic notes documents.
- [ ] Candidate generation uses the effective Ragas provider/model configuration and the existing local evaluation embedding model; it does not require the Notes Helper FAISS index.
- [ ] Candidate generation reads only synthetic fixture documents and never reads or modifies the user's notes.
- [ ] Generated cases remain drafts until a maintainer verifies the query and reference answer and adds expected source paths and evidence snippets.
- [ ] Approved candidates augment the existing reviewed benchmark; the current direct, multi-hop, unanswerable, and ambiguous seed cases remain.
- [ ] The QA fixture validator accepts additional approved cases with unique IDs and the required ground-truth fields.
- [ ] The normal evaluation command reads only the approved, fixed fixture and never regenerates QA.
- [ ] Each evaluation run still builds an isolated Notes Helper index from the synthetic notes fixture and records a fixture digest.
- [ ] Automated tests validate candidate-to-fixture mapping and fixture loading using offline sample output, without live generation calls.
