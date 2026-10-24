# 05: Synthetic notes corpus and Classic evaluation baseline

**What to build:** Give maintainers a repeatable Classic RAG benchmark built from fictional notes and reviewed QA, isolated from personal notes, with per-question and aggregate results saved for later comparison.

**Blocked by:** 02: Classic RAG empty-evidence abstention.

Status: ready-for-agent

- [ ] A fixed synthetic notes corpus and 16 human-reviewed QA cases cover direct lookup, multi-hop integration, unanswerable, and ambiguous questions, with four cases per category.
- [ ] Each QA case records expected answerability, a reference answer, expected source paths, and optional evidence snippets; ground truth does not depend on chunk IDs.
- [ ] The benchmark uses isolated notes and index storage and never reads or modifies the user's notes.
- [ ] The initial --eval benchmark runs Classic RAG over the fixture through ingestion, chunking, indexing, and retrieval.
- [ ] Per-case and aggregate results include faithfulness, answer relevancy, context precision, context recall, evidence/source coverage, and abstention behavior.
- [ ] Each run saves a log, per-case results, and a summary in a unique UTC-timestamped repository-local directory; generated run artifacts are ignored by Git.
- [ ] The existing base.json smoke data remains independently available and is not mixed into notes RAG scores.
- [ ] Automated tests use temporary notes and storage locations and do not require live LLM credentials.
