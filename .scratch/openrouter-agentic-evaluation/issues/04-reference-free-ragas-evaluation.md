# 04: Reference-free Ragas evaluation

**What to build:** Evaluate answer grounding, answer relevance, and context utilization with reference-free Ragas metrics while retaining deterministic checks against the reviewed QA fixture.

**Blocked by:** 03: Configurable Ragas judge providers.

**Status:** resolved

- [x] Ragas uses reference-free faithfulness, answer relevance, and context utilization metrics.
- [x] Ragas judge inputs contain the question, generated response, and appropriate retrieved contexts but do not contain the fixture reference answer.
- [x] Reference-dependent context recall, reference-based context precision, and AnswerCorrectness are not included in this evaluation.
- [x] The fixture reference answer remains available for human inspection in per-case evaluation results.
- [x] Expected source paths and evidence snippets are checked deterministically without using chunk IDs or line numbers as persistent ground-truth keys.
- [x] Expected answerability, abstention correctness, and evidence/source coverage remain deterministic benchmark results.
- [x] Ragas answer/context metrics run only for expected-answerable cases; unanswerable and ambiguous cases remain evaluated through deterministic checks.
- [x] The answer-relevancy embedding model remains the existing local evaluation embedding model.
- [x] Automated tests verify that the Ragas judge receives no reference answer and that ground-truth checks, metric applicability, and separate usage reporting remain correct.

## Comments

- Resolved in commit 2c97764. Evaluation uses reference-free Faithfulness, AnswerRelevancy, and ContextUtilization while retaining deterministic checks and human-review references; the full suite passed (79 tests).
