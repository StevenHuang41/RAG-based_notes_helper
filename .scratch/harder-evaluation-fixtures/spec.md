# Realistic Evaluation Fixture Dataset

Status: ready-for-agent

## Problem Statement

The current synthetic fixture is too small and too easy to represent realistic retrieval behavior. Its eight notes mostly contain short, isolated facts, and its 16 questions do not sufficiently test source attribution, overlapping topics, stale information, multi-document reasoning, or abstention when evidence is missing or ambiguous.

As a result, evaluation can report favorable scores without showing whether Notes Helper can find the correct evidence among related documents in a realistic notes corpus.

## Solution

The existing Aurora Planner fixture has seven note files. Expand it to exactly eight notes by adding one related note and enriching the existing seven. Make the notes more detailed and realistic, with related facts distributed across documents, repeated topics, status and version differences, and some genuinely unresolved or missing information. Expand the reviewed QA fixture to 50 cases with answer and evidence annotations.

Keep the fixture compatible with the existing evaluation workflow. The evaluation must read the configured chunk size and overlap, index only the synthetic fixture through the existing isolated evaluation process, and complete its per-query evaluation. The fixture is intended to support later retrieval and Ragas evaluation; this change does not redesign indexing artifacts or reports.

## User Stories

1. As an evaluator, I want the fixture to contain eight notes, so that the benchmark stays small enough to review while containing more complex content.
2. As an evaluator, I want the notes to describe one coherent Aurora Planner project, so that overlap between files resembles a real project workspace.
3. As an evaluator, I want project facts to appear in multiple relevant notes, so that retrieval must distinguish the correct source rather than match a topic alone.
4. As an evaluator, I want notes to use realistic document forms such as plans, meeting records, decisions, status updates, and checklists, so that the corpus does not read like a list of isolated benchmark facts.
5. As an evaluator, I want at least seven notes to contain more than 200 characters and one note to be allowed below 200 characters, so that the fixture includes both substantial context and a short-note edge case.
6. As an evaluator, I want related notes to repeat names, dates, amounts, and project topics with meaningful differences, so that queries can test whether the right fact was retrieved from the right file.
7. As an evaluator, I want proposed, approved, superseded, and current information to be distinguishable by date, version, or status, so that queries can test temporal and status-aware retrieval.
8. As an evaluator, I want some notes to contain stale information that can be resolved from a newer or approved source, so that a system must retrieve and prioritize the valid fact.
9. As an evaluator, I want some cases to remain genuinely ambiguous or unsupported by the corpus, so that evaluation measures whether the system abstains instead of guessing.
10. As an evaluator, I want the corpus to include multi-document dependencies, so that multi-hop questions require evidence from more than one note.
11. As an evaluator, I want the QA fixture to contain 50 reviewed cases, so that evaluation covers more retrieval patterns than the current seed set.
12. As a maintainer, I want the 50 cases distributed as 10 direct, 15 multi-hop, 12 ambiguous, and 13 unanswerable, so that the benchmark covers answerable lookup, synthesis, ambiguity, and abstention.
13. As an evaluator, I want source-attribution and other difficulty dimensions represented as challenge tags that can overlap categories, so that cases can be analyzed by challenge type without making categories mutually exclusive.
14. As an evaluator, I want every case to include a stable unique ID and a non-empty question, so that results can be compared and traced across runs.
15. As an evaluator, I want every case to include an answerability label and reviewed reference answer, so that answerable and abstention behavior have explicit expected outcomes.
16. As an evaluator, I want every answerable case to identify its expected source files and evidence spans, so that retrieval can be scored against reviewed source evidence.
17. As an evaluator, I want every evidence span to identify its source, inclusive physical line range, and exact text, so that annotations can be checked against the fixture documents.
18. As an evaluator, I want unanswerable cases to explain what information is missing, so that expected abstention is auditable.
19. As an evaluator, I want ambiguous cases to identify the conflicting or underspecified evidence, so that ambiguity is intentional and not an annotation mistake.
20. As an evaluator, I want the fixture references preserved for applicable Ragas metrics, so that the same reviewed dataset can support grounded generation evaluation.
21. As an evaluator, I want Answer Correctness to use the reviewed reference answer while reference-free metrics use only their required inputs, so that each Ragas metric receives suitable data.
22. As an evaluator, I want Ragas answer metrics to run only for answerable cases, so that ambiguous and unanswerable examples are assessed through answerability and abstention behavior.
23. As an evaluator, I want retrieval evidence checks to remain deterministic and source-aware, so that LLM judgments do not define whether the expected source was retrieved.
24. As an evaluator, I want the evaluation runner to use the configured chunk size and overlap, so that the fixture is indexed under the active settings.
25. As a maintainer, I want the evaluation process to build an isolated index from the fixture and complete the cases without touching user notes or the persistent user index, so that the benchmark is safe to run locally.
26. As a maintainer, I want fixture validation to identify broken source paths, line ranges, or evidence text, so that edits to long notes cannot silently invalidate ground truth.
27. As a maintainer, I want all existing reviewed seed cases and their IDs retained, so that the expanded fixture remains a continuation of the existing regression benchmark.
28. As an evaluator, I want the fixture content and annotations to be version controlled, so that a result can be tied to a reviewed dataset revision.

## Implementation Decisions

- Expand the current seven-note Aurora Planner corpus to exactly eight files, adding one related note and enriching the existing notes.
- Write notes as realistic, related project documents with repeated topics and facts whose source, date, version, or approval status matters.
- Include both resolvable stale/current information and truly ambiguous or missing information. Do not introduce arbitrary contradictions that have no documented resolution.
- Keep at least seven notes above 200 characters and include one short note below 200 characters. Do not require a separate fixture run at a hard-coded 1000-character chunk size and 200-character overlap.
- Expand the QA fixture to 50 reviewed cases, retaining the existing seed IDs and categories. Use the agreed distribution: 10 direct, 15 multi-hop, 12 ambiguous, and 13 unanswerable.
- Keep category for the primary evaluation type and add overlapping challenge tags for source attribution, temporal/version distinction, multi-document evidence, and chunk-boundary difficulty where applicable.
- Preserve reviewed reference answers, answerability, expected sources, and line-anchored evidence. Add an explicit explanation of missing information for unanswerable cases and the reason for ambiguity for ambiguous cases.
- Use source paths and 1-based inclusive line ranges as the ground-truth evidence identity; chunk IDs are not fixture annotations.
- Preserve the existing fixture validation contract and extend it only as needed for new challenge tags and explanatory annotations.
- Keep the evaluation runner as the sole high-level fixture-to-evaluation test seam. Use offline fake generation and judge providers when exercising evaluation behavior; do not require live credentials for automated tests.
- Validate the fixture through the existing evaluation runner using the active chunk settings. The user will update .env to the intended chunk settings; this spec does not add a dedicated 1000/200 settings run.
- For applicable Ragas metrics, pass the reviewed reference answer only to metrics that require it, such as Answer Correctness. Keep reference-free metrics independent of the reference answer.
- Follow the accepted line-anchored evidence and mode-specific evaluation decisions in ADR 0003. Keep the fixture isolated from user notes as required by ADR 0002.
- Defer index artifact destination changes, evaluation report redesign, metric changes, and production indexing refactors to later work.

## Testing Decisions

- Good tests verify externally observable fixture and evaluation behavior: reviewed cases load, annotations match source content, the evaluation process uses the configured chunk settings, and all cases complete without accessing user notes or persistent user storage.
- Use the existing high-level evaluation runner as the single fixture-to-evaluation seam.
- Exercise the runner with offline fake generation and judge providers. Keep the fixture index-building path under test and avoid live provider credentials or network calls.
- Verify that all 50 case IDs are present in the completed run and that answerable, ambiguous, and unanswerable cases retain their expected evaluation behavior.
- Verify that invalid fixture paths, evidence line ranges, evidence text, duplicate IDs, and missing required annotations are rejected with case-specific errors.
- Verify that reference answers are supplied to reference-dependent Ragas metrics and are not supplied to reference-free metrics.
- Reuse the existing evaluation-runner and QA-fixture-validation test patterns. Do not add separate tests for a hard-coded 1000/200 configuration pair.

## Out of Scope

- Changing the target corpus size from eight notes.
- Redesigning the production index format or moving evaluation index artifacts to a new destination.
- Changing the evaluation report layout, per-query telemetry, or aggregate metrics.
- Changing the Ragas metric set or judge-provider configuration.
- Automatically generating or trusting unreviewed QA references.
- Running evaluation over user notes or modifying the user's persistent index.
- Editing .env; the user will set the desired chunk size and overlap.
- Making statistical claims about production retrieval quality from this synthetic benchmark alone.

## Further Notes

- The existing fixture already uses source-line evidence annotations and a QA loader that validates source paths, line bounds, and exact evidence text. Keep these checks authoritative as note content and annotations expand.
- Existing evaluation code builds a fresh fixture-only index in temporary storage and reads the active settings. Preserve this isolation.
- The accepted ADR 0003 uses reviewed reference answers for Ragas Answer Correctness and reference-free Faithfulness and Answer Relevancy. The fixture should remain compatible with that distinction.
- This spec covers the fixture-first phase only. Index artifact placement and evaluation-report improvements can be specified after the expanded fixture is reviewed.
