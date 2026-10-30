# Separate evaluation assets from pytest code

Status: ready-for-agent

## Problem Statement

The reviewed evaluation corpus and QA ground truth currently live beneath `tests/fixtures/`, even though they define the RAG benchmark rather than pytest-only test inputs. Generated evaluation reports use a separate root-level location, while a candidate-authoring workflow adds a third, unreviewed data lifecycle. This makes it hard to see what belongs to Python tests, what is version-controlled benchmark truth, and what is generated output.

The project needs a simple, stable evaluation layout that keeps the evaluation corpus, reviewed QA set, and generated reports visibly separate from application data, persistent indexes, and pytest code.

## Solution

Create one repository-root evaluation-assets boundary with three directories: `evals/notes/` for the version-controlled evaluation corpus, `evals/qa/` for the version-controlled reviewed QA ground truth, and `evals/results/` for Git-ignored run artifacts.

Keep pytest code under `tests/`, and keep the development-only evaluation implementation importable from the source package. Evaluation continues to build an isolated temporary index from the evaluation corpus and never reads or mutates the user's `data/` or persistent `storage/` directories. Remove the candidate-authoring generator and its draft-output lifecycle.

## User Stories

1. As an evaluation reviewer, I want all version-controlled evaluation inputs under one `evals/` boundary, so that I can distinguish them from pytest code and application data.
2. As an evaluation reviewer, I want the synthetic note corpus stored under `evals/notes/`, so that the documents evaluated by the RAG workflow are obvious.
3. As an evaluation reviewer, I want the reviewed QA ground truth stored under `evals/qa/`, so that questions, reference answers, answerability labels, and expected evidence are clearly identified as benchmark truth.
4. As an evaluation reviewer, I want reports stored under `evals/results/`, so that I can inspect completed runs without searching among source inputs.
5. As a maintainer, I want `tests/` to contain pytest code only, so that a test fixture is not confused with the authoritative evaluation benchmark.
6. As a maintainer, I want one copy of every evaluation note and QA record, so that duplicated fixtures cannot drift apart.
7. As an evaluator, I want Classic and Agentic evaluation to continue using the same reviewed corpus and QA set, so that their results remain comparable.
8. As an evaluator, I want evaluation to construct an isolated temporary index from `evals/notes/`, so that my personal notes and persistent index never affect benchmark results.
9. As an evaluator, I want the reviewed QA set to retain corpus-relative source identifiers such as `budget.md`, so that ground truth identifies documents without coupling itself to repository layout.
10. As a maintainer, I want the evaluation asset locator to reject a missing or incomplete checked-out benchmark clearly, so that evaluation never silently falls back to user data or storage.
11. As an evaluator, I want `rag-app --eval` and `rag-app --agentic --eval` to remain the only supported evaluation commands, so that every run has a fixed, auditable input boundary.
12. As an evaluator, I want generated reports to retain their timestamped run directory, collision refusal, human-readable report, machine-readable results, and process log, so that every run remains auditable.
13. As a maintainer, I want reports to be Git-ignored, so that provider-dependent generated output does not create repository noise or accidental benchmark changes.
14. As a maintainer, I want unreviewed Ragas candidate generation removed, so that the fixed benchmark has no separate draft lifecycle, live-provider authoring cost, or extra output directory.
15. As a maintainer, I want modifying corpus content to require explicit review of affected QA evidence, so that line-anchored evidence cannot silently become invalid.
16. As a documentation reader, I want repository guidance to describe the new evaluation boundary consistently, so that instructions, ignore rules, architecture decisions, and code agree.
17. As a future maintainer, I want the prior layout decision preserved and superseded by a new ADR, so that the reason for this architectural change remains traceable.
18. As a test author, I want a single high-level evaluation seam with offline fakes, so that layout changes are validated as observable behavior without live providers.

## Implementation Decisions

- Establish `evals/notes/`, `evals/qa/`, and `evals/results/` as the complete root-level evaluation-assets layout. The reviewed QA file remains named `notes_qa.json` beneath `evals/qa/`.
- Move, rather than copy, the reviewed corpus and QA set out of `tests/fixtures/`. Pytest tests may consume evaluation assets but do not own duplicate benchmark data.
- Preserve the semantic content of reviewed QA records. Their source fields remain paths relative to the evaluation corpus root, not repository-relative paths.
- Introduce one explicit repository evaluation-asset locator. It resolves and validates the evaluation corpus and reviewed QA set before an evaluation run; it must not fall back to application data, persistent storage, or the caller's arbitrary current directory.
- Keep the evaluation implementation as development-only source-package code. Do not move importable Python modules to the root evaluation-assets directory.
- Make the evaluation runner build its isolated temporary index from the resolved evaluation corpus. Continue to use the existing ingestion, chunking, indexing, evidence validation, and evaluation workflows.
- Preserve the existing Classic-only and Agentic-only CLI modes, sampling behavior, report schema, metric definitions, local-time timestamp convention, collision refusal, and three-artifact report contract. Only the default report root changes to `evals/results/`.
- Ignore all generated contents of `evals/results/`; create result directories only when a run starts. Do not track placeholder files merely to retain an empty generated directory.
- Do not move or delete pre-existing, Git-ignored root-level `eval/` artifacts. They are historical local output and outside this migration's destructive scope.
- Remove the candidate-generator module, its documentation, tests, draft output location, and any related configuration or ignore rules. Direct, reviewed edits to the fixed evaluation corpus and QA set remain the only benchmark-maintenance path.
- Update all documentation and metadata that reference the former fixture, report, or candidate-draft locations. Record a new ADR that supersedes the older repository-local evaluation layout decision while retaining that historical ADR unchanged.
- Retain the reviewed QA validator: a changed note that makes an expected source, line range, or evidence text invalid must fail clearly and require an explicit reviewed QA update.

## Testing Decisions

- Good tests assert externally observable evaluation behavior: which checked-in corpus and QA set are loaded, that the index is isolated, which artifacts are produced, and how missing assets fail. They do not assert private locator implementation details or hard-code parent-directory traversal.
- Use the existing high-level evaluation-runner seam with offline fake providers and a temporary report output root as the primary acceptance seam.
- Verify the evaluation runner loads the relocated corpus and QA set, creates the isolated index from that corpus, and writes the required report artifacts to the new default results root.
- Verify both Classic and Agentic paths continue to use the same reviewed evaluation assets and preserve their existing workflow selection behavior.
- Verify a missing corpus directory, missing QA file, or malformed reviewed QA set produces an actionable failure and never falls back to `data/` or `storage/`.
- Verify QA source identifiers remain corpus-relative and that existing line-anchored validation still detects stale expected evidence after corpus changes.
- Verify generated reports under the new results root are ignored by Git, while `evals/notes/` and `evals/qa/` remain version-controlled.
- Verify no candidate-generator command, module entry point, draft output behavior, or related documentation remains.
- Reuse existing evaluation-runner, fixture-validation, CLI, report, and candidate-generator test patterns where applicable; replace candidate-generator tests with removal-oriented coverage only where needed. Do not use live providers or network access.

## Out of Scope

- Changing reviewed questions, reference answers, answerability labels, evidence line spans, categories, retrieval metrics, Ragas metrics, report schema, sampling semantics, or Agentic RAG behavior.
- Evaluating the user's `data/` directory or persistent `storage/` snapshot.
- Adding path-override flags for arbitrary evaluation corpora, QA sets, or report roots.
- Migrating, deleting, interpreting, or committing old root-level `eval/` artifacts.
- Adding a replacement automated candidate-generation, draft-review, or benchmark-evolution workflow.
- Changing runtime package inclusion rules beyond what is necessary to preserve the existing development-only evaluation command.

## Further Notes

- The evaluation corpus is not application data. The reviewed QA set is not generated output. The results directory is not benchmark ground truth.
- A source value such as `budget.md` is resolved relative to the evaluation corpus root; it becomes `evals/notes/budget.md` at runtime without embedding that repository layout in reviewed QA records.
- The evaluation benchmark is intentionally fixed. If its corpus is intentionally edited later, reviewed QA evidence must be updated in the same deliberate change after validator failures identify affected records.
- The previously accepted repository-local evaluation ADR remains historical context. The new ADR will supersede its layout assertions, not rewrite its past decision.
