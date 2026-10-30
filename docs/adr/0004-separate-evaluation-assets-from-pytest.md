# Separate evaluation assets from pytest code

Status: Accepted

The evaluation corpus and reviewed QA set live under `evals/notes/` and `evals/qa/`, while generated reports live under Git-ignored `evals/results/`; pytest retains test code only. This supersedes the layout assertions of ADR-0002: it keeps the fixed benchmark visibly separate from user `data/`, persistent `storage/`, pytest implementation tests, and generated output, without changing the isolated fixture-backed evaluation contract.

## Considered Options

- Keep the benchmark under `tests/fixtures/` and reports under root `eval/`.
- Store repository-layout prefixes in reviewed QA source identifiers.

The first option conflates benchmark truth with pytest-owned inputs and splits evaluation assets across the repository. The second couples reviewed evidence to a checkout layout; QA source identifiers remain relative to the evaluation corpus root instead.

## Consequences

- Corpus edits must be accompanied by an explicit review of affected QA evidence; validation rejects stale line-anchored evidence.
- Candidate generation and its unreviewed draft lifecycle are absent. The fixed corpus and reviewed QA set are the only benchmark-maintenance inputs.
