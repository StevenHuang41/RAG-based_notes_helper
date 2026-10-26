# Reproducible sampling for RAG evaluation

Status: resolved

## Problem Statement

The fixture-backed `--eval` workflow always runs every reviewed case. This makes quick, small-sample checks take longer and consume more evaluation resources than needed. When a subset is used, evaluators also need to reproduce and audit exactly which cases were selected.

## Solution

Add `--sample N` and `--seed INTEGER` options to fixture evaluation. Sampling selects a reproducible, uniform subset without replacement when a sample count is supplied. The run report records the requested sampling configuration and selected case IDs. Without `--sample`, evaluation continues to run the full fixture, including when `--seed` is supplied alone.

## User Stories

1. As an evaluator, I want to run a small subset of reviewed cases, so that I can get faster feedback while iterating.
2. As an evaluator, I want to set a random seed for a sampled run, so that I can reproduce the same selected cases.
3. As an evaluator, I want sampling to work for both Classic and Agentic fixture evaluation, so that I can check either workflow with less time and provider usage.
4. As an evaluator, I want sampling to select cases uniformly without replacement, so that no case is duplicated and each fixture case has the same selection chance.
5. As an evaluator, I want selected cases to execute in fixture order, so that result ordering is predictable and easy to inspect.
6. As an evaluator, I want the report to show the requested sample count and seed, so that I can understand how the run was configured.
7. As an evaluator, I want the report to list selected case IDs, so that I can audit or repeat an unseeded run.
8. As an evaluator, I want a seed without a sample count to run the full fixture, so that a seed alone does not silently reduce coverage.
9. As an evaluator, I want invalid or oversized sample counts to fail clearly, so that a run cannot silently evaluate a different number of cases than requested.
10. As an evaluator, I want sampling flags outside fixture evaluation to fail clearly, so that settings are never silently ignored by normal query or legacy smoke modes.
11. As a maintainer, I want sampling to preserve the reviewed fixture and isolated evaluation index behavior, so that quick runs do not change evaluation ground truth or touch personal notes.
12. As an evaluator, I want an unspecified seed to allow a fresh random subset, so that I can explore different cases while retaining the exact selected IDs in the report.

## Implementation Decisions

- Add `--sample` with a positive integer argument and `--seed` with an integer argument to the CLI.
- Apply both options to fixture evaluation, including `--agentic --eval`; reject either option unless `--eval` is active. `--eval-base` remains unchanged and rejects these options.
- A sample count selects that many fixture cases uniformly without replacement. Reject a count greater than the available fixture case count.
- With `--sample` and no seed, select a fresh random subset. With both options, the same fixture content, sample count, and seed select the same cases.
- If `--sample` is omitted, evaluate the full fixture. A seed alone does not randomize, truncate, or reorder the fixture.
- Run selected cases in their original fixture order after selection.
- Preserve the fixture data and existing isolated-index behavior. Do not describe a small sample as statistically representative.
- Record the configured sample count and seed, plus the selected case IDs, as run metadata in the human-readable report and machine-readable results.
- Do not add an ADR; this is a reversible CLI and evaluation-run behavior choice.

## Testing Decisions

- Good tests assert externally observable CLI and run-result behavior, not the random-number implementation or private helper structure.
- Use the CLI boundary to verify option parsing, forwarding, and rejection when fixture evaluation is not active.
- Use the evaluation runner boundary with a fixed fixture and offline fakes to verify full-fixture behavior, positive and oversized count validation, no duplicate selections, seed reproducibility, fixture-order execution, and recorded metadata.
- Verify Classic and Agentic evaluation receive the same selected case IDs for identical sample settings.
- Verify reports include selected case IDs for both seeded and unseeded samples, and do not claim statistical representativeness.
- Reuse existing patterns in `test_cli_agentic.py` for CLI entry behavior and `test_eval_runner.py` for fixture-run behavior. Do not use live providers or network access.

## Out of Scope

- Sampling the legacy `--eval-base` dataset or changing its user-index behavior.
- Stratified sampling or guaranteeing coverage of every fixture category in a sample.
- Changing fixture cases, category definitions, evaluation metrics, report directory naming, or index isolation.
- Claiming that a sampled run is a statistically representative benchmark.
- Sampling/reordering cases when `--sample` is absent.

## Further Notes

- The selected IDs are necessary for auditing unseeded selections and for interpreting category coverage in a small sample.
- Reproducibility is defined against the same fixture contents and sampling implementation; reports retain selected IDs as the durable record of what ran.
- Implemented on `integration/reproducible-evaluation-sampling`. The full test suite passed (145 tests); no live providers were used.
