# 01: Add reproducible sampling to fixture evaluation

**What to build:** Let evaluators run a small, reproducible subset of reviewed fixture cases for Classic or Agentic evaluation, and inspect exactly which cases ran in the report.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Add `--sample N` and `--seed INTEGER` to `--eval` and `--agentic --eval`; reject either flag unless fixture `--eval` is active, including with `--eval-base`.
- [x] Require a positive sample count no greater than the fixture size. Select uniformly without replacement and execute selected cases in fixture order.
- [x] With a seed and sample count, select the same case IDs for the same fixture and settings. Without a seed, permit a fresh subset on each run.
- [x] When `--sample` is omitted, run the full fixture in its original order; `--seed` alone does not change selection or order.
- [x] Record the requested sample count, seed, and selected case IDs in the human-readable report and machine-readable results, including selected IDs for unseeded runs.
- [x] Preserve reviewed fixture data and isolated evaluation index behavior; do not claim small samples are statistically representative.
- [x] Add offline-fake coverage for CLI validation/forwarding, runner selection and report metadata, and matching Classic/Agentic selections for identical settings.

## Comments

- Implemented on `integration/reproducible-evaluation-sampling`; full pytest passed (145 tests), with no live providers.
