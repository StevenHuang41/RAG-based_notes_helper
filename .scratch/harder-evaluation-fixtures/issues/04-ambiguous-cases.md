# 04: Add ambiguous cases with reviewed rationales

**What to build:** Let evaluators check that Notes Helper abstains when related project notes conflict or omit enough context to determine a single supported answer.

**Blocked by:** 01: Establish the realistic fixture foundation

**Status:** resolved

- [ ] Add eight reviewed ambiguous cases, bringing the ambiguous category to twelve cases.
- [ ] Record why each case is ambiguous and identify the conflicting or underspecified fixture evidence.
- [ ] Mark cases as unanswerable for deterministic answerability and abstention evaluation when no single answer is supported.
- [ ] Keep any relevant source references auditable without presenting contradictory evidence as a uniquely correct answer.
- [ ] Confirm fixture validation accepts the cases and the evaluation runner processes them using the active chunk settings.

## Comments

- Implemented as `61b4f5b` and merged as `6584e07`. Added eight reviewed ambiguous cases, bringing the category to twelve. Evaluation and fixture validation tests passed (21 tests); `git diff --check` was clean.
