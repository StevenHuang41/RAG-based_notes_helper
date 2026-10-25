# 01: Line-anchored fixture evidence and chunks

**What to build:** Add validated source-line provenance to reviewed QA evidence and newly indexed chunks, so evaluation can compare evidence by file and line rather than unstable chunk IDs.

**Status:** resolved

- [x] Replace each fixture `expected_evidence` text entry with an object containing `source`, 1-based inclusive `start_line`/`end_line`, and exact `text`.
- [x] Migrate all existing seed cases without removing IDs or changing the four categories and their minimum counts.
- [x] Validate source paths remain within the synthetic fixture corpus, line bounds exist, and annotated text matches the indicated source lines; failures identify case ID and path.
- [x] Require every answerable case to provide at least one evidence span; unanswerable cases may omit evidence.
- [x] Extend chunk records and all text loaders to retain 1-based inclusive source line ranges; overlapping lines may appear in adjacent chunks.
- [x] Preserve enough line boundaries in chunk text to audit the recorded range against the source file.
- [x] Persist and retrieve chunk line metadata through index metadata storage.
- [x] Keep legacy persistent indexes queryable; represent missing legacy line positions as unknown and never fabricate a line range. Fresh fixture indexes must always have valid line ranges.
- [x] Add automated coverage for single-line, multi-line, overlapping, invalid, and text-mismatched annotations, plus metadata round-tripping.

## Comments

- Resolved in commit `17e8f30`, merged to the integration branch at `fecb6d8`. Focused coverage passed (27 tests); the implementer also reported the full suite passed (100 tests). `git diff --check` passed.
