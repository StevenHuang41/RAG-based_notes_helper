# 01: Line-anchored fixture evidence and chunks

**What to build:** Add validated source-line provenance to reviewed QA evidence and newly indexed chunks, so evaluation can compare evidence by file and line rather than unstable chunk IDs.

**Status:** ready-for-agent

- [ ] Replace each fixture `expected_evidence` text entry with an object containing `source`, 1-based inclusive `start_line`/`end_line`, and exact `text`.
- [ ] Migrate all existing seed cases without removing IDs or changing the four categories and their minimum counts.
- [ ] Validate source paths remain within the synthetic fixture corpus, line bounds exist, and annotated text matches the indicated source lines; failures identify case ID and path.
- [ ] Extend chunk records and all text loaders to retain 1-based inclusive source line ranges; overlapping lines may appear in adjacent chunks.
- [ ] Preserve enough line boundaries in chunk text to audit the recorded range against the source file.
- [ ] Persist and retrieve chunk line metadata through index metadata storage.
- [ ] Keep legacy persistent indexes queryable; represent missing legacy line positions as unknown and never fabricate a line range. Fresh fixture indexes must always have valid line ranges.
- [ ] Add automated coverage for single-line, multi-line, overlapping, invalid, and text-mismatched annotations, plus metadata round-tripping.

