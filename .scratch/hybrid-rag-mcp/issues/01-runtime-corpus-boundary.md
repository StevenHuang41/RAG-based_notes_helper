# 01: Enforce the allowed runtime note corpus

**What to build:** Runtime note searches use only the approved local corpus, and index updates stop returning chunks from notes that have been removed or are no longer allowed.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Runtime ingestion resolves supported note files beneath the approved `data/` root and never reads or indexes files outside that root.
- [x] Path traversal and symlinks that resolve outside the approved root cannot expand the searchable corpus.
- [x] Index rebuilds remove chunks for deleted or excluded notes, including when removals are the only corpus change.
- [x] Runtime searches return no chunks from sources outside the current allowed corpus after a rebuild.
- [x] Fixture-backed evaluation continues to use its isolated evaluation corpus and does not read or mutate runtime notes or the persistent runtime index.
- [x] Automated tests exercise allowed files, out-of-root paths, removed files, and deletion-only index updates through observable ingestion, rebuild, and search behavior.

## Answer

Runtime ingestion is constrained to `PROJECT_ROOT/data`; configured roots and
explicit runtime paths outside it fail closed, file symlinks must resolve
inside the selected root, and symlink directories are not traversed. Smart
rebuild now detects deleted or excluded notes even when no files were added or
changed, and writes an empty searchable index if the runtime corpus is empty.
Fixture evaluation uses a separate explicit isolated-corpus loader.

Offline verification passed with the existing Python 3.14 environment:
`python -m pytest -q` (full suite) and `git diff --check`. The Python 3.13 `uv`
environment could not be populated because uncached packages were unavailable
without network access.
