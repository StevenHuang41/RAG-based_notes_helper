# 01: Enforce the allowed runtime note corpus

**What to build:** Runtime note searches use only the approved local corpus, and index updates stop returning chunks from notes that have been removed or are no longer allowed.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Runtime ingestion resolves supported note files beneath the approved `data/` root and never reads or indexes files outside that root.
- [ ] Path traversal and symlinks that resolve outside the approved root cannot expand the searchable corpus.
- [ ] Index rebuilds remove chunks for deleted or excluded notes, including when removals are the only corpus change.
- [ ] Runtime searches return no chunks from sources outside the current allowed corpus after a rebuild.
- [ ] Fixture-backed evaluation continues to use its isolated evaluation corpus and does not read or mutate runtime notes or the persistent runtime index.
- [ ] Automated tests exercise allowed files, out-of-root paths, removed files, and deletion-only index updates through observable ingestion, rebuild, and search behavior.
