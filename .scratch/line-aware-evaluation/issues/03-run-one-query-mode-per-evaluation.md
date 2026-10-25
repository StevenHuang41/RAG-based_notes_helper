# 03: Run one query mode per evaluation

**What to build:** Route evaluation through exactly one workflow selected by the CLI while keeping both workflows isolated to the same reviewed fixture corpus.

**Blocked by:** 01: Line-anchored fixture evidence and chunks.

**Status:** ready-for-agent

- [ ] `rag-app --eval` runs only Classic RAG.
- [ ] `rag-app --agentic --eval` runs only Agentic RAG.
- [ ] Each run uses the same fixed fixture QA and synthetic notes, builds a fresh temporary index, and avoids user `data/` and persistent `storage/`.
- [ ] Record the selected workflow in run metadata and each case record.
- [ ] Preserve Agentic's per-search result order and search boundaries for ranking metrics and report details.
- [ ] Keep `--eval-base` behavior separate and unchanged.
- [ ] Add CLI/evaluator coverage proving one workflow is called per invocation and the user index is not loaded.

