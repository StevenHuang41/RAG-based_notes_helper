# 03: Run one query mode per evaluation

**What to build:** Route evaluation through exactly one workflow selected by the CLI while keeping both workflows isolated to the same reviewed fixture corpus.

**Blocked by:** 01: Line-anchored fixture evidence and chunks.

**Status:** resolved

- [x] `rag-app --eval` runs only Classic RAG.
- [x] `rag-app --agentic --eval` runs only Agentic RAG.
- [x] Each run uses the same fixed fixture QA and synthetic notes, builds a fresh temporary index, and avoids user `data/` and persistent `storage/`.
- [x] Record the selected workflow in run metadata and each case record.
- [x] Preserve Agentic's per-search result order and search boundaries for ranking metrics and report details.
- [x] Keep `--eval-base` behavior separate and unchanged.
- [x] Add CLI/evaluator coverage proving one workflow is called per invocation and the user index is not loaded.
