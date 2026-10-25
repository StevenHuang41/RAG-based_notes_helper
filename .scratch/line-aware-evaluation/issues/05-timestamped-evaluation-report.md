# 05: Timestamped evaluation artifacts and README report

**What to build:** Write a unique per-run directory with process logs, machine-readable case results, and a README that presents aggregate metrics before verbose query details.

**Blocked by:** 02: Line-aware retrieval metrics and answer correctness; 03: Run one query mode per evaluation; 04: Per-query latency and provider token usage.

**Status:** ready-for-agent

- [ ] Name run directories `eval/YYYY-MM-DD-HHmmss/` using `Asia/Taipei` local time.
- [ ] Refuse to overwrite a directory with the same timestamp and return a clear collision error.
- [ ] Write `run.log`, `results.json`, and `README.md`; remove the duplicate `summary.json` artifact.
- [ ] Ensure `results.json` includes run metadata and complete per-query records, including retrieved chunks, scores, source line ranges, latency, judge results, and usage records.
- [ ] Write a README aggregate section first with mode, timestamp, fixture digest, configuration, category counts, overall metrics, and per-category metrics.
- [ ] After aggregate results, include every query's question, reference/generated answer or abstention, latency, Retrieval metrics and ranked chunk details, citations, Generation metrics, LLM-as-judge scores/details, and token usage.
- [ ] Include chunk ID, source path, inclusive line interval, similarity score, and chunk text for each retrieved result.
- [ ] Link the README to `results.json` and `run.log`; keep generated reports ignored by Git.
- [ ] Log run start, fixture load, isolated index build, each case, metric stages, successful artifact writing, failures, and interruption without note contents or credentials.
- [ ] Keep reports and records clear about unavailable metrics/usage and document that FAISS similarity is not a calibrated probability.
- [ ] Test README section order and detail, JSON records, log contents, timezone naming, and collision behavior.

