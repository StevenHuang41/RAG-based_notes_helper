# Line-Aware, Mode-Specific RAG Evaluation

Status: resolved

## Problem Statement

The current fixture evaluation runs Classic and Agentic RAG together and writes a timestamped directory containing a log, JSON results, and a JSON summary. The QA fixture identifies expected files and evidence snippets, but not source line locations. Retrieved chunks likewise do not carry line ranges, so retrieval coverage and ranking cannot be measured precisely against reviewed evidence. The report does not provide one human-readable, per-run document that combines aggregate metrics with query-level retrieval, generation, latency, judge, and token details.

## Goals

- Evaluate exactly one workflow per invocation: `rag-app --eval` for Classic and `rag-app --agentic --eval` for Agentic.
- Keep evaluation isolated to the version-controlled synthetic notes and reviewed QA fixture; never read or mutate the user's notes or persistent index.
- Anchor expected evidence and newly indexed chunks to 1-based, inclusive source line ranges.
- Measure retrieval quality with line-aware Recall, Precision, and NDCG, while retaining useful Ragas context diagnostics.
- Measure generated answers with Ragas Faithfulness, Answer Relevancy, and reference-based Answer Correctness.
- Produce a readable, auditable run report with aggregate results first and detailed results for every test query after it.
- Record per-query process latency and provider-reported token usage, including Agentic controller calls and judge calls.
- Make each evaluation run immutable by assigning it a unique local-time directory name and refusing to overwrite an existing run.

## Non-Goals

- Changing the legacy `--eval-base` smoke workflow or its use of the configured user index.
- Treating the initial fixture as a statistically representative benchmark.
- Estimating tokens with a local tokenizer when a provider does not report usage.
- Persisting hidden chain-of-thought or complete judge prompts.
- Adding a new RAG workflow, external evidence source, MCP integration, or cross-query memory.

## User Stories

1. As an evaluator, I want to select Classic or Agentic with the CLI flags, so each report represents one workflow clearly.
2. As an evaluator, I want each fixture evidence annotation to name its source path, inclusive start/end lines, and exact text, so ground truth can be checked against the source.
3. As an evaluator, I want every newly indexed chunk to retain its source path and inclusive line range, so retrieved evidence can be compared with fixture evidence.
4. As an evaluator, I want line-aware Recall, Precision, and NDCG, so overlap and wrong-source retrieval are represented accurately.
5. As an evaluator, I want the report to show which chunks were retrieved, their IDs, source paths, line ranges, and similarity scores, so I can inspect ranking behavior.
6. As an evaluator, I want Ragas answer metrics and judge details, so I can inspect grounding, relevance, and correctness of generated answers.
7. As an evaluator, I want stage latency and provider-reported token usage per query and in aggregates, so I can compare workflow cost and speed.
8. As an evaluator, I want a `README.md`, `results.json`, and `run.log` in each run directory, so I can read, process, and audit the same evaluation.
9. As a maintainer, I want the existing four fixture categories and minimum coverage retained, and all evidence annotations validated against source files, so regression coverage remains trustworthy.

## Domain Definitions

- **Evidence span**: a contiguous, inclusive range of 1-based physical lines in one source file, paired with the exact annotated text.
- **Retrieved chunk coverage**: the union of source-line positions covered by retrieved chunks, keyed by source path. Overlap within the same source is counted once.
- **Retrieval Recall**: relevant ground-truth line positions covered by retrieved chunks divided by all unique ground-truth line positions.
- **Retrieval Precision**: relevant ground-truth line positions covered by retrieved chunks divided by all unique line positions covered by retrieved chunks. Lines from non-ground-truth files count in the denominator but never in the numerator.
- **NDCG@k**: normalized discounted cumulative gain over the ranked retrieved chunks, where each chunk's graded relevance is the number of previously uncovered ground-truth line positions it newly covers; `k` is the configured retrieval `top_k`.
- **Run timestamp**: the local `Asia/Taipei` wall-clock time formatted as `YYYY-MM-DD-HHmmss`.

## Functional Requirements

### Mode selection and isolation

- `rag-app --eval` evaluates only Classic RAG.
- `rag-app --agentic --eval` evaluates only Agentic RAG.
- Both commands use the same fixed QA fixture and synthetic notes corpus, and each builds a fresh isolated fixture index.
- Neither mode reads or modifies user `data/` or persistent `storage/`.
- `--eval-base` remains separate and unchanged.

### Fixture and source-line metadata

- Each expected evidence annotation is an object with `source`, `start_line`, `end_line`, and `text`.
- Line ranges are 1-based and inclusive. A single-line annotation has equal start and end lines.
- One case may have multiple evidence spans across one or more files.
- The QA fixture keeps `direct`, `multi_hop`, `unanswerable`, and `ambiguous` categories, with at least four cases per category and all current seed IDs retained.
- Fixture validation checks unique IDs, required case fields, source paths within the synthetic fixture corpus, valid line bounds, and exact agreement between annotated text and the indicated source lines.
- Newly created chunks carry `start_line` and `end_line`; overlap lines may appear in adjacent chunks.
- Chunk text preserves source line boundaries sufficiently to map its line range back to the original file.
- Existing user indexes remain queryable during metadata evolution. Evaluation always builds a fresh index, so all evaluation chunks must have valid line metadata. Re-indexing behavior for legacy indexes must be explicit and must not silently claim unknown line ranges as known.

### Retrieval scoring

- Retrieval results preserve rank order and expose `chunk_id`, `source`, `start_line`, `end_line`, and the existing FAISS similarity score.
- Ground-truth and retrieved intervals are compared only when their source paths match.
- For Recall and Precision, line positions are inclusive and deduplicated by `(source, line_number)`. Overlapping retrieved chunks from the same file do not count a line more than once.
- A retrieved line from a file absent from the ground truth contributes to the Precision denominator, not the numerator.
- Recall is the count of retrieved ground-truth line positions divided by the count of all ground-truth line positions.
- Precision is the count of retrieved ground-truth line positions divided by the count of all unique retrieved line positions.
- NDCG uses the retrieval rank order and newly covered relevant-line count per chunk as graded relevance; chunks that only repeat already-covered evidence add no relevance gain.
- Classic computes NDCG over its single ranked retrieval list. Agentic preserves each search call as a separate ranked list, computes NDCG per search, and reports the arithmetic mean over applicable searches for the query; Recall and Precision use the deduplicated union of chunks across searches.
- Empty or absent ground truth yields a not-applicable value rather than a fabricated zero. Empty retrieval yields zero recall and not-applicable precision/NDCG when their denominators are empty.
- Ragas `ContextUtilization` remains available for applicable answerable cases and is presented under Retrieval diagnostics.

### Generation and judge metrics

- Generation metrics for expected-answerable cases are Ragas `Faithfulness`, `AnswerRelevancy`, and `AnswerCorrectness`.
- `AnswerCorrectness` receives the fixture's reviewed reference answer. The other metrics receive only the inputs required by their metric contracts.
- Unanswerable and ambiguous cases are evaluated with deterministic answerability and abstention checks and do not receive answerable-case Ragas scores.
- The report records judge provider/model, per-metric scores and latency, aggregate judge latency, and provider-reported judge token usage.
- Include a metric rationale only when the Ragas metric returns one. Do not store hidden reasoning or full prompts.

### Per-query telemetry

- Record retrieval latency, generation latency, judge latency, and end-to-end query latency.
- Agentic runs retain latency for each search and aggregate search latency for the query.
- Record latency for each controller, answer-generation, and judge LLM call; expose controller latency separately from vector-search latency.
- Record every provider-reported LLM usage call with call type (`controller`, `answer`, or `judge`), provider/model, and input/output/total token counts when available. Judge calls also identify the Ragas metric that triggered them, so usage can be aggregated per metric.
- Provide per-query, per-judge-metric, category, and whole-run token totals, with generation/controller usage separated from judge usage.
- Missing usage remains explicitly `unavailable` or `partial`; do not estimate token counts.
- Keep Ragas judge usage distinct from workflow generation usage.

### Run artifacts and README

- Each run writes to `eval/YYYY-MM-DD-HHmmss/` using `Asia/Taipei` local time.
- If that exact directory already exists, fail without modifying it and report the collision clearly.
- Each completed run contains `README.md`, `results.json`, and `run.log`; generated run artifacts remain ignored by Git.
- `README.md` begins with workflow, run timestamp, fixture digest, configuration, category counts, overall metrics, and per-category metrics.
- After aggregate results, `README.md` includes a detail section for every query. Each detail shows the query, reference answer, generated answer/abstention, per-query latency, retrieval metrics, the ranked retrieved-chunk table (chunk ID, source path, line range, similarity score, and chunk text), citations, generation metrics, judge metrics/details, and token usage.
- `results.json` contains the complete machine-readable per-case records and run metadata, including retrieved chunks and usage records.
- `run.log` records evaluation start, fixture loading, isolated index build, each case, metric stages, artifact writing, failures, and interruption. Log entries identify case IDs and stages without dumping source-note contents or credentials.
- The README links to `results.json` and `run.log`. The separate `summary.json` is removed to avoid duplicating the human-readable summary.
- Any displayed retrieval score is named `similarity_score` (or equivalently explicit UI text) and documented as the FAISS score; do not imply it is calibrated probability.

### CLI output

- Evaluation progress is visible in terminal use and identifies the selected workflow and fixture-index stages.
- The fixture evaluation must not print a misleading user-index message such as `Index not found. Building index from notes ...`.
- Print the successful index-build message only once.
- Keep evaluation reports in the run directory; do not print query or note content as progress output.

## Acceptance Criteria

- The two CLI invocations produce one workflow each, and both use identical fixture inputs and isolated indexes.
- The fixture validator rejects missing, invalid, out-of-bounds, wrong-source, or text-mismatched evidence annotations.
- Chunk line metadata round-trips through index metadata and is available in evaluation retrieval results.
- Unit-level scoring examples cover overlapping same-file intervals, multiple ground-truth files, retrieved chunks from an unrelated file, inclusive single-line spans, and empty denominators.
- NDCG tests verify ranking sensitivity and that duplicate overlap does not add relevance gain twice.
- AnswerCorrectness is given the reviewed reference answer; reference-free metrics do not receive it.
- Per-query and aggregate records expose latency and token usage by call type, retaining partial/unavailable states.
- Each successful run writes a unique timestamped directory with `README.md`, `results.json`, and `run.log`; an existing directory is not overwritten.
- README aggregate sections precede query details, and each query detail includes retrieved chunk content and the agreed metrics/telemetry.
- Legacy `--eval-base` behavior remains unchanged.
- Indexing prints one success message and evaluation progress identifies the synthetic fixture index.

## Risks and Trade-offs

- Line-anchored annotations are precise but need maintenance when fixture notes move or are edited; validation should fail early with case ID and source path.
- Adding line metadata to persistent index records requires a compatibility policy for old metadata. Evaluation is insulated by rebuilding a fresh fixture index, while normal index migration must remain explicit.
- A verbose README can grow large as fixtures expand. The requested inline chunk detail favors auditability; `results.json` remains the complete data source for external analysis.
- Ragas judge metrics are model-dependent, can consume tokens, and are diagnostics rather than objective proof of correctness.
- Inclusive line-based metrics intentionally penalize irrelevant retrieved lines and wrong-source chunks; NDCG additionally evaluates their order.
- The four-category fixture remains a regression set, not a statistically representative benchmark.

## Testing Approach

- Use synthetic fixture files and deterministic fake providers for offline automated coverage.
- Verify source-line boundaries through normal ingestion, chunking, metadata storage, and retrieval.
- Verify each CLI mode dispatches one workflow and never accesses the configured user index.
- Verify metrics with small hand-computed interval examples and ranked chunks.
- Verify Ragas payload composition, metric applicability, judge usage, latency capture, and AnswerCorrectness reference input without live API credentials.
- Verify report generation, ordering, artifact naming, collision behavior, and duplicate index-output removal.
- Run the full test suite and `git diff --check` during implementation; do not treat live provider scores as deterministic assertions.

## Out of Scope

- Changing candidate QA generation or adding unreviewed generated cases to the fixed fixture.
- Automatic fixture repair when source line numbers change.
- Statistical significance claims, cross-run dashboards, or a hosted evaluation service.
- Logging secrets, full judge prompts, hidden chain-of-thought, or user note contents.
