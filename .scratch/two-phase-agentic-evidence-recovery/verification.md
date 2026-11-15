# Two-phase Agentic verification

Status: Implementation and verification completed locally, with the live limitations below.

## Checkout and scope

- Integration branch: `integration/two-phase-agentic-evidence-recovery`.
- Starting commit: `e2fa2f6`.
- Implementation is currently uncommitted. The earlier instruction not to commit remains in effect; the optional request for local-commit authorization has not received an answer.
- Existing planning changes and the earlier Search Controller-latency-row fix were preserved. Isolated implementer worktrees were integrated using scoped patches rather than Git commits/merges. No remote changes or merge into main occurred.

## Deterministic verification

- Approved seams: public `run_query()` results/callbacks and `run_evaluation()` JSON/Markdown artifacts.
- Final full suite: `UV_CACHE_DIR=/tmp/rag-notes-uv-cache uv run --no-sync --offline pytest` — **264 passed, 1 warning**.
- The warning is an existing LangChain pending deprecation concerning `allowed_objects`; it is reported, not silently suppressed.
- `git diff --check` passes.
- No configured lint/typechecking command was found; Ruff, mypy, and Pyright are not installed. No lint/typecheck pass is claimed.
- All 17 runtime and reviewed benchmark files match the pre-implementation hashes after tests and live runs, with no additions or removals.

## Review

- Independent Standards and Spec reviews examined the working-tree implementation relative to `e2fa2f6`, including new tests and the approved spec. This adapts the committed-diff review flow to the user's no-commit constraint.
- Standards: one P2 correctness finding — provider failure lost actual submitted generation context.
- Spec: the same P2 plus a separate P2 — evaluation dropped sanitized `failure_category`.
- A single implementer fixed both; red/green public-seam regression tests verify cause retention, actual token-truncated context retention, warning safety, and unchanged Classic exception behavior. The subsequent full suite passed.

## Live acceptance

- Classic baseline executed sequentially before Agentic: `UV_CACHE_DIR=/tmp/rag-notes-uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync --offline rag-app --eval --sample 8 --seed 10`.
- Selected IDs: direct-01, direct-03, multi-hop-04, multi-hop-10, multi-hop-12, multi-hop-13, ambiguous-08, unanswerable-13.
- Active answer/judge model: Ollama `gemma4:e2b`; configured context window 8192, output budget 1024, generation chunk limit 5. Private environment configuration was not edited.
- Classic artifacts: `evals/results/2026-11-15-013735/`; Agentic artifacts: `evals/results/2026-11-15-014442/`. Both completed, with identical selected IDs, fixture hash, models, generation limits, and context/output budgets.
- Classic: 8 normal completions, no degradation, all 8 eligible/scored per Ragas metric.
- Agentic comparison run: 5 accepted generated answers, 3 `answer_attempts_exhausted` warnings, no terminal provider/review errors, and 5 Retrieval degradations. The reliability measure excludes degraded executions, resulting in 3/8; it is not a count of correct answers. Three warnings are not accepted abstentions.
- Retrieval degradations: 2 `review_context_budget_exceeded`; 3 `schema_invalid_action`. Search counts were 1–3, Answer counts 1–3, selected/actual context sizes at most 5. Each query rendered both pool tables. Ragas scored 5 accepted answers per metric; 3 warnings were unavailable.
- Actual live adaptive Search, pool growth, conditional final rerank, same-context revisions, and revision exhaustion were observed. No accepted partial answer was observed in this sample; deterministic tests establish that contract, not live model quality.

| Paired metric on the same five accepted-case IDs | Classic | Agentic |
| --- | ---: | ---: |
| Faithfulness | 0.8000 | 0.8000 |
| Answer Relevancy | 0.4812 | 0.5030 |
| Answer Correctness | 0.8204 | 0.8427 |
| Context Utilization | 0.6400 | 0.6342 |

- These are single-run model-judge observations on five selected successful cases, not evidence of a stable Agentic advantage; the three warnings and five degradations must remain visible.
- Post-run artifact correction: warnings with `answer_attempts_exhausted` are now unavailable answerability, not correct abstentions. Only labels/derived summaries/Markdown were recomputed, without changing model answers, Ragas scores, or timings. Original Agentic JSON is retained as `results-before-answerability-fix.json`; the correction is annotated in the report/run log.
- Additional live probes used the isolated fixture and current settings: unintelligible query `zxqvzxqv qqqqqq !!!???` returned `needs_clarification` after 1 Search and 0 Answers; the combined product-owner/support-volume question used 3 Searches and 3 Answers before returning a safe revision-exhausted warning. Probe commands/results are preserved as `/tmp/two-phase-live-probes.py`, `/tmp/two-phase-live-probes.log`, and `/tmp/two-phase-live-probes.json`.
- After the comparison, reviewer input was improved to explicitly include `last_search` (query and thresholds), matching the validator's requirement that a recovery plan differ. Public-seam tests first failed without this field, then the full 264-test suite passed. The eight-case Ragas comparison above predates this prompt improvement and is not presented as a re-evaluation of the final prompt.
- Final-runtime live smoke: `How many customer support requests were received during the pilot?` returned the accepted answer-model abstention after 3 Searches and 1 Answer, with `search_budget_exhausted`, no Retrieval degradation, and `review_passed=true`. This replaces the earlier controller reference failure for that observed query. Exact command/result are preserved as `/tmp/two-phase-live-final-smoke.py`, `/tmp/two-phase-live-final-smoke.log`, and `/tmp/two-phase-live-final-smoke.json`; no Ragas re-judging was performed for this smoke test.
- Empty final pool, rerank-skipped pool sizes, and accepted-partial paths have deterministic coverage; not all were observed live. Context overflow still intentionally degrades, and model reviewers may reject useful candidates. These limitations remain visible rather than being suppressed.

## Local integration and cleanup

- Tickets 01–07 are resolved for implementation/verification scope; live semantic quality is limited as described above.
- No local commits, commit-based merges, push, or merge into main occurred. The complete implementation remains in the integration checkout as uncommitted changes.
- Only the four task-created worktrees were removed after their patches/untracked files were archived under `/tmp/two-phase-worktree-archives/` with a manifest. The pre-existing detached worktree was preserved. Agent branch refs remain at their original base commit; no unmerged commits were discarded.
