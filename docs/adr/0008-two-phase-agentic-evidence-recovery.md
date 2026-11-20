# Separate Agentic evidence recovery from answer generation

Status: Accepted — implemented locally; live-provider limitations are recorded in the feature verification report.

Amended by [ADR-0009](0009-remove-local-context-window-budget.md): local context-window budget checks and budget-driven trimming are removed. Complete-pool review and provider-failure fallback remain in effect.

## Context

The Agentic workflow completes bounded evidence recovery before answer generation, then reviews answers without reopening Retrieval. This amends ADR-0005 and ADR-0007's direct-first-generation, empty-result, evidence-selection, and post-generation Retrieval recovery decisions. It also amends ADR-0003's exclusion of abstentions from Ragas: accepted Answer-model-generated abstentions enter per-metric applicability checks, while deterministic and controller-origin abstentions remain excluded. Their remaining constraints continue to apply. The local implementation follows this decision; acceptance of the architecture does not guarantee model judgments or higher Ragas scores.

The initial empty-search decision below is amended: an empty first Search immediately returns a deterministic search-limitation response suggesting a revised question, without a controller, subsequent Search, or Answer call. A later empty Search preserves the existing nonempty pool.

## Confirmed design direction

- Search executes at most `AGENTIC_MAX_SEARCH_ATTEMPTS` times, including the first Search; the default remains three. Controller decisions and their validation retries are separate from executed Search attempts.
- Accumulate original retrieved chunks and provenance in a per-query pool, deduplicating by `(source, chunk_id)` across searches without replacing earlier evidence. Preserve first-seen order and append Search provenance for repeated chunks. Preserve Search audits; do not maintain a separate round coverage state.
- After each Search, assess the accumulated pool against the original question as `zero_evidence`, `partial_evidence`, or `sufficient_evidence`. Record the assessed question and missing information. These assessments are model judgments, not ground truth.
- Normal Retrieval stops only when pool coverage is sufficient (`sufficient_evidence`) or the Search budget is exhausted (`search_budget_exhausted`). Keep coverage separate from the stop reason: exhausted searches can leave zero or partial evidence. A missing or invalid next plan is degraded execution, not normal budget exhaustion.
- For zero or partial coverage, the controller may rewrite the query and adjust Vector minimum score and query-relative BM25 minimum-score ratio within `[0, 1]`. Retrieval limits remain independently configured; `LLM_MAX_CHUNKS` is not a per-search limit.
- An empty first Search returns deterministic text describing the unsuccessful search and suggesting a revised question without claiming corpus absence; it consumes no Answer attempt and is excluded from generated-answer Ragas averages.
- Retrieval review emits coverage, missing information, and a next Search plan rather than selecting chunk references. Allow one validation retry for invalid decisions. If review or intelligibility assessment remains invalid, or a provider failure prevents review, record degraded execution and continue to final context selection using the available pool; an empty pool follows the deterministic empty-pool outcome. Unconfirmed coverage is `unknown`, not `zero_evidence`.
- The pool itself remains intact. Submit the complete reviewer input without a local context-window budget check; provider failure uses the existing degraded Retrieval fallback. ADR-0009 supersedes this ADR's earlier local `review_context_budget_exceeded` rule. Do not silently truncate reviewer input and claim the whole pool was assessed.
- If pool size exceeds `LLM_MAX_CHUNKS`, rerank against the original question and pass only the top `LLM_MAX_CHUNKS` chunks to generation. Otherwise, skip this final rerank. Pool coverage does not guarantee coverage after context selection or truncation.
- Agentic reports show two aligned chunk metadata tables: the accumulated pool after Retrieval and the selected pool after final reranking/top-k selection. Both include evidence-line coverage and chunk provenance rather than raw chunk text. Show both tables even when final reranking is skipped, and label that skip explicitly. Retain the separate exact Answer generation context for evidence auditing.
- Generate an answer whenever the selected pool is non-empty, even if its chunks are judged irrelevant. Answers must disclose unsupported or missing portions of the requested information.
- Answer review checks relevance, completeness relative to available evidence, disclosure of missing information, and factual support against the exact generation context. It may request same-context revisions within an independent Answer attempt budget; it cannot reopen Retrieval. Partial answers with explicit limitations may be accepted.
- Record `answer_coverage` (`full`, `partial`, or `none`) independently from `review_passed`. Evidence-supported partial answers with explicit gaps may pass. Unsupported factual claims or undisclosed gaps require revision using the same context.
- Keep a revision-exhausted terminal outcome when the Answer attempt budget is exhausted without an accepted candidate. Never expose the rejected candidate.
- Allow one validation retry for invalid Answer review decisions. If review remains invalid or fails at the provider boundary, return a fixed user-facing warning rather than the unreviewed candidate. Empty answer-model output and generation provider failures also return a fixed warning. Preserve distinct internal causes for review failure, generation failure, and exhausted Answer attempts; do not expose internal failure codes in user-facing text.
- Use a warning such as “本次未能產生通過驗證的答案，請稍後再試。” for these unsuccessful generation/review outcomes. Fixed warnings are not accepted generated answers and do not enter their Ragas averages.
- All accepted generated final answers, including partial answers and abstentions, enter evaluation. Report per-metric applicability and `scored`, `unavailable`, or `failed` status with reasons. Retrieval-controller degradation is reported independently and does not automatically exclude an otherwise accepted generated answer from Ragas. Deterministic empty-pool and unintelligible-query outcomes are excluded from generated-answer Ragas averages.

## Trade-offs

Simpler Retrieval actions remove chunk-reference selection as a validation requirement, but do not guarantee valid controller responses or correct coverage judgments. Final top-k reranking can discard necessary evidence, so answer review must assess the actual generation context rather than inherit the pool's coverage judgment.

## Scope and validation

This decision changes Agentic orchestration and its evaluation reporting; Classic remains the comparison baseline. The existing configurable Search and Answer attempt settings and their revised roles are documented in `.env.example`; private `.env` configuration is preserved. Implementation, deterministic verification, and live observations are recorded in the feature's verification report. Local commits and remote integration are separate operations.
