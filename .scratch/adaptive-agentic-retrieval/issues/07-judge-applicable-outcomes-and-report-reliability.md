# 07: Judge applicable outcomes and report workflow reliability

**What to build:** Evaluate every non-empty Classic and Agentic outcome with the Ragas diagnostics whose required inputs exist, while keeping deterministic behavior checks and operational reliability separate from model-based answer diagnostics. Render a truthful, stable report that preserves partial Retrieval measurements, shows adaptive thresholds, distinguishes evaluation execution from workflow outcome, and does not hide workflow errors inside aggregate answer scores.

**Blocked by:** 06: Reopen retrieval from Answer review

**Status:** resolved

- [x] Answer Relevancy and Answer Correctness run for every non-empty answer, abstention, clarification, insufficient-evidence response, exhausted-answer response, and degraded workflow-error response.
- [x] Faithfulness and Context Utilization run only when the case has a non-empty Final generation context.
- [x] Missing context produces an explicit unavailable reason for Faithfulness and Context Utilization rather than an invented high score or a zero.
- [x] Expected answerability no longer pre-emptively excludes an otherwise applicable Ragas diagnostic.
- [x] Deterministic answerability, reason-code, response-origin, workflow-error, and evidence checks remain separate from Ragas scores.
- [x] Normal semantic-response Ragas aggregates exclude workflow-error cases while each workflow-error case retains its diagnostic scores.
- [x] Evaluation run status reports artifact-process success independently from workflow outcome.
- [x] A completed run containing query errors reports `completed_with_errors`, workflow-error count, and workflow reliability over selected cases.
- [x] The Search query registry shows Search ID, query, Vector threshold, and BM25 threshold ratio for every executed search.
- [x] Every query renders Vector, BM25, Fusion, Rerank, and Generation metric rows in a stable order.
- [x] Executed stages retain real metrics after a later failure; only unexecuted or uncomputable stages display a dash.
- [x] Retrieval stage metrics include every actual Search attempt, while Final context evidence metrics use only the exact Final generation context.
- [x] Overall report labels use Recall, Precision, and NDCG capitalization consistently.
- [x] Machine-readable schema version and structured records include new outcome metadata, search diagnostics, workflow status, reliability, and metric applicability reasons.
- [x] `run_evaluation()` tests assert agreement between structured results and Markdown for normal answers, abstentions, clarification, empty context, controller failure after retrieval, Retrieval failure, and mixed-success runs.
- [x] The governing evaluation ADR is superseded or amended to define the new applicability contract while retaining Ragas as evaluation-only diagnostics.

## Comments

- Implemented in `6847da2` through the public `run_evaluation()`, machine-readable `results.json`, and rendered `README.md` seams.
- Schema version 8 records artifact `run_status`, workflow outcome, workflow-error count, reliability numerator/denominator/ratio, per-case outcome metadata, Search thresholds and diagnostics, and explicit per-metric applicability reasons.
- Answer Relevancy and Answer Correctness now judge every non-empty normal or degraded response. Faithfulness and Context Utilization require a non-empty exact Final generation context; missing context remains unavailable with an explicit reason.
- Semantic-response Ragas means exclude workflow-error cases, while each workflow-error case and a separate error-only aggregate retain diagnostic scores. Deterministic answerability, evidence, reason-code, response-origin, and workflow-error checks remain independent.
- Markdown reports distinguish completed artifact generation from `completed_with_errors`, show workflow reliability, preserve every executed Search and its thresholds, render stable Vector/BM25/Fusion/Rerank/Generation rows, and retain measured Retrieval metrics after later failures.
- ADR-0003 now defines outcome/input-based Ragas applicability and separates execution status, workflow reliability, and semantic diagnostics while keeping Ragas evaluation-only.
- Verification before resolution: full offline pytest suite passed (237 test items); focused evaluation/report/Ragas suite passed (42); `py_compile` and `git diff --check` passed.
- Evaluation corpus, reviewed QA ground truth, index isolation, sampling semantics, and live provider settings were not changed. Live Ragas/provider behavior remains unverified; automated verification used offline fakes.

