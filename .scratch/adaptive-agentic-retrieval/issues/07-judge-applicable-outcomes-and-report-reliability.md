# 07: Judge applicable outcomes and report workflow reliability

**What to build:** Evaluate every non-empty Classic and Agentic outcome with the Ragas diagnostics whose required inputs exist, while keeping deterministic behavior checks and operational reliability separate from model-based answer diagnostics. Render a truthful, stable report that preserves partial Retrieval measurements, shows adaptive thresholds, distinguishes evaluation execution from workflow outcome, and does not hide workflow errors inside aggregate answer scores.

**Blocked by:** 06: Reopen retrieval from Answer review

**Status:** ready-for-agent

- [ ] Answer Relevancy and Answer Correctness run for every non-empty answer, abstention, clarification, insufficient-evidence response, exhausted-answer response, and degraded workflow-error response.
- [ ] Faithfulness and Context Utilization run only when the case has a non-empty Final generation context.
- [ ] Missing context produces an explicit unavailable reason for Faithfulness and Context Utilization rather than an invented high score or a zero.
- [ ] Expected answerability no longer pre-emptively excludes an otherwise applicable Ragas diagnostic.
- [ ] Deterministic answerability, reason-code, response-origin, workflow-error, and evidence checks remain separate from Ragas scores.
- [ ] Normal semantic-response Ragas aggregates exclude workflow-error cases while each workflow-error case retains its diagnostic scores.
- [ ] Evaluation run status reports artifact-process success independently from workflow outcome.
- [ ] A completed run containing query errors reports `completed_with_errors`, workflow-error count, and workflow reliability over selected cases.
- [ ] The Search query registry shows Search ID, query, Vector threshold, and BM25 threshold ratio for every executed search.
- [ ] Every query renders Vector, BM25, Fusion, Rerank, and Generation metric rows in a stable order.
- [ ] Executed stages retain real metrics after a later failure; only unexecuted or uncomputable stages display a dash.
- [ ] Retrieval stage metrics include every actual Search attempt, while Final context evidence metrics use only the exact Final generation context.
- [ ] Overall report labels use Recall, Precision, and NDCG capitalization consistently.
- [ ] Machine-readable schema version and structured records include new outcome metadata, search diagnostics, workflow status, reliability, and metric applicability reasons.
- [ ] `run_evaluation()` tests assert agreement between structured results and Markdown for normal answers, abstentions, clarification, empty context, controller failure after retrieval, Retrieval failure, and mixed-success runs.
- [ ] The governing evaluation ADR is superseded or amended to define the new applicability contract while retaining Ragas as evaluation-only diagnostics.

