# Generate before reviewing Agentic evidence

## Problem Statement

Agentic RAG currently asks its controller to judge evidence sufficiency before
the first Answer generation. That extra decision can classify a clear query as
ambiguous or fail validation before the answer model sees evidence. In the
reviewed 2026-11-08 sample, Agentic did not generate an answer for any selected
case: it asked for clarification on a clear answerable question, asked for
clarification on a clear unanswerable question, and failed during controller
validation on an ambiguous question.

The current flow therefore cannot evaluate the Answer model's ability to
answer or abstain on those cases. It also conflates query ambiguity with
insufficient or absent answer evidence.

## Solution

Keep the shared original-query Retrieval search and ranking baseline. When the
initial search produces candidates, send the first Generation top-k directly to
the Answer model and buffer its candidate. Then use Answer review to accept,
revise, expand evidence, request a bounded Search, clarify a genuinely
ambiguous query, or abstain. No candidate is returned before Answer review
accepts it.

An empty initial candidate pool is a structural Retrieval-empty outcome:
return `no_retrieval_results` without calling the controller or Answer model.
Do not interpret empty retrieval as query ambiguity or proof that the notes
corpus lacks an answer. For non-empty candidate pools, do not run a semantic
evidence-sufficiency controller before Answer generation.

Classic RAG, the shared retrieval pipeline, answer model, answer prompt, and
Generation top-k remain unchanged. The change applies to the bounded Agentic
workflow and its evaluation contract.

## User Stories

1. As a Notes Helper user, I want a clear query with retrieved candidates to
   reach Answer generation immediately, so that a pre-generation controller
   decision cannot prevent the answer model from seeing the evidence.
2. As a user asking an answerable question, I want the first Agentic Answer
   context to use the same ranked Generation top-k as Classic RAG, so that the
   comparison keeps a shared retrieval and context baseline.
3. As a user, I want each candidate Answer buffered until review accepts it,
   so that rejected or revised text is never shown as the final answer.
4. As a user, I want Answer review to inspect the exact ordered context used
   for that candidate, so that acceptance and grounding refer to the evidence
   the Answer model actually received.
5. As a user, I want Answer review to revise a candidate only when its current
   context is sufficient, so that answer-quality fixes do not spend Retrieval
   budget unnecessarily.
6. As a user, I want Answer review to expand from the current candidate pool
   when useful evidence was left out of the generation context, so that the
   workflow can recover evidence without another Search.
7. As a user, I want Answer review to request an adaptive Search only when
   necessary evidence is absent from the active candidate pool, so that new
   Retrieval work remains bounded and evidence-directed.
8. As a user, I want clarification only when my query is materially
   ambiguous or underspecified, so that missing answer evidence is not
   misrepresented as a question-clarity problem.
9. As a user, I want a clear query with no retrieved candidates to receive a
   Retrieval-empty outcome, so that zero hits are not treated as ambiguity or
   as proof that the corpus contains no answer.
10. As a user, I want evidence gaps to end in a clear abstention, so that the
    workflow does not turn an unsuccessful bounded Search into a claim that no
    answer exists in the notes.
11. As a user, I want every generated answer returned by Agentic RAG to have
    passed Answer review, so that unsupported or incomplete candidates remain
    hidden.
12. As a user, I want Search and Answer attempts bounded independently,
    including the initial Search and first Answer generation, so that latency
    and model usage remain predictable.
13. As a maintainer, I want Search and Answer budgets preserved across review
    and adaptive Search transitions, so that reopening Retrieval cannot reset
    or evade either limit.
14. As a maintainer, I want workflow outcomes to distinguish empty Retrieval,
    ambiguity, insufficient evidence, exhausted Answer attempts, and
    controller failure, so that users and evaluators can interpret each
    terminal state correctly.
15. As a maintainer, I want controller failures to expose a sanitized failure
    category and retain the safe workflow trace, so that validation failures
    can be diagnosed without persisting prompts or model outputs.
16. As an evaluator, I want valid unanswerable cases with retrieved candidates
    to reach Answer generation and review, so that the workflow's model
    abstention behavior can be assessed.
17. As an evaluator, I want genuine clarification outcomes and deterministic
    Retrieval-empty outcomes excluded from answer-model Ragas scoring, so that
    controller and retrieval results are not mistaken for generated answers.
18. As an evaluator, I want accepted Answer-model responses with references to
    remain eligible for applicable Ragas metrics regardless of expected
    answerability, so that supported model abstentions can be measured.
19. As an evaluator, I want context-dependent metrics unavailable when no
    final Answer generation context exists, so that missing evidence is not
    reported as a zero or fabricated score.
20. As an evaluator, I want paired Classic and Agentic runs to use identical
    case IDs, corpus, model, and retrieval settings, so that workflow
    differences remain interpretable.
21. As a maintainer, I want evaluation ground truth left unchanged, so that
    the benchmark remains an independent reference for the redesigned
    workflow.
22. As a maintainer, I want the existing public query and evaluation entry
    points to remain the behavioral test seams, so that the design can be
    verified without adding lower-level seams solely for tests.
23. As a CLI user, I want progress to show the actual Search, Answer
    generation, and Answer review phases in order, so that it does not imply a
    pre-generation controller decision that no longer occurs.

## Implementation Decisions

### Initial Search and Answer generation

- Perform the same deterministic input validation and original-query first
  Search used by the shared baseline.
- Use the shared Retrieval and ranking pipeline and its ordered candidates.
  If the initial ranked candidate pool is non-empty, pass its first Generation
  top-k directly to Answer generation. Do not ask the controller to select
  evidence or judge sufficiency before this first Answer attempt.
- Preserve the exact ordered generation context for audit and later review.
  The reviewer must receive that exact context and the current candidate pool.
- If the initial ranked candidate pool is empty, return a deterministic
  `no_retrieval_results` outcome with no controller call, Answer call, or
  Answer attempt. Do not ask for clarification solely because retrieval is
  empty. This describes Retrieval coverage, not corpus absence or index
  health.
- Keep Agentic candidate text buffered until accepted. Never return or log a
  rejected candidate as the user-facing answer.
- Report progress in the actual order: initial Search, Answer generation, then
  Answer review. Do not show an initial controller review phase.

### Answer review and recovery

- The reviewer may accept, request a focused revision with the same evidence,
  expand from evidence already in the candidate pool, request another adaptive
  Search, clarify, or abstain.
- Use `revise` only when the current context can support a sufficient answer;
  revision feedback must identify the required correction.
- Use evidence expansion when the required candidate is already in the
  current pool but is absent from the exact generation context. Select
  additional candidates ahead of the existing context so Generation top-k can
  include them.
- Request adaptive Search only when necessary evidence is absent from the
  current candidate pool. Keep query rewrite, thresholds, and `replace` or
  `extend` behavior within the existing validated Search contract.
- Clarify only when the query itself is ambiguous or materially
  underspecified. A missing fact, irrelevant evidence, or zero-hit Search is
  not by itself a clarification condition.
- Treat an accepted Answer-model abstention as `insufficient_evidence` with
  `response_origin=answer_model`. A controller-selected abstention is also
  `insufficient_evidence`, with its controller origin. Preserve provenance so
  evaluation can distinguish them.

### Outcomes, budgets, and failures

- Keep distinct terminal reason codes for `answered`,
  `no_retrieval_results`, `needs_clarification`, `insufficient_evidence`,
  `answer_attempts_exhausted`, `controller_failure`, and `retrieval_failure`.
- Use `no_retrieval_results` only when no executed Search has produced
  candidates. When candidates exist but cannot support an accepted answer,
  report `insufficient_evidence` or the applicable bounded terminal reason.
- Preserve separately configurable Search and Answer total-attempt budgets.
  Defaults remain three, including the initial Search and first Answer
  generation, with the existing supported range. A later Search is allowed
  only when a Search attempt and a later Answer attempt both remain.
- Count a Search only when Retrieval executes and an Answer attempt only when
  candidate generation is invoked. Controller calls and validation retries
  consume neither budget. A structural empty-pool outcome consumes no Answer
  attempt.
- Preserve per-query isolation, read-only Retrieval, evidence-only grounding,
  and the existing safe terminal behavior. If the reviewer cannot produce a
  valid action after its validation retry, return `controller_failure`; do
  not accept the buffered candidate as a fallback.
- Record a safe machine-readable controller failure category, such as provider
  failure, invalid JSON, schema-invalid action, invalid evidence reference, or
  out-of-budget action. Retain the workflow trace and exact generation
  context, but never persist raw prompts or controller/Answer outputs as
  diagnostics.
- Keep Ragas evaluation-only. Ragas does not participate in runtime Answer
  review or routing.

### Evaluation contract

- Keep each evaluation case isolated and run Classic and Agentic on the same
  reviewed case IDs, corpus, answer model, shared prompt, Retrieval settings,
  and Generation top-k. Do not change reviewed QA data to improve outcomes.
- Preserve Retrieval-stage metrics for retrieved candidates separately from
  final-context Recall, Precision, and NDCG for the exact context passed to
  Answer generation. When no generation context exists, keep context metrics
  unavailable.
- Only score a final response with Ragas when it is a non-empty accepted
  Answer-model response, a reference answer is available, and the workflow
  completed without error. Apply context-dependent metrics only when the
  corresponding final generation context exists.
- This eligibility includes an accepted Answer-model abstention for a valid
  unanswerable case when the required reference and context inputs exist.
  Skip Ragas for controller clarification, controller-selected abstention,
  deterministic Retrieval-empty outcomes, exhausted rejected candidates, and
  workflow errors. Report a per-metric reason for each skipped or unavailable
  metric.
- Keep normal bounded outcomes separate from workflow errors in reliability
  reporting. A completed evaluation artifact run does not imply every case
  reached a normal outcome.

## Testing Decisions

### Test seam

Use two existing public behavior seams. At `agentic_answer()`, inject fake
Search, controller, and Answer providers and assert returned outcomes,
attempts, exact generation context, safe traces, and that rejected candidates
are not exposed. At `run_evaluation()`, inject deterministic query results and
evaluation workers and assert case outcome mapping, Ragas eligibility and skip
reasons, and report records. Do not add a private node-level seam just for
these tests.

A good test asserts externally observable behavior and stable outcome
contracts, not graph node ordering or private helper calls. Fake providers
must make controller and Answer responses deterministic; no live provider is
required for offline acceptance.

### Required behavioral coverage

- `direct-06`: first Search returns candidates, the Answer model runs before
  any controller call, the expected evidence is in the exact Generation
  context, and Answer review accepts the grounded answer.
- `ambiguous-01`: a candidate is generated and reviewed; genuine ambiguity
  returns `needs_clarification`, never exposes the candidate, and is not sent
  to Ragas as an Answer-model response.
- `unanswerable-13`: non-empty candidates reach the Answer model despite
  insufficient answer evidence. Assert an accepted Answer-model abstention
  with `response_origin=answer_model` and applicable Ragas metrics; separately
  assert that a controller-selected abstention is not Ragas-eligible.
- Empty initial Retrieval returns `no_retrieval_results`, performs no
  controller or Answer generation, consumes no Answer attempt, and is not
  reported as clarification or corpus absence.
- Same-context revision requires focused feedback; candidate-pool expansion
  includes only known candidates not already in the generation context; a new
  Search occurs only when the candidate pool lacks needed evidence.
- Progress callbacks show initial Search, Answer generation, and Answer
  review without a pre-generation controller phase or candidate text.
- Search and Answer attempt counts remain independent across expansions,
  revisions, and adaptive Searches. No action can exceed a remaining budget.
- A controller-invalid replay after Answer generation returns
  `controller_failure` with a sanitized category and retained safe trace; it
  does not return the buffered candidate or persist raw output.
- Evaluation eligibility covers accepted Answer-model responses, supported
  model abstentions, genuine clarifications, empty Retrieval, exhausted
  candidates, and controller failures. Context metrics remain unavailable
  without a final context.
- Same-seed Classic and Agentic runs select the same reviewed case IDs and do
  not read or mutate runtime `data/` or `storage/`.

## Out of Scope

- Implementing this spec or creating implementation tickets in this design
  step.
- Changing Classic RAG, the shared Answer model or prompt, Retrieval models,
  starting thresholds, reranking, or Generation top-k.
- Changing evaluation ground truth, claiming that Agentic guarantees better
  answers, or treating a bounded failure as proof that the corpus lacks an
  answer.
- Adding external Search, write side effects, persistent cross-query memory,
  or general-purpose autonomous-agent behavior.
- Running live-provider acceptance as a deterministic unit test.

## Further Notes

- The reviewed `direct-06`, `ambiguous-01`, and `unanswerable-13` reports use
  the same sample and seed. The diagnosis comes from stored reports and
  current-source inspection; this spec does not claim a fresh live Ollama
  reproduction.
- The structural no-hit exception preserves the existing empty-Retrieval
  contract. It does not restore a semantic pre-generation evidence checkpoint:
  any non-empty candidate pool proceeds directly to Answer generation.
- A non-empty candidate pool for an ambiguous or unanswerable question incurs
  an Answer generation call before review can clarify or abstain. Its added
  latency and token use are part of the Agentic workflow cost and remain
  visible in evaluation reports.
- Keep the resolved Agentic-finalization spec and its resolved issues as
  historical records. This spec is a follow-up contract.
