# Finalize the bounded Agentic RAG workflow

Status: resolved

## Goal

Finalize Agentic RAG as a bounded, evidence-grounded workflow that can adapt
retrieval and recover from a poor candidate answer, while preserving a fair
Classic-versus-Agentic comparison. Agentic RAG is expected to use more model
calls. Better answer quality is a hypothesis to measure, not a guarantee.

## Shared comparison contract

- Classic and Agentic use the same runtime note corpus, Hybrid retrieval
  implementation, Cross-encoder, starting retrieval thresholds, answer model,
  answer prompt, and Generation top-k.
- Both modes perform the same deterministic input validation and begin with
  one Retrieval search using the original query and configured starting
  thresholds.
- Do not add an Agentic-only semantic query router before that first search.
  The Agentic controller may identify an unclear query after inspecting the
  first search and ask for clarification.
- Classic remains one Retrieval search followed by one grounded Answer
  generation. It does not call the Agentic controller or answer reviewer.
- Relax answer-style constraints in the shared generation prompt so either
  mode can produce a natural, organized answer. Both modes retain the same
  evidence-only grounding requirement and must not add unsupported facts.

## Agentic query workflow

1. Start with the original query and the shared Hybrid Retrieval pipeline.
2. The controller examines the query, session context when available, retrieved
   chunks, filter diagnostics, and remaining budgets. It chooses to ask for
   clarification, request an adaptive search, select evidence for generation,
   or abstain.
3. Later searches may rewrite the query, adjust Vector and query-relative BM25
   thresholds, and choose `replace` or `extend`. Retrieval top-k values and
   models remain fixed by configuration. Search results and provenance remain
   auditable even when a later search replaces active evidence.
4. The controller selects evidence. The application constructs the exact
   ordered Answer generation context and applies `LLM_MAX_CHUNKS` before
   generation.
5. Generate a buffered candidate with the shared answer model and prompt.
6. The reviewer evaluates relevance, completeness, and grounding against the
   exact context supplied to that generation call. It may:
   - accept the candidate;
   - request a focused revision using the same context;
   - select additional evidence already present in the current candidate pool;
   - request a new adaptive search when the required evidence is absent from
     that pool;
   - clarify or abstain.
7. Every returned generated answer must pass review. Do not expose rejected
   candidates. When an attempt budget is exhausted, return the existing
   deterministic bounded outcome.

Answer quality failures and evidence gaps use separate recovery paths. A
revision must include focused feedback; do not repeat an unchanged generation.
Only a missing or inadequate evidence pool should consume another Retrieval
attempt. When evidence cannot be recovered within budget, abstain.

## Attempt budgets and configuration

- Configure separate total-attempt limits for Retrieval and Answer generation.
- `AGENTIC_MAX_SEARCH_ATTEMPTS` defaults to `3`; `AGENTIC_MAX_ANSWER_ATTEMPTS`
  defaults to `3`.
- Each setting counts the initial attempt. Values must be between `1` and `5`.
- Record the effective limits in evaluation metadata and REPL progress output.
- Add configuration fields to Settings and document the variables in
  `.env.example` and README. Never edit the user's actual `.env` file.
- Controller calls and validation retries do not consume Search or Answer
  attempts. A Retrieval request from Answer review consumes the next Search
  attempt and requires a later Answer attempt to remain.

## Agentic REPL conversation context

- Agentic REPL keeps an in-memory window of at most five user/assistant pairs,
  bounded to fit the configured model context. It is cleared when the REPL
  session ends.
- Use conversation history only to resolve follow-up intent and references.
  It is not evidence; each query must retrieve note evidence for its own answer.
- Pass the history as explicitly labeled conversational context to the
  controller and, when needed to interpret the current question, the answer
  generator. Retrieved note chunks remain the only factual evidence.
- One-shot queries and evaluation cases start with empty conversation history.

## REPL progress

Agentic REPL progress reports the observable workflow: query clarification when
needed, the selected read-only search action, Search attempt and query, returned
chunk count, Answer attempt, answer review decision, and terminal outcome. For
long model calls, show that work is still in progress. Do not display hidden
chain-of-thought, raw prompts, or rejected candidate answers.

## Evaluation contract

- Evaluate Classic and Agentic separately against the same reviewed evaluation
  corpus and QA cases. Use the same sample and seed so both runs select the same
  case IDs. Do not pass conversation history between evaluation cases.
- Keep the embedding model, initial retrieval configuration, Cross-encoder,
  answer model, answer prompt, and Generation top-k the same between runs.
- Compare line-grounded Retrieval stage and final-context Recall, Precision,
  and NDCG; compare generated answer quality with the applicable Faithfulness,
  Answer Relevancy, Answer Correctness, and Context Utilization metrics.
- Report runtime controller, reviewer, and answer-generation token usage
  separately from Ragas evaluation-judge usage. Missing provider token counts
  remain unavailable rather than estimated.
- Do not modify reviewed evaluation questions, reference answers, answerability
  labels, or evidence spans to improve scores. Treat quality and token deltas as
  observations, not guarantees.

## Acceptance criteria

- Both modes use the same initial Retrieval query and configuration and the
  same answer model, shared answer prompt, and Generation top-k.
- Classic performs one search and one grounded generation without controller
  or review calls.
- Agentic can adaptively search, revise an answer, expand currently retrieved
  evidence, clarify, or abstain, while honoring separate configured budgets.
- The answer reviewer receives exactly the ordered context supplied to the
  corresponding answer-generation call.
- Agentic REPL follow-ups can use bounded session history, but no previous
  answer or chunk is accepted as current evidence.
- Progress reports actions and attempt counts without exposing model reasoning.
- `.env.example` documents all new environment settings; the actual `.env` is
  unchanged.
- Same-seed Classic and Agentic evaluations select identical case IDs and
  expose Retrieval quality, answer quality, and runtime token costs separately.

## Out of scope

- A pre-search LLM router used only by Agentic RAG.
- Different answer-generation models, prompts, or context limits between modes.
- External search, write tools, or general-purpose autonomous task execution.
- Persistent conversation memory or reuse of evidence across queries.
- Changes to reviewed evaluation ground truth or claims that Agentic RAG
  guarantees better answers.
