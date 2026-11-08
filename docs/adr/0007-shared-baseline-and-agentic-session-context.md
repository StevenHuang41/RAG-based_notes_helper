# Keep a shared RAG baseline and bound Agentic session context

Status: Accepted

## Context

Classic and Agentic RAG are compared using the same local note corpus and
reviewed evaluation cases. The comparison should measure the value and cost of
Agentic control over retrieval and answer review without also changing the
initial retrieval path, answer model, answer prompt, or generation context
limit.

Agentic REPL also needs to interpret follow-up questions such as references to
an earlier subject. Persistent or cross-session memory is unnecessary for that
interaction and could blur the boundary between conversation and note evidence.

## Decision

- Use the same deterministic query validation and original-query first Search
  for Classic and Agentic. Do not add a separate pre-search semantic router only
  to Agentic. Agentic may clarify after its controller inspects the initial
  search and available conversation context.
- Keep the Hybrid Retrieval pipeline, starting thresholds, Cross-encoder,
  answer model, shared answer-generation prompt, and Generation top-k equal in
  both modes. The shared answer prompt may allow flexible organization and
  formatting, but both modes remain grounded only in the supplied note context.
- Classic remains one Search followed by one Answer generation. Agentic adds
  bounded controller decisions, adaptive Search, and answer review. These extra
  calls are part of Agentic's measured runtime cost.
- Keep the LangGraph state fresh for each query. Agentic REPL may supply an
  in-memory window of at most five user/assistant pairs to help interpret
  follow-ups. Clear the window when the REPL ends. Conversation history is
  contextual information, never factual evidence; each query retrieves its own
  note evidence. Evaluation cases always start without history.
- Require the answer reviewer to assess the exact ordered context passed to the
  candidate's generation call. It distinguishes answer-quality revisions using
  the same context from evidence recovery using additional candidate chunks or
  a new Search.
- Make Retrieval and Answer total-attempt budgets independently configurable.
  Defaults are three attempts, including the initial call, with an allowed
  range of one through five. New variables are implemented in Settings and
  documented in `.env.example`; never edit a user's `.env`.
- Compare workflows on the same reviewed QA case IDs, corpus, and model and
  retrieval settings. Report evidence metrics, answer metrics, runtime token
  use, and evaluation-judge token use separately. A full workflow comparison
  measures the combined Agentic controller and reviewer; isolating the effect
  of each component requires a separate ablation.

## Consequences

This keeps a common baseline and makes Agentic's extra reasoning and tool use
visible in both quality and cost results. Session history supports natural
follow-ups without persisting state across application runs or reusing old note
evidence.

The Agentic path will use more model calls and may still fail to improve quality.
The reviewer is a model-based check, not an independent correctness guarantee.
Evaluation results must be used to assess whether the added calls provide
enough benefit.

This decision amends ADR-0005's fixed three-attempt maximums into configurable
total-attempt limits whose defaults remain three. ADR-0005 continues to govern
LangGraph orchestration, bounded read-only retrieval, action validation, and
buffered answer review.
