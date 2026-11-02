# Use LangGraph for bounded Agentic RAG orchestration

Status: Accepted

ADR-0001 kept the first Agentic RAG implementation framework-light until graph orchestration became useful; the hybrid retrieval, evidence-selection, answer-generation, and answer-review transitions now justify LangGraph's explicit state graph. Use a reusable compiled `StateGraph` for Agentic RAG while retaining Pydantic action validation, one validation retry for malformed controller output, read-only note search, answers restricted to evidence retrieved in the current query, and abstention when evidence is insufficient.

Bound the workflow with separate budgets: at most three Search attempts including the original-query search, and at most three Answer attempts including the first candidate generation. After each candidate generation, the controller reviews the answer against the selected evidence and either accepts it, requests a focused revision, or abstains. If the third candidate still requires revision, return a deterministic insufficient-answer response asking for a more specific question. Answer attempts reuse the evidence selected before generation; they do not reopen retrieval or use Ragas judges at runtime.

Agentic answers are buffered until review completes, because streaming a candidate before acceptance would expose text that the workflow may reject. Invoke the graph with fresh per-query state and no checkpointer or thread persistence; do not add external tracing or monitoring integrations. Classic RAG remains the default workflow and retains its existing streaming behavior.
