# Use LangGraph for bounded Agentic RAG orchestration

Status: Accepted

ADR-0001 kept the first Agentic RAG implementation framework-light until graph orchestration became useful; the hybrid retrieval stages and evidence-selection transitions now justify LangGraph's explicit state graph. Use a reusable compiled `StateGraph` for Agentic RAG while retaining Pydantic action validation, a single validation retry, a three-search limit including the initial retrieval, read-only note search, answers restricted to evidence retrieved in the current query, and abstention when evidence is insufficient. Invoke the graph with fresh per-query state and no checkpointer or thread persistence; do not add external tracing or monitoring integrations. Classic RAG remains the default workflow.
