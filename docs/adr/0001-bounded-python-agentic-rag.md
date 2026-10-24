# Start Agentic RAG with a bounded Python loop

The first Agentic RAG workflow will use a framework-light Python loop with Pydantic-validated JSON actions, a three-search limit, and the existing read-only note-search and answer-generation services; Classic RAG remains the default. This supports iterative retrieval without LangChain, LangGraph, or MCP overhead, while keeping orchestration isolated so LangGraph or MCP can be added later if graph orchestration or external tool interoperability becomes a concrete requirement.
