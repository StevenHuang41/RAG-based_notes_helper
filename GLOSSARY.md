# Notes RAG

This context covers searching personal notes and generating evidence-grounded answers from them.

## Query modes

**Classic RAG**:
A fixed query workflow that retrieves note chunks once and generates an answer from that evidence.
_Avoid_: Original RAG, legacy RAG

**Agentic RAG**:
A per-query workflow in which an agent plans note exploration, may retrieve repeatedly based on evidence sufficiency, and synthesizes a final answer. It uses indexed local notes as its evidence source and retains no state across separate queries.
_Avoid_: Agent mode, conversational agent

**Evidence-grounded answer**:
An answer whose factual claims are supported by retrieved note evidence; when evidence is insufficient, the system abstains.
_Avoid_: Hallucination-free answer

## Retrieval

**Note Search Tool**:
A read-only interface that searches indexed notes and returns candidate chunks with source metadata and relevance scores for an agent to examine.
