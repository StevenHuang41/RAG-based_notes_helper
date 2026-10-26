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

**Retrieval search**:
One invocation of the Note Search Tool that produces an independently ranked candidate list. Searches within one query are identified by their invocation order.

**Retrieval top-k**:
The maximum number of candidate chunks returned by each retrieval search.

## Evaluation

**Evidence span**:
A contiguous range of source lines that supports an expected answer. Its source and line bounds identify where the evidence appears, while its text records the reviewed content.

**Expected answerability**:
The fixture's label for whether its reviewed evidence is sufficient for the workflow to answer rather than abstain.

**Retrieved chunk**:
A note chunk returned by a retrieval search, identified by its source metadata and rank within that search. A retrieved chunk is a candidate for the answer and is not necessarily passed to the answer generator.

**Answer generation context**:
The ordered evidence blocks actually provided to the final answer-generation call. It may contain all retrieved chunks or a selected, reordered, or transformed subset.

**Context prefix**:
The first k chunks of an answer generation context, in the exact order provided to the answer generator. Prefix metrics describe the evidence available after including those chunks.

**Evaluation run**:
An assessment of a RAG query mode against prepared questions and expected evidence, with results available for review.

**Evaluation sample**:
A subset of prepared evaluation cases selected for one run; it is not a claim that the cases statistically represent a broader question population.
