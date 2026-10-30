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

**Final context evidence metric**:
A line-grounded Recall, Precision, or NDCG score computed from the complete answer generation context. It measures the evidence the answer model received after Agentic RAG selected from retrieved chunks.

**Evaluation run**:
An assessment of a RAG query mode against the version-controlled synthetic fixture, with reviewed questions, reference answers, and expected evidence available for review. It does not evaluate a user's persistent index snapshot.

**Evaluation corpus**:
The version-controlled synthetic notes used to build the isolated index for an evaluation run. It is stored under `evals/notes/` and is distinct from a user's notes under `data/`.

**Evaluation ground truth QA set**:
The version-controlled benchmark definition that pairs reviewed questions, reference answers, answerability labels, and expected evidence with the evaluation corpus. It is stored under `evals/qa/`.

**Evaluation result**:
The generated, Git-ignored report artifacts for one evaluation run, including its human-readable report, machine-readable results, and process log. It is stored beneath `evals/results/` and is not benchmark ground truth.

**Persistent index snapshot**:
The persisted FAISS index and its matching chunk metadata under `storage/`, representing the note corpus at indexing time. It is queryable without rereading source files, but does not establish that current source files remain synchronized with the snapshot.

**Evaluation sample**:
A subset of prepared evaluation cases selected for one run; it is not a claim that the cases statistically represent a broader question population.
