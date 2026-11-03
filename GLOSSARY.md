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

**Answer attempt**:
One candidate answer generation followed by an evidence-grounded Answer review.
_Avoid_: Answer retry

**Answer review**:
A per-query controller decision that accepts a candidate answer, requests a revision, reopens Retrieval with a validated adaptive search, asks for clarification, or abstains based on the selected note evidence and remaining budgets.
_Avoid_: Ragas judge, answer grading

## Retrieval

**Note Search Tool**:
A read-only interface that searches the Allowed note corpus and returns ranked candidate chunks with source metadata for an agent or connected client to examine.

**MCP search tool**:
A read-only external interface that runs Hybrid retrieval over the Allowed note corpus and returns the final selected chunks with provenance to a connected model client. It does not generate an answer.

**Retrieval search**:
One invocation of the Note Search Tool for a query. A retrieval search can contain separate Search method results and downstream ranking stages.

**Adaptive search**:
A later Agentic RAG Retrieval search with a validated rewritten query, Vector minimum score, query-relative BM25 minimum-score ratio, and Evidence mode. It consumes one Search attempt only when Retrieval executes.

**Evidence mode**:
The per-search choice to `replace` the active candidate pool with the new fused candidates or `extend` it with deduplicated prior and new fused candidates before reranking against the original question. All executed Search audits remain available in either mode.

**Allowed note corpus**:
The set of note files approved as evidence sources for runtime retrieval. It is distinct from the Evaluation corpus used by benchmark runs.

**Search method result**:
An independently ranked list of candidate chunks produced by one retrieval method within a Retrieval search. Its ranks apply only to that method's list.

**Hybrid retrieval**:
A Retrieval search that combines semantic and lexical Search method results into a shared candidate ranking, then reorders candidates using query and chunk relevance.

**Reciprocal Rank Fusion (RRF)**:
A method for combining independently ranked candidate lists by their ranks, without comparing the lists' raw relevance scores.

**Cross-encoder reranking**:
A second-stage ranking that evaluates each query and candidate chunk together and reorders retrieved candidates. It cannot introduce chunks that were absent from its input candidate set.

**Retrieval stage**:
A named ranked result within Hybrid retrieval, such as a Search method result, a fused result, or a reranked result. Each stage has its own candidate list and result limit.

**Retrieval top-k**:
The maximum number of candidate chunks returned at a specified Retrieval stage. The value may differ between Search method results and later stages.

**Generation top-k**:
The maximum number of chunks selected from the final ranked results for the Answer generation context.

## Evaluation

**Evidence span**:
A contiguous range of source lines that supports an expected answer. Its source and line bounds identify where the evidence appears, while its text records the reviewed content.

**Expected answerability**:
The fixture's label for whether its reviewed evidence is sufficient for the workflow to answer rather than abstain.

**Retrieved chunk**:
A note chunk returned by a retrieval search, identified by its source metadata and rank within that search. A retrieved chunk is a candidate for the answer and is not necessarily passed to the answer generator.

**Retrieval stage metrics**:
Recall, Precision, and NDCG calculated from the ranked candidate list produced at one Retrieval stage, against the reviewed expected evidence for that query.

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

**Structured outcome**:
The non-empty query result plus stable reason code, response origin, and optional workflow-error category. Normal bounded outcomes and degraded workflow-error outcomes remain distinguishable even when both provide user-facing text.

**Evaluation run status**:
Whether the evaluation process completed artifact generation or failed. It is independent from the workflow outcome of the selected query cases.

**Workflow reliability**:
The ratio of selected evaluation cases that reached a normal bounded outcome without a workflow error. A completed artifact run can have a `completed_with_errors` workflow outcome and reliability below one.
