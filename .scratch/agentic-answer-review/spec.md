# Bounded Agentic answer review

## Problem Statement

Agentic RAG can decide whether retrieved evidence is sufficient, but it currently returns the first generated answer without checking whether that answer is direct and supported by the selected evidence. Retrieval is bounded while answer generation has no explicit retry model.

## Solution

Keep retrieval and answer generation as separate bounded phases. Retrieval permits at most three Search attempts including the original query. After the controller selects evidence, generate and review at most three candidate answers. The runtime controller reviews each candidate against the selected evidence and chooses `accept`, `revise`, or `abstain`. A revision supplies focused feedback to the next Answer attempt without reopening retrieval. If the third candidate still needs revision, return a deterministic insufficient-answer response that asks for a more specific question.

Buffer Agentic candidate answers until review completes. Classic RAG behavior remains unchanged. Ragas judges remain evaluation-only diagnostics and do not participate in runtime answer review.

## Acceptance Criteria

- Search attempts remain capped at three including the initial search.
- Answer attempts are capped at three including the first candidate generation.
- Every returned Agentic generated answer has passed evidence-grounded Answer review.
- Review can accept, request a focused revision, or abstain.
- Invalid review JSON receives one validation retry, matching the controller action contract.
- Exhausting Answer attempts returns a deterministic insufficient-answer response without exposing the rejected candidate as the answer.
- Controller and answer calls record phase/attempt attribution, measured latency, and provider-reported token usage.
- Agentic answers are buffered until accepted; Classic streaming remains unchanged.
- Offline tests cover acceptance, revision, exhaustion, abstention, call attribution, and per-query isolation.

## Out of Scope

- Persistent or cross-query memory.
- More than three Search attempts or three Answer attempts.
- Reopening retrieval after Answer review begins.
- Runtime Ragas evaluation, cosine-similarity thresholds, or calibrated confidence claims.
- External tools, write side effects, or changes to Classic RAG.
