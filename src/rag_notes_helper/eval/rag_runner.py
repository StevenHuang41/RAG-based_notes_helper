from dataclasses import dataclass
from typing import List

from rag_notes_helper.rag.retrieval import retrieve
from rag_notes_helper.rag.answer import rag_answer


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: List[str]
    sources: List[str]
    scores: List[float]


def run_single_query(rag, meta_store, query: str) -> QAResult:
    # retrieve
    hits = retrieve(rag, meta_store, query=query)

    # generate answer
    result = rag_answer(query, hits=hits)

    # extract contexts
    contexts = [h["text"] for h in hits]

    # optional meta
    sources = [h["source"] for h in hits]
    scores = [h["score"] for h in hits]

    return QAResult(
        question=query,
        answer=result["answer"],
        contexts=contexts,
        sources=sources,
        scores=scores,
    )
