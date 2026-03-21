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
    contexts = [hit["text"] for hit in hits]

    # optional metadata
    sources = [hit["source"] for hit in hits]
    scores = [hit["score"] for hit in hits]

    return QAResult(
        question=query,
        answer=result["answer"],
        contexts=contexts,
        sources=sources,
        scores=scores,
    )
