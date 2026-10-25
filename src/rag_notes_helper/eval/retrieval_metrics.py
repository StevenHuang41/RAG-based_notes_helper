"""Line-aware retrieval metrics for the reviewed notes fixture."""

from collections.abc import Mapping, Sequence
import math
from typing import Any


LinePosition = tuple[str, int]


def _span_positions(span: Mapping[str, Any]) -> set[LinePosition]:
    source = span.get("source")
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(start_line, int)
        or not isinstance(end_line, int)
        or start_line < 1
        or end_line < start_line
    ):
        return set()
    return {(source, line_number) for line_number in range(start_line, end_line + 1)}


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def _ideal_relevances(
    chunks: Sequence[Mapping[str, Any]],
    ground_truth: set[LinePosition],
    *,
    top_k: int,
) -> list[int]:
    remaining = [
        _span_positions(chunk) & ground_truth
        for chunk in chunks[:top_k]
    ]
    covered: set[LinePosition] = set()
    ideal: list[int] = []

    while remaining and len(ideal) < top_k:
        gains = [len(lines - covered) for lines in remaining]
        best_index = max(range(len(gains)), key=gains.__getitem__)
        best_gain = gains[best_index]
        ideal.append(best_gain)
        covered.update(remaining.pop(best_index))

    return ideal


def _ndcg_for_search(
    ranked_chunks: Sequence[Mapping[str, Any]],
    ground_truth: set[LinePosition],
    *,
    top_k: int,
) -> float | None:
    if not ranked_chunks or not ground_truth:
        return None

    chunks = ranked_chunks[:top_k]
    covered: set[LinePosition] = set()
    actual_relevances: list[int] = []
    for chunk in chunks:
        newly_covered = _span_positions(chunk) & ground_truth - covered
        actual_relevances.append(len(newly_covered))
        covered.update(newly_covered)

    ideal_relevances = _ideal_relevances(
        chunks,
        ground_truth,
        top_k=top_k,
    )
    ideal_dcg = _dcg(ideal_relevances)
    if ideal_dcg == 0:
        return None
    return _dcg(actual_relevances) / ideal_dcg


def calculate_retrieval_metrics(
    *,
    expected_evidence: Sequence[Mapping[str, Any]],
    retrieval_lists: Sequence[Sequence[Mapping[str, Any]]],
    top_k: int,
) -> dict[str, float | None]:
    """Score retrieved inclusive source-line spans against fixture evidence.

    Recall and precision use the unique line union across all search calls.
    NDCG is computed independently for each ranked search list, then averaged
    over lists with at least one relevant retrieved line.
    """
    ground_truth = set().union(*(_span_positions(span) for span in expected_evidence))
    retrieved = set().union(*(
        _span_positions(chunk)
        for ranked_chunks in retrieval_lists
        for chunk in ranked_chunks[:top_k]
    ))

    if not ground_truth:
        recall = None
        precision = None
    else:
        relevant_retrieved = ground_truth & retrieved
        recall = len(relevant_retrieved) / len(ground_truth)
        precision = (
            len(relevant_retrieved) / len(retrieved)
            if retrieved
            else None
        )

    ndcg_values = [
        score
        for ranked_chunks in retrieval_lists
        if (score := _ndcg_for_search(ranked_chunks, ground_truth, top_k=top_k))
        is not None
    ]

    return {
        "recall": recall,
        "precision": precision,
        "ndcg": sum(ndcg_values) / len(ndcg_values) if ndcg_values else None,
    }
