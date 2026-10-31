from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any, overload

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.index import RagIndex
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.utils.timer import deco_time_block


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_RRF_CONSTANT = 60


@dataclass(frozen=True)
class HybridSearchResults(Sequence[dict[str, Any]]):
    """Ranked method results and their RRF fusion for one retrieval search.

    Iteration and indexing expose the fused candidates, preserving the sequence
    behavior expected by answer generation and the Agentic controller.
    """

    vector_results: list[dict[str, Any]]
    bm25_results: list[dict[str, Any]]
    fused_results: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.fused_results)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.fused_results)

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        return self.fused_results[index]


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


def _bm25_scores(query: str, documents: list[dict[str, Any]]) -> list[float]:
    query_terms = Counter(_tokens(query))
    if not query_terms or not documents:
        return [0.0] * len(documents)

    tokenized_documents = [_tokens(document.get("text", "")) for document in documents]
    document_frequencies = Counter(
        term
        for tokens in tokenized_documents
        for term in set(tokens)
    )
    average_length = sum(map(len, tokenized_documents)) / len(tokenized_documents)
    if average_length == 0:
        return [0.0] * len(documents)

    document_count = len(documents)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []

    for tokens in tokenized_documents:
        term_frequencies = Counter(tokens)
        document_length = len(tokens)
        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = term_frequencies.get(term, 0)
            if not term_frequency:
                continue

            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            normalization = k1 * (
                1 - b + b * document_length / average_length
            )
            score += query_frequency * inverse_document_frequency * (
                term_frequency * (k1 + 1) / (term_frequency + normalization)
            )
        scores.append(score)

    return scores


def _document_identity(document: dict[str, Any], index: int) -> tuple[Any, ...]:
    if "source" in document and "chunk_id" in document:
        return (document["source"], document["chunk_id"])
    if "doc_id" in document and "chunk_id" in document:
        return (document["doc_id"], document["chunk_id"])
    return ("faiss", index)


def _ranked_method_results(
    *,
    indexes: list[int],
    scores: list[float],
    records: list[dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    results = []
    for rank, (index, score) in enumerate(zip(indexes, scores), start=1):
        result = dict(records[index])
        result["rank"] = rank
        result["score"] = float(score)
        result[f"{method}_rank"] = rank
        result[f"{method}_score"] = float(score)
        results.append(result)
    return results


def _fuse_results(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    vector_indexes: list[int],
    bm25_indexes: list[int],
    records: list[dict[str, Any]],
    fusion_top_k: int,
) -> list[dict[str, Any]]:
    fused: dict[tuple[Any, ...], dict[str, Any]] = {}

    for method_results, candidate_indexes, method in (
        (vector_results, vector_indexes, "vector"),
        (bm25_results, bm25_indexes, "bm25"),
    ):
        for candidate, document_index in zip(method_results, candidate_indexes):
            rank = candidate[f"{method}_rank"]
            key = _document_identity(records[document_index], document_index)
            result = fused.setdefault(
                key,
                {
                    **records[document_index],
                    "vector_rank": None,
                    "vector_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "fusion_score": 0.0,
                },
            )
            result[f"{method}_rank"] = rank
            result[f"{method}_score"] = candidate[f"{method}_score"]
            result["fusion_score"] += 1.0 / (_RRF_CONSTANT + rank)

    ranked = sorted(
        fused.values(),
        key=lambda result: result["fusion_score"],
        reverse=True,
    )[:fusion_top_k]
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
        result["score"] = result["fusion_score"]
    return ranked


@deco_time_block
def retrieve(
    rag: RagIndex,
    meta_store: MetaStore,
    *,
    query: str,
    top_k: int | None = None,
) -> HybridSearchResults:
    """Run Vector and BM25 searches, then fuse their ranked candidates with RRF.

    ``top_k`` is retained as a compatibility override for the final fused
    result limit. Production configuration uses independent Vector, BM25, and
    fusion limits.
    """
    settings = get_settings()
    vector_top_k = settings.vector_top_k
    bm25_top_k = settings.bm25_top_k
    fusion_top_k = settings.fusion_top_k if top_k is None else top_k

    model = rag.embed_model
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    vector_scores, vector_indexes_raw = rag.index.search(query_embedding, vector_top_k)  # type: ignore

    records = [meta_store.get(index) for index in range(rag.index.ntotal)]  # type: ignore

    vector_indexes: list[int] = []
    vector_score_values: list[float] = []
    for score, index in zip(vector_scores[0], vector_indexes_raw[0]):
        if index < 0 or score < settings.min_retrieval_score:
            continue
        vector_indexes.append(int(index))
        vector_score_values.append(float(score))

    vector_results = _ranked_method_results(
        indexes=vector_indexes,
        scores=vector_score_values,
        records=records,
        method="vector",
    )

    bm25_score_values = _bm25_scores(query, records)
    bm25_indexes = [
        index
        for index, score in enumerate(bm25_score_values)
        if score > 0
    ]
    bm25_indexes.sort(key=lambda index: (-bm25_score_values[index], index))
    bm25_indexes = bm25_indexes[:bm25_top_k]
    bm25_result_scores = [bm25_score_values[index] for index in bm25_indexes]
    bm25_results = _ranked_method_results(
        indexes=bm25_indexes,
        scores=bm25_result_scores,
        records=records,
        method="bm25",
    )

    fused_results = _fuse_results(
        vector_results,
        bm25_results,
        vector_indexes,
        bm25_indexes,
        records,
        fusion_top_k,
    )
    return HybridSearchResults(
        vector_results=vector_results,
        bm25_results=bm25_results,
        fused_results=fused_results,
    )
