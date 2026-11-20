from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import math
from numbers import Real
import re
from time import perf_counter
from typing import Any, Protocol, overload

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.index import RagIndex
from rag_notes_helper.rag.meta_store import MetaStore
from rag_notes_helper.utils.timer import deco_time_block


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_RRF_CONSTANT = 60


class CrossEncoderModel(Protocol):
    def predict(self, sentences: list[tuple[str, str]]) -> Sequence[float]: ...


@dataclass(frozen=True)
class HybridSearchResults(Sequence[dict[str, Any]]):
    """Ranked method results and their RRF fusion for one retrieval search.

    Iteration and indexing expose reranked candidates, preserving the sequence
    behavior expected by answer generation and the Agentic controller.
    """

    vector_results: list[dict[str, Any]]
    bm25_results: list[dict[str, Any]]
    fused_results: list[dict[str, Any]]
    reranked_results: list[dict[str, Any]]
    generation_top_k: int
    stage_latencies_ms: dict[str, float]
    vector_top_k: int
    bm25_top_k: int
    fusion_top_k: int
    effective_thresholds: dict[str, float | None] = field(default_factory=dict)
    configured_top_k: dict[str, int] = field(default_factory=dict)
    filter_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.reranked_results)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.reranked_results)

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        return self.reranked_results[index]

    def to_stage_record(
        self,
        *,
        search_id: int,
        search_query: str,
    ) -> dict[str, Any]:
        stage_results = {
            "vector": (self.vector_results, self.vector_top_k),
            "bm25": (self.bm25_results, self.bm25_top_k),
            "rrf": (self.fused_results, self.fusion_top_k),
            "reranked": (self.reranked_results, self.fusion_top_k),
        }
        stages = {
            name: {
                "top_k": top_k,
                "returned_count": len(results),
                "results": results,
                "latency_ms": self.stage_latencies_ms[name],
            }
            for name, (results, top_k) in stage_results.items()
        }
        return {
            "search_id": search_id,
            "search_query": search_query,
            "generation_top_k": self.generation_top_k,
            "effective_thresholds": self.effective_thresholds,
            "configured_top_k": self.configured_top_k,
            "filter_diagnostics": self.filter_diagnostics,
            "stages": stages,
        }


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str) -> CrossEncoderModel:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


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


def _validate_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number from 0 to 1.")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"{name} must be a finite number from 0 to 1.")
    return threshold


def _score_bounds(scores: Sequence[float]) -> dict[str, float] | None:
    if not scores:
        return None
    return {"highest": float(max(scores)), "lowest": float(min(scores))}


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


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    cross_encoder: CrossEncoderModel | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    model = (
        cross_encoder
        if cross_encoder is not None
        else get_cross_encoder(get_settings().cross_encoder_model)
    )
    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = list(model.predict(pairs))
    if len(scores) != len(candidates):
        raise ValueError(
            "CrossEncoder must return one ranking score for each candidate."
        )

    ranked = [
        {**candidate, "reranker_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    ranked.sort(key=lambda candidate: candidate["reranker_score"], reverse=True)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rerank_rank"] = rank
    return ranked


@deco_time_block
def retrieve(
    rag: RagIndex,
    meta_store: MetaStore,
    *,
    query: str,
    top_k: int | None = None,
    cross_encoder: CrossEncoderModel | None = None,
    vector_min_score: float | None = None,
    bm25_min_score_ratio: float | None = None,
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
    if (vector_min_score is None) != (bm25_min_score_ratio is None):
        raise ValueError(
            "vector_min_score and bm25_min_score_ratio must be provided together."
        )
    if vector_min_score is None:
        vector_min_score = settings.vector_min_score
        bm25_min_score_ratio = settings.bm25_min_score_ratio
    vector_min_score = _validate_threshold("vector_min_score", vector_min_score)
    bm25_min_score_ratio = _validate_threshold(
        "bm25_min_score_ratio",
        bm25_min_score_ratio,
    )

    vector_started = perf_counter()
    model = rag.embed_model
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    vector_scores, vector_indexes_raw = rag.index.search(query_embedding, vector_top_k)  # type: ignore
    vector_latency_ms = round((perf_counter() - vector_started) * 1000, 3)

    bm25_started = perf_counter()
    records = [meta_store.get(index) for index in range(rag.index.ntotal)]  # type: ignore

    observed_vector = [
        (float(score), int(index))
        for score, index in zip(vector_scores[0], vector_indexes_raw[0])
        if index >= 0
    ]
    vector_indexes: list[int] = []
    vector_score_values: list[float] = []
    for score, index in zip(vector_scores[0], vector_indexes_raw[0]):
        if index < 0 or score < vector_min_score:
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
    positive_bm25_indexes = [
        index
        for index, score in enumerate(bm25_score_values)
        if score > 0
    ]
    positive_bm25_indexes.sort(
        key=lambda index: (-bm25_score_values[index], index)
    )
    observed_bm25_indexes = positive_bm25_indexes[:bm25_top_k]
    highest_bm25_score = (
        bm25_score_values[positive_bm25_indexes[0]]
        if positive_bm25_indexes
        else None
    )
    bm25_effective_min_score = (
        highest_bm25_score * bm25_min_score_ratio
        if highest_bm25_score is not None
        else None
    )
    bm25_indexes = (
        [
            index
            for index in observed_bm25_indexes
            if bm25_score_values[index] >= bm25_effective_min_score
        ]
        if bm25_effective_min_score is not None
        else []
    )
    bm25_result_scores = [bm25_score_values[index] for index in bm25_indexes]
    bm25_results = _ranked_method_results(
        indexes=bm25_indexes,
        scores=bm25_result_scores,
        records=records,
        method="bm25",
    )
    bm25_latency_ms = round((perf_counter() - bm25_started) * 1000, 3)

    fusion_started = perf_counter()
    fused_results = _fuse_results(
        vector_results,
        bm25_results,
        vector_indexes,
        bm25_indexes,
        records,
        fusion_top_k,
    )
    fusion_latency_ms = round((perf_counter() - fusion_started) * 1000, 3)

    rerank_started = perf_counter()
    reranked_results = rerank_candidates(
        query,
        fused_results,
        cross_encoder=cross_encoder,
    )
    rerank_latency_ms = round((perf_counter() - rerank_started) * 1000, 3)

    return HybridSearchResults(
        vector_results=vector_results,
        bm25_results=bm25_results,
        fused_results=fused_results,
        reranked_results=reranked_results,
        generation_top_k=settings.llm.max_chunks,
        stage_latencies_ms={
            "vector": vector_latency_ms,
            "bm25": bm25_latency_ms,
            "rrf": fusion_latency_ms,
            "reranked": rerank_latency_ms,
        },
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        fusion_top_k=fusion_top_k,
        effective_thresholds={
            "vector_min_score": vector_min_score,
            "bm25_min_score_ratio": bm25_min_score_ratio,
            "bm25_effective_min_score": (
                float(bm25_effective_min_score)
                if bm25_effective_min_score is not None
                else None
            ),
        },
        configured_top_k={
            "vector": vector_top_k,
            "bm25": bm25_top_k,
            "fusion": fusion_top_k,
            "rerank": fusion_top_k,
            "generation": settings.llm.max_chunks,
        },
        filter_diagnostics={
            "vector": {
                "pre_filter_count": len(observed_vector),
                "post_filter_count": len(vector_indexes),
                "observed_top_k_score_bounds": _score_bounds(
                    [score for score, _index in observed_vector]
                ),
                "highest_excluded_score": max(
                    (
                        score
                        for score, _index in observed_vector
                        if score < vector_min_score
                    ),
                    default=None,
                ),
            },
            "bm25": {
                "pre_filter_count": len(observed_bm25_indexes),
                "post_filter_count": len(bm25_indexes),
                "observed_top_k_score_bounds": _score_bounds(
                    [bm25_score_values[index] for index in observed_bm25_indexes]
                ),
                "highest_excluded_score": max(
                    (
                        bm25_score_values[index]
                        for index in observed_bm25_indexes
                        if index not in bm25_indexes
                    ),
                    default=None,
                ),
            },
        },
    )
