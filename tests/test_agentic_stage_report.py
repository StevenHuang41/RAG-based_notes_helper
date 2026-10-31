import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from rag_notes_helper.eval import eval_runner
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE


def _settings():
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="fixture",
            model="fixture-answer",
            effective_eval_provider="fixture",
            effective_eval_model="fixture-judge",
            max_chunks=1,
        ),
        embed_model_name="fixture-embedding",
        chunk_size=200,
        chunk_overlap=50,
        top_k=3,
        vector_top_k=2,
        bm25_top_k=2,
        fusion_top_k=3,
        min_retrieval_score=0.0,
    )


def _hit(source, chunk_id, start, end, text, *, vector_rank, bm25_rank,
         rank, rerank_rank):
    return {
        "source": source,
        "chunk_id": chunk_id,
        "start_line": start,
        "end_line": end,
        "text": text,
        "rank": rank,
        "score": 0.5 + (1 / (rank + 10)),
        "vector_rank": vector_rank,
        "vector_score": 0.8,
        "bm25_rank": bm25_rank,
        "bm25_score": 2.0,
        "fusion_score": 1 / (rank + 60),
        "rerank_rank": rerank_rank,
        "reranker_score": 0.9,
    }


def _search_record(search_id, search_query, vector, bm25, rrf, reranked):
    stage_results = {
        "vector": (vector, 2),
        "bm25": (bm25, 2),
        "rrf": (rrf, 3),
        "reranked": (reranked, 3),
    }
    return {
        "search_id": search_id,
        "search_query": search_query,
        "generation_top_k": 1,
        "stages": {
            name: {
                "top_k": cutoff,
                "returned_count": len(hits),
                "latency_ms": search_id * stage_index / 10,
                "results": hits,
            }
            for stage_index, (name, (hits, cutoff)) in enumerate(
                stage_results.items(), start=1
            )
        },
    }


def test_agentic_evaluation_preserves_search_stages_union_metrics_context_and_latency(
    monkeypatch,
    tmp_path,
):
    evidence_case = {
        "id": "agentic-repeated-searches",
        "category": "multi-hop",
        "challenge_tags": ["repeated-search"],
        "question": "Who owns the release and what is the backup?",
        "reference_answer": "Mina owns the release; Kai is the backup.",
        "answerable": True,
        "expected_sources": ["notes/release.md"],
        "expected_evidence": [{
            "source": "notes/release.md",
            "start_line": 10,
            "end_line": 11,
            "text": "Reviewed release ownership evidence.",
        }],
    }
    abstention_case = {
        "id": "agentic-abstention",
        "category": "unanswerable",
        "challenge_tags": [],
        "question": "What is the undocumented emergency code?",
        "reference_answer": "The notes do not contain an emergency code.",
        "answerable": False,
        "expected_sources": [],
        "expected_evidence": [],
    }
    cases = [evidence_case, abstention_case]
    settings = _settings()

    noise = _hit(
        "notes/noise.md", 99, 1, 1, "RETRIEVED_ONLY_NOISE",
        vector_rank=1, bm25_rank=2, rank=1, rerank_rank=2,
    )
    line_ten = _hit(
        "notes/release.md", 10, 10, 10, "RETRIEVED_ONLY_LINE_TEN",
        vector_rank=2, bm25_rank=1, rank=2, rerank_rank=1,
    )
    line_eleven = _hit(
        "notes/release.md", 11, 11, 11, "FINAL_CONTEXT_ONLY_TEXT",
        vector_rank=3, bm25_rank=3, rank=3, rerank_rank=3,
    )
    answer_searches = [
        _search_record(
            1,
            evidence_case["question"],
            [noise, line_ten],
            [line_ten, noise],
            [noise, line_ten, line_eleven],
            [line_ten, noise, line_eleven],
        ),
        _search_record(
            2,
            "release backup owner",
            [line_eleven, noise],
            [noise, line_eleven],
            [noise, line_eleven, line_ten],
            [line_eleven, noise, line_ten],
        ),
    ]
    empty_search = _search_record(1, abstention_case["question"], [], [], [], [])

    @contextmanager
    def isolated_index():
        yield settings, object(), object()

    def fake_run_query(
        _rag,
        _meta_store,
        *,
        query,
        agentic,
        stream,
        on_generation,
        on_retrieval,
    ):
        assert agentic is True
        if query == evidence_case["question"]:
            on_generation({"call_type": "controller", "latency_ms": 1.2})
            on_retrieval({
                "search_index": 1, "latency_ms": 10.0, "result_count": 3,
            })
            on_generation({"call_type": "controller", "latency_ms": 2.3})
            on_retrieval({
                "search_index": 2, "latency_ms": 11.0, "result_count": 3,
            })
            on_generation({"call_type": "answer", "latency_ms": 3.4})
            return {
                "answer": "Mina owns the release; Kai is the backup.",
                "citations": [line_eleven],
                "retrieved_evidence": [noise, line_ten, line_eleven],
                "search_results": [
                    [noise, line_ten, line_eleven],
                    [line_eleven, noise, line_ten],
                ],
                "search_queries": [
                    evidence_case["question"], "release backup owner",
                ],
                "retrieval_stages": answer_searches,
                "generation_context": [{
                    "source": line_eleven["source"],
                    "chunk_id": line_eleven["chunk_id"],
                    "start_line": line_eleven["start_line"],
                    "end_line": line_eleven["end_line"],
                    "text": line_eleven["text"],
                    "prompt_block": line_eleven["text"],
                    "reranker_score": line_eleven["reranker_score"],
                }],
            }

        # This represents the bounded initial-search abstention path: there is
        # no controller or answer generation call, and retrieval latency was
        # unavailable in the callback.
        on_retrieval({
            "search_index": 1, "latency_ms": None, "result_count": 0,
        })
        return {
            "answer": ABSTENTION_MESSAGE,
            "citations": [],
            "retrieved_evidence": [],
            "search_results": [[]],
            "search_queries": [query],
            "retrieval_stages": [empty_search],
            "generation_context": [],
        }

    def fake_ragas(cases_to_score, *, usage_recorder):
        metric_names = (
            "faithfulness", "answer_relevancy", "answer_correctness",
            "context_utilization",
        )
        for metric in metric_names:
            usage_recorder({
                "row_index": 0,
                "metric": metric,
                "latency_ms": 7.8 if metric == "faithfulness" else None,
                "token_usage": {
                    "status": "available",
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
            })
        return [
            {
                **{metric: 0.8 for metric in metric_names},
                "metric_reasons": {},
                "metric_latencies_ms": {
                    "faithfulness": 7.8,
                    "answer_relevancy": None,
                    "answer_correctness": None,
                    "context_utilization": None,
                },
            },
            {
                **{metric: None for metric in metric_names},
                "metric_reasons": {},
                "metric_latencies_ms": {metric: None for metric in metric_names},
            },
        ]

    monkeypatch.setattr(eval_runner, "load_qa_fixture", lambda: cases)
    monkeypatch.setattr(eval_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(eval_runner, "_isolated_fixture_index", isolated_index)
    monkeypatch.setattr("rag_notes_helper.cli.run_query", fake_run_query)
    monkeypatch.setattr(eval_runner, "_run_ragas_isolated", fake_ragas)

    run_dir = eval_runner.run_evaluation(output_root=tmp_path, agentic=True)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    answer_result, abstain_result = results["cases"]

    assert [
        (search["search_id"], search["search_query"])
        for search in answer_result["retrieval_stages"]
    ] == [
        (1, evidence_case["question"]),
        (2, "release backup owner"),
    ]
    first_reranked = answer_result["retrieval_stages"][0]["stages"]["reranked"]
    assert [
        (name, stage["top_k"], stage["returned_count"])
        for name, stage in answer_result["retrieval_stages"][0]["stages"].items()
    ] == [
        ("vector", 2, 2),
        ("bm25", 2, 2),
        ("rrf", 3, 3),
        ("reranked", 3, 3),
    ]
    assert first_reranked["top_k"] == 3
    assert first_reranked["returned_count"] == 3
    assert first_reranked["results"][0]["rank"] == 1
    assert first_reranked["results"][0]["fusion_rank"] == 2
    assert first_reranked["results"][0]["reranker_score"] == 0.9

    vector_metrics = answer_result["retrieval_stage_metrics"]["vector"]
    assert vector_metrics["search_count"] == 2
    assert vector_metrics["recall"] == 1.0
    assert vector_metrics["precision"] == pytest.approx(2 / 3)
    assert vector_metrics["ndcg"] == pytest.approx(0.8154648767857288)

    vector_per_search = [
        search["stages"]["vector"]["metrics"]["ndcg"]
        for search in answer_result["retrieval_stages"]
    ]
    assert vector_per_search[0] == pytest.approx(0.6309297535714575)
    assert vector_per_search[1] == 1.0
    assert answer_result["retrieval_stage_metrics"]["bm25"]["recall"] == 1.0
    assert answer_result["retrieval_stage_metrics"]["rrf"]["precision"] == pytest.approx(
        2 / 3
    )
    assert answer_result["retrieval_stage_metrics"]["reranked"]["ndcg"] == pytest.approx(
        0.9197207891481876
    )
    assert answer_result["retrieval_stages"][1]["stages"]["reranked"][
        "results"
    ][0]["chunk_id"] == line_eleven["chunk_id"]

    comparison = answer_result["rerank_comparison"]["searches"]
    assert [item["common_top_k"] for item in comparison] == [1, 1]
    assert all(item["candidate_pool_matches"] for item in comparison)
    assert [item["reranked_metrics"]["recall"] for item in comparison] == [
        0.5, 0.5,
    ]
    assert [item["fused_metrics"]["recall"] for item in comparison] == [
        0.0, 0.0,
    ]
    assert answer_result["retrieval_stage_metrics"]["rrf"]["recall"] == 1.0
    assert answer_result["final_context_metrics"]["recall"] == 0.5
    assert answer_result["final_context_metrics"]["precision"] == 1.0
    assert answer_result["final_context_metrics"]["ndcg"] == 1.0

    assert answer_result["latency"]["retrieval_calls"] == [
        {"search_index": 1, "latency_ms": 10.0, "result_count": 3},
        {"search_index": 2, "latency_ms": 11.0, "result_count": 3},
    ]
    assert answer_result["latency"]["controller_ms"] == 3.5
    assert answer_result["latency"]["answer_generation_ms"] == 3.4
    assert answer_result["latency"]["judge_metrics_ms"]["faithfulness"] == 7.8
    assert answer_result["latency"]["judge_ms"] is None
    assert answer_result["latency"]["rag_query_ms"] is not None
    assert answer_result["latency"]["end_to_end_ms"] is None

    assert abstain_result["answer"] == ABSTENTION_MESSAGE
    assert abstain_result["generation_context"] == []
    assert abstain_result["final_context_metrics"]["recall"] is None
    assert abstain_result["retrieval_stage_metrics"]["vector"]["recall"] is None
    assert abstain_result["latency"]["retrieval_ms"] is None
    assert abstain_result["latency"]["controller_ms"] is None
    assert abstain_result["latency"]["answer_generation_ms"] is None
    assert abstain_result["latency"]["judge_ms"] is None
    assert abstain_result["latency"]["end_to_end_ms"] == pytest.approx(
        abstain_result["latency"]["rag_query_ms"]
    )

    report = (run_dir / "README.md").read_text(encoding="utf-8")
    assert "Search ID | Search query" in report
    assert "##### Vector results" in report
    assert "##### BM25 results" in report
    assert "##### RRF results" in report
    assert "##### Reranked results" in report
    assert "Union recall" in report
    assert "Mean per-search NDCG" in report
    assert "Pre-rerank and post-rerank comparison" in report
    assert "Common cutoff" in report
    assert "Search 1 | Retrieval search" in report
    assert "Search 2 | Retrieval search" in report
    assert "Controller latency" in report
    assert "Judge latency" in report
    assert "RAG query latency" in report
    assert "End-to-end evaluation latency" in report
    assert "RETRIEVED_ONLY_NOISE" not in report
    assert "RETRIEVED_ONLY_LINE_TEN" not in report
    assert report.count("FINAL_CONTEXT_ONLY_TEXT") == 1
