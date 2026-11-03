import json
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock
import pytest

import numpy as np
import faiss

from rag_notes_helper.rag.retrieval import rerank_candidates, retrieve
from rag_notes_helper.rag.index import RagIndex
from rag_notes_helper.core.config import LLMSettings, Settings

@pytest.fixture
def mock_meta_store():
    class DummyMetaStore:
        def get(self, faiss_id: int) -> dict:
            data = [{"text": "high"},{"text": "low"}]
            return data[faiss_id]

    return DummyMetaStore()


def test_retrieve_filters_by_score(monkeypatch, mock_meta_store):
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.5")

    dim = 3
    index = faiss.IndexFlatIP(dim)
    vectors = np.array([[1, 0, 0], [0.1, 0, 0]]).astype("float32")

    index.add(vectors) # type: ignore

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(
        [[0.9, 0.1, 0.1]]
    ).astype("float32")

    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        PropertyMock(return_value=mock_model),
    )

    rag = RagIndex(index=index)

    results = retrieve(
        query="any query will lead to the mock return value",
        rag=rag,
        top_k=2,
        meta_store=mock_meta_store,
    )

    assert len(results) == 1
    assert results[0]["text"] == "high"
    assert results[0]["vector_score"] > 0.5


def test_retrieve_applies_explicit_vector_threshold_and_reports_diagnostics(
    monkeypatch,
):
    records = [
        {"doc_id": "high", "chunk_id": 0, "text": "high"},
        {"doc_id": "low", "chunk_id": 0, "text": "low"},
    ]

    class MetaStore:
        def get(self, index):
            return dict(records[index])

    index = faiss.IndexFlatIP(2)
    index.add(np.array([[0.9, 0.0], [0.4, 0.0]], dtype="float32"))
    model = MagicMock()
    model.encode.return_value = np.array([[1.0, 0.0]], dtype="float32")
    monkeypatch.setattr(RagIndex, "embed_model", PropertyMock(return_value=model))
    monkeypatch.setattr(
        "rag_notes_helper.rag.retrieval.get_settings",
        lambda: SimpleNamespace(
            vector_top_k=2,
            bm25_top_k=2,
            fusion_top_k=2,
            min_retrieval_score=0.2,
            bm25_min_score_ratio=0.0,
            llm=SimpleNamespace(max_chunks=1),
        ),
    )

    result = retrieve(
        RagIndex(index),
        MetaStore(),
        query="absent",
        vector_min_score=0.5,
        bm25_min_score_ratio=0.0,
        cross_encoder=SimpleNamespace(predict=lambda pairs: [1.0] * len(pairs)),
    )

    assert [item["doc_id"] for item in result.vector_results] == ["high"]
    assert result.effective_thresholds == {
        "vector_min_score": 0.5,
        "bm25_min_score_ratio": 0.0,
        "bm25_effective_min_score": None,
    }
    vector_diagnostics = result.filter_diagnostics["vector"]
    assert vector_diagnostics["pre_filter_count"] == 2
    assert vector_diagnostics["post_filter_count"] == 1
    assert vector_diagnostics["observed_top_k_score_bounds"] == pytest.approx(
        {"highest": 0.9, "lowest": 0.4}
    )
    assert vector_diagnostics["highest_excluded_score"] == pytest.approx(0.4)


def test_bm25_threshold_is_relative_to_each_query_high_score(monkeypatch):
    records = [
        {"doc_id": "strong", "chunk_id": 0, "text": "rare rare rare ownership"},
        {"doc_id": "weak", "chunk_id": 0, "text": "rare summary"},
        {"doc_id": "none", "chunk_id": 0, "text": "unrelated"},
    ]

    class MetaStore:
        def get(self, index):
            return dict(records[index])

    index = faiss.IndexFlatIP(2)
    index.add(np.zeros((3, 2), dtype="float32"))
    model = MagicMock()
    model.encode.return_value = np.array([[1.0, 0.0]], dtype="float32")
    monkeypatch.setattr(RagIndex, "embed_model", PropertyMock(return_value=model))
    monkeypatch.setattr(
        "rag_notes_helper.rag.retrieval.get_settings",
        lambda: SimpleNamespace(
            vector_top_k=3,
            bm25_top_k=3,
            fusion_top_k=3,
            min_retrieval_score=0.0,
            bm25_min_score_ratio=0.0,
            llm=SimpleNamespace(max_chunks=2),
        ),
    )
    cross_encoder = SimpleNamespace(predict=lambda pairs: [1.0] * len(pairs))

    first = retrieve(
        RagIndex(index),
        MetaStore(),
        query="rare",
        vector_min_score=1.0,
        bm25_min_score_ratio=0.8,
        cross_encoder=cross_encoder,
    )
    scaled = retrieve(
        RagIndex(index),
        MetaStore(),
        query="rare rare rare",
        vector_min_score=1.0,
        bm25_min_score_ratio=0.8,
        cross_encoder=cross_encoder,
    )

    assert [item["doc_id"] for item in first.bm25_results] == ["strong"]
    assert [item["doc_id"] for item in scaled.bm25_results] == ["strong"]
    assert scaled.bm25_results[0]["bm25_score"] == pytest.approx(
        first.bm25_results[0]["bm25_score"] * 3
    )
    assert first.filter_diagnostics["bm25"]["pre_filter_count"] == 2
    assert first.filter_diagnostics["bm25"]["post_filter_count"] == 1
    assert first.filter_diagnostics["bm25"]["highest_excluded_score"] == (
        pytest.approx(0.5022939549)
    )


@pytest.mark.parametrize(
    ("vector_min_score", "bm25_min_score_ratio"),
    [
        (-0.1, 0.0),
        (1.1, 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (0.0, -0.1),
        (0.0, 1.1),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, None),
        (None, 0.0),
    ],
)
def test_invalid_threshold_requests_fail_before_retrieval(
    monkeypatch,
    vector_min_score,
    bm25_min_score_ratio,
):
    monkeypatch.setattr(
        "rag_notes_helper.rag.retrieval.get_settings",
        lambda: SimpleNamespace(
            vector_top_k=2,
            bm25_top_k=2,
            fusion_top_k=2,
            min_retrieval_score=0.2,
            bm25_min_score_ratio=0.0,
            llm=SimpleNamespace(max_chunks=1),
        ),
    )

    with pytest.raises(ValueError, match="finite number|provided together"):
        retrieve(
            object(),
            object(),
            query="query",
            vector_min_score=vector_min_score,
            bm25_min_score_ratio=bm25_min_score_ratio,
        )


def test_hybrid_retrieval_keeps_method_ranks_and_fuses_shared_candidates(monkeypatch):
    records = [
        {"doc_id": "d0", "chunk_id": 0, "source": "notes/0.md", "text": "project summary", "start_line": 1, "end_line": 1},
        {"doc_id": "d1", "chunk_id": 0, "source": "notes/1.md", "text": "rareterm rareterm ownership", "start_line": 2, "end_line": 2},
        {"doc_id": "d2", "chunk_id": 0, "source": "notes/2.md", "text": "rareterm release plan", "start_line": 3, "end_line": 3},
        {"doc_id": "d3", "chunk_id": 0, "source": "notes/3.md", "text": "unrelated material", "start_line": 4, "end_line": 4},
    ]

    class MetaStore:
        def get(self, index):
            return dict(records[index])

    index = faiss.IndexFlatIP(2)
    index.add(np.array([[0.98, 0.01], [0.10, 0.02], [0.80, 0.05], [0.60, 0.0]], dtype="float32"))
    model = MagicMock()
    model.encode.return_value = np.array([[1.0, 0.0]], dtype="float32")
    monkeypatch.setattr(RagIndex, "embed_model", PropertyMock(return_value=model))
    monkeypatch.setattr(
        "rag_notes_helper.rag.retrieval.get_settings",
        lambda: SimpleNamespace(
            vector_top_k=4,
            bm25_top_k=2,
            fusion_top_k=2,
            min_retrieval_score=0.5,
            cross_encoder_model="fixture/cross-encoder",
            llm=SimpleNamespace(max_chunks=5),
        ),
    )

    result = retrieve(RagIndex(index), MetaStore(), query="rareterm")

    assert [item["rank"] for item in result.vector_results] == [1, 2, 3]
    assert [item["doc_id"] for item in result.vector_results] == ["d0", "d2", "d3"]
    assert [item["doc_id"] for item in result.bm25_results] == ["d1", "d2"]
    assert [item["doc_id"] for item in result.fused_results] == ["d2", "d0"]
    assert result.fused_results[0]["vector_rank"] == 2
    assert result.fused_results[0]["bm25_rank"] == 2
    assert result.fused_results[0]["vector_score"] == pytest.approx(0.8)
    assert result.fused_results[0]["bm25_score"] > 0
    assert result[0] is result.reranked_results[0]


def test_reranking_uses_cross_encoder_model_from_settings(monkeypatch):
    factory_calls = []

    class FakeCrossEncoder:
        def predict(self, pairs):
            assert pairs == [("query", "candidate")]
            return [0.9]

    model = FakeCrossEncoder()
    monkeypatch.setattr(
        "rag_notes_helper.rag.retrieval.get_settings",
        lambda: SimpleNamespace(cross_encoder_model="fixture/configured-model"),
    )

    def fake_factory(model_name):
        factory_calls.append(model_name)
        return model

    monkeypatch.setattr("rag_notes_helper.rag.retrieval.get_cross_encoder", fake_factory)
    ranked = rerank_candidates(
        "query",
        [{"source": "notes/a.md", "chunk_id": 1, "text": "candidate"}],
    )

    assert factory_calls == ["fixture/configured-model"]
    assert ranked[0]["reranker_score"] == 0.9


def test_hybrid_retrieval_reranks_fused_candidates_without_score_thresholding(
    monkeypatch,
):
    records = [
        {
            "doc_id": "first",
            "chunk_id": 0,
            "source": "notes/first.md",
            "text": "first candidate text",
            "start_line": 1,
            "end_line": 1,
        },
        {
            "doc_id": "second",
            "chunk_id": 0,
            "source": "notes/second.md",
            "text": "second candidate text",
            "start_line": 2,
            "end_line": 2,
        },
        {
            "doc_id": "third",
            "chunk_id": 0,
            "source": "notes/third.md",
            "text": "third candidate text",
            "start_line": 3,
            "end_line": 3,
        },
    ]

    class MetaStore:
        def get(self, index):
            return dict(records[index])

    index = faiss.IndexFlatIP(2)
    index.add(
        np.array(
            [[0.95, 0.0], [0.8, 0.0], [0.7, 0.0]],
            dtype="float32",
        )
    )
    model = MagicMock()
    model.encode.return_value = np.array([[1.0, 0.0]], dtype="float32")
    monkeypatch.setattr(RagIndex, "embed_model", PropertyMock(return_value=model))
    monkeypatch.setattr(
        "rag_notes_helper.rag.retrieval.get_settings",
        lambda: SimpleNamespace(
            vector_top_k=3,
            bm25_top_k=3,
            fusion_top_k=3,
            min_retrieval_score=0.5,
            llm=SimpleNamespace(max_chunks=2),
        ),
    )

    class FakeCrossEncoder:
        def predict(self, pairs):
            assert pairs == [
                ("Which candidate?", "first candidate text"),
                ("Which candidate?", "second candidate text"),
                ("Which candidate?", "third candidate text"),
            ]
            return [-0.25, 0.9, 0.4]

    result = retrieve(
        RagIndex(index),
        MetaStore(),
        query="Which candidate?",
        cross_encoder=FakeCrossEncoder(),
    )

    assert [item["doc_id"] for item in result.fused_results] == [
        "first",
        "second",
        "third",
    ]
    assert [item["doc_id"] for item in result.reranked_results] == [
        "second",
        "third",
        "first",
    ]
    assert [item["reranker_score"] for item in result.reranked_results] == [
        0.9,
        0.4,
        -0.25,
    ]
    assert result.reranked_results[-1]["vector_score"] == pytest.approx(0.95)

    stage_record = result.to_stage_record(
        search_id=1,
        search_query="Which candidate?",
    )

    assert list(stage_record["stages"]) == ["vector", "bm25", "rrf", "reranked"]
    assert stage_record["search_id"] == 1
    assert stage_record["search_query"] == "Which candidate?"
    assert stage_record["generation_top_k"] == 2
    assert {
        name: stage["top_k"]
        for name, stage in stage_record["stages"].items()
    } == {"vector": 3, "bm25": 3, "rrf": 3, "reranked": 3}
    assert {
        name: stage["returned_count"]
        for name, stage in stage_record["stages"].items()
    } == {"vector": 3, "bm25": 3, "rrf": 3, "reranked": 3}
    assert all(stage["latency_ms"] >= 0 for stage in stage_record["stages"].values())


def test_hybrid_retrieval_returns_no_candidates_when_both_methods_miss(monkeypatch):
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[0.1, 0.0]], dtype="float32"))
    model = MagicMock()
    model.encode.return_value = np.array([[1.0, 0.0]], dtype="float32")
    monkeypatch.setattr(RagIndex, "embed_model", PropertyMock(return_value=model))
    monkeypatch.setattr(
        "rag_notes_helper.rag.retrieval.get_settings",
        lambda: SimpleNamespace(
            vector_top_k=2,
            bm25_top_k=3,
            fusion_top_k=4,
            min_retrieval_score=0.5,
            llm=SimpleNamespace(max_chunks=5),
        ),
    )

    class MetaStore:
        def get(self, _index):
            return {
                "doc_id": "miss",
                "chunk_id": 0,
                "source": "notes/miss.md",
                "text": "unrelated evidence",
                "start_line": 1,
                "end_line": 1,
            }

    result = retrieve(RagIndex(index), MetaStore(), query="absentterm")

    assert result.vector_results == []
    assert result.bm25_results == []
    assert result.fused_results == []
    assert list(result) == []
    assert result.effective_thresholds["bm25_effective_min_score"] is None
    assert result.filter_diagnostics["bm25"] == {
        "pre_filter_count": 0,
        "post_filter_count": 0,
        "observed_top_k_score_bounds": None,
        "highest_excluded_score": None,
    }


def test_hybrid_retrieval_defaults_do_not_couple_candidates_to_generation_limit(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    settings = Settings(
        _env_file=None,
        llm=LLMSettings(provider="ollama", model="llama3", max_chunks=5),
        notes_dir=notes_dir,
        storage_dir=tmp_path / "storage",
        logs_dir=tmp_path / "logs",
    )

    assert settings.vector_top_k == 30
    assert settings.bm25_top_k == 20
    assert settings.fusion_top_k == 15
    assert settings.vector_min_score == 0.2
    assert settings.bm25_min_score_ratio == 0.0
    assert settings.min_retrieval_score == settings.vector_min_score
    assert settings.llm.max_chunks == 5


def test_retrieval_starting_thresholds_load_from_environment(monkeypatch, tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    monkeypatch.setenv("VECTOR_MIN_SCORE", "0.35")
    monkeypatch.setenv("BM25_MIN_SCORE_RATIO", "0.45")

    settings = Settings(
        _env_file=None,
        llm=LLMSettings(provider="ollama", model="llama3", max_chunks=5),
        notes_dir=notes_dir,
        storage_dir=tmp_path / "storage",
        logs_dir=tmp_path / "logs",
    )

    assert settings.vector_min_score == 0.35
    assert settings.bm25_min_score_ratio == 0.45


def test_legacy_top_k_environment_setting_maps_to_fusion_limit(monkeypatch, tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    monkeypatch.setenv("TOP_K", "7")

    settings = Settings(
        _env_file=None,
        llm=LLMSettings(provider="ollama", model="llama3", max_chunks=5),
        notes_dir=notes_dir,
        storage_dir=tmp_path / "storage",
        logs_dir=tmp_path / "logs",
    )

    assert settings.fusion_top_k == 7
    assert settings.top_k == 7


def test_classic_and_agentic_workflows_use_the_shared_hybrid_search(monkeypatch):
    from rag_notes_helper import cli
    from rag_notes_helper.rag import agentic

    records = [
        {"doc_id": "vector", "chunk_id": 0, "source": "notes/vector.md", "text": "vector evidence", "start_line": 1, "end_line": 1},
        {"doc_id": "lexical", "chunk_id": 0, "source": "notes/lexical.md", "text": "rareterm ownership evidence", "start_line": 1, "end_line": 1},
    ]

    class MetaStore:
        def get(self, index):
            return dict(records[index])

    index = faiss.IndexFlatIP(2)
    index.add(np.array([[0.95, 0.0], [0.1, 0.0]], dtype="float32"))
    rag = RagIndex(index)
    model = MagicMock()
    model.encode.return_value = np.array([[1.0, 0.0]], dtype="float32")
    monkeypatch.setattr(RagIndex, "embed_model", PropertyMock(return_value=model))
    monkeypatch.setattr(
        "rag_notes_helper.rag.retrieval.get_settings",
        lambda: SimpleNamespace(
            vector_top_k=2,
            bm25_top_k=2,
            fusion_top_k=2,
            min_retrieval_score=0.5,
            llm=SimpleNamespace(max_chunks=5),
        ),
    )

    def answer(query, *, hits, stream=False, **_kwargs):
        return {"answer": f"answer to {query}", "citations": []}

    class StableCrossEncoder:
        def predict(self, pairs):
            return list(reversed(range(len(pairs))))

    monkeypatch.setattr(cli, "rag_answer", answer)
    cross_encoder = StableCrossEncoder()
    classic = cli.run_query(
        rag,
        MetaStore(),
        query="rareterm",
        agentic=False,
        cross_encoder=cross_encoder,
    )

    class Controller:
        provider_name = "fixture"

        def __init__(self):
            self.calls = 0

        def generate_raw(self, prompt):
            self.calls += 1
            request = json.loads(prompt[1]["content"])
            if "candidate_answer" in request:
                return json.dumps({"action": "accept"})
            if self.calls == 1:
                    return json.dumps({
                        "action": "search",
                        "query": "ownership",
                        "vector_min_score": 0.5,
                        "bm25_min_score_ratio": 0.0,
                        "evidence_mode": "extend",
                    })
            evidence = request["evidence"]
            return json.dumps({
                "action": "answer",
                "evidence": [
                    {"source": item["source"], "chunk_id": item["chunk_id"]}
                    for item in evidence
                ],
            })

    controller = Controller()
    monkeypatch.setattr(agentic, "get_llm", lambda: controller)
    monkeypatch.setattr(agentic, "rag_answer", answer)
    agentic_result = cli.run_query(
        rag,
        MetaStore(),
        query="rareterm",
        agentic=True,
        cross_encoder=cross_encoder,
    )

    expected_sources = {"notes/vector.md", "notes/lexical.md"}
    assert {hit["source"] for hit in classic["retrieved_evidence"]} == expected_sources
    assert controller.calls == 3
    assert len(agentic_result["search_results"]) == 2
    assert all(
        {hit["source"] for hit in search_result} == expected_sources
        for search_result in agentic_result["search_results"]
    )
    assert all(
        hit["vector_rank"] is not None or hit["bm25_rank"] is not None
        for search_result in agentic_result["search_results"]
        for hit in search_result
    )
    assert len(classic["retrieval_stages"]) == 1
    assert len(agentic_result["retrieval_stages"]) == 2
    assert [
        stage["search_query"]
        for stage in agentic_result["retrieval_stages"]
    ] == ["rareterm", "ownership"]
    assert all(
        set(stage["stages"]) == {"vector", "bm25", "rrf", "reranked"}
        for stage in agentic_result["retrieval_stages"]
    )
    classic_thresholds = classic["retrieval_stages"][0]["effective_thresholds"]
    assert classic_thresholds["vector_min_score"] == 0.5
    assert classic_thresholds["bm25_min_score_ratio"] == 0.0
    assert classic_thresholds["bm25_effective_min_score"] == 0.0
    assert agentic_result["retrieval_stages"][0]["effective_thresholds"] == (
        classic["retrieval_stages"][0]["effective_thresholds"]
    )
    assert classic["retrieval_stages"][0]["configured_top_k"] == {
        "vector": 2,
        "bm25": 2,
        "fusion": 2,
        "rerank": 2,
        "generation": 5,
    }
    assert set(classic["retrieval_stages"][0]["filter_diagnostics"]) == {
        "vector",
        "bm25",
    }

