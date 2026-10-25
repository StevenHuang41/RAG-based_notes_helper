import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from rag_notes_helper.eval import eval_runner
from rag_notes_helper.rag.agentic import MAX_SEARCHES
from rag_notes_helper.rag.index import RagIndex


class FixtureEmbeddingModel:
    def encode(self, texts, *, normalize_embeddings, convert_to_numpy):
        return np.ones((len(texts), 4), dtype=np.float32)


class FixtureAnswerModel:
    provider_name = "fixture"
    model = "fixture-answer"
    last_usage = {
        "status": "available",
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
    }
    last_output = ""

    def generate(self, prompt, line_width):
        self.last_output = "A synthetic answer based on the fixture notes."
        return self.last_output

    def stream(self, prompt, line_width):
        yield "A synthetic answer based on the fixture notes."


class FixtureController:
    provider_name = "fixture"
    model = "fixture-controller"
    last_usage = {
        "status": "available",
        "input_tokens": 3,
        "output_tokens": 4,
        "total_tokens": 7,
    }
    last_output = ""

    def generate_raw(self, prompt):
        request = json.loads(prompt[-1]["content"])
        evidence = request["evidence"]
        self.last_output = json.dumps({
            "action": "answer",
            "evidence": [
                {"source": hit["source"], "chunk_id": hit["chunk_id"]}
                for hit in evidence[:1]
            ],
        })
        return self.last_output


@pytest.mark.parametrize(
    "agentic,workflow",
    [(False, "classic"), (True, "agentic")],
)
def test_selected_workflow_uses_synthetic_notes_and_writes_run_artifacts(
    monkeypatch,
    tmp_path,
    agentic,
    workflow,
):
    user_notes = Path(os.environ["NOTES_DIR"])
    user_storage = Path(os.environ["STORAGE_DIR"])
    private_note = user_notes / "private.md"
    private_note.write_text("This must not enter the benchmark.", encoding="utf-8")
    private_storage_file = user_storage / "user-index-sentinel"
    private_storage_file.write_text("Keep this index untouched.", encoding="utf-8")

    embedding_cache_sentinel = object()
    monkeypatch.setattr(RagIndex, "_model", embedding_cache_sentinel)
    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        property(lambda self: FixtureEmbeddingModel()),
    )
    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: FixtureAnswerModel(),
    )
    monkeypatch.setattr(
        "rag_notes_helper.rag.agentic.get_llm",
        lambda: FixtureController(),
    )

    fixture_index_observations = []
    real_load_or_build_index = eval_runner.load_or_build_index

    def observe_fixture_index_build():
        fixture_index_observations.append(RagIndex._model is None)
        return real_load_or_build_index()

    monkeypatch.setattr(
        eval_runner,
        "load_or_build_index",
        observe_fixture_index_build,
    )

    worker_requests = []

    def fake_ragas_worker(_command, *, input, capture_output, text, check):
        request = json.loads(input)
        worker_requests.append(request)
        score_sets = [
            [
                {metric: 1.0 for metric in batch["metric_names"]}
                for _row in batch["rows"]
            ]
            for batch in request["batches"]
        ]
        judge_usage = (
            {
                "status": "available",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            }
            if len(worker_requests) == 1
            else {
                "status": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
        )
        return SimpleNamespace(
            stdout=json.dumps({
                "scores": score_sets,
                "judge_calls": [judge_usage],
            }),
            returncode=0,
        )

    monkeypatch.setattr(eval_runner.subprocess, "run", fake_ragas_worker)
    output_root = tmp_path / "eval-runs"

    run_dir = eval_runner.run_evaluation(output_root=output_root, agentic=agentic)

    assert run_dir.parent == output_root
    assert re.fullmatch(r"\d{8}T\d{12}Z", run_dir.name)
    assert (run_dir / "run.log").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "summary.json").is_file()
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert results["judge_usage"] == summary["judge_usage"]
    assert summary["run_metadata"]["judge_provider"] == "ollama"
    assert summary["run_metadata"]["judge_model"] == "llama3.1"
    assert summary["run_metadata"]["agentic_max_searches"] == MAX_SEARCHES
    assert summary["run_metadata"]["workflow"] == workflow
    assert len(results["cases"]) == 16
    assert {case["workflow"] for case in results["cases"]} == {workflow}
    assert len({case["case_id"] for case in results["cases"]}) == 16
    assert all(case["answer"] for case in results["cases"])
    assert all(case["citations"] for case in results["cases"])
    assert all(case["retrieved_evidence"] for case in results["cases"])
    assert all(case["reference_answer"] for case in results["cases"])
    assert all("expected_source_coverage" in case for case in results["cases"])
    assert all("expected_evidence_coverage" in case for case in results["cases"])
    assert all("answerability_correct" in case for case in results["cases"])
    assert all(
        set(case["metrics"]) == {
            "faithfulness",
            "answer_relevancy",
            "context_utilization",
        }
        for case in results["cases"]
    )
    assert all(
        all(value == 1.0 for value in case["metrics"].values())
        for case in results["cases"]
        if case["expected_answerable"]
    )
    assert all(
        all(value is None for value in case["metrics"].values())
        for case in results["cases"]
        if not case["expected_answerable"]
    )
    assert all(
        case["answerability_correct"] is True
        for case in results["cases"]
        if case["expected_answerable"]
    )
    assert all(
        case["answerability_correct"] is False
        and case["correct_abstention"] is False
        for case in results["cases"]
        if not case["expected_answerable"]
    )
    assert set(summary["workflows"]) == {workflow}
    assert summary["workflows"][workflow]["case_count"] == 16
    assert summary["judge_usage"]["total"]["token_usage"] == {
        "status": "available",
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert summary["judge_usage"]["by_workflow"][workflow]["calls"][0][
        "provider"
    ] == "ollama"
    assert summary["judge_usage"]["by_workflow"][workflow]["calls"][0][
        "model"
    ] == "llama3.1"
    selected_case = results["cases"][0]
    assert selected_case["generation_usage"]["call_count"] == (2 if agentic else 1)
    assert selected_case["generation_usage"]["token_usage"] == {
        "status": "available",
        "input_tokens": 13 if agentic else 10,
        "output_tokens": 6 if agentic else 2,
        "total_tokens": 19 if agentic else 12,
    }
    if agentic:
        assert selected_case["generation_usage"]["character_usage"]["input"] > 0
        assert selected_case["generation_usage"]["character_usage"]["output"] > 0
    assert summary["workflows"][workflow]["generation_usage"]["call_count"] == (
        32 if agentic else 16
    )
    assert summary["workflows"][workflow]["generation_usage"]["token_usage"][
        "total_tokens"
    ] == (304 if agentic else 192)
    assert summary["category_counts"] == {
        "direct": 4,
        "multi_hop": 4,
        "unanswerable": 4,
        "ambiguous": 4,
    }
    assert private_note.read_text(encoding="utf-8") == (
        "This must not enter the benchmark."
    )
    assert private_storage_file.read_text(encoding="utf-8") == (
        "Keep this index untouched."
    )
    assert Path(os.environ["NOTES_DIR"]) == user_notes
    assert Path(os.environ["STORAGE_DIR"]) == user_storage
    assert "api_key" not in json.dumps(summary["run_metadata"]).lower()
    assert fixture_index_observations == [True]
    assert RagIndex._model is embedding_cache_sentinel

    workflow_batches = worker_requests[0]["batches"]
    retrieved_rows = workflow_batches[0]["rows"]
    answer_rows = workflow_batches[1]["rows"]
    assert len(retrieved_rows) == 8
    assert workflow_batches[0]["metric_names"] == [
        "answer_relevancy",
        "context_utilization",
    ]
    assert workflow_batches[1]["metric_names"] == ["faithfulness"]
    assert all(
        not {"ground_truth", "reference", "reference_answer"}.intersection(row)
        for row in [*retrieved_rows, *answer_rows]
    )
    assert selected_case["reference_answer"] == (
        "Lena Ortiz is the product owner for Aurora Planner."
    )
    assert [row["contexts"] for row in answer_rows] == [
        case["answer_contexts"]
        for case in results["cases"]
        if case["expected_answerable"]
    ]
    if agentic:
        assert any(
            len(answer_row["contexts"]) < len(retrieved_row["contexts"])
            for answer_row, retrieved_row in zip(answer_rows, retrieved_rows)
        )
    else:
        assert all(
            answer_row["contexts"] == retrieved_row["contexts"]
            for answer_row, retrieved_row in zip(answer_rows, retrieved_rows)
        )
