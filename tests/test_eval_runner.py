import json
from contextlib import contextmanager
from io import StringIO
import os
from pathlib import Path
import re
from types import SimpleNamespace
from datetime import datetime
from zoneinfo import ZoneInfo

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
    capsys,
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
    real_build_and_save_rag = eval_runner.build_and_save_rag

    def observe_fixture_index_build(*, success_message="Index built and saved"):
        fixture_index_observations.append(RagIndex._model is None)
        return real_build_and_save_rag(success_message=success_message)

    monkeypatch.setattr(
        eval_runner,
        "build_and_save_rag",
        observe_fixture_index_build,
    )

    worker_requests = []

    def fake_ragas_worker(_command, *, input, capture_output, text, check):
        request = json.loads(input)
        worker_requests.append(request)
        score_sets = [
            [
                {
                    **{metric: 1.0 for metric in batch["metric_names"]},
                    "metric_reasons": {
                        metric: f"reason for {metric}"
                        for metric in batch["metric_names"]
                    },
                    "metric_latencies_ms": {
                        metric: 1.5 for metric in batch["metric_names"]
                    },
                }
                for _row in batch["rows"]
            ]
            for batch in request["batches"]
        ]
        judge_usage = [
            {
                "metric": metric,
                "row_index": row_index,
                "token_usage": {
                    "status": "available",
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
            }
            for batch in request["batches"]
            for row_index, _row in enumerate(batch["rows"])
            for metric in batch["metric_names"]
        ]
        return SimpleNamespace(
            stdout=json.dumps({
                "scores": score_sets,
                "judge_calls": judge_usage,
            }),
            returncode=0,
        )

    monkeypatch.setattr(eval_runner.subprocess, "run", fake_ragas_worker)
    output_root = tmp_path / "eval-runs"

    class Terminal(StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    monkeypatch.setattr(eval_runner.sys, "stderr", terminal)

    run_dir = eval_runner.run_evaluation(output_root=output_root, agentic=agentic)
    console = capsys.readouterr()

    assert run_dir.parent == output_root
    assert "Index not found. Building index from notes" not in console.out
    assert console.out.count("Fixture index built and saved.") == 1
    assert f"Starting {workflow} fixture evaluation" in terminal.getvalue()
    assert "Building isolated synthetic fixture index" in terminal.getvalue()
    assert "A synthetic answer based on the fixture notes." not in (
        console.out + terminal.getvalue()
    )
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{6}", run_dir.name)
    assert (run_dir / "run.log").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "README.md").is_file()
    assert not (run_dir / "summary.json").exists()
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    summary = results["summary"]
    readme = (run_dir / "README.md").read_text(encoding="utf-8")
    assert results["judge_usage"] == summary["judge_usage"]
    assert results["run_metadata"]["judge_provider"] == "ollama"
    assert results["run_metadata"]["judge_model"] == "llama3.1"
    assert results["run_metadata"]["agentic_max_searches"] == MAX_SEARCHES
    assert results["run_metadata"]["workflow"] == workflow
    assert results["run_metadata"]["timezone"] == "Asia/Taipei"
    assert "[results.json](results.json)" in readme
    assert "[run.log](run.log)" in readme
    assert readme.index("## Overall metrics") < readme.index("## Query details")
    assert "## Query details" in readme
    assert "similarity_score (FAISS)" in readme
    assert "Chunk text" in readme
    assert "question" in readme
    assert "answer_correctness" in readme
    assert "retrieval_ms" in readme
    assert "direct-01" in readme
    assert any(
        hit["source"] in readme
        for case in results["cases"]
        for search_hits in case["search_results"]
        for hit in search_hits
    )
    assert "reason for faithfulness" in readme
    assert "## Metrics by category" in readme
    assert "Controller/answer tokens" in readme
    assert "Judge tokens" in readme
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "Evaluation artifacts written successfully." in run_log
    assert "Who is the product owner for Aurora Planner?" not in run_log
    fixture_cases = eval_runner.load_qa_fixture()
    assert len(results["cases"]) == len(fixture_cases)
    assert [case["case_id"] for case in results["cases"]] == [
        case["id"] for case in fixture_cases
    ]
    assert sum(case["category"] == "multi_hop" for case in results["cases"]) == 15
    assert {case["workflow"] for case in results["cases"]} == {workflow}
    assert len({case["case_id"] for case in results["cases"]}) == len(fixture_cases)
    assert all(case["challenge_tags"] for case in results["cases"])
    assert all(
        case["ambiguity_reason"]
        for case in results["cases"]
        if case["category"] == "ambiguous"
    )
    assert all(
        case["missing_information"]
        for case in results["cases"]
        if case["category"] == "unanswerable"
    )
    assert all(case["answer"] for case in results["cases"])
    assert all(case["citations"] for case in results["cases"])
    assert all(case["retrieved_evidence"] for case in results["cases"])
    assert all(case["reference_answer"] for case in results["cases"])
    assert all("expected_source_coverage" in case for case in results["cases"])
    assert all("expected_evidence_coverage" in case for case in results["cases"])
    assert all("answerability_correct" in case for case in results["cases"])
    assert all(
        set(case["retrieval_metrics"]) == {
            "recall",
            "precision",
            "ndcg",
            "context_utilization",
        }
        for case in results["cases"]
    )
    assert all(
        set(case["generation_metrics"]) == {
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        }
        for case in results["cases"]
    )
    assert all(
        all(value == 1.0 for value in case["generation_metrics"].values())
        for case in results["cases"]
        if case["expected_answerable"]
    )
    assert all(
        all(value is None for value in case["generation_metrics"].values())
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
    assert summary["workflows"][workflow]["case_count"] == len(fixture_cases)
    direct_summary = summary["workflows"][workflow]["by_category"]["direct"]
    assert direct_summary["generation_usage"]["token_usage"]["total_tokens"] == (
        sum(case["category"] == "direct" for case in fixture_cases)
        * (19 if agentic else 12)
    )
    direct_answerable_count = sum(
        case["category"] == "direct" and case["answerable"]
        for case in fixture_cases
    )
    answerable_count = sum(case["answerable"] for case in fixture_cases)
    assert direct_summary["judge_usage"]["token_usage"]["total_tokens"] == (
        direct_answerable_count * 12
    )
    assert summary["judge_usage"]["total"]["token_usage"] == {
        "status": "available",
        "input_tokens": answerable_count * 8,
        "output_tokens": answerable_count * 4,
        "total_tokens": answerable_count * 12,
    }
    assert summary["judge_usage"]["by_workflow"][workflow]["calls"][0][
        "provider"
    ] == "ollama"
    assert summary["judge_usage"]["by_workflow"][workflow]["calls"][0][
        "model"
    ] == "llama3.1"
    assert summary["judge_usage"]["by_metric"]["answer_correctness"][
        "token_usage"
    ]["total_tokens"] == answerable_count * 3
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
    assert selected_case["judge_usage"]["call_count"] == 4
    assert selected_case["judge_usage"]["token_usage"]["total_tokens"] == 12
    assert selected_case["judge_usage_by_metric"]["faithfulness"][
        "token_usage"
    ]["total_tokens"] == 3
    assert set(selected_case["judge_metrics"]) == {
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_utilization",
    }
    assert all(
        item["latency_ms"] == 1.5
        for item in selected_case["judge_metrics"].values()
    )
    assert selected_case["latency"]["judge_ms"] == 6.0
    assert selected_case["latency"]["end_to_end_ms"] >= selected_case["latency"][
        "rag_query_ms"
    ]
    assert summary["workflows"][workflow]["generation_usage"]["call_count"] == (
        len(fixture_cases) * (2 if agentic else 1)
    )
    assert summary["workflows"][workflow]["generation_usage"]["token_usage"][
        "total_tokens"
    ] == len(fixture_cases) * (19 if agentic else 12)
    assert summary["category_counts"] == {
        category: sum(case["category"] == category for case in fixture_cases)
        for category in ("direct", "multi_hop", "unanswerable", "ambiguous")
    }
    assert private_note.read_text(encoding="utf-8") == (
        "This must not enter the benchmark."
    )
    assert private_storage_file.read_text(encoding="utf-8") == (
        "Keep this index untouched."
    )
    assert Path(os.environ["NOTES_DIR"]) == user_notes
    assert Path(os.environ["STORAGE_DIR"]) == user_storage
    assert "api_key" not in json.dumps(results["run_metadata"]).lower()
    assert fixture_index_observations == [True]
    assert RagIndex._model is embedding_cache_sentinel

    workflow_batches = worker_requests[0]["batches"]
    retrieved_rows = workflow_batches[0]["rows"]
    answer_rows = workflow_batches[1]["rows"]
    assert len(retrieved_rows) == answerable_count
    assert len(answer_rows) == answerable_count
    assert workflow_batches[0]["metric_names"] == [
        "answer_relevancy",
        "context_utilization",
    ]
    assert workflow_batches[1]["metric_names"] == ["faithfulness"]
    assert workflow_batches[2]["metric_names"] == ["answer_correctness"]
    assert all(
        not {"ground_truth", "reference", "reference_answer"}.intersection(row)
        for row in [*retrieved_rows, *answer_rows]
    )
    assert all("reference_answer" not in row for row in workflow_batches[1]["rows"])
    correctness_rows = workflow_batches[2]["rows"]
    assert [row["reference_answer"] for row in correctness_rows] == [
        case["reference_answer"]
        for case in results["cases"]
        if case["expected_answerable"]
    ]
    assert selected_case["reference_answer"] == (
        "Lena Ortiz is the product owner."
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


@pytest.mark.parametrize(
    ("case_id", "field", "error_fragment"),
    [
        ("ambiguous-01", "ambiguity_reason", "ambiguity_reason"),
        ("unanswerable-01", "missing_information", "missing_information"),
    ],
)
def test_evaluation_rejects_cases_without_auditable_rationale(
    monkeypatch,
    tmp_path,
    case_id,
    field,
    error_fragment,
):
    records = json.loads(eval_runner.QA_FIXTURE.read_text(encoding="utf-8"))
    case = next(record for record in records if record["id"] == case_id)
    del case[field]
    qa_fixture = tmp_path / "notes_qa.json"
    qa_fixture.write_text(json.dumps(records), encoding="utf-8")
    real_loader = eval_runner.load_qa_fixture
    monkeypatch.setattr(
        eval_runner,
        "load_qa_fixture",
        lambda: real_loader(qa_fixture),
    )

    def should_not_build_index(*, success_message="Index built and saved"):
        pytest.fail("invalid fixture should be rejected before indexing")

    monkeypatch.setattr(eval_runner, "build_and_save_rag", should_not_build_index)

    with pytest.raises(ValueError, match=f"{case_id}.*{error_fragment}"):
        eval_runner.run_evaluation(output_root=tmp_path / "evaluation")


def test_run_directory_uses_taipei_seconds_and_refuses_collision(
    monkeypatch,
    tmp_path,
):
    fixed_now = datetime(2026, 10, 25, 15, 42, 25, tzinfo=ZoneInfo("Asia/Taipei"))

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(eval_runner, "datetime", FixedDateTime)
    output_root = tmp_path / "eval"

    timestamp, run_dir = eval_runner._new_run_directory(output_root)

    assert timestamp == "2026-10-25-154225"
    assert run_dir.name == timestamp
    with pytest.raises(FileExistsError):
        eval_runner._new_run_directory(output_root)
    assert list(output_root.iterdir()) == [run_dir]


def test_artifact_failure_does_not_log_successful_write(
    monkeypatch,
    tmp_path,
):
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            provider="fixture",
            model="fixture-answer",
            effective_eval_provider="fixture",
            effective_eval_model="fixture-judge",
            max_chunks=5,
        ),
        embed_model_name="fixture-embedding",
        chunk_size=100,
        chunk_overlap=10,
        top_k=5,
        min_retrieval_score=0.0,
    )

    @contextmanager
    def isolated_fixture_index():
        yield settings, object(), object()

    def fail_readme_write(_path, _results):
        raise OSError("simulated README write failure")

    monkeypatch.setattr(eval_runner, "load_qa_fixture", lambda: [])
    monkeypatch.setattr(
        eval_runner,
        "_isolated_fixture_index",
        isolated_fixture_index,
    )
    monkeypatch.setattr(eval_runner, "write_readme", fail_readme_write)

    output_root = tmp_path / "eval-runs"
    with pytest.raises(OSError, match="simulated README write failure"):
        eval_runner.run_evaluation(output_root=output_root)

    run_dir = next(output_root.iterdir())
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "Evaluation artifacts written successfully." not in run_log
    assert "Evaluation failed (OSError)." in run_log
