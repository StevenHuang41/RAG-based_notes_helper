import json
from contextlib import contextmanager
from io import StringIO
import os
from pathlib import Path
import re
import shutil
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
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")

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
    real_build_and_save_rag = eval_runner.build_and_save_rag_from_isolated_corpus

    def observe_fixture_index_build(
        corpus_root,
        *,
        success_message="Index built and saved",
    ):
        fixture_index_observations.append(RagIndex._model is None)
        return real_build_and_save_rag(
            corpus_root,
            success_message=success_message,
        )

    monkeypatch.setattr(
        eval_runner,
        "build_and_save_rag_from_isolated_corpus",
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
    assert results["schema_version"] == 5
    for case in results["cases"]:
        assert len(case["search_queries"]) == len(case["search_results"])
        assert case["search_queries"][0] == case["question"]
        assert all(
            isinstance(query, str) and query
            for query in case["search_queries"]
        )

        retrieved_by_reference = {
            (hit["source"], hit["chunk_id"]): hit
            for search_hits in case["search_results"]
            for hit in search_hits
        }
        context_references = [
            (item["source"], item["chunk_id"])
            for item in case["generation_context"]
        ]
        assert context_references == [
            (citation["source"], citation["chunk_id"])
            for citation in case["citations"]
        ]
        expected_answer_contexts = []
        for item in case["generation_context"]:
            reference = (item["source"], item["chunk_id"])
            if item["content_changed"]:
                assert item["text"] is not None
                assert item["prompt_block"] is not None
                expected_answer_contexts.append(item["text"])
            else:
                assert item["text"] is None
                assert item["prompt_block"] is None
                expected_answer_contexts.append(
                    retrieved_by_reference[reference]["text"]
                )
        assert case["answer_contexts"] == expected_answer_contexts

    summary = results["summary"]
    readme = (run_dir / "README.md").read_text(encoding="utf-8")
    fixture_cases = eval_runner.load_qa_fixture()
    assert fixture_cases
    answerable_count = sum(case["answerable"] for case in fixture_cases)
    unanswerable_count = len(fixture_cases) - answerable_count
    unanswerable_category_count = sum(
        case["category"] == "unanswerable" for case in fixture_cases
    )
    assert results["judge_usage"] == summary["judge_usage"]
    assert results["run_metadata"]["judge_provider"] == "ollama"
    assert results["run_metadata"]["judge_model"] == "llama3.1"
    assert results["run_metadata"]["agentic_max_searches"] == MAX_SEARCHES
    assert results["run_metadata"]["workflow"] == workflow
    assert results["run_metadata"]["timezone"] == "Asia/Taipei"
    assert results["run_metadata"]["chunk_size"] == 200
    assert results["run_metadata"]["chunk_overlap"] == 50
    assert "[results.json](results.json)" in readme
    assert "[run.log](run.log)" in readme
    assert readme.index("## Overall metrics") < readme.index(
        "## Evaluation process and metric guide"
    ) < readme.index("## Query details")
    assert "## Query details" in readme
    assert "Search query" in readme
    assert "Retrieved chunk text" in readme
    assert "Context passed to answer generation" in readme
    assert "Ragas diagnostics" in readme
    assert "Execution audit: latency and token usage" in readme
    assert "question" in readme
    assert "answer_correctness" in readme
    assert "| Answer Correctness | 1.0000 |" in readme
    assert "| Answer Correctness | 1.0000 |" in readme
    assert "similarity_score (FAISS)" not in readme
    assert "expected_source_coverage" not in readme
    assert "expected_evidence_coverage" not in readme
    assert "| Context Utilization | 1.0000 |" in readme
    assert "Eligible cases / selected cases" not in readme
    assert "### Answerability outcomes" not in readme
    assert "Judge: context_utilization" in readme
    assert "#### Citations" not in readme
    assert readme.count("#### Execution audit: latency and token usage") == len(
        results["cases"]
    )
    assert "direct-01" in readme
    assert any(
        hit["source"] in readme
        for case in results["cases"]
        for search_hits in case["search_results"]
        for hit in search_hits
    )
    assert "reason for faithfulness" not in readme
    assert "## Metrics by category" in readme
    assert "Judge usage by metric" in readme
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "Evaluation artifacts written successfully." in run_log
    assert "Who is the product owner for Aurora Planner?" not in run_log
    assert len(results["cases"]) == len(fixture_cases)
    assert [case["case_id"] for case in results["cases"]] == [
        case["id"] for case in fixture_cases
    ]
    assert sum(case["category"] == "multi_hop" for case in results["cases"]) == 15
    assert sum(case["category"] == "unanswerable" for case in results["cases"]) == 13
    assert {case["workflow"] for case in results["cases"]} == {workflow}
    assert len({case["case_id"] for case in results["cases"]}) == len(fixture_cases)
    assert all(case["challenge_tags"] for case in results["cases"])
    boundary_case = next(
        case for case in results["cases"] if case["case_id"] == "multi-hop-03"
    )
    assert "chunk_boundary" in boundary_case["challenge_tags"]
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
    workflow_summary = summary["workflows"][workflow]
    assert workflow_summary["case_count"] == len(fixture_cases)
    assert workflow_summary["ragas_metric_eligible_case_counts"] == {
        metric: answerable_count
        for metric in (
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "context_utilization",
        )
    }
    assert workflow_summary["answerability_outcomes"] == {
        "expected_answerable": {
            "case_count": answerable_count,
            "answered_count": answerable_count,
            "abstained_count": 0,
            "unavailable_count": 0,
            "error_count": 0,
            "unavailable_or_error_count": 0,
            "correct_behavior_count": answerable_count,
        },
        "expected_unanswerable": {
            "case_count": unanswerable_count,
            "answered_count": unanswerable_count,
            "abstained_count": 0,
            "unavailable_count": 0,
            "error_count": 0,
            "unavailable_or_error_count": 0,
            "correct_behavior_count": 0,
        },
    }
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

    workflow_batches = [
        batch
        for request in worker_requests
        for batch in request["batches"]
    ]
    answer_relevancy_rows = workflow_batches[0]["rows"]
    context_utilization_rows = workflow_batches[1]["rows"]
    answer_rows = workflow_batches[2]["rows"]
    assert len(context_utilization_rows) == answerable_count
    assert len(answer_relevancy_rows) == answerable_count
    assert len(answer_rows) == answerable_count
    assert [batch["metric_names"] for batch in workflow_batches] == [
        ["answer_relevancy"],
        ["context_utilization"],
        ["faithfulness"],
        ["answer_correctness"],
    ]
    assert all(
        not {"ground_truth", "reference", "reference_answer"}.intersection(row)
        for row in [
            *answer_relevancy_rows,
            *context_utilization_rows,
            *answer_rows,
        ]
    )
    correctness_rows = workflow_batches[3]["rows"]
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
    assert [row["contexts"] for row in context_utilization_rows] == [
        case["answer_contexts"]
        for case in results["cases"]
        if case["expected_answerable"]
    ]


def test_default_evaluation_run_writes_artifacts_under_results(monkeypatch):
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            provider="fixture",
            model="fixture-answer",
            effective_eval_provider="fixture",
            effective_eval_model="fixture-judge",
            max_chunks=2,
        ),
        embed_model_name="fixture-embedding",
        chunk_size=200,
        chunk_overlap=50,
        top_k=3,
        min_retrieval_score=0.0,
    )

    @contextmanager
    def isolated_index():
        yield settings, object(), object()

    monkeypatch.setattr(eval_runner, "load_qa_fixture", lambda: [])
    monkeypatch.setattr(eval_runner, "_isolated_fixture_index", isolated_index)
    monkeypatch.setattr(
        eval_runner,
        "_attach_ragas_scores",
        lambda *args, **kwargs: None,
    )

    run_dir = eval_runner.run_evaluation()
    try:
        assert run_dir.parent.name == "results"
        assert run_dir.parent.parent.name == "evals"
        assert (run_dir / "README.md").is_file()
        assert (run_dir / "results.json").is_file()
        assert (run_dir / "run.log").is_file()
    finally:
        shutil.rmtree(run_dir)


@pytest.mark.parametrize("agentic,workflow", [(False, "classic"), (True, "agentic")])
@pytest.mark.parametrize("seed", [271, None])
def test_sampled_run_records_reproducible_case_selection(
    monkeypatch, tmp_path, agentic, workflow, seed
):
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            provider="fixture",
            model="fixture-answer",
            effective_eval_provider="fixture",
            effective_eval_model="fixture-judge",
            max_chunks=2,
        ),
        embed_model_name="fixture-embedding",
        chunk_size=200,
        chunk_overlap=50,
        top_k=3,
        min_retrieval_score=0.0,
    )

    @contextmanager
    def isolated_index():
        yield settings, object(), object()

    def record_case(case, *, workflow, rag, meta_store):
        return {
            "case_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "reference_answer": case["reference_answer"],
            "answer": "fixture answer",
            "expected_answerable": case["answerable"],
            "predicted_answerable": case["answerable"],
            "answerability_correct": True,
            "abstained": False,
            "correct_abstention": False,
            "expected_sources": [],
            "expected_evidence": [],
            "retrieval_metrics": {},
            "expected_source_coverage": 0.0,
            "expected_evidence_coverage": 0.0,
            "latency": {},
            "generation_usage": {"calls": [], "call_count": 0},
            "judge_usage": {"calls": []},
            "generation_metrics": {},
            "search_queries": [],
            "search_results": [],
            "generation_context": [],
            "judge_metrics": {},
            "workflow": workflow,
        }

    monkeypatch.setattr(eval_runner, "_isolated_fixture_index", isolated_index)
    monkeypatch.setattr(eval_runner, "_record_case", record_case)
    monkeypatch.setattr(eval_runner, "_attach_ragas_scores", lambda *args, **kwargs: None)

    run_dir = eval_runner.run_evaluation(
        output_root=tmp_path / workflow,
        agentic=agentic,
        sample=5,
        seed=seed,
    )

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    readme = (run_dir / "README.md").read_text(encoding="utf-8")
    fixture_cases = eval_runner.load_qa_fixture()
    expected_ids = [case["id"] for case in fixture_cases]
    sampled_ids = results["run_metadata"]["selected_case_ids"]

    if seed == 271:
        assert sampled_ids == [
            "ambiguous-04",
            "multi-hop-05",
            "multi-hop-13",
            "unanswerable-05",
            "unanswerable-07",
        ]
    assert len(sampled_ids) == 5
    assert len(set(sampled_ids)) == 5
    assert sampled_ids == [case_id for case_id in expected_ids if case_id in sampled_ids]
    assert [case["case_id"] for case in results["cases"]] == sampled_ids
    assert results["run_metadata"]["sample_count"] == 5
    assert results["run_metadata"]["seed"] == seed
    assert results["run_metadata"]["workflow"] == workflow
    assert results["summary"]["case_count"] == 5
    assert "**Cases:** 5" in readme
    assert "Sample size | 5" in readme
    assert f"Sampling seed | {'Not set' if seed is None else seed}" in readme
    assert ", ".join(sampled_ids) in readme


def test_seed_without_sample_runs_full_fixture(monkeypatch, tmp_path):
    recorded_ids = []

    @contextmanager
    def isolated_index():
        yield SimpleNamespace(
            llm=SimpleNamespace(
                provider="fixture",
                model="fixture-answer",
                effective_eval_provider="fixture",
                effective_eval_model="fixture-judge",
                max_chunks=2,
            ),
            embed_model_name="fixture-embedding",
            chunk_size=200,
            chunk_overlap=50,
            top_k=3,
            min_retrieval_score=0.0,
        ), object(), object()

    def record_case(case, *, workflow, rag, meta_store):
        recorded_ids.append(case["id"])
        return {
            "case_id": case["id"], "category": case["category"],
            "question": case["question"],
            "reference_answer": case["reference_answer"], "answer": "answer",
            "expected_answerable": case["answerable"],
            "predicted_answerable": case["answerable"],
            "answerability_correct": True, "abstained": False,
            "correct_abstention": False, "expected_sources": [],
            "expected_evidence": [], "retrieval_metrics": {},
            "expected_source_coverage": 0.0, "expected_evidence_coverage": 0.0,
            "latency": {}, "generation_usage": {"calls": [], "call_count": 0},
            "judge_usage": {"calls": []}, "generation_metrics": {},
            "search_queries": [], "search_results": [], "generation_context": [],
            "judge_metrics": {}, "workflow": workflow,
        }

    monkeypatch.setattr(eval_runner, "_isolated_fixture_index", isolated_index)
    monkeypatch.setattr(eval_runner, "_record_case", record_case)
    monkeypatch.setattr(eval_runner, "_attach_ragas_scores", lambda *args, **kwargs: None)

    run_dir = eval_runner.run_evaluation(output_root=tmp_path, seed=19)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    readme = (run_dir / "README.md").read_text(encoding="utf-8")
    fixture_ids = [case["id"] for case in eval_runner.load_qa_fixture()]

    assert recorded_ids == fixture_ids
    assert results["run_metadata"]["selected_case_ids"] == fixture_ids
    assert results["run_metadata"]["sample_count"] is None
    assert results["run_metadata"]["seed"] == 19
    assert "Sample size | Full fixture" in readme
    assert "Sampling seed | 19" in readme


def test_run_summary_counts_errors_and_unavailable_answers_separately(
    monkeypatch,
    tmp_path,
):
    cases = [
        {
            "id": "answerable-error",
            "category": "direct",
            "question": "Who owns the review?",
            "reference_answer": "Mina owns the review.",
            "answerable": True,
            "expected_sources": [],
            "expected_evidence": [],
        },
        {
            "id": "unanswerable-unavailable",
            "category": "unanswerable",
            "question": "What is the private code?",
            "reference_answer": "No private code is documented.",
            "answerable": False,
            "expected_sources": [],
            "expected_evidence": [],
        },
    ]
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            provider="fixture",
            model="fixture-answer",
            effective_eval_provider="fixture",
            effective_eval_model="fixture-judge",
            max_chunks=2,
        ),
        embed_model_name="fixture-embedding",
        chunk_size=200,
        chunk_overlap=50,
        top_k=3,
        min_retrieval_score=0.0,
    )

    @contextmanager
    def isolated_index():
        yield settings, object(), object()

    def record_case(case, *, workflow, rag, meta_store):
        record = {
            "case_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "reference_answer": case["reference_answer"],
            "answer": None,
            "expected_answerable": case["answerable"],
            "predicted_answerable": None,
            "answerability_correct": None,
            "abstained": None,
            "correct_abstention": None,
            "expected_sources": [],
            "expected_evidence": [],
            "retrieval_metrics": {
                "recall": None,
                "precision": None,
                "ndcg": None,
                "context_utilization": None,
            },
            "generation_metrics": {
                "faithfulness": None,
                "answer_relevancy": None,
                "answer_correctness": None,
            },
            "expected_source_coverage": None,
            "expected_evidence_coverage": None,
            "latency": {
                "retrieval_ms": 0.0,
                "controller_ms": 0.0,
                "answer_generation_ms": 0.0,
                "generation_ms": 0.0,
                "judge_ms": 0.0,
                "rag_query_ms": 0.0,
                "end_to_end_ms": 0.0,
            },
            "generation_usage": {"calls": [], "call_count": 0},
            "judge_usage": {"calls": [], "call_count": 0},
            "judge_usage_by_metric": {},
            "judge_metrics": {},
            "search_queries": [case["question"]],
            "search_results": [[]],
            "generation_context": [],
            "workflow": workflow,
        }
        if case["id"] == "answerable-error":
            record["workflow_error"] = "RuntimeError"
        return record

    monkeypatch.setattr(eval_runner, "load_qa_fixture", lambda: cases)
    monkeypatch.setattr(eval_runner, "_isolated_fixture_index", isolated_index)
    monkeypatch.setattr(eval_runner, "_record_case", record_case)
    monkeypatch.setattr(
        eval_runner,
        "_attach_ragas_scores",
        lambda *args, **kwargs: None,
    )

    run_dir = eval_runner.run_evaluation(output_root=tmp_path)

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    workflow_summary = results["summary"]["workflows"]["classic"]
    assert workflow_summary["answerability_outcomes"] == {
        "expected_answerable": {
            "case_count": 1,
            "answered_count": 0,
            "abstained_count": 0,
            "unavailable_count": 0,
            "error_count": 1,
            "unavailable_or_error_count": 1,
            "correct_behavior_count": 0,
        },
        "expected_unanswerable": {
            "case_count": 1,
            "answered_count": 0,
            "abstained_count": 0,
            "unavailable_count": 1,
            "error_count": 0,
            "unavailable_or_error_count": 1,
            "correct_behavior_count": 0,
        },
    }
    assert workflow_summary["by_category"]["direct"][
        "answerability_outcomes"
    ]["expected_unanswerable"] == {
        "case_count": 0,
        "answered_count": None,
        "abstained_count": None,
        "unavailable_count": None,
        "error_count": None,
        "unavailable_or_error_count": None,
        "correct_behavior_count": None,
    }


def test_whitespace_answer_is_unavailable_and_not_ragas_eligible(
    monkeypatch,
    tmp_path,
):
    case = {
        "id": "answerable-whitespace",
        "category": "direct",
        "challenge_tags": [],
        "question": "Who owns the review?",
        "reference_answer": "Mina owns the review.",
        "answerable": True,
        "expected_sources": [],
        "expected_evidence": [],
    }
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            provider="fixture",
            model="fixture-answer",
            effective_eval_provider="fixture",
            effective_eval_model="fixture-judge",
            max_chunks=2,
        ),
        embed_model_name="fixture-embedding",
        chunk_size=200,
        chunk_overlap=50,
        top_k=3,
        min_retrieval_score=0.0,
    )

    @contextmanager
    def isolated_index():
        yield settings, object(), object()

    hit = {
        "source": "notes/review.md",
        "chunk_id": 1,
        "start_line": 1,
        "end_line": 1,
        "score": 0.9,
        "text": "Mina owns the review.",
    }

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
        return {
            "answer": "  \n ",
            "citations": [],
            "retrieved_evidence": [hit],
            "search_results": [[hit]],
            "search_queries": [query],
            "generation_context": [{
                "source": hit["source"],
                "chunk_id": hit["chunk_id"],
            }],
        }

    worker_requests = []

    def fake_ragas_worker(_command, *, input, capture_output, text, check):
        request = json.loads(input)
        worker_requests.append(request)
        score_sets = [
            [{metric: 1.0 for metric in batch["metric_names"]} for _ in batch["rows"]]
            for batch in request["batches"]
        ]
        return SimpleNamespace(
            stdout=json.dumps({"scores": score_sets, "judge_calls": []}),
            returncode=0,
        )

    monkeypatch.setattr(eval_runner, "load_qa_fixture", lambda: [case])
    monkeypatch.setattr(eval_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(eval_runner, "_isolated_fixture_index", isolated_index)
    monkeypatch.setattr("rag_notes_helper.cli.run_query", fake_run_query)
    monkeypatch.setattr(eval_runner.subprocess, "run", fake_ragas_worker)

    run_dir = eval_runner.run_evaluation(output_root=tmp_path)

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    record = results["cases"][0]
    workflow_summary = results["summary"]["workflows"]["classic"]
    assert record["answer"] == "  \n "
    assert record["predicted_answerable"] is None
    assert record["answerability_correct"] is None
    assert record["abstained"] is None
    assert workflow_summary["answerability_outcomes"]["expected_answerable"] == {
        "case_count": 1,
        "answered_count": 0,
        "abstained_count": 0,
        "unavailable_count": 1,
        "error_count": 0,
        "unavailable_or_error_count": 1,
        "correct_behavior_count": 0,
    }
    assert workflow_summary["ragas_metric_eligible_case_counts"] == {
        "faithfulness": 0,
        "answer_relevancy": 0,
        "answer_correctness": 0,
        "context_utilization": 0,
    }
    assert record["generation_metrics"] == {
        "faithfulness": None,
        "answer_relevancy": None,
        "answer_correctness": None,
    }
    assert worker_requests == []
    assert "| Faithfulness | — |" in run_dir.joinpath("README.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "sample,message",
    [(0, "positive integer"), (-1, "positive integer"), (51, "fixture size")],
)
def test_evaluation_rejects_invalid_sample_count_before_creating_run(
    monkeypatch, tmp_path, sample, message
):
    monkeypatch.setattr(
        eval_runner,
        "load_qa_fixture",
        lambda: [{"id": "case-1"}, {"id": "case-2"}],
    )

    with pytest.raises(ValueError, match=message):
        eval_runner.run_evaluation(output_root=tmp_path, sample=sample, seed=1)

    assert not any(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{6}", path.name)
        for path in tmp_path.iterdir()
    )


def test_ragas_metrics_skip_context_dependent_scores_when_retrieval_is_empty(
    monkeypatch,
):
    cases = [
        {
            "id": "empty-context",
            "expected_answerable": True,
            "question": "What is the approved date?",
            "answer": "I cannot find the answer in the notes.",
            "contexts": [],
            "answer_contexts": [],
            "reference_answer": "2026-06-01",
        },
        {
            "id": "with-context",
            "expected_answerable": True,
            "question": "Who owns QA?",
            "answer": "Priya owns QA.",
            "contexts": ["Priya owns QA verification."],
            "answer_contexts": ["Priya owns QA verification."],
            "reference_answer": "Priya Shah owns QA verification.",
        },
        {
            "id": "missing-answer",
            "expected_answerable": True,
            "question": "Which date was approved?",
            "answer": None,
            "contexts": [],
            "answer_contexts": [],
            "reference_answer": "2026-06-01",
        },
        {
            "id": "unanswerable",
            "expected_answerable": False,
            "question": "What is the private launch code?",
            "answer": "I cannot find the answer in the notes.",
            "contexts": [],
            "answer_contexts": [],
            "reference_answer": "",
        },
    ]
    batches_seen = []

    def fake_batches(batches, *, usage_recorder):
        outputs = []
        for batch in batches:
            metric_name = batch["metric_names"][0]
            batches_seen.append(batch)
            scores = []
            for row_index, _row in enumerate(batch["rows"]):
                usage_recorder({
                    "metric": metric_name,
                    "row_index": row_index,
                    "token_usage": {"status": "unavailable"},
                })
                scores.append({
                    metric_name: 0.5,
                    "metric_reasons": {},
                    "metric_latencies_ms": {metric_name: 1.0},
                })
            outputs.append(scores)
        return outputs

    monkeypatch.setattr(eval_runner, "_run_ragas_batches_isolated", fake_batches)
    judge_calls = []

    scores = eval_runner._run_ragas_isolated(
        cases,
        usage_recorder=judge_calls.append,
    )

    assert [batch["metric_names"] for batch in batches_seen] == [
        ["answer_relevancy"],
        ["context_utilization"],
        ["faithfulness"],
        ["answer_correctness"],
    ]
    assert [len(batch["rows"]) for batch in batches_seen] == [2, 1, 1, 2]
    assert scores[0]["answer_relevancy"] == 0.5
    assert scores[0]["answer_correctness"] == 0.5
    assert "context_utilization" not in scores[0]
    assert "faithfulness" not in scores[0]
    assert "no answer context" in scores[0]["metric_reasons"][
        "context_utilization"
    ]
    assert "no answer context" in scores[0]["metric_reasons"]["faithfulness"]
    assert scores[2] == {
        "metric_reasons": {
            name: "Skipped because no generated answer was available."
            for name in eval_runner.METRIC_NAMES
        },
        "metric_latencies_ms": {},
    }
    assert scores[1]["context_utilization"] == 0.5
    assert scores[1]["faithfulness"] == 0.5
    assert scores[3] == {}
    assert [call["row_index"] for call in judge_calls if call["metric"] == "context_utilization"] == [1]
    assert [call["row_index"] for call in judge_calls if call["metric"] == "faithfulness"] == [1]


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

    def should_not_build_index(*_args, **_kwargs):
        pytest.fail("invalid fixture should be rejected before indexing")

    monkeypatch.setattr(
        eval_runner,
        "build_and_save_rag_from_isolated_corpus",
        should_not_build_index,
    )

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


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("transformed_context", [False, True])
def test_run_evaluation_records_ordered_context_prefix_metrics(
    monkeypatch,
    tmp_path,
    agentic,
    transformed_context,
):
    case = {
        "id": "direct-context-prefix",
        "category": "direct",
        "challenge_tags": [],
        "question": "Which lines support the answer?",
        "reference_answer": "The evidence is on lines two and three.",
        "answerable": True,
        "expected_sources": ["notes/alpha.md"],
        "expected_evidence": [{
            "source": "notes/alpha.md",
            "start_line": 2,
            "end_line": 3,
            "text": "shared\nanswer evidence",
        }],
    }
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            provider="fixture",
            model="fixture-answer",
            effective_eval_provider="fixture",
            effective_eval_model="fixture-judge",
            max_chunks=2,
        ),
        embed_model_name="fixture-embedding",
        chunk_size=200,
        chunk_overlap=50,
        top_k=2,
        min_retrieval_score=0.0,
    )

    @contextmanager
    def isolated_index():
        yield settings, object(), object()

    first_hit = {
        "source": "notes/alpha.md",
        "chunk_id": 1,
        "start_line": 4,
        "end_line": 4,
        "score": 0.95,
        "text": "irrelevant context",
        "stage": "retrieval",
        "method": "FAISS vector search",
    }
    second_hit = {
        "source": "notes/alpha.md",
        "chunk_id": 2,
        "start_line": 2,
        "end_line": 3,
        "score": 0.90,
        "text": "shared\nanswer evidence",
        "stage": "retrieval",
        "method": "FAISS vector search",
    }

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
        context_second_text = (
            "answer evidence" if transformed_context else second_hit["text"]
        )
        context = [first_hit, second_hit]
        generation_context = [
            {
                "source": hit["source"],
                "chunk_id": hit["chunk_id"],
                "start_line": hit["start_line"],
                "end_line": hit["end_line"],
                "text": (
                    context_second_text if hit is second_hit else hit["text"]
                ),
                "prompt_block": hit["text"],
            }
            for hit in context
        ]
        search_results = (
            [[first_hit], [second_hit]] if agentic else [[first_hit, second_hit]]
        )
        search_queries = (
            [query, "controller follow-up query"] if agentic else [query]
        )
        citations = [
            {
                "source": hit["source"],
                "chunk_id": hit["chunk_id"],
                "score": hit["score"],
                "start_line": hit["start_line"],
                "end_line": hit["end_line"],
            }
            for hit in context
        ]
        return {
            "answer": "The evidence is on lines two and three.",
            "citations": citations,
            "retrieved_evidence": [first_hit, second_hit],
            "search_results": search_results,
            "search_queries": search_queries,
            "generation_context": generation_context,
        }

    monkeypatch.setattr(eval_runner, "load_qa_fixture", lambda: [case])
    monkeypatch.setattr(eval_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(eval_runner, "_isolated_fixture_index", isolated_index)
    monkeypatch.setattr(
        "rag_notes_helper.cli.run_query",
        fake_run_query,
    )
    monkeypatch.setattr(
        eval_runner,
        "_attach_ragas_scores",
        lambda *args, **kwargs: None,
    )

    run_dir = eval_runner.run_evaluation(
        output_root=tmp_path,
        agentic=agentic,
    )

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    case_result = results["cases"][0]
    prefixes = case_result["context_prefix_metrics"]
    assert results["run_metadata"]["top_k"] == 2
    assert [prefix["context_prefix_k"] for prefix in prefixes] == [1, 2]
    assert prefixes[0] == {
        "context_prefix_k": 1,
        "evidence_lines_covered": 0,
        "ground_truth_line_count": 2,
        "recall": 0.0,
        "precision": 0.0,
        "ndcg": 0.0,
        "line_provenance_available": True,
    }
    if transformed_context:
        assert prefixes[1] == {
            "context_prefix_k": 2,
            "evidence_lines_covered": None,
            "ground_truth_line_count": 2,
            "recall": None,
            "precision": None,
            "ndcg": None,
            "line_provenance_available": False,
        }
    else:
        assert prefixes[1] == {
            "context_prefix_k": 2,
            "evidence_lines_covered": 2,
            "ground_truth_line_count": 2,
            "recall": 1.0,
            "precision": 2 / 3,
            "ndcg": pytest.approx(0.6309297535714575),
            "line_provenance_available": True,
        }

    assert case_result["final_context_metrics"] == prefixes[-1]
    assert results["summary"]["workflows"][
        "agentic" if agentic else "classic"
    ]["final_context_metrics"] == {
        name: prefixes[-1][name]
        for name in ("recall", "precision", "ndcg")
    }


@pytest.mark.parametrize("agentic", [False, True])
def test_run_evaluation_uses_final_answer_context_for_context_utilization(
    monkeypatch,
    tmp_path,
    agentic,
):
    case = {
        "id": "final-context-utilization",
        "category": "direct",
        "challenge_tags": [],
        "question": "What final context reaches the answer?",
        "reference_answer": "Only the selected chunk.",
        "answerable": True,
        "expected_sources": ["notes/alpha.md"],
        "expected_evidence": [{
            "source": "notes/alpha.md",
            "start_line": 1,
            "end_line": 1,
            "text": "selected evidence",
        }],
    }
    settings = SimpleNamespace(
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
        top_k=2,
        min_retrieval_score=0.0,
    )
    selected_hit = {
        "source": "notes/alpha.md", "chunk_id": 1,
        "start_line": 1, "end_line": 1, "score": 0.95,
        "text": "selected evidence",
    }
    rejected_hit = {
        "source": "notes/alpha.md", "chunk_id": 2,
        "start_line": 2, "end_line": 2, "score": 0.90,
        "text": "candidate only",
    }

    @contextmanager
    def isolated_index():
        yield settings, object(), object()

    def fake_run_query(*_args, **_kwargs):
        return {
            "answer": "Only the selected chunk.",
            "citations": [selected_hit],
            "retrieved_evidence": [selected_hit, rejected_hit],
            "search_results": [[selected_hit, rejected_hit]],
            "search_queries": [case["question"]],
            "generation_context": [selected_hit],
        }

    captured_batches = []

    def fake_worker(_command, *, input, capture_output, text, check):
        request = json.loads(input)
        captured_batches.extend(request["batches"])
        return SimpleNamespace(
            stdout=json.dumps({
                "scores": [[{
                    **{metric: 1.0 for metric in batch["metric_names"]},
                    "metric_reasons": {},
                    "metric_latencies_ms": {},
                } for _row in batch["rows"]]
                for batch in request["batches"]],
                "judge_calls": [],
            }),
            returncode=0,
        )

    monkeypatch.setattr(eval_runner, "load_qa_fixture", lambda: [case])
    monkeypatch.setattr(eval_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(eval_runner, "_isolated_fixture_index", isolated_index)
    monkeypatch.setattr("rag_notes_helper.cli.run_query", fake_run_query)
    monkeypatch.setattr(eval_runner.subprocess, "run", fake_worker)

    eval_runner.run_evaluation(output_root=tmp_path, agentic=agentic)

    context_utilization_rows = next(
        batch["rows"] for batch in captured_batches
        if batch["metric_names"] == ["context_utilization"]
    )
    assert context_utilization_rows == [{
        "question": case["question"],
        "answer": "Only the selected chunk.",
        "contexts": [selected_hit["text"]],
    }]
