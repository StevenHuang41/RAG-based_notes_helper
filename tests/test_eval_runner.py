import json
import os
from pathlib import Path
import re

import numpy as np

from rag_notes_helper.eval import eval_runner
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
                for hit in evidence
            ],
        })
        return self.last_output


def test_paired_benchmark_uses_synthetic_notes_and_writes_run_artifacts(
    monkeypatch,
    tmp_path,
):
    user_notes = Path(os.environ["NOTES_DIR"])
    user_storage = Path(os.environ["STORAGE_DIR"])
    private_note = user_notes / "private.md"
    private_note.write_text("This must not enter the benchmark.", encoding="utf-8")
    private_storage_file = user_storage / "user-index-sentinel"
    private_storage_file.write_text("Keep this index untouched.", encoding="utf-8")

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

    def fake_ragas(cases, *, usage_recorder):
        if cases[0]["workflow"] == "classic":
            usage_recorder({
                "status": "available",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            })
        else:
            usage_recorder({
                "status": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            })
        return [
            {
                "faithfulness": 1.0,
                "answer_relevancy": 1.0,
                "context_precision": 1.0,
                "context_recall": 1.0,
            }
            for _ in cases
        ]

    monkeypatch.setattr(eval_runner, "_run_ragas_isolated", fake_ragas)
    output_root = tmp_path / "eval-runs"

    run_dir = eval_runner.run_evaluation(output_root=output_root)

    assert run_dir.parent == output_root
    assert re.fullmatch(r"\d{8}T\d{12}Z", run_dir.name)
    assert (run_dir / "run.log").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "summary.json").is_file()
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert results["judge_usage"] == summary["judge_usage"]
    assert len(results["cases"]) == 32
    assert {case["workflow"] for case in results["cases"]} == {
        "classic",
        "agentic",
    }
    assert len({case["case_id"] for case in results["cases"]}) == 16
    assert all(case["answer"] for case in results["cases"])
    assert all(case["citations"] for case in results["cases"])
    assert all(case["retrieved_evidence"] for case in results["cases"])
    assert all(
        set(case["metrics"]) == {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }
        for case in results["cases"]
    )
    assert summary["workflows"]["classic"]["case_count"] == 16
    assert summary["workflows"]["agentic"]["case_count"] == 16
    assert summary["judge_usage"]["total"]["token_usage"] == {
        "status": "partial",
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert (
        summary["judge_usage"]["by_workflow"]["agentic"]["token_usage"]["status"]
        == "unavailable"
    )
    classic_case = next(
        case for case in results["cases"] if case["workflow"] == "classic"
    )
    agentic_case = next(
        case for case in results["cases"] if case["workflow"] == "agentic"
    )
    assert classic_case["generation_usage"]["call_count"] == 1
    assert agentic_case["generation_usage"]["call_count"] == 2
    assert classic_case["generation_usage"]["token_usage"] == {
        "status": "available",
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
    }
    assert agentic_case["generation_usage"]["token_usage"] == {
        "status": "available",
        "input_tokens": 13,
        "output_tokens": 6,
        "total_tokens": 19,
    }
    assert agentic_case["generation_usage"]["character_usage"]["input"] > 0
    assert agentic_case["generation_usage"]["character_usage"]["output"] > 0
    assert summary["workflows"]["agentic"]["generation_usage"]["call_count"] == 32
    assert summary["workflows"]["classic"]["generation_usage"]["token_usage"]["total_tokens"] == 192
    assert summary["workflows"]["agentic"]["generation_usage"]["token_usage"]["total_tokens"] == 304
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
