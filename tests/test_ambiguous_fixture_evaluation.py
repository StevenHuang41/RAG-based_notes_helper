import json
from types import SimpleNamespace

import numpy as np

from rag_notes_helper.eval import eval_runner
from rag_notes_helper.rag.answer import ABSTENTION_MESSAGE
from rag_notes_helper.rag.index import RagIndex


class FixtureEmbeddingModel:
    def encode(self, texts, *, normalize_embeddings, convert_to_numpy):
        return np.ones((len(texts), 4), dtype=np.float32)


class AbstainingAnswerModel:
    provider_name = "fixture"
    model = "fixture-abstainer"
    last_usage = {
        "status": "available",
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }
    last_output = ABSTENTION_MESSAGE

    def generate(self, prompt, line_width):
        self.last_output = ABSTENTION_MESSAGE
        return self.last_output


def test_ambiguous_fixture_cases_are_auditable_and_abstain_through_evaluation(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        RagIndex,
        "embed_model",
        property(lambda self: FixtureEmbeddingModel()),
    )
    monkeypatch.setattr(
        "rag_notes_helper.rag.answer.get_llm",
        lambda: AbstainingAnswerModel(),
    )

    def fake_ragas_worker(_command, *, input, capture_output, text, check):
        request = json.loads(input)
        scores = []
        for batch in request["batches"]:
            metrics = batch["metric_names"]
            scores.append([
                {
                    **{metric: 1.0 for metric in metrics},
                    "metric_reasons": {
                        metric: "fixture judge result" for metric in metrics
                    },
                    "metric_latencies_ms": {metric: 0.1 for metric in metrics},
                }
                for _row in batch["rows"]
            ])
        return SimpleNamespace(
            stdout=json.dumps({"scores": scores, "judge_calls": []}),
            returncode=0,
        )

    monkeypatch.setattr(eval_runner.subprocess, "run", fake_ragas_worker)
    run_dir = eval_runner.run_evaluation(output_root=tmp_path / "runs")
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    ambiguous = [case for case in results["cases"] if case["category"] == "ambiguous"]
    assert len(ambiguous) == 12
    assert all(not case["expected_answerable"] for case in ambiguous)
    assert all(case["ambiguity_reason"].strip() for case in ambiguous)
    assert all(case["expected_sources"] for case in ambiguous)
    assert all(case["expected_evidence"] for case in ambiguous)
    assert all(case["answer"] == ABSTENTION_MESSAGE for case in ambiguous)
    assert all(case["correct_abstention"] for case in ambiguous)
