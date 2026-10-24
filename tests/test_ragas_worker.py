import io
import json
import sys
from types import ModuleType, SimpleNamespace

from rag_notes_helper.eval import ragas_worker


def test_ragas_worker_preserves_results_for_each_metric_batch(monkeypatch):
    request = {
        "batches": [
            {
                "rows": [{"question": "Q1"}, {"question": "Q2"}],
                "metric_names": ["context_utilization"],
            },
            {
                "rows": [{"question": "Q1"}],
                "metric_names": ["faithfulness"],
            },
        ]
    }
    calls = []

    class FixtureDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    def run_ragas(dataset, *, metric_names, usage_recorder):
        calls.append((dataset, metric_names))
        usage_recorder({"total_tokens": len(dataset)})
        return SimpleNamespace(
            scores=[
                {metric: 0.5 for metric in metric_names}
                for _row in dataset
            ]
        )

    fake_datasets = ModuleType("datasets")
    fake_datasets.Dataset = FixtureDataset
    fake_runner = ModuleType("rag_notes_helper.eval.ragas_runner")
    fake_runner.run_ragas = run_ragas
    stdout = io.StringIO()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "rag_notes_helper.eval.ragas_runner", fake_runner)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(sys, "stdout", stdout)

    ragas_worker.main()

    response = json.loads(stdout.getvalue())
    assert calls == [
        (request["batches"][0]["rows"], ["context_utilization"]),
        (request["batches"][1]["rows"], ["faithfulness"]),
    ]
    assert response == {
        "scores": [
            [{"context_utilization": 0.5}, {"context_utilization": 0.5}],
            [{"faithfulness": 0.5}],
        ],
        "judge_calls": [{"total_tokens": 2}, {"total_tokens": 1}],
    }
