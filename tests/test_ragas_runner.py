import sys
from types import ModuleType
from types import SimpleNamespace

from rag_notes_helper.eval import ragas_runner


def test_ragas_runner_evaluates_only_the_requested_metrics(monkeypatch):
    captured = {}
    fake_metrics = ModuleType("ragas.metrics")
    faithfulness = object()
    fake_metrics.faithfulness = faithfulness
    fake_metrics.answer_relevancy = object()
    fake_metrics.context_precision = object()
    fake_metrics.context_recall = object()
    fake_ragas = ModuleType("ragas")
    fake_ragas.__path__ = []
    fake_cost = ModuleType("ragas.cost")

    class FakeTokenUsage:
        def __init__(self, *, input_tokens, output_tokens):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    fake_cost.TokenUsage = FakeTokenUsage
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics)
    monkeypatch.setitem(sys.modules, "ragas.cost", fake_cost)
    monkeypatch.setattr(
        ragas_runner,
        "evaluate",
        lambda **kwargs: captured.update(kwargs) or "evaluated",
    )
    monkeypatch.setattr(ragas_runner, "get_eval_llm", lambda: object())
    monkeypatch.setattr(ragas_runner, "get_embeddings", lambda: object())
    monkeypatch.setattr(
        ragas_runner,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(eval_model="judge-model", eval_provider="fixture")
        ),
    )

    result = ragas_runner.run_ragas(
        dataset=object(),
        metric_names=["faithfulness"],
    )

    assert result == "evaluated"
    assert captured["metrics"] == [faithfulness]
