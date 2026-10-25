import sys
from types import ModuleType, SimpleNamespace

from rag_notes_helper.eval import ragas_runner


def test_ragas_runner_scores_only_requested_current_metrics(monkeypatch):
    metric_calls = []
    constructed = []

    class FakeMetric:
        def __init__(self, llm, embeddings=None):
            constructed.append((llm, embeddings))

        async def ascore(self, **kwargs):
            metric_calls.append(kwargs)
            return SimpleNamespace(value=0.75)

    collections = ModuleType("ragas.metrics.collections")
    collections.Faithfulness = FakeMetric
    collections.AnswerRelevancy = FakeMetric
    collections.ContextUtilization = FakeMetric
    collections.AnswerCorrectness = FakeMetric
    fake_ragas = ModuleType("ragas")
    fake_ragas.__path__ = []
    fake_metrics_package = ModuleType("ragas.metrics")
    fake_metrics_package.__path__ = []
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics_package)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections)

    judge = object()
    embeddings = object()
    usage = []
    monkeypatch.setattr(
        ragas_runner,
        "get_eval_llm",
        lambda *, usage_recorder: (usage_recorder({"status": "unavailable"}) or judge),
    )
    monkeypatch.setattr(ragas_runner, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(
        ragas_runner,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(
                effective_eval_provider="openrouter",
                effective_eval_model="provider/judge-model",
            )
        ),
    )

    result = ragas_runner.run_ragas(
        dataset=[{
            "question": "Why is the launch delayed?",
            "answer": "The validation step is unfinished.",
            "contexts": ["Validation is still in progress."],
            "ground_truth": "The validation step is unfinished.",
        }],
        metric_names=["faithfulness"],
        usage_recorder=usage.append,
    )

    assert result.scores == [{"faithfulness": 0.75}]
    assert constructed == [(judge, None)]
    assert metric_calls == [{
        "user_input": "Why is the launch delayed?",
        "response": "The validation step is unfinished.",
        "retrieved_contexts": ["Validation is still in progress."],
    }]
    assert usage == [{"status": "unavailable"}]


def test_ragas_runner_uses_reference_free_metric_signatures(monkeypatch):
    metric_calls = {}

    def record(metric_name, **kwargs):
        metric_calls[metric_name] = kwargs
        return SimpleNamespace(value=0.5)

    class FakeFaithfulness:
        def __init__(self, llm, embeddings=None):
            pass

        async def ascore(self, *, user_input, response, retrieved_contexts):
            return record(
                "faithfulness",
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )

    class FakeAnswerRelevancy:
        def __init__(self, llm, embeddings=None):
            pass

        async def ascore(self, *, user_input, response):
            return record(
                "answer_relevancy",
                user_input=user_input,
                response=response,
            )

    class FakeContextUtilization:
        def __init__(self, llm, embeddings=None):
            pass

        async def ascore(self, *, user_input, response, retrieved_contexts):
            return record(
                "context_utilization",
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )

    collections = ModuleType("ragas.metrics.collections")
    collections.Faithfulness = FakeFaithfulness
    collections.AnswerRelevancy = FakeAnswerRelevancy
    collections.ContextUtilization = FakeContextUtilization
    collections.AnswerCorrectness = FakeFaithfulness
    fake_ragas = ModuleType("ragas")
    fake_ragas.__path__ = []
    fake_metrics_package = ModuleType("ragas.metrics")
    fake_metrics_package.__path__ = []
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics_package)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections)
    monkeypatch.setattr(ragas_runner, "get_eval_llm", lambda **_kwargs: object())
    monkeypatch.setattr(ragas_runner, "get_embeddings", object)
    monkeypatch.setattr(
        ragas_runner,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(
                effective_eval_provider="openrouter",
                effective_eval_model="provider/judge-model",
            )
        ),
    )

    row = {
        "question": "What happened?",
        "answer": "The launch was delayed.",
        "contexts": ["Validation is still in progress."],
        "ground_truth": "The launch was delayed because validation is unfinished.",
    }
    result = ragas_runner.run_ragas(
        dataset=[row],
        metric_names=[
            "faithfulness",
            "answer_relevancy",
            "context_utilization",
        ],
    )

    assert result.scores == [{
        "faithfulness": 0.5,
        "answer_relevancy": 0.5,
        "context_utilization": 0.5,
    }]
    assert metric_calls == {
        "faithfulness": {
            "user_input": "What happened?",
            "response": "The launch was delayed.",
            "retrieved_contexts": ["Validation is still in progress."],
        },
        "answer_relevancy": {
            "user_input": "What happened?",
            "response": "The launch was delayed.",
        },
        "context_utilization": {
            "user_input": "What happened?",
            "response": "The launch was delayed.",
            "retrieved_contexts": ["Validation is still in progress."],
        },
    }


def test_ragas_runner_scores_reference_correctness_and_preserves_reason(monkeypatch):
    constructed = []
    metric_calls = []

    class FakeMetric:
        def __init__(self, llm, embeddings=None):
            constructed.append((llm, embeddings))

        async def ascore(self, **kwargs):
            metric_calls.append(kwargs)
            return SimpleNamespace(
                value=0.8,
                reason="The answer matches the supported facts.",
            )

    collections = ModuleType("ragas.metrics.collections")
    collections.AnswerCorrectness = FakeMetric
    collections.AnswerRelevancy = FakeMetric
    collections.ContextUtilization = FakeMetric
    collections.Faithfulness = FakeMetric
    fake_ragas = ModuleType("ragas")
    fake_ragas.__path__ = []
    fake_metrics_package = ModuleType("ragas.metrics")
    fake_metrics_package.__path__ = []
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics_package)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections)

    judge = object()
    embeddings = object()
    monkeypatch.setattr(ragas_runner, "get_eval_llm", lambda **_kwargs: judge)
    monkeypatch.setattr(ragas_runner, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(
        ragas_runner,
        "get_settings",
        lambda: SimpleNamespace(
            llm=SimpleNamespace(
                effective_eval_provider="openrouter",
                effective_eval_model="provider/judge-model",
            )
        ),
    )

    result = ragas_runner.run_ragas(
        dataset=[{
            "question": "Why was the launch delayed?",
            "answer": "Validation is incomplete.",
            "contexts": ["Validation is still in progress."],
            "reference_answer": "The launch was delayed because validation is incomplete.",
        }],
        metric_names=["answer_correctness"],
    )

    assert result.scores == [{"answer_correctness": 0.8}]
    assert result.reasons == [{
        "answer_correctness": "The answer matches the supported facts."
    }]
    assert constructed == [(judge, embeddings)]
    assert metric_calls == [{
        "user_input": "Why was the launch delayed?",
        "response": "Validation is incomplete.",
        "reference": "The launch was delayed because validation is incomplete.",
    }]
