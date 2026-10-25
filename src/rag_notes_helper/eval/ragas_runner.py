import asyncio
from collections.abc import Callable, Sequence
from types import SimpleNamespace

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.utils.logger import get_logger


logger = get_logger("cli")

GENERATION_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
)
RETRIEVAL_METRIC_NAMES = ("context_utilization",)
LEGACY_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_utilization",
)
METRIC_NAMES = (*GENERATION_METRIC_NAMES, *RETRIEVAL_METRIC_NAMES)


def get_eval_llm(*, usage_recorder: Callable[[dict], None] | None = None):
    from rag_notes_helper.eval.llm import get_eval_llm as create_llm

    return create_llm(usage_recorder=usage_recorder)


def get_embeddings():
    from rag_notes_helper.eval.embedding import get_embeddings as create_embeddings

    return create_embeddings()


def _dataset_rows(dataset) -> list[dict]:
    if hasattr(dataset, "to_list"):
        return dataset.to_list()
    return list(dataset)


def _metric_inputs(metric_name: str, row: dict) -> dict:
    user_input = row.get("question", row.get("user_input"))
    response = row.get("answer", row.get("response"))
    retrieved_contexts = row.get("contexts", row.get("retrieved_contexts"))

    if metric_name == "answer_correctness":
        return {
            "user_input": user_input,
            "response": response,
            "reference": row.get("reference_answer", row.get("ground_truth")),
        }
    if metric_name == "answer_relevancy":
        return {"user_input": user_input, "response": response}
    if metric_name == "faithfulness":
        return {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        }
    if metric_name == "context_utilization":
        return {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        }
    raise ValueError(f"Unsupported Ragas metric: {metric_name}")


def run_ragas(
    dataset,
    *,
    metric_names: Sequence[str],
    usage_recorder: Callable[[dict], None] | None = None,
):
    """Run Ragas 0.4 collection metrics over a batch of dataset rows."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        AnswerCorrectness,
        ContextUtilization,
        Faithfulness,
    )

    settings = get_settings()
    provider = settings.llm.effective_eval_provider
    model = settings.llm.effective_eval_model
    logger.info(f"[EVAL] {model} {provider}")

    llm = get_eval_llm(usage_recorder=usage_recorder)
    embeddings = (
        get_embeddings()
        if {"answer_relevancy", "answer_correctness"}.intersection(metric_names)
        else None
    )
    metric_types = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_utilization": ContextUtilization,
        "answer_correctness": AnswerCorrectness,
    }
    metrics = {}
    for name in metric_names:
        metric_type = metric_types[name]
        if name in {"answer_relevancy", "answer_correctness"}:
            metrics[name] = metric_type(llm=llm, embeddings=embeddings)
        else:
            metrics[name] = metric_type(llm=llm)

    async def score_rows():
        results = []
        reasons = []
        for row in _dataset_rows(dataset):
            row_scores = {}
            row_reasons = {}
            for name in metric_names:
                result = await metrics[name].ascore(**_metric_inputs(name, row))
                row_scores[name] = result.value
                reason = getattr(result, "reason", None)
                if reason is not None:
                    row_reasons[name] = reason
            results.append(row_scores)
            reasons.append(row_reasons)
        return results, reasons

    scores, reasons = asyncio.run(score_rows())
    return SimpleNamespace(scores=scores, reasons=reasons)
