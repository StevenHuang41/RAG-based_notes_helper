from collections.abc import Callable, Mapping, Sequence

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.utils.logger import get_logger
from rag_notes_helper.rag.llm.base import (
    INPUT_TOKEN_ALIASES,
    OUTPUT_TOKEN_ALIASES,
    TOTAL_TOKEN_ALIASES,
    BaseLLM,
    normalize_token_usage,
)

logger = get_logger("cli")


def evaluate(*args, **kwargs):
    """Load Ragas lazily so CLI startup and FAISS retrieval stay lightweight."""
    from ragas import evaluate as ragas_evaluate

    return ragas_evaluate(*args, **kwargs)


def get_eval_llm():
    from rag_notes_helper.eval.llm import get_eval_llm as create_llm

    return create_llm()


def get_embeddings():
    from rag_notes_helper.eval.embedding import get_embeddings as create_embeddings

    return create_embeddings()


def _usage_sources(source):
    if source is None:
        return
    yield source
    for key in ("usage", "token_usage"):
        nested = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
        if nested is not None:
            yield nested


def _read_usage(response) -> dict:
    sources = []
    for group in getattr(response, "generations", []) or []:
        for generation in group:
            message = getattr(generation, "message", None)
            sources.extend(_usage_sources(getattr(message, "usage_metadata", None)))
            sources.extend(_usage_sources(getattr(message, "response_metadata", None)))
    sources.extend(_usage_sources(getattr(response, "llm_output", None)))

    def first_value(*names):
        return next(
            (
                value
                for source in sources
                if (value := BaseLLM._value(source, *names)) is not None
            ),
            None,
        )

    return normalize_token_usage({
        "input_tokens": first_value(*INPUT_TOKEN_ALIASES),
        "output_tokens": first_value(*OUTPUT_TOKEN_ALIASES),
        "total_tokens": first_value(*TOTAL_TOKEN_ALIASES),
    })


def run_ragas(
    dataset,
    *,
    metric_names: Sequence[str],
    usage_recorder: Callable[[dict], None] | None = None,
):
    from ragas.cost import TokenUsage
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    settings = get_settings()
    logger.info(f"[EVAL] {settings.llm.eval_model} {settings.llm.eval_provider}")

    llm = get_eval_llm()
    embeddings = get_embeddings()

    def token_usage_parser(response):
        usage = _read_usage(response)
        if usage_recorder is not None:
            usage_recorder(usage)
        return TokenUsage(
            input_tokens=usage["input_tokens"] or 0,
            output_tokens=usage["output_tokens"] or 0,
        )

    metrics_by_name = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    result = evaluate(
        dataset=dataset,
        metrics=[metrics_by_name[name] for name in metric_names],
        llm=llm,
        embeddings=embeddings,
        token_usage_parser=token_usage_parser,
        show_progress=False,
    )

    return result
