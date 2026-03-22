from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.eval.llm import get_eval_llm
from rag_notes_helper.eval.embedding import get_embeddings
from rag_notes_helper.utils.logger import get_logger

logger = get_logger("cli")

def run_ragas(dataset):
    settings = get_settings()
    logger.info(f"[EVAL] {settings.llm.eval_model} {settings.llm.eval_provider}")

    llm = get_eval_llm()
    embeddings = get_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    return result
