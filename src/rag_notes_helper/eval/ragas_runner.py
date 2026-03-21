from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from rag_notes_helper.core.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_llm():
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.llm.model,
        temperature=0,  # need to be deterministic
        google_api_key=settings.llm.api_key,
    )


def get_embeddings():
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=settings.llm.api_key,
    )


def run_ragas(dataset):
    llm = get_llm()
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
