from rag_notes_helper.core.config import settings
from rag_notes_helper.rag.llm.hf import HuggingFaceLLM
from rag_notes_helper.rag.llm.ollama import OllamaLLM
from rag_notes_helper.rag.llm.openai_api import OpenAILLM


def get_llm():
    provider = settings.LLM_PROVIDER
    if not provider:
        raise RuntimeError("LLM_PROVIDER is required in .env")

    provider = provider.lower()

    model = settings.LLM_MODEL
    if not model:
        raise RuntimeError("LLM_MODEL is required in .env")

    api_key = settings.LLM_API_KEY
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required in .env")


    if provider == "hf":
        return HuggingFaceLLM(
            model=model,
            api_key=api_key,
        )

    elif provider == "openai":
        return OpenAILLM(
            model=model,
            api_key=api_key
        )

    elif provider == "ollama":

        if not settings.OLLAMA_BASE_URL:
            raise RuntimeError("OLLAMA_BASE_URL is required in .env")

        return OllamaLLM(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")

