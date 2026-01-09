from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.llm.hf import HuggingFaceLLM
from rag_notes_helper.rag.llm.ollama import OllamaLLM
from rag_notes_helper.rag.llm.openai_api import OpenAILLM


def get_llm():
    settings = get_settings()

    provider = settings.LLM_PROVIDER
    model = settings.LLM_MODEL
    api_key = settings.LLM_API_KEY

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

