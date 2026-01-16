from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.llm.hf import HuggingFaceLLM
from rag_notes_helper.rag.llm.ollama import OllamaLLM
from rag_notes_helper.rag.llm.openai_api import OpenAILLM


def get_llm():
    settings = get_settings()

    provider = settings.llm.provider
    model = settings.llm.model
    api_key = settings.llm.api_key_str

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

        if not settings.ollama_base_url:
            raise RuntimeError("OLLAMA_BASE_URL is required in .env")

        return OllamaLLM(
            model=model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm.provider}")

