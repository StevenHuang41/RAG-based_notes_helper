from rag_notes_helper.core.config import settings
from rag_notes_helper.rag.llm.hf import HuggingFaceLLM
from rag_notes_helper.rag.llm.openai_api import OpenAILLM


def get_llm():
    if settings.LLM_PROVIDER == "hf":
        return HuggingFaceLLM()

    elif settings.LLM_PROVIDER == "openai":
        return OpenAILLM()

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")

