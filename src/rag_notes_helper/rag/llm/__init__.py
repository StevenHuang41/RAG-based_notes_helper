
def get_llm():
    from rag_notes_helper.core.config import get_settings
    settings = get_settings()

    provider = settings.llm.provider

    kws = {
        "model": settings.llm.model,
        "api_key": settings.llm.api_key_str,
        "max_tokens": settings.llm.max_tokens,
        "temperature": settings.llm.temperature,
        "line_width": settings.line_width,
    }

    if provider == "hf":
        from .hf_api import HuggingFaceLLM
        return HuggingFaceLLM(**kws)

    elif provider == "gemini":
        from .gemini_api import GeminiLLM
        return GeminiLLM(**kws)

    elif provider == "openai":
        from .openai_api import OpenAILLM
        return OpenAILLM(**kws)

    elif provider == "ollama":
        from .ollama_api import OllamaLLM
        return OllamaLLM(base_url=settings.ollama_base_url, **kws)

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm.provider}")


