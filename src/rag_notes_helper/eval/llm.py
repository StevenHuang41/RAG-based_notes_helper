from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from rag_notes_helper.core.config import get_settings

def get_eval_llm():
    settings = get_settings()

    provider = settings.llm.eval_provider

    eval_api_key = settings.llm.eval_api_key or settings.llm.api_key

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.llm.eval_model,
            temperature=0,
            google_api_key=eval_api_key,
            timeout=60,
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=settings.llm.eval_model,
            temperature=0,
            api_key=eval_api_key,
            timeout=60,
        )

    elif provider == "ollama":
        print("[Warning]: Ollama eval may be unreliable")
        return ChatOllama(
            model=settings.llm.eval_model,
            base_url=settings.ollama_base_url,
        )

    else:
        raise ValueError("Unsupported eval LLM")
