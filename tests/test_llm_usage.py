from types import SimpleNamespace

from rag_notes_helper.rag.llm.base import BaseLLM
from rag_notes_helper.rag.llm.gemini_api import GeminiLLM
from rag_notes_helper.rag.llm.hf_api import HuggingFaceLLM
from rag_notes_helper.rag.llm.ollama_api import OllamaLLM
from rag_notes_helper.rag.llm.openai_api import OpenAILLM


def _initialize(llm_type):
    llm = object.__new__(llm_type)
    BaseLLM.__init__(llm, model="fixture-model", api_key=None)
    return llm


def _assert_usage(llm, *, input_tokens, output_tokens, total_tokens):
    assert llm.last_usage == {
        "status": "available",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def test_openai_generation_normalizes_reported_usage():
    llm = _initialize(OpenAILLM)
    llm.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                output_text="answer",
                usage=SimpleNamespace(
                    input_tokens=5,
                    output_tokens=2,
                    total_tokens=7,
                ),
            )
        )
    )

    assert llm.generate_raw([{"role": "user", "content": "question"}]) == "answer"
    _assert_usage(llm, input_tokens=5, output_tokens=2, total_tokens=7)


def test_huggingface_generation_normalizes_reported_usage():
    llm = _initialize(HuggingFaceLLM)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                    usage={
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                    },
                )
            )
        )
    )

    assert llm.generate_raw([{"role": "user", "content": "question"}]) == "answer"
    _assert_usage(llm, input_tokens=5, output_tokens=2, total_tokens=7)


def test_gemini_generation_normalizes_reported_usage():
    llm = _initialize(GeminiLLM)
    llm.client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text="answer",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=5,
                    candidates_token_count=2,
                    total_token_count=7,
                ),
            )
        )
    )

    assert llm.generate_raw([{"role": "user", "content": "question"}]) == "answer"
    _assert_usage(llm, input_tokens=5, output_tokens=2, total_tokens=7)


def test_ollama_generation_normalizes_reported_usage(monkeypatch):
    import rag_notes_helper.rag.llm.ollama_api as ollama_api

    llm = _initialize(OllamaLLM)
    llm.base_url = "http://localhost:11434"

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {"content": "answer"},
                "prompt_eval_count": 5,
                "eval_count": 2,
            }

    monkeypatch.setattr(ollama_api.requests, "post", lambda *args, **kwargs: Response())

    assert llm.generate_raw([{"role": "user", "content": "question"}]) == "answer"
    assert llm.last_usage == {
        "status": "partial",
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": None,
    }


def test_missing_provider_usage_stays_unavailable():
    llm = _initialize(OpenAILLM)
    llm.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text="answer", usage=None)
        )
    )

    assert llm.generate_raw([{"role": "user", "content": "question"}]) == "answer"
    assert llm.last_usage == {
        "status": "unavailable",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
