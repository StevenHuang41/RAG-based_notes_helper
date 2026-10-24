import asyncio
import sys
from types import ModuleType, SimpleNamespace

from pydantic import BaseModel
import pytest

from rag_notes_helper.eval import llm as eval_llm


class JudgeResponse(BaseModel):
    verdict: str


def _settings(provider, model, api_key):
    return SimpleNamespace(
        llm=SimpleNamespace(
            effective_eval_provider=provider,
            effective_eval_model=model,
            effective_eval_api_key_str=api_key,
        ),
        ollama_base_url="http://ollama.test:11434",
    )


def _install_fake_ragas_base(monkeypatch):
    package = ModuleType("ragas")
    package.__path__ = []
    module = ModuleType("ragas.llms")

    class InstructorBaseRagasLLM:
        pass

    module.InstructorBaseRagasLLM = InstructorBaseRagasLLM
    monkeypatch.setitem(sys.modules, "ragas", package)
    monkeypatch.setitem(sys.modules, "ragas.llms", module)


@pytest.mark.parametrize(
    ("provider", "model", "api_key", "expected_options"),
    [
        ("openai", "gpt-4o-mini", "shared-key", {"api_key": "shared-key"}),
        (
            "ollama",
            "llama3.1",
            None,
            {"api_key": "ollama", "base_url": "http://ollama.test:11434/v1"},
        ),
    ],
)
def test_openai_compatible_judge_routes_generation_providers(
    monkeypatch, provider, model, api_key, expected_options
):
    import openai

    constructors = []

    def fake_client(**kwargs):
        constructors.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))

    monkeypatch.setattr(openai, "OpenAI", fake_client)
    monkeypatch.setattr(openai, "AsyncOpenAI", fake_client)
    monkeypatch.setattr(
        eval_llm,
        "get_settings",
        lambda: _settings(provider, model, api_key),
    )
    _install_fake_ragas_base(monkeypatch)

    llm = eval_llm.get_eval_llm()

    assert llm.model == model
    assert constructors == [expected_options, expected_options]


def test_openrouter_judge_uses_configured_model_and_records_response_usage(
    monkeypatch,
):
    import openai

    constructors = []
    requests = []
    usage = []

    def response():
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=JudgeResponse(verdict="supported"))
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=13,
                completion_tokens=4,
                total_tokens=17,
            ),
        )

    class FakeSyncCompletions:
        def parse(self, **kwargs):
            requests.append(kwargs)
            return response()

    class FakeAsyncCompletions:
        async def parse(self, **kwargs):
            requests.append(kwargs)
            return response()

    def fake_openai(**kwargs):
        constructors.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=FakeSyncCompletions())
        )

    def fake_async_openai(**kwargs):
        constructors.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=FakeAsyncCompletions())
        )

    monkeypatch.setattr(openai, "AsyncOpenAI", fake_async_openai)
    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    monkeypatch.setattr(
        eval_llm,
        "get_settings",
        lambda: _settings("openrouter", "meta-llama/llama-3.1-8b-instruct", "judge-key"),
    )
    _install_fake_ragas_base(monkeypatch)

    llm = eval_llm.get_eval_llm(usage_recorder=usage.append)
    sync_result = llm.generate("Judge whether the answer is supported.", JudgeResponse)
    result = asyncio.run(
        llm.agenerate("Judge whether the answer is supported.", JudgeResponse)
    )

    assert sync_result == JudgeResponse(verdict="supported")
    assert result == JudgeResponse(verdict="supported")
    assert constructors == [
        {"api_key": "judge-key", "base_url": "https://openrouter.ai/api/v1"},
        {"api_key": "judge-key", "base_url": "https://openrouter.ai/api/v1"},
    ]
    assert requests[0]["model"] == "meta-llama/llama-3.1-8b-instruct"
    assert requests[0]["response_format"] is JudgeResponse
    assert usage == [
        {
            "status": "available",
            "input_tokens": 13,
            "output_tokens": 4,
            "total_tokens": 17,
        }
    ] * 2


def test_gemini_judge_uses_structured_response_schema(monkeypatch):
    from google import genai

    requests = []
    usage = []

    class FakeSyncModels:
        def generate_content(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                parsed={"verdict": "supported"},
                text='{"verdict":"supported"}',
                usage_metadata=SimpleNamespace(
                    prompt_token_count=9,
                    candidates_token_count=3,
                    total_token_count=12,
                ),
            )

    class FakeAsyncModels:
        async def generate_content(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                parsed={"verdict": "supported"},
                text='{"verdict":"supported"}',
                usage_metadata=SimpleNamespace(
                    prompt_token_count=9,
                    candidates_token_count=3,
                    total_token_count=12,
                ),
            )

    fake_client = SimpleNamespace(
        models=FakeSyncModels(),
        aio=SimpleNamespace(models=FakeAsyncModels()),
    )
    constructors = []

    def fake_client_factory(**kwargs):
        constructors.append(kwargs)
        return fake_client

    monkeypatch.setattr(genai, "Client", fake_client_factory)
    monkeypatch.setattr(
        eval_llm,
        "get_settings",
        lambda: _settings("gemini", "gemini-2.5-flash", "judge-key"),
    )
    _install_fake_ragas_base(monkeypatch)

    llm = eval_llm.get_eval_llm(usage_recorder=usage.append)
    sync_result = llm.generate("Judge whether the answer is supported.", JudgeResponse)
    result = asyncio.run(llm.agenerate("Judge whether the answer is supported.", JudgeResponse))

    assert sync_result == JudgeResponse(verdict="supported")
    assert result == JudgeResponse(verdict="supported")
    assert constructors == [{"api_key": "judge-key"}]
    assert requests[0]["model"] == "gemini-2.5-flash"
    assert requests[0]["contents"] == "Judge whether the answer is supported."
    assert requests[0]["config"].response_mime_type == "application/json"
    assert requests[0]["config"].response_schema is JudgeResponse
    assert usage == [
        {
            "status": "available",
            "input_tokens": 9,
            "output_tokens": 3,
            "total_tokens": 12,
        }
    ] * 2


def test_huggingface_judge_validates_structured_response_and_records_usage(
    monkeypatch,
):
    import huggingface_hub

    constructors = []
    requests = []
    usage = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"verdict":"supported"}')
                    )
                ],
                usage={
                    "prompt_tokens": 11,
                    "completion_tokens": 3,
                    "total_tokens": 14,
                },
            )

    def fake_inference_client(**kwargs):
        constructors.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )

    monkeypatch.setattr(huggingface_hub, "InferenceClient", fake_inference_client)
    monkeypatch.setattr(
        eval_llm,
        "get_settings",
        lambda: _settings("hf", "meta-llama/Llama-3.1-8B-Instruct", "hf-key"),
    )
    _install_fake_ragas_base(monkeypatch)

    llm = eval_llm.get_eval_llm(usage_recorder=usage.append)
    result = asyncio.run(llm.agenerate("Judge whether the answer is supported.", JudgeResponse))

    assert result == JudgeResponse(verdict="supported")
    assert constructors == [{"provider": "auto", "api_key": "hf-key"}]
    assert requests[0]["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert requests[0]["response_format"]["type"] == "json"
    assert requests[0]["response_format"]["value"]["properties"] == {
        "verdict": {"title": "Verdict", "type": "string"}
    }
    assert usage == [
        {
            "status": "available",
            "input_tokens": 11,
            "output_tokens": 3,
            "total_tokens": 14,
        }
    ]


def test_judge_marks_provider_usage_unavailable_when_response_omits_usage(
    monkeypatch,
):
    import openai

    usage = []

    class FakeCompletions:
        async def parse(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(parsed=JudgeResponse(verdict="supported"))
                    )
                ]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        eval_llm,
        "get_settings",
        lambda: _settings("openai", "gpt-4o-mini", "judge-key"),
    )
    _install_fake_ragas_base(monkeypatch)

    llm = eval_llm.get_eval_llm(usage_recorder=usage.append)
    asyncio.run(llm.agenerate("Judge this answer.", JudgeResponse))

    assert usage == [
        {
            "status": "unavailable",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
    ]


def test_structured_output_failure_is_reported_without_provider_fallback(monkeypatch):
    import openai

    requests = []

    class FakeCompletions:
        async def parse(self, **kwargs):
            requests.append(kwargs)
            raise RuntimeError("The selected model does not support JSON schema output.")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        eval_llm,
        "get_settings",
        lambda: _settings("openrouter", "provider/unsupported-model", "judge-key"),
    )
    _install_fake_ragas_base(monkeypatch)

    llm = eval_llm.get_eval_llm()

    with pytest.raises(RuntimeError, match="does not support JSON schema"):
        asyncio.run(llm.agenerate("Judge this answer.", JudgeResponse))

    assert len(requests) == 1
    assert requests[0]["model"] == "provider/unsupported-model"
