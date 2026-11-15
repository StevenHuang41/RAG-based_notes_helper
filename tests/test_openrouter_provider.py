from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from rag_notes_helper.core.config import LLMSettings
from rag_notes_helper.rag.agentic import agentic_answer
from rag_notes_helper.rag.answer import rag_answer
from rag_notes_helper.rag.llm import openrouter_api
from rag_notes_helper.rag.llm.openrouter_api import OpenRouterLLM


def _chat_response(content, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


def _stream_event(content=None, usage=None):
    choices = (
        [SimpleNamespace(delta=SimpleNamespace(content=content))]
        if content is not None
        else []
    )
    return SimpleNamespace(choices=choices, usage=usage)


def _install_fake_openai(monkeypatch, create):
    client_options = []

    class FakeClient:
        def __init__(self, **kwargs):
            client_options.append(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )

    monkeypatch.setattr(openrouter_api, "OpenAI", FakeClient)
    return client_options


def test_openrouter_generation_uses_chat_completions_and_reports_usage(monkeypatch):
    requests = []
    usage = SimpleNamespace(
        prompt_tokens=9,
        completion_tokens=4,
        total_tokens=13,
    )
    client_options = _install_fake_openai(
        monkeypatch,
        lambda **request: requests.append(request)
        or _chat_response("grounded answer", usage),
    )
    llm = OpenRouterLLM(
        model="anthropic/claude-test",
        api_key="test-key",
        max_tokens=77,
        temperature=0.4,
    )
    prompt = [{"role": "user", "content": "What do my notes say?"}]

    assert llm.generate_raw(prompt) == "grounded answer"
    assert client_options == [{
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "timeout": 120,
    }]
    assert requests == [{
        "model": "anthropic/claude-test",
        "messages": prompt,
        "max_completion_tokens": 77,
        "temperature": 0.4,
    }]
    assert llm.last_usage == {
        "status": "available",
        "input_tokens": 9,
        "output_tokens": 4,
        "total_tokens": 13,
    }


def test_llm_settings_accept_openrouter_model_identifier():
    settings = LLMSettings(
        provider="openrouter",
        model="anthropic/claude-test",
        api_key=SecretStr("test-key"),
    )

    assert settings.provider == "openrouter"
    assert settings.model == "anthropic/claude-test"
    assert settings.api_key_str == "test-key"


def test_provider_factory_routes_openrouter_settings(monkeypatch):
    from rag_notes_helper.core import config
    from rag_notes_helper.rag import llm

    class GenerationSettings:
        provider = "openrouter"
        model = "anthropic/claude-test"
        api_key_str = "test-key"
        max_tokens = 77
        temperature = 0.4

    class Settings:
        llm = GenerationSettings()
        line_width = 91

    expected = object()
    calls = []
    monkeypatch.setattr(config, "get_settings", lambda: Settings())
    monkeypatch.setattr(
        openrouter_api,
        "OpenRouterLLM",
        lambda **kwargs: calls.append(kwargs) or expected,
    )

    assert llm.get_llm() is expected
    assert calls == [{
        "model": "anthropic/claude-test",
        "api_key": "test-key",
        "max_tokens": 77,
        "temperature": 0.4,
        "line_width": 91,
    }]


def test_openrouter_streams_chat_deltas_and_captures_usage(monkeypatch):
    requests = []
    usage = SimpleNamespace(
        prompt_tokens=6,
        completion_tokens=2,
        total_tokens=8,
    )
    events = [
        _stream_event("Agentic "),
        _stream_event("answer"),
        _stream_event(usage=usage),
    ]
    _install_fake_openai(
        monkeypatch,
        lambda **request: requests.append(request) or iter(events),
    )
    llm = OpenRouterLLM(
        model="anthropic/claude-test",
        api_key="test-key",
        max_tokens=77,
        temperature=0.4,
    )
    prompt = [{"role": "user", "content": "Answer from notes."}]

    assert "".join(llm.stream(prompt)) == "Agentic answer"
    assert requests == [{
        "model": "anthropic/claude-test",
        "messages": prompt,
        "max_completion_tokens": 77,
        "temperature": 0.4,
        "stream": True,
        "stream_options": {"include_usage": True},
    }]
    assert llm.last_output == "Agentic answer"
    assert llm.last_usage == {
        "status": "available",
        "input_tokens": 6,
        "output_tokens": 2,
        "total_tokens": 8,
    }


def test_openrouter_missing_usage_stays_unavailable(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        lambda **request: _chat_response("answer"),
    )
    llm = OpenRouterLLM(model="openai/test-model", api_key="test-key")

    assert llm.generate_raw([{"role": "user", "content": "question"}]) == "answer"
    assert llm.last_usage == {
        "status": "unavailable",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_openrouter_api_errors_are_not_suppressed(monkeypatch):
    def fail_request(**request):
        raise RuntimeError("OpenRouter request failed")

    _install_fake_openai(monkeypatch, fail_request)
    llm = OpenRouterLLM(model="openai/test-model", api_key="test-key")

    with pytest.raises(RuntimeError, match="OpenRouter request failed"):
        llm.generate_raw([{"role": "user", "content": "question"}])


def test_classic_and_agentic_answers_use_openrouter_factory(monkeypatch):
    from rag_notes_helper.core import config

    hit = {
        "source": "fixture.md",
        "chunk_id": 0,
        "text": "The project review is on Tuesday.",
        "score": 0.9,
    }
    responses = iter([
        "Classic answer",
        '{"coverage":"sufficient_evidence","missing_information":[]}',
        "Agentic answer",
        '{"action":"accept"}',
    ])
    requests = []
    client_options = _install_fake_openai(
        monkeypatch,
        lambda **request: requests.append(request)
        or _chat_response(next(responses)),
    )
    settings = SimpleNamespace(
        llm=LLMSettings(
            provider="openrouter",
            model="anthropic/claude-test",
            api_key=SecretStr("test-key"),
        ),
        line_width=80,
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr("rag_notes_helper.rag.answer.get_settings", lambda: settings)

    classic_result = rag_answer("Question", hits=[hit])
    agentic_result = agentic_answer(
        "Question",
        rag=None,
        meta_store=None,
        search_fn=lambda query: [hit],
    )

    assert classic_result["answer"] == "Classic answer"
    assert agentic_result["answer"] == "Agentic answer"
    assert len(client_options) == 3
    assert all(
        options["base_url"] == "https://openrouter.ai/api/v1"
        for options in client_options
    )
    assert [request["model"] for request in requests] == [
        "anthropic/claude-test",
        "anthropic/claude-test",
        "anthropic/claude-test",
        "anthropic/claude-test",
    ]
