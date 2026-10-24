from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from rag_notes_helper.core.config import get_settings
from rag_notes_helper.rag.llm.base import normalize_token_usage


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
UsageRecorder = Callable[[dict[str, Any]], None]


class _OpenAICompatibleRagasLLM:
    """Ragas structured-output adapter for OpenAI-compatible chat APIs."""

    def __init__(
        self,
        *,
        model: str,
        sync_client,
        async_client,
        usage_recorder: UsageRecorder | None,
    ):
        self.model = model
        self.sync_client = sync_client
        self.async_client = async_client
        self.usage_recorder = usage_recorder

    def _record_usage(self, response) -> None:
        if self.usage_recorder is not None:
            self.usage_recorder(
                normalize_token_usage(getattr(response, "usage", None))
            )

    def _request_kwargs(self, prompt: str, response_model: type[ResponseModel]):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": response_model,
            "temperature": 0,
            "max_tokens": 1024,
        }

    def _parse_response(self, response, response_model: type[ResponseModel]):
        self._record_usage(response)
        if not getattr(response, "choices", None):
            raise ValueError("The judge provider returned no completion choices.")

        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)
        if parsed is not None:
            return response_model.model_validate(parsed)

        content = getattr(message, "content", None)
        if not content:
            raise ValueError("The judge provider returned no structured response.")
        return response_model.model_validate_json(content)

    def generate(self, prompt: str, response_model: type[ResponseModel]) -> ResponseModel:
        response = self.sync_client.chat.completions.parse(
            **self._request_kwargs(prompt, response_model)
        )
        return self._parse_response(response, response_model)

    async def agenerate(
        self,
        prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        response = await self.async_client.chat.completions.parse(
            **self._request_kwargs(prompt, response_model)
        )
        return self._parse_response(response, response_model)


def _openai_compatible_judge(
    *,
    provider: str,
    model: str,
    api_key: str | None,
    ollama_base_url: str,
    usage_recorder: UsageRecorder | None,
) -> Any:
    from openai import AsyncOpenAI, OpenAI

    base_url = {
        "openai": None,
        "openrouter": "https://openrouter.ai/api/v1",
        "ollama": f"{ollama_base_url.rstrip('/')}/v1",
    }[provider]
    client_options = {"api_key": api_key or "ollama"}
    if base_url is not None:
        client_options["base_url"] = base_url

    return _instantiate_ragas_adapter(
        _OpenAICompatibleRagasLLM,
        model=model,
        sync_client=OpenAI(**client_options),
        async_client=AsyncOpenAI(**client_options),
        usage_recorder=usage_recorder,
    )


class _GeminiRagasLLM:
    """Ragas structured-output adapter for the Google GenAI API."""

    def __init__(
        self,
        *,
        model: str,
        client,
        types_module,
        usage_recorder: UsageRecorder | None,
    ):
        self.model = model
        self.client = client
        self.types = types_module
        self.usage_recorder = usage_recorder

    def _config(self, response_model: type[ResponseModel]):
        return self.types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=response_model,
        )

    def _parse_response(self, response, response_model: type[ResponseModel]):
        usage = normalize_token_usage(getattr(response, "usage_metadata", None))
        if self.usage_recorder is not None:
            self.usage_recorder(usage)

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return response_model.model_validate(parsed)
        content = getattr(response, "text", None)
        if not content:
            raise ValueError("The Gemini judge returned no structured response.")
        return response_model.model_validate_json(content)

    def generate(self, prompt: str, response_model: type[ResponseModel]) -> ResponseModel:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._config(response_model),
        )
        return self._parse_response(response, response_model)

    async def agenerate(
        self,
        prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._config(response_model),
        )
        return self._parse_response(response, response_model)


def _gemini_judge(
    *, model: str, api_key: str | None, usage_recorder: UsageRecorder | None
) -> Any:
    from google import genai
    from google.genai import types

    return _instantiate_ragas_adapter(
        _GeminiRagasLLM,
        model=model,
        client=genai.Client(api_key=api_key),
        types_module=types,
        usage_recorder=usage_recorder,
    )


class _HuggingFaceRagasLLM:
    """Ragas structured-output adapter for Hugging Face chat completions."""

    def __init__(
        self,
        *,
        model: str,
        client,
        usage_recorder: UsageRecorder | None,
    ):
        self.model = model
        self.client = client
        self.usage_recorder = usage_recorder

    def _parse_response(self, response, response_model: type[ResponseModel]):
        usage = normalize_token_usage(getattr(response, "usage", None))
        if self.usage_recorder is not None:
            self.usage_recorder(usage)

        if not getattr(response, "choices", None):
            raise ValueError("The Hugging Face judge returned no completion choices.")
        content = getattr(response.choices[0].message, "content", None)
        if not content:
            raise ValueError("The Hugging Face judge returned no structured response.")
        return response_model.model_validate_json(content)

    def generate(self, prompt: str, response_model: type[ResponseModel]) -> ResponseModel:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json",
                "value": response_model.model_json_schema(),
            },
            temperature=0,
            max_tokens=1024,
        )
        return self._parse_response(response, response_model)

    async def agenerate(
        self,
        prompt: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        return await asyncio.to_thread(self.generate, prompt, response_model)


def _huggingface_judge(
    *, model: str, api_key: str | None, usage_recorder: UsageRecorder | None
) -> Any:
    from huggingface_hub import InferenceClient

    return _instantiate_ragas_adapter(
        _HuggingFaceRagasLLM,
        model=model,
        client=InferenceClient(provider="auto", api_key=api_key),
        usage_recorder=usage_recorder,
    )


def _instantiate_ragas_adapter(adapter_type, **kwargs):
    from ragas.llms import InstructorBaseRagasLLM

    ragas_adapter = type(
        adapter_type.__name__,
        (adapter_type, InstructorBaseRagasLLM),
        {"__module__": adapter_type.__module__},
    )
    return ragas_adapter(**kwargs)


def get_eval_llm(*, usage_recorder: UsageRecorder | None = None) -> Any:
    settings = get_settings()
    provider = settings.llm.effective_eval_provider
    model = settings.llm.effective_eval_model
    api_key = settings.llm.effective_eval_api_key_str

    if provider in {"openai", "openrouter", "ollama"}:
        return _openai_compatible_judge(
            provider=provider,
            model=model,
            api_key=api_key,
            ollama_base_url=settings.ollama_base_url,
            usage_recorder=usage_recorder,
        )

    if provider == "gemini":
        return _gemini_judge(
            model=model,
            api_key=api_key,
            usage_recorder=usage_recorder,
        )

    if provider == "hf":
        return _huggingface_judge(
            model=model,
            api_key=api_key,
            usage_recorder=usage_recorder,
        )

    raise ValueError(f"Unsupported evaluation LLM provider: {provider}")
