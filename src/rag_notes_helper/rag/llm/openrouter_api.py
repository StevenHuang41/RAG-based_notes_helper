from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from rag_notes_helper.rag.llm.base import BaseLLM


class OpenRouterLLM(BaseLLM):
    provider_name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(self, **kws):
        super().__init__(**kws)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120,
        )

    def _generate(self, prompt: list[dict[str, Any]], **kws) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,  # type: ignore[arg-type]
            max_completion_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
        )

        self._capture_usage(getattr(response, "usage", None))
        choices = getattr(response, "choices", [])
        if not choices:
            return ""
        content = getattr(choices[0].message, "content", None)
        return content if isinstance(content, str) else ""

    def _stream(self, prompt: list[dict[str, Any]], **kws) -> Iterator[str]:
        self.last_usage = self._unavailable_usage()
        self.last_output = ""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,  # type: ignore[arg-type]
            max_completion_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
            stream=True,
            stream_options={"include_usage": True},
        )

        for event in response:
            usage = getattr(event, "usage", None)
            if usage is not None:
                self._capture_usage(usage)

            for choice in getattr(event, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None)
                if isinstance(content, str):
                    self.last_output += content
                    yield content
