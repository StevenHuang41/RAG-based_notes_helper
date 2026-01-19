from collections.abc import Iterator
from typing import Any

from huggingface_hub import InferenceClient

from rag_notes_helper.rag.llm.base import BaseLLM


class HuggingFaceLLM(BaseLLM):
    def __init__(
        self,
        api_key: str | None,
        **kws,
    ):
        super().__init__(**kws)
        self.client = InferenceClient(
            provider="auto", # pick the best hardware avaliable on HF
            api_key = api_key,
            timeout=120,
        )

    def _generate(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> str:

        response = self.client.chat.completions.create(
            model = self.model,
            messages=prompt,
            max_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
        )

        content = response.choices[0].message.content
        return content or ""


    def _stream(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> Iterator[str]:
        
        response = self.client.chat.completions.create(
            model = self.model,
            messages=prompt,
            max_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
            stream=True,
        )

        for chunk in response:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue

            for char in chunk.choices[0].delta.content:
                yield char
