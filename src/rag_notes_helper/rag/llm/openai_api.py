from collections.abc import Iterable, Iterator, Mapping
from typing import Any, cast

from openai import OpenAI

from rag_notes_helper.rag.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key: str | None,
        **kws,
    ):
        super().__init__(**kws)
        self.client = OpenAI(
            api_key=api_key,
            timeout=120
        )


    def _generate(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> str:

        response = self.client.responses.create(
            model=self.model,
            input=prompt, # type: ignore
            max_output_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
        )

        content = response.output_text
        return content or ""


    def _stream(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> Iterator[str]:

        response = self.client.responses.create(
            model=self.model,
            input=prompt, # type: ignore
            max_output_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
            stream=True,
        )

        for event in response:
            if event.type == "response.output_text.delt":
                yield from event.delta

            elif event.type == "response.error":
                yield f"\n[API Error]: {event.error.message}"
