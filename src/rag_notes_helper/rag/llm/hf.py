from collections.abc import Iterable, Iterator
import textwrap
from typing import Any

from huggingface_hub import InferenceClient

class HuggingFaceLLM:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        line_width: int = 80,
    ) -> None:
        self.model = model
        self.client = InferenceClient(
            provider="auto", # pick the best hardware avaliable on HF
            api_key = api_key,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.line_width = line_width

    def generate(
        self,
        prompt: Iterable[dict[str, Any]],
        **kws,
    ) -> str:

        line_width = kws.get("line_width", self.line_width)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(prompt),
            max_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
            stream=False,
        )

        response_text = (
            response.choices[0].message.content
            or "LLM returns emtpy response"
        )
        return textwrap.fill(response_text, width=line_width)

    def stream(
        self,
        prompt: Iterable[dict[str, Any]],
        **kws,
    ) -> Iterator[str]:

        line_width = kws.get("line_width", self.line_width)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(prompt),
            max_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
            stream=True,
        )

        current_len = 0
        buffer = ""
        has_content = False
        for chunk in response:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue

            content = chunk.choices[0].delta.content
            has_content = True

            for char in content:
                if char == "\n":
                    yield buffer + "\n"
                    buffer = ""
                    current_len = 0

                elif char == " ":
                    # buffer is a complete word
                    if current_len + len(buffer) + 1 > line_width:
                        # last word of line exceed
                        yield "\n" + buffer.lstrip() + " "
                        current_len = len(buffer.lstrip()) + 1
                    else :
                        # last word of line fits
                        yield buffer + " "
                        current_len += len(buffer) + 1

                    buffer = ""

                else :
                    # buffer is not complete for a word
                    buffer += char

        if buffer:
            if current_len + len(buffer) > line_width:
                yield "\n" + buffer.lstrip()
            else :
                yield buffer

        if not has_content:
            yield "LLM return empty response"

