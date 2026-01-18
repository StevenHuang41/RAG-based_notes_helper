from collections.abc import Iterable
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
        prompt: list[dict[str, Any]],
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
        prompt: list[dict[str, Any]],
        **kws,
    ) -> Iterable[str]:

        line_width = kws.get("line_width", self.line_width)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(prompt),
            max_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
            stream=True,
        )                            

        current_len = 0
        has_content = False
        for chunk in response:
            if chunk.choices and (line := chunk.choices[0].delta.content):
                has_content = True
                words = line.split(" ")
                for i, word in enumerate(words):
                    display_word = word if i == 0 else " " + word

                    if current_len + len(display_word) > line_width:
                        yield "\n"

                        display_word = word.lstrip()
                        current_len = len(display_word)
                    else :
                        current_len += len(display_word)

                    yield display_word

        if not has_content:
            yield "LLM return empty response"

