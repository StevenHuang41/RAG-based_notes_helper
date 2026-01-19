from typing import Any
from typing import Iterator
import textwrap
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(
        self,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        line_width: int = 80,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.line_width = line_width


    @abstractmethod
    def _generate(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> str:
        raise NotImplementedError


    def generate(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> str:
        line_width = kws.get("line_width", self.line_width)

        content = self._generate(prompt, **kws)

        response_text = content or "LLM return empty response"

        return textwrap.fill(response_text, width=line_width)


    @abstractmethod
    def _stream(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> Iterator[str]:
        raise NotImplementedError


    def stream(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> Iterator[str]:

        line_width = kws.get("line_width", self.line_width)

        current_len = 0
        buffer = ""
        has_content = False

        for char in self._stream(prompt, **kws):
            has_content = True
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

