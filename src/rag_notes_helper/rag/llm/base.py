from typing import Any
from typing import Iterator
from collections.abc import Mapping
import textwrap
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    provider_name = "unknown"

    def __init__(
        self,
        model: str,
        api_key: str | None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        line_width: int = 80,
    ):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.line_width = line_width
        self.last_usage = self._unavailable_usage()
        self.last_output = ""


    @staticmethod
    def _unavailable_usage() -> dict[str, Any]:
        return {
            "status": "unavailable",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }


    @staticmethod
    def _value(source: Any, *names: str) -> int | None:
        for name in names:
            if isinstance(source, Mapping):
                value = source.get(name)
            else:
                value = getattr(source, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None


    def _capture_usage(
        self,
        usage: Any,
        *,
        input_names: tuple[str, ...] = (
            "input_tokens",
            "prompt_tokens",
            "prompt_token_count",
            "prompt_eval_count",
        ),
        output_names: tuple[str, ...] = (
            "output_tokens",
            "completion_tokens",
            "candidates_token_count",
            "eval_count",
        ),
        total_names: tuple[str, ...] = (
            "total_tokens",
            "total_token_count",
        ),
    ) -> None:
        if usage is None:
            self.last_usage = self._unavailable_usage()
            return

        input_tokens = self._value(usage, *input_names)
        output_tokens = self._value(usage, *output_names)
        total_tokens = self._value(usage, *total_names)
        status = (
            "available"
            if input_tokens is not None
            and output_tokens is not None
            and total_tokens is not None
            else "partial"
            if any(value is not None for value in (
                input_tokens,
                output_tokens,
                total_tokens,
            ))
            else "unavailable"
        )
        self.last_usage = {
            "status": status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }


    @abstractmethod
    def _generate(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> str:
        raise NotImplementedError


    def generate_raw(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> str:
        """Generate text without terminal-oriented line wrapping."""
        self.last_usage = self._unavailable_usage()
        self.last_output = ""
        if not (content := self._generate(prompt, **kws)):
            return "LLM return empty response"

        self.last_output = content
        return content


    def generate(
        self,
        prompt: list[dict[str, Any]],
        **kws,
    ) -> str:
        line_width = kws.get("line_width", self.line_width)

        self.last_usage = self._unavailable_usage()
        self.last_output = ""
        if not (content := self._generate(prompt, **kws)):
            return "LLM return empty response"

        self.last_output = content
        paragraphs = content.splitlines()
        wrap_paragraphs = [textwrap.fill(p, width=line_width) for p in paragraphs]

        return "\n".join(wrap_paragraphs)


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


def make_generation_usage_record(
    llm: Any,
    prompt: list[dict[str, Any]],
    output: str,
    *,
    call_type: str,
) -> dict[str, Any]:
    """Build a safe per-call record without estimating tokens from characters."""
    input_text = "".join(str(message.get("content", "")) for message in prompt)
    usage = getattr(llm, "last_usage", None) or BaseLLM._unavailable_usage()
    output_text = getattr(llm, "last_output", output)
    return {
        "call_type": call_type,
        "provider": getattr(llm, "provider_name", "unknown"),
        "model": getattr(llm, "model", None),
        "token_usage": dict(usage),
        "characters": {
            "input": len(input_text),
            "output": len(output_text),
        },
    }

