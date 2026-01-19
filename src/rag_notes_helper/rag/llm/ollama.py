import requests
import json
import textwrap
from typing import Any
from collections.abc import Iterator, Iterable


class OllamaLLM:
    def __init__(
        self,
        model: str,
        base_url: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        line_width: int = 80,
    ) -> None:

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.line_width = line_width


    def _get_payload(self, prompt, stream, **kws):
        payload = {
            "model": self.model,
            "messages": prompt,
            "stream": stream,
            "options": {
                "num_predict": kws.get("max_tokens", self.max_tokens),
                "temperature": kws.get("temperature", self.temperature),
            }
        }
        return payload


    def generate(
        self,
        prompt: Iterable[dict[str, Any]],
        **kws,
    ) -> str:

        line_width = kws.get("line_width", self.line_width)
        payload = self._get_payload(prompt, stream=False, **kws)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        response_text = (
            response.json().get("message", {}).get("content", "").strip()
            or "LLM return empty response"
        )

        return textwrap.fill(response_text, width=line_width)


    def stream(
        self,
        prompt: Iterable[dict[str, Any]],
        **kws,
    ) -> Iterator[str]:

        line_width = kws.get("line_width", self.line_width)
        payload = self._get_payload(prompt, stream=True, **kws)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        current_len = 0
        buffer = ""
        has_content = False
        for line in response.iter_lines():
            if not line:
                continue

            chunk = json.loads(line.decode("utf-8"))
            content = chunk.get("message", {}).get("content", "")

            if content:
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

