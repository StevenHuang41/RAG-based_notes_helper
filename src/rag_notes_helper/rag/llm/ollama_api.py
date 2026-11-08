import requests
import json
from typing import Any
from collections.abc import Iterator

from rag_notes_helper.rag.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    provider_name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", **kws) -> None:
        super().__init__(**kws)
        self.base_url = base_url.rstrip("/")


    def _get_payload(self, prompt, *, stream=False, think=None, format=None):
        payload = {
            "model": self.model,
            "messages": prompt,
            "stream": stream,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            }
        }
        if think is not None:
            payload["think"] = think
        if format is not None:
            payload["format"] = format
        return payload


    def _generate(self, prompt: list[dict[str, Any]], **kws) -> str:
        requested_format = kws.get("format")
        payload = self._get_payload(
            prompt,
            think=kws.get("think"),
            format=requested_format,
        )

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if not isinstance(requested_format, dict) or status_code not in {
                400,
                422,
            }:
                raise

            payload = self._get_payload(
                prompt,
                think=kws.get("think"),
                format="json",
            )
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

        payload = response.json()
        self._capture_usage(payload)
        content = payload.get("message", {}).get("content", "")
        return content or ""


    def _stream(self, prompt: list[dict[str, Any]], **kws) -> Iterator[str]:
        payload = self._get_payload(prompt, stream=True)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            chunk = json.loads(line.decode("utf-8"))
            text = chunk.get("message", {}).get("content", "")

            for char in text:
                yield char
