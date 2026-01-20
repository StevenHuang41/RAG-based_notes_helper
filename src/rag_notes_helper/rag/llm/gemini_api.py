from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import types

from rag_notes_helper.rag.llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, **kws):
        model = kws.pop("model", "gemini-2.0-flash")
        super().__init__(model=model, **kws)
        self.client = genai.Client(api_key=self.api_key)


    def _convert_prompt(self, prompt: list[dict[str, Any]]):
        system_instruction = None
        contents = []

        for p in prompt:
            if p["role"] == "system":
                system_instruction = p["content"]
            else :
                contents.append(
                    types.Content(
                        role=p["role"],
                        parts=[types.Part.from_text(text=p["content"])],
                    )
                )

        return system_instruction, contents


    def _generate(self, prompt: list[dict[str, Any]], **kws) -> str:
        system_instruction, contents = self._convert_prompt(prompt)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return response.text or ""


    def _stream(self, prompt: list[dict[str, Any]], **kws) -> Iterator[str]:
        system_instruction, contents = self._convert_prompt(prompt)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=kws.get("max_tokens", self.max_tokens),
            temperature=kws.get("temperature", self.temperature),
        )

        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )

        for chunk in response:
            if chunk.text:
                for char in chunk.text:
                    yield char

            elif hasattr(chunk, "candidates") and chunk.candidates:
                reason = chunk.candidates[0].finish_reason
                if reason == "SAFETY":
                    yield "\n[Blocked by Gemini Safety Filters]"
