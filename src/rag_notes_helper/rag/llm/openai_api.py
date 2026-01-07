from collections.abc import Iterable
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

class OpenAILLM:
    def __init__(self, model: str, api_key: str) -> None:

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
        )


    def generate(
        self,
        prompt: Iterable[ChatCompletionMessageParam],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        response = self.client.chat.completions.create(
            model = self.model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return (
            response.choices[0].message.content
            or "LLM return empty response"
        )



