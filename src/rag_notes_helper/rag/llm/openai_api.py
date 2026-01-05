from openai import OpenAI

class OpenAILLM:
    def __init__(self, model: str, api_key: str) -> None:

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
        )


    def generate(
        self,
        prompt: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        response = self.client.chat.completions.create(
            model = self.model,
            message=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return (
            response.choices[0].message.content
            or "LLM return empty response"
        )



