from huggingface_hub import InferenceClient

class HuggingFaceLLM:
    def __init__(self, model: str, api_key: str) -> None:

        self.model = model
        self.client = InferenceClient(
            provider="auto",
            api_key = api_key,
        )

    def generate(
        self,
        prompt: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # huggingface return string
        return (
            response.choices[0].message.content
            or "LLM returns emtpy response"
        )

