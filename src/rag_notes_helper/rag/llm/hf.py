from huggingface_hub import InferenceClient

from rag_notes_helper.core.config import settings


class HuggingFaceLLM:
    def __init__(self) -> None:

        if not settings.HUGGINGFACE_API_KEY:
            raise RuntimeError("HUGGINGFACE_API_KEY not set in .env")

        self.client = InferenceClient(
            provider="auto",
            api_key = settings.HUGGINGFACE_API_KEY
        )

        self.model = settings.HF_MODEL

    def generate(self, prompt: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )


        # huggingface return string
        return response.choices[0].message.content \
                or "LLM returns emtpy response"

