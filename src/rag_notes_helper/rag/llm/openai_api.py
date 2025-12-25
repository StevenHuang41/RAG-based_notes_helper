from openai import OpenAI

from rag_notes_helper.core.config import settings

class OpenAILLM:
    def __init__(self) -> None:

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set in .env")

        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
        )
        
        self.model = settings.OPENAI_MODEL

    def generate(self, prompt: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model = self.model,
            message=prompt,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

        return response.choices[0].message.content \
                or "LLM return empty response"

    

