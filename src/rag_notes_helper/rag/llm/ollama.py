import requests

class OllamaLLM:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _flatten_prompt(self, prompt: list[dict[str, str]]) -> str:
        lines = []
        for d in prompt:
            role = d["role"].upper()
            content = d["content"]
            lines.append(f"{role}:\n{content}")

        return "\n\n".join(lines)

    def generate(
        self,
        prompt: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        text_prompt = self._flatten_prompt(prompt)

        payload = {
            "model": self.model,
            "prompt": text_prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        return (
            response.json()["response"].strip()
            or "LLM return empty response"
        )




