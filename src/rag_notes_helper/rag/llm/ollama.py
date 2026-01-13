import requests

class OllamaLLM:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
         
        payload = {
            "model": self.model,    
            "messages": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        return (
            response.json().get("message", {}).get("content", "").strip()
            or "LLM return empty response"
        )




