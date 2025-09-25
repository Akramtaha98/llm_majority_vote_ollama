from __future__ import annotations
import requests

class OllamaClient:
    def __init__(self, model: str = "llama3.2:3b-instruct", temperature: float = 0.7, max_tokens: int = 8, host: str = "http://localhost:11434"):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.host = host.rstrip("/")

    def classify_once(self, system_prompt: str, user_prompt: str) -> str:
        prompt = (
            system_prompt + "\n\n"
            "Instruction: Respond with ONLY one label from the allowed list. No punctuation, no extra words.\n\n"
            + user_prompt
        )
        r = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature}
            },
            timeout=120,
        )
        r.raise_for_status()
        txt = (r.json() or {}).get("response", "") or ""
        first = txt.strip().splitlines()[0]
        if ":" in first:
            first = first.split(":")[-1].strip()
        return first
