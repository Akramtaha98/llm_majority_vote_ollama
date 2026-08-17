from __future__ import annotations
import re
import requests

# Strips a <think>...</think> reasoning trace (e.g. DeepSeek-R1) before label parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2:3b-instruct",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 4096,
        max_tokens: int = 8,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.num_ctx = num_ctx
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
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "repeat_penalty": self.repeat_penalty,
                    "num_ctx": self.num_ctx,
                },
            },
            timeout=120,
        )
        r.raise_for_status()
        txt = (r.json() or {}).get("response", "") or ""
        # Strip any <think>...</think> reasoning trace (present for reasoning models such
        # as DeepSeek-R1 when the Ollama version/endpoint does not already separate it out)
        # before taking the first non-empty line as the label.
        txt = _THINK_RE.sub("", txt).strip()
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        first = lines[0] if lines else ""
        if ":" in first:
            first = first.split(":")[-1].strip()
        return first
