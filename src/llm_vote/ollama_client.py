from __future__ import annotations
import re
import requests


def _clean_response(txt: str) -> str:
    """
    Strip DeepSeek-R1 chain-of-thought <think>...</think> blocks and return
    the first clean label line.

    DeepSeek-R1 (and its distilled variants) prefix every response with a
    reasoning block inside <think>...</think>.  Taking splitlines()[0] naively
    picks up the opening <think> tag, which never matches any label and causes
    100% ABSTAIN.  This function removes those blocks first.
    """
    # Remove <think>...</think> blocks (multi-line, greedy-off)
    cleaned = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

    # If nothing remained (model only output a think block), try raw text
    if not cleaned:
        cleaned = txt.strip()

    # Return first non-empty, non-tag line
    for line in cleaned.splitlines():
        line = line.strip()
        if line and not line.startswith("<"):
            # Strip "Answer: World" style prefixes
            if ":" in line:
                line = line.split(":")[-1].strip()
            return line

    return cleaned.split("\n")[0].strip()


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2:3b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 50,        # raised: DeepSeek needs tokens to close </think>
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.host = host.rstrip("/")

    def classify_once(self, system_prompt: str, user_prompt: str) -> str:
        prompt = (
            system_prompt + "\n\n"
            "Instruction: Respond with ONLY one label from the allowed list. "
            "No punctuation, no extra words.\n\n"
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
                    "num_predict": self.max_tokens,
                },
            },
            timeout=300,   # generous timeout for reasoning models
        )
        r.raise_for_status()
        txt = (r.json() or {}).get("response", "") or ""
        return _clean_response(txt)
