from __future__ import annotations
import re
import requests

# Strips a <think>...</think> reasoning trace (e.g. DeepSeek-R1) before label parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# BUGFIX (post-submission audit): a single small max_tokens default (8) is enough for a
# short-label response from a non-reasoning model, but silently truncates a reasoning
# model's <think>...</think> trace before it ever reaches its final label, breaking
# parsing on every DeepSeek-R1 call that would otherwise use the constructor default.
# All results reported in the accompanying paper predate this fix (they passed an
# explicit, sufficiently large max_tokens at the call site); this only changes behavior
# for future runs that rely on the constructor default.
_REASONING_MODEL_SUBSTRINGS = ("deepseek", "-r1", "r1:")


def _default_max_tokens(model: str) -> int:
    """Per-model-family default generation length.

    Reasoning models (e.g. DeepSeek-R1) need room for a <think>...</think> trace before
    their final label; non-reasoning models (e.g. LLaMA-3.2) do not.
    """
    m = model.lower()
    if any(s in m for s in _REASONING_MODEL_SUBSTRINGS):
        return 1024
    return 32


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2:3b-instruct",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 4096,
        max_tokens: int | None = None,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens if max_tokens is not None else _default_max_tokens(model)
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
                    # BUGFIX (reviewer-flagged): max_tokens was accepted as a
                    # constructor parameter but never forwarded to the Ollama API,
                    # so server-side generation length was never actually bounded
                    # by it. Ollama's generation-length option is `num_predict`,
                    # not `max_tokens` -- forward it explicitly.
                    "num_predict": self.max_tokens,
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
