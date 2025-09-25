from __future__ import annotations
from typing import List

EN_SYSTEM = (
    "You are a careful text classifier. "
    "When given a list of allowed labels and a text, you MUST return exactly one of those labels. "
    "Respond with ONLY the label, no punctuation, no extra words."
)
AR_SYSTEM = EN_SYSTEM

def build_prompt(text: str, labels: List[str], task_name: str) -> str:
    labels_line = ", ".join(labels)
    return (
        f"Task: Single-label classification for {task_name}.\n"
        f"Choose ONE label from: [{labels_line}]\n"
        f"Text:\n{text}\n"
        f"Answer with exactly one label from the list above."
    )
