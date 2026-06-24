\
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re

def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip()).lower()

def majority_vote_single(
    client,
    system_prompt: str,
    user_prompt: str,
    labels: List[str],
    K: int = 3,
    abstain_threshold: float = 0.0,
    early_stop: bool = True,
) -> Tuple[Optional[str], Dict[str, int]]:
    """
    Run K stochastic calls to the LLM and return (prediction_or_None, vote_counts).
    Early stopping triggers as soon as any label has strict majority (> K/2).
    """
    norm_map = {normalize_label(l): l for l in labels}
    vote_counts: Dict[str, int] = {l: 0 for l in labels}

    for i in range(1, K + 1):
        raw = client.classify_once(system_prompt, user_prompt)
        norm = normalize_label(raw)
        if norm in norm_map:
            vote_counts[norm_map[norm]] += 1
        else:
            for l in labels:
                if normalize_label(l) == norm:
                    vote_counts[l] += 1
                    break

        if early_stop and max(vote_counts.values()) > K / 2.0:
            break

    if sum(vote_counts.values()) == 0:
        return None, vote_counts

    mode = max(vote_counts.values())
    winners = [l for l, c in vote_counts.items() if c == mode]
    final_label = None
    for l in labels:
        if l in winners:
            final_label = l
            break

    top_share = mode / K if K > 0 else 0.0
    if abstain_threshold > 0.0 and top_share < abstain_threshold:
        return None, vote_counts

    return final_label, vote_counts
