from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re

def normalize_label(label: str) -> str:
    """Whitespace-normalize only. Matching against the fixed label set is
    case-sensitive, per the manuscript's Materials and Methods (Section 4.3):
    a raw model output must match a label's exact casing to be counted as a
    valid vote for that label."""
    return re.sub(r"\s+", " ", label.strip())

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

    MMV's abstention rule is unconditional and uses the ORIGINAL, fixed K as its
    denominator (not the number of calls that happened to parse into a valid
    label): a prediction is only returned when one label's vote count is a
    strict majority of K (mode > K/2). `abstain_threshold`, if set above 0.5,
    can additionally require an even higher vote share than a bare majority;
    it can never loosen the K/2 rule below a strict majority.
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

    # Strict-majority abstention (Equation 1): mode must exceed K/2, using the
    # original fixed K, not sum(vote_counts.values()). This is unconditional --
    # it is MMV's core definition, not an opt-in behind abstain_threshold.
    if mode <= K / 2.0:
        return None, vote_counts

    top_share = mode / K if K > 0 else 0.0
    if abstain_threshold > 0.5 and top_share < abstain_threshold:
        return None, vote_counts

    return final_label, vote_counts
