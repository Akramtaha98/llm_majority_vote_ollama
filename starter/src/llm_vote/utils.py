from __future__ import annotations
from typing import Dict, Tuple

def votes_to_confidence(votes: Dict[str, int], K: int) -> Tuple[float, str, int]:
    if not votes:
        return 0.0, "", 0
    best_label = max(votes, key=lambda k: votes[k])
    top = votes[best_label]
    return (top / K), best_label, top
