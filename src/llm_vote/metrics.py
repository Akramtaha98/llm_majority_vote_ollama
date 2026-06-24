from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

def accuracy(y_true: List[str], y_pred: List[Optional[str]]) -> float:
    total = 0
    correct = 0
    for t, p in zip(y_true, y_pred):
        if p is None:
            continue
        total += 1
        if p == t:
            correct += 1
    return (correct / total) if total else 0.0

def macro_f1(y_true: List[str], y_pred: List[Optional[str]], labels: List[str]) -> float:
    filt_true, filt_pred = [], []
    for t, p in zip(y_true, y_pred):
        if p is None:
            continue
        filt_true.append(t)
        filt_pred.append(p)
    if not filt_true:
        return 0.0
    return f1_score(filt_true, filt_pred, labels=labels, average="macro", zero_division=0)

def mcc(y_true: List[str], y_pred: List[Optional[str]], label_to_idx: Dict[str, int]) -> float:
    filt_true, filt_pred = [], []
    for t, p in zip(y_true, y_pred):
        if p is None:
            continue
        filt_true.append(label_to_idx.get(t, -1))
        filt_pred.append(label_to_idx.get(p, -1))
    if not filt_true or len(set(filt_true)) < 2:
        return 0.0
    return matthews_corrcoef(filt_true, filt_pred)

def expected_calibration_error(confidences, corrects, n_bins: int = 15) -> float:
    """ECE with 15-bin fixed-width binning (matches paper specification)."""
    confidences = np.asarray(confidences)
    corrects = np.asarray(corrects)
    if len(confidences) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b_lo, b_hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= b_lo) & (confidences <= b_hi)
        if mask.sum() == 0:
            continue
        acc = corrects[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)
