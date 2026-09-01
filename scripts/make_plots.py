"""
make_plots.py — Generate the manuscript's per-run diagnostic figures from a
single MMV predictions CSV (as written by scripts/eval_dataset.py).

BUGFIX (reviewer-flagged): the previous version plotted a single bar of
`df["correct"].mean() * 100`, i.e. accuracy over *every* row including
ABSTAIN rows treated as incorrect, with no coverage, no confidence/reliability
information, and no relationship to K. That does not reproduce any figure in
the manuscript and does not use the paper's own definition of "conditional
accuracy" (accuracy computed over covered/non-abstained predictions only,
Section 4.3). This version instead:
  1. Uses the paper's conditional-accuracy definition (accuracy on
     `pred != "ABSTAIN"` rows only), matching get_metrics() in
     scripts/compute_results.py / scripts/regenerate_all.py.
  2. Plots a reliability diagram (mean accuracy vs. mean confidence per bin,
     15 bins, matching the ECE binning in src/llm_vote/metrics.py) --
     this is what the manuscript's calibration figures actually show.
  3. Reports coverage and conditional accuracy as text annotations, so the
     figure is self-describing rather than a single ambiguous bar.

Usage:
    python scripts/make_plots.py --preds runs/ag_news_ollama_deepseek-r1-7b_k5.csv --out fig.png
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_BINS = 15


def conditional_accuracy(df: pd.DataFrame) -> tuple[float, float, int, int]:
    """Returns (conditional_accuracy_pct, coverage_pct, n_covered, n_total),
    matching the paper's definition (Section 4.3): accuracy is computed only
    over non-abstained ("covered") predictions, not over all rows."""
    total = len(df)
    covered = df[df["pred"] != "ABSTAIN"]
    n_cov = len(covered)
    if n_cov == 0:
        return 0.0, 0.0, 0, total
    acc = covered["correct"].mean() * 100.0
    coverage = (n_cov / total) * 100.0 if total else 0.0
    return acc, coverage, n_cov, total


def reliability_bins(df: pd.DataFrame, n_bins: int = N_BINS):
    """Per-bin (mean confidence, mean accuracy, count), using the same
    (lo, hi]-except-first-bin boundary convention as
    src/llm_vote/metrics.py::expected_calibration_error, so the reliability
    diagram is consistent with the manuscript's reported ECE."""
    covered = df[df["pred"] != "ABSTAIN"].copy()
    if covered.empty:
        return [], [], []
    conf = covered["confidence"].to_numpy()
    corr = covered["correct"].astype(float).to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf, mean_acc, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi) if lo == 0.0 else (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        mean_conf.append(conf[mask].mean())
        mean_acc.append(corr[mask].mean() * 100.0)
        counts.append(int(mask.sum()))
    return mean_conf, mean_acc, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="Path to a run CSV from scripts/eval_dataset.py")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.preds)
    acc, coverage, n_cov, total = conditional_accuracy(df)
    mean_conf, mean_acc, counts = reliability_bins(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    # Left panel: reliability diagram (accuracy vs. confidence, perfect calibration = diagonal)
    ax1.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Perfect calibration")
    if mean_conf:
        ax1.scatter([c * 100 for c in mean_conf], mean_acc, s=[max(20, c) for c in counts])
    ax1.set_xlabel("Mean confidence (%)")
    ax1.set_ylabel("Mean accuracy (%)")
    ax1.set_title("Reliability diagram")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=7)

    # Right panel: conditional accuracy + coverage summary, matching Table 3's definitions
    ax2.bar(["Conditional\naccuracy"], [acc])
    ax2.bar(["Coverage"], [coverage])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("%")
    ax2.set_title(f"n={total}, covered={n_cov}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
    print(f"Conditional accuracy (covered only): {acc:.2f}%   Coverage: {coverage:.2f}%   "
          f"(n={total}, covered={n_cov})")


if __name__ == "__main__":
    main()
