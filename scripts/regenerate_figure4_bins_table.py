"""
Regenerate the per-bin sample-count table accompanying Figure 4's reliability
diagrams (Table 13). Reuses make_figures_3rep.py's pooled_reliability_bins()
logic exactly (same 15 equal-width confidence bins, same pooling across the
three post-fix repeats) so the reported n values match the points plotted in
Figure 4 one-for-one; this script only adds columns for the confidence-bin
edges and mean accuracy, and prints the result as a CSV instead of a plot.
"""
import sys, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
import regenerate_all as ra
import regenerate_3rep as r3

DATA_DIR = "runs/three_repeat"
OUT_CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "figure4_bins_table.csv")


def pooled_reliability_bins_detailed(dfs, n_bins=15):
    import pandas as pd
    covered = pd.concat([df[df["mmv_pred"] != "ABSTAIN"] for df in dfs], ignore_index=True)
    if covered.empty:
        return []
    conf = covered["confidence_mmv"].to_numpy()
    corr = covered["mmv_correct"].astype(float).to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi) if lo == 0.0 else (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_label = f"[{lo:.2f}, {hi:.2f}]" if lo == 0.0 else f"({lo:.2f}, {hi:.2f}]"
        out.append((bin_label, int(mask.sum()), round(float(conf[mask].mean()), 3),
                     round(float(corr[mask].mean() * 100.0), 1)))
    return out


def main():
    all_reps = r3.load_all_reps(DATA_DIR)
    rel_panels = [
        ("AG News", "LLaMA-3.2:3B"),
        ("AG News", "DeepSeek-R1:7B"),
        ("DBpedia", "DeepSeek-R1:7B"),
        ("GoEmotions", "DeepSeek-R1:7B"),
    ]
    rows = []
    for ds, model in rel_panels:
        for k in (1, 3, 5):
            dfs = all_reps[(ds, model, k)]
            for bin_label, n, mean_conf, mean_acc in pooled_reliability_bins_detailed(dfs):
                rows.append([ds, model, k, bin_label, n, mean_conf, mean_acc])

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Model", "k", "Confidence bin", "n", "Mean confidence", "Mean accuracy (%)"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
