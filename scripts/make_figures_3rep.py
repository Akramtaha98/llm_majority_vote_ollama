"""
Regenerate Figures 2, 3, 4, 5 from the 3-repeat post-fix data, reusing
regenerate_3rep.py's already-verified per-sample processing (which itself
reuses regenerate_all.py's process_condition) so no computation is
duplicated or reimplemented differently from Table 1/2/5/7/8/9.

Figure 2 (Accuracy vs k) and Figure 3 (ECE vs k) plot the post-fix Table 1
Mean(3) values directly. Figure 4 (reliability diagrams) and Figure 5 (vote
agreement) pool the three reps' covered samples per condition/k (increases N,
smooths the bins/histogram) rather than averaging three separate diagrams.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import regenerate_all as ra
import regenerate_3rep as r3

DATA_DIR = "runs/reviewer_r1_reruns"
OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_reps = r3.load_all_reps(DATA_DIR)
t1 = r3.build_table1_3rep(all_reps)

TEAL = "#1f6f78"
ORANGE = "#e08214"

# ---------------- Figure 2: Accuracy vs k ----------------
fig, axes = plt.subplots(1, 3, figsize=(13.293, 4.193))
panels = [("AG News", True), ("DBpedia", False), ("GoEmotions", False)]
for ax, (ds, has_llama) in zip(axes, panels):
    if has_llama:
        sub = t1[(t1["Dataset"] == ds) & (t1["Model"] == "LLaMA-3.2:3B")].sort_values("k")
        ax.plot(sub["k"], sub["Acc_mean"], marker="o", color=TEAL, linewidth=2, markersize=9, label="LLaMA 3.2:3B")
    sub2 = t1[(t1["Dataset"] == ds) & (t1["Model"] == "DeepSeek-R1:7B")].sort_values("k")
    if len(sub2):
        ax.plot(sub2["k"], sub2["Acc_mean"], marker="s", color=ORANGE, linewidth=2, markersize=9, label="DeepSeek-R1:7B")
    ax.set_title(ds, fontsize=13, fontweight="bold")
    ax.set_xlabel("k (ensemble size)")
    ax.set_xticks([1, 3, 5])
    if ax is axes[0]:
        ax.set_ylabel("Accuracy (%)")
axes[0].legend(loc="lower right", fontsize=9)
plt.suptitle("")
plt.tight_layout()
plt.savefig(OUT_DIR / "figure2_accuracy_vs_k.png", dpi=300)
plt.close(fig)
print("Saved figure2_accuracy_vs_k.png")

# ---------------- Figure 3: ECE vs k ----------------
fig, axes = plt.subplots(1, 3, figsize=(9.933, 2.727))
for ax, (ds, has_llama) in zip(axes, panels):
    if has_llama:
        sub = t1[(t1["Dataset"] == ds) & (t1["Model"] == "LLaMA-3.2:3B")].sort_values("k")
        ax.plot(sub["k"], sub["ECE_mean"], marker="o", color=TEAL, linewidth=2, markersize=9, label="LLaMA 3.2:3B")
    sub2 = t1[(t1["Dataset"] == ds) & (t1["Model"] == "DeepSeek-R1:7B")].sort_values("k")
    if len(sub2):
        ax.plot(sub2["k"], sub2["ECE_mean"], marker="s", color=ORANGE, linewidth=2, markersize=9, label="DeepSeek-R1:7B")
    ax.set_title(ds, fontsize=13, fontweight="bold")
    ax.set_xlabel("k (ensemble size)")
    ax.set_xticks([1, 3, 5])
    if ax is axes[0]:
        ax.set_ylabel("ECE (%)")
    ax.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "figure3_ece_vs_k.png", dpi=300)
plt.close(fig)
print("Saved figure3_ece_vs_k.png")

# ---------------- Figure 4: reliability diagrams (pooled 3 reps) ----------------
def pooled_reliability_bins(dfs, n_bins=15):
    covered = pd.concat([d[d["mmv_pred"] != "ABSTAIN"] for d in dfs], ignore_index=True)
    if covered.empty:
        return [], [], []
    conf = covered["confidence_mmv"].to_numpy()
    corr = covered["mmv_correct"].astype(float).to_numpy()
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

rel_panels = [
    ("AG News", "LLaMA-3.2:3B", "AG News\nLLaMA 3.2:3B"),
    ("AG News", "DeepSeek-R1:7B", "AG News\nDeepSeek-R1:7B"),
    ("DBpedia", "DeepSeek-R1:7B", "DBpedia\nDeepSeek-R1:7B"),
    ("GoEmotions", "DeepSeek-R1:7B", "GoEmotions\nDeepSeek-R1:7B"),
]
kcolors = {1: "#d62728", 3: "#ff7f0e", 5: "#2ca02c"}
fig, axes = plt.subplots(2, 2, figsize=(6.833, 5.16))
axes = axes.flatten()
for ax, (ds, model, title) in zip(axes, rel_panels):
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for k in (1, 3, 5):
        dfs = all_reps[(ds, model, k)]
        mean_conf, mean_acc, counts = pooled_reliability_bins(dfs)
        if not mean_conf:
            continue
        acc_frac = [a / 100.0 for a in mean_acc]
        ax.plot(mean_conf, acc_frac, marker="o", color=kcolors[k], linewidth=1.8, markersize=7, label=f"k={k}")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "figure4_reliability.png", dpi=300)
plt.close(fig)
print("Saved figure4_reliability.png")

# ---------------- Figure 5: vote agreement, AG News DeepSeek k=3,5 (pooled 3 reps) ----------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
fig.suptitle("Vote Agreement Distribution: AG News (DeepSeek-R1:7B, pooled across 3 post-fix repeats)",
             fontsize=12, fontweight="bold")
for ax, k in zip(axes, (3, 5)):
    dfs = all_reps[("AG News", "DeepSeek-R1:7B", k)]
    top_votes = pd.concat([d["top_votes"] for d in dfs], ignore_index=True) if "top_votes" in dfs[0].columns else None
    if top_votes is None:
        # top_votes not carried through process_condition's output frame; recompute from votes directly
        raw = [pd.read_csv(Path(DATA_DIR) / f"ag_news_deepseek_matched_k{k}_rep{rep}.csv") for rep in (1, 2, 3)]
        top_votes = pd.concat([r["top_votes"] for r in raw], ignore_index=True)
    counts = top_votes.value_counts().sort_index()
    labels = [f"{v}/{k}" for v in range(1, k + 1)]
    heights = [int(counts.get(v, 0)) for v in range(1, k + 1)]
    ax.bar(labels, heights, color=ORANGE)
    ax.set_title(f"k = {k}  (n = {sum(heights)})", fontsize=11)
    ax.set_xlabel("Vote Count")
    ax.set_ylabel("Frequency")
plt.subplots_adjust(wspace=0.35, top=0.80, bottom=0.15)
plt.savefig(OUT_DIR / "figure5_vote_agreement.png", dpi=300)
plt.close(fig)
print("Saved figure5_vote_agreement.png")
