"""
Post-fix regeneration of Figure 6 (AG News confusion matrix, LLaMA 3.2:3B, k=5),
pooling MMV-covered predictions across the three independent post-fix repeats,
consistent with Figures 4 and 5's pooling methodology and Table 4's CV-HB script.
The original Figure 6 was left unchanged from the original submission (LLaMA was
not affected by the DeepSeek-specific generation-length bug), but every other
numbered figure in this paper has since been regenerated from post-fix pooled
data, so this closes that last remaining gap for full consistency.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regenerate_3rep as r3

AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

all_reps = r3.load_all_reps("runs/reviewer_r1_reruns")
dfs = all_reps[("AG News", "LLaMA-3.2:3B", 5)]
covered = pd.concat([d[d["mmv_pred"] != "ABSTAIN"] for d in dfs], ignore_index=True)
n = len(covered)
print("pooled covered n =", n)

cm = np.zeros((4, 4), dtype=int)
label_idx = {l: i for i, l in enumerate(AG_NEWS_LABELS)}
for _, row in covered.iterrows():
    gold = row["gold"]
    pred = row["mmv_pred"]
    if gold in label_idx and pred in label_idx:
        cm[label_idx[gold], label_idx[pred]] += 1
print(cm, cm.sum())

fig, ax = plt.subplots(figsize=(1169/300, 1078/300))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(AG_NEWS_LABELS, rotation=45, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(AG_NEWS_LABELS)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"AG News Confusion Matrix\nLLaMA 3.2:3B, k=5 (n={n} covered, pooled across 3 post-fix repeats)",
             fontsize=9, fontweight="bold")
vmax = cm.max()
for i in range(4):
    for j in range(4):
        val = cm[i, j]
        color = "white" if val > vmax * 0.5 else "black"
        ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("out_figs/figure6_confusion_matrix.png", dpi=300)
print("saved")
