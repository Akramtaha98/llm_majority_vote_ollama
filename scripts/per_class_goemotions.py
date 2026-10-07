"""
GoEmotions DeepSeek-R1:7B, k=5: per-class precision/recall/F1 on covered predictions (first-listed gold,
fixed 28-label set, per repeat then mean over the 3 repeats), per-class coverage (answered / all items of
that first-listed gold class), and macro-F1 / MCC under two gold conventions: first-listed gold (as in
Table 1) and any-gold-consistent (a covered prediction found anywhere in the gold set is scored as its
own label; otherwise scored against the first-listed gold label).
Usage: python3 scripts/per_class_goemotions.py [out.csv]
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import regenerate_3rep as r3, regenerate_all as ra
from sklearn.metrics import precision_recall_fscore_support, f1_score, matthews_corrcoef

LABELS = ra.GOEMO_LABELS
reps = r3.load_all_reps("runs/three_repeat")[("GoEmotions", "DeepSeek-R1:7B", 5)]
first = lambda s: str(s).split("|")[0]
P, R, F, S, C, T = [], [], [], [], [], []
mf_first, mf_any, mcc_first, mcc_any = [], [], [], []
for d in reps:
    cov = d[d["mmv_pred"] != "ABSTAIN"]
    yt = cov["gold"].map(first); yp = cov["mmv_pred"]
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=LABELS, zero_division=0)
    P.append(p); R.append(r); F.append(f); S.append(s)
    allc = d["gold"].map(first).value_counts().reindex(LABELS).fillna(0).to_numpy()
    covc = yt.value_counts().reindex(LABELS).fillna(0).to_numpy()
    C.append(covc); T.append(allc)
    yt_any = [pm if pm in set(str(gm).split("|")) else first(g) for pm, gm, g in zip(cov["mmv_pred"], cov["gold_multi"], cov["gold"])]
    mf_first.append(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0))
    mf_any.append(f1_score(yt_any, yp, labels=LABELS, average="macro", zero_division=0))
    li = {l: i for i, l in enumerate(LABELS)}
    mcc_first.append(matthews_corrcoef([li[x] for x in yt], [li[x] for x in yp]))
    mcc_any.append(matthews_corrcoef([li[x] for x in yt_any], [li[x] for x in yp]))
df = pd.DataFrame({"class": LABELS, "precision": np.mean(P, 0) * 100, "recall": np.mean(R, 0) * 100,
                   "f1": np.mean(F, 0) * 100, "mean_support": np.mean(S, 0), "total_support": np.sum(S, 0),
                   "items_in_class_total": np.sum(T, 0), "covered_total": np.sum(C, 0)})
df["coverage_pct"] = np.where(df["items_in_class_total"] > 0, df["covered_total"] / df["items_in_class_total"].replace(0, np.nan) * 100, np.nan)
df = df.sort_values("f1", ascending=False)
print(df.round(1).to_string(index=False))
print(f"\nmacro-F1 mean-of-class-F1 = {df['f1'].mean():.2f}")
print(f"macro-F1 first-gold = {np.mean(mf_first)*100:.2f}, any-gold-consistent = {np.mean(mf_any)*100:.2f}")
print(f"MCC first-gold = {np.mean(mcc_first):.3f}, any-gold-consistent = {np.mean(mcc_any):.3f}")
if len(sys.argv) > 1: df.to_csv(sys.argv[1], index=False)
