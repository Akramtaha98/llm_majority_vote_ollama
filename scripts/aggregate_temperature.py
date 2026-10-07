"""Summarize runs/temperature_extended/*.csv -> runs/temperature_extended/summary.csv
Metrics per file (k=5): coverage, covered accuracy, all-items accuracy, MMV ECE (15 bins),
mean normalized vote entropy, AURC of the self-consistency vote share (all valid items)."""
import glob, json, math, re, sys
import numpy as np, pandas as pd

def ece(conf, corr, bins=15):
    conf, corr = np.asarray(conf, float), np.asarray(corr, float); e = np.linspace(0, 1, bins + 1); t = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.any(): t += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return t

def aurc(conf, corr):
    conf, corr = np.asarray(conf, float), np.asarray(corr, float); area = 0.0; pc, pr = 0.0, None
    for v in np.unique(conf)[::-1]:
        m = conf >= v; c = m.mean(); r = 1 - corr[m].mean()
        area += c * r if pr is None else (c - pc) * (r + pr) / 2; pc, pr = c, r
    return area

def ent(v):
    try: d = json.loads(v)
    except Exception: return float("nan")
    tot = sum(d.values())
    if tot == 0 or len(d) < 2: return float("nan")
    return -sum(c / tot * math.log(c / tot) for c in d.values() if c) / math.log(len(d))

rows = []
for f in sorted(glob.glob("runs/temperature_extended/*_temp*.csv")):
    m = re.match(r".*/(ag_news|dbpedia|goemotions)_(deepseek|llama)_k(\d+)_temp([\d.]+)\.csv", f)
    if not m: continue
    ds, tag, k, T = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
    d = pd.read_csv(f); n = len(d)
    cov = d[d["pred"] != "ABSTAIN"]
    sc_pred, sc_conf, sc_ok = [], [], []
    for _, r in d.iterrows():
        v = json.loads(r["votes"]); tot = sum(v.values())
        if tot == 0: continue
        top = max(v.values()); lab = next(l for l, c in v.items() if c == top)
        gold = set(str(r["gold_multi"]).split("|")) if isinstance(r["gold_multi"], str) and r["gold_multi"] else {r["gold"]}
        sc_conf.append(top / r["K"]); sc_ok.append(int(lab in gold))
    rows.append({"dataset": ds, "model": tag, "k": k, "T": T, "n": n,
                 "coverage_%": round(100 * len(cov) / n, 2),
                 "covered_acc_%": round(100 * cov["correct"].mean(), 2) if len(cov) else None,
                 "all_items_acc_%": round(100 * d["correct"].sum() / n, 2),
                 "MMV_ECE_%": round(100 * ece(cov["confidence"], cov["correct"]), 2) if len(cov) else None,
                 "mean_vote_entropy": round(float(np.nanmean(d["votes"].apply(ent))), 4),
                 "SC_AURC_%": round(100 * aurc(sc_conf, sc_ok), 2)})
s = pd.DataFrame(rows).sort_values(["dataset", "model", "T"])
print(s.to_string(index=False)); s.to_csv("runs/temperature_extended/summary.csv", index=False)
