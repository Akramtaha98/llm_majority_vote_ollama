"""
Pooled (3 repeats), covered-sample ECE at k=5 for three confidence signals computed from
the released vote-count records: MMV vote-share (top votes / K), classical variation ratio
(1 - mode/total cast votes), and normalized vote entropy (1 - H/ln|labels|). 15 equal-width bins.
Also prints exact-vs-cumulative vote-agreement groups (coverage/accuracy) for GoEmotions.
Usage: python3 scripts/compute_confidence_signals.py
"""
import sys, json
import numpy as np
sys.path.insert(0, "scripts")
import regenerate_3rep as r3

def ece(conf, corr, nb=15):
    edges = np.linspace(0, 1, nb + 1); n = len(conf); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(corr[m].mean() - conf[m].mean())
    return e * 100

def signals(dfs):
    out = {"mmv": [], "vr": [], "ent": []}; Y = []
    for d in dfs:
        for _, r in d.iterrows():
            if r["mmv_pred"] == "ABSTAIN": continue
            v = r["votes"]; v = json.loads(v) if isinstance(v, str) else v
            c = np.array(list(v.values()), float); tot = c.sum(); L = len(c)
            p = c[c > 0] / tot; H = -(p * np.log(p)).sum()
            out["mmv"].append(c.max() / r["K"]); out["vr"].append(c.max() / tot)
            out["ent"].append(1 - H / np.log(L)); Y.append(float(r["mmv_correct"]))
    return {k: np.array(v) for k, v in out.items()}, np.array(Y)

if __name__ == "__main__":
    reps = r3.load_all_reps("runs/three_repeat")
    print("condition,n_covered,ECE_mmv,ECE_vr,ECE_entropy")
    for key in [("AG News","LLaMA-3.2:3B",5),("AG News","DeepSeek-R1:7B",5),("DBpedia","DeepSeek-R1:7B",5),("GoEmotions","DeepSeek-R1:7B",5)]:
        s, y = signals(reps[key])
        print(f"{key[0]} {key[1]},{len(y)},{ece(s['mmv'],y):.2f},{ece(s['vr'],y):.2f},{ece(s['ent'],y):.2f}")
    # GoEmotions vote-agreement groups, per repeat then mean, over n=300 items
    dfs = reps[("GoEmotions","DeepSeek-R1:7B",5)]
    print("\nGoEmotions k=5: threshold, coverage(%), accuracy(%) [mean of 3 repeats; n=300 each]")
    for label, lo, hi in [("exactly 3/5",3,3),("exactly 4/5",4,4),("exactly 5/5",5,5),(">=3/5",3,5),(">=4/5",4,5),(">=5/5",5,5)]:
        cov, acc = [], []
        for d in dfs:
            m = (d["top_votes"] >= lo) & (d["top_votes"] <= hi) & (d["mmv_pred"] != "ABSTAIN")
            cov.append(m.sum() / len(d) * 100); acc.append(d.loc[m, "mmv_correct"].astype(float).mean() * 100)
        print(f"{label},{np.mean(cov):.2f},{np.mean(acc):.2f}")
