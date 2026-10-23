"""Generate tiny synthetic runs/unified files to smoke-test analyze_unified.py (no GPU needed)."""
import json, os, sys, numpy as np, pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth_unified"; os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)
L = {"ag_news": ["World", "Sports", "Business", "Sci/Tech"], "dbpedia": [f"c{i}" for i in range(14)], "goemotions": [f"e{i}" for i in range(28)]}
for ds, labs in L.items():
    for tag, acc in (("deepseek", .8), ("llama", .6)):
        for rep in (1, 2, 3):
            rows = []
            for i in range(600):
                g = labs[rng.integers(len(labs))]; gm = ""
                if ds == "goemotions":
                    gl = [g] + ([labs[rng.integers(28)]] if rng.random() < .3 else []); gm = "|".join(gl)
                p = acc * (0.5 + rng.random()) if ds != "goemotions" else 0.3
                calls = [g if rng.random() < min(p, .97) else labs[rng.integers(len(labs))] for _ in range(5)]
                calls = [c if rng.random() > .02 else None for c in calls]
                tk = [int(rng.integers(150, 900)) if tag == "deepseek" else 3 for _ in range(5)]
                tr = [int(t > 850) for t in tk]
                row = dict(id=i, gold=g, gold_multi=gm, n_gold=len(gm.split("|")) if gm else 1, calls=json.dumps(calls), toks=json.dumps(tk),
                           trunc=json.dumps(tr), p_answer_seq=json.dumps([round(float(rng.random()), 3) for _ in range(5)]), ans_tail=json.dumps(["x"] * 5))
                if rep == 1:
                    row["direct_pred"] = g if rng.random() < acc else labs[0]; row["direct_conf"] = float(np.clip(rng.normal(.8, .15), .05, 1)); row["direct_margin"] = .1
                rows.append(row)
            pd.DataFrame(rows).to_csv(f"{out}/{ds}_{tag}_rep{rep}.csv", index=False)
print("ok", out)
