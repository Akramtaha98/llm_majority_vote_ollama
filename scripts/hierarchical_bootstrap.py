"""
Two-level (items x repeats) bootstrap for the k=5 minus k=1 accuracy difference.
Each resample (1) draws items with replacement, then (2) for every drawn item draws one of the
three generation repeats independently for each k, so both item-sampling and generation
variability enter the interval. Reports covered-only accuracy and abstention-as-error accuracy.
10,000 resamples, seed 42. Usage: python3 scripts/hierarchical_bootstrap.py
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
import regenerate_3rep as r3

def arr(dfs):
    ids = sorted(set.intersection(*[set(d["id"]) for d in dfs]))
    cov = np.zeros((len(dfs), len(ids))); cor = np.zeros_like(cov)
    for r, d in enumerate(dfs):
        d = d.set_index("id").loc[ids]
        cov[r] = (d["mmv_pred"] != "ABSTAIN").to_numpy(float)
        cor[r] = (d["mmv_correct"].astype(float).to_numpy()) * cov[r]
    return cov, cor

def acc(cov, cor, ii, rr, covered_only):
    c = cov[rr, ii].sum(); k = cor[rr, ii].sum()
    return (k / c if covered_only else k / len(ii)) * 100 if c > 0 else np.nan

if __name__ == "__main__":
    reps = r3.load_all_reps("runs/three_repeat"); rng = np.random.default_rng(42); B = 10000
    print("dataset,model,mode,point_diff_pp,ci_low,ci_high")
    for ds, model in [("GoEmotions","DeepSeek-R1:7B"),("AG News","DeepSeek-R1:7B"),("DBpedia","DeepSeek-R1:7B"),("AG News","LLaMA-3.2:3B")]:
        c1, y1 = arr(reps[(ds, model, 1)]); c5, y5 = arr(reps[(ds, model, 5)])
        ids1 = sorted(set.intersection(*[set(d["id"]) for d in reps[(ds,model,1)]]) & set.intersection(*[set(d["id"]) for d in reps[(ds,model,5)]]))
        # align on common ids
        def sub(dfs):
            ids = ids1; cov = np.zeros((3, len(ids))); cor = np.zeros_like(cov)
            for r, d in enumerate(dfs):
                d = d.set_index("id").loc[ids]
                cov[r] = (d["mmv_pred"] != "ABSTAIN").to_numpy(float); cor[r] = d["mmv_correct"].astype(float).to_numpy() * cov[r]
            return cov, cor
        c1, y1 = sub(reps[(ds,model,1)]); c5, y5 = sub(reps[(ds,model,5)]); n = len(ids1)
        for mode in (True, False):
            pt = np.mean([acc(c5,y5,np.arange(n),np.full(n,r),mode) - acc(c1,y1,np.arange(n),np.full(n,r),mode) for r in range(3)])
            ds_ = []
            for _ in range(B):
                ii = rng.integers(0, n, n); r1 = rng.integers(0, 3, n); r5 = rng.integers(0, 3, n)
                ds_.append(acc(c5,y5,ii,r5,mode) - acc(c1,y1,ii,r1,mode))
            lo, hi = np.nanpercentile(ds_, [2.5, 97.5])
            print(f"{ds},{model},{'covered-only' if mode else 'abstention-as-error'},{pt:.2f},{lo:.2f},{hi:.2f}")
