"""Analysis for the unified revision runs (reads runs/unified/*.csv, writes tables to --out).

Outputs (CSV):
  T_main.csv        per dataset/model/k: coverage, covered acc, all-items acc (abstain=error), ECE, AUROC, AURC, mean +- sd over repeats
  T_baselines.csv   k=1 logprob (direct label prob; post-reasoning answer prob) vs verbal-free SC/MMV at k=3,5 (ECE/AUROC/AURC + bootstrap CI)
  T_cost.csv        mean generated tokens per item per k (token-based cost), accuracy per 1k tokens
  T_trunc.csv       DeepSeek truncation rate, truncated-vs-abstained association (Fisher), per dataset
  T_tests.csv       paired McNemar (abstention-as-error and covered-both), Holm-corrected within family, per repeat
  T_conformal.csv   split-calibrated selective classification with a risk guarantee (R1-C8): coverage/risk vs target alpha
  T_goemo.csv       GoEmotions: any-gold vs primary-gold vs single-annotation subset
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, beta, binomtest
from scipy.stats import rankdata

KS = (1, 3, 5)


def ece(conf, y, bins=15):
    conf, y = np.asarray(conf, float), np.asarray(y, float)
    if len(conf) == 0: return np.nan
    e = np.linspace(0, 1, bins + 1); t = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.any(): t += m.mean() * abs(y[m].mean() - conf[m].mean())
    return 100 * t


def auroc_err(conf, y):
    err = 1 - np.asarray(y, int)
    if err.sum() in (0, len(err)): return np.nan
    r = rankdata(1 - np.asarray(conf, float)); n1 = err.sum(); n0 = len(err) - n1
    return (r[err == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def aurc(conf, y):
    conf, y = np.asarray(conf, float), np.asarray(y, float)
    if len(conf) == 0: return np.nan
    area, pc, pr = 0.0, 0.0, None
    for v in np.unique(conf)[::-1]:
        m = conf >= v; c = m.mean(); r = 1 - y[m].mean()
        area += c * r if pr is None else (c - pc) * (r + pr) / 2
        pc, pr = c, r
    return 100 * area


def load(runs):
    out = {}
    for f in sorted(glob.glob(f"{runs}/*_rep*.csv")):
        m = re.match(r".*/(ag_news|dbpedia|goemotions)_(deepseek|llama|large)_rep(\d+)\.csv", f)
        if not m: continue
        d = pd.read_csv(f, keep_default_na=False)
        for c in ("calls", "toks", "trunc", "p_answer_seq"): d[c] = d[c].apply(json.loads)
        out[(m.group(1), m.group(2), int(m.group(3)))] = d
    return out


def gold_set(r):
    return set(r.gold_multi.split("|")) if r.gold_multi else {r.gold}


def vote_table(d, k, primary=False):
    """per-item: top label, share (top/k), covered(strict majority of k), correct, valid_any."""
    rows = []
    for r in d.itertuples():
        calls = [c for c in r.calls[:k] if c]
        gs = {r.gold} if primary else gold_set(r)
        if not calls:
            rows.append((None, 0.0, False, 0, False)); continue
        vc = pd.Series(calls).value_counts(); top = vc.max()
        lab = sorted([l for l, c in vc.items() if c == top])[0]  # deterministic tie-break (SC)
        rows.append((lab, top / k, top > k / 2, int(lab in gs), True))
    return pd.DataFrame(rows, columns=["lab", "share", "cov", "ok", "valid"])


def metrics_k(d, k, primary=False):
    v = vote_table(d, k, primary); n = len(v)
    c = v[v["cov"]]
    sc = v[v.valid]
    return dict(n=n, cov=100 * len(c) / n, acc_cov=100 * c.ok.mean() if len(c) else np.nan,
                acc_all=100 * c.ok.sum() / n,                      # abstain/parse-fail = error
                sc_acc_full=100 * sc.ok.sum() / n,
                mmv_ece=ece(c.share, c.ok), mmv_auroc=auroc_err(c.share, c.ok), mmv_aurc=aurc(c.share, c.ok),
                sc_ece=ece(sc.share, sc.ok), sc_auroc=auroc_err(sc.share, sc.ok), sc_aurc=aurc(sc.share, sc.ok))


def boot_ci(fn, items, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(items); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n); vals.append(fn(idx))
    return np.nanpercentile(vals, [2.5, 97.5])


def ucb_cp(err, n, delta):  # one-sided Clopper-Pearson upper bound
    return 1.0 if n == 0 else beta.ppf(1 - delta, err + 1, n - err) if err < n else 1.0


def ltt_threshold(conf, err, alpha, delta=0.1):
    """Fixed-sequence (Learn-then-Test) selective threshold: scan thresholds from strictest to loosest,
    stop at first whose selective-risk UCB exceeds alpha. Returns chosen threshold (or inf = abstain on all)."""
    chosen = np.inf
    for t in sorted(np.unique(conf), reverse=True):
        m = conf >= t
        if ucb_cp(int(err[m].sum()), int(m.sum()), delta) <= alpha: chosen = t
        else: break
    return chosen


def conformal_sets(votes_cal, y_cal_lab, votes_te, y_te_lab, labels, alpha):
    """Split conformal (LAC). score = 1 - share(true label). Returns (coverage, mean set size)."""
    s = np.array([1 - v.get(y, 0) for v, y in zip(votes_cal, y_cal_lab)])
    n = len(s); q = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    cov, size = [], []
    for v, y in zip(votes_te, y_te_lab):
        S = [l for l in labels if 1 - v.get(l, 0) <= q]
        cov.append(y in S); size.append(len(S))
    return np.mean(cov), np.mean(size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/unified"); ap.add_argument("--out", default="runs/unified_tables")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = load(a.runs); assert R, f"no runs in {a.runs}"
    combos = sorted({(ds, m) for ds, m, _ in R})

    # ---- main table
    rows = []
    for ds, m in combos:
        reps = [r for (d_, m_, r) in R if d_ == ds and m_ == m]
        for k in KS:
            ms = pd.DataFrame([metrics_k(R[(ds, m, r)], k) for r in reps])
            row = dict(dataset=ds, model=m, k=k, n=int(ms.n.iloc[0]), reps=len(reps))
            for c in ms.columns[1:]:
                row[c] = round(ms[c].mean(), 2); row[c + "_sd"] = round(ms[c].std(ddof=1), 2) if len(reps) > 1 else np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(f"{a.out}/T_main.csv", index=False)

    # ---- baselines: logprob (single call) vs vote share (k=3,5), rep 1 items, pooled bootstrap CI
    rows = []
    for ds, m in combos:
        d = R[(ds, m, 1)] if (ds, m, 1) in R else None
        if d is None or "direct_conf" not in d: continue
        gs = [gold_set(r) for r in d.itertuples()]
        n = len(d)
        sigs = {}
        ok_dir = np.array([int(p in g) for p, g in zip(d.direct_pred, gs)])
        sigs["logprob_label_prob (k=1, no reasoning)"] = (d.direct_conf.values, ok_dir, np.ones(n, bool))
        # answer-sequence probability of first sampled call (post-reasoning for DeepSeek)
        pf = np.array([x[0] if x[0] is not None else np.nan for x in d.p_answer_seq]); c1 = [x[0] for x in d.calls]
        ok1 = np.array([int(c in g) if c else 0 for c, g in zip(c1, gs)]); val = ~np.isnan(pf) & np.array([bool(c) for c in c1])
        sigs["logprob_answer_seq (k=1, sampled call)"] = (np.nan_to_num(pf), ok1, val)
        vf = f"{a.runs}/{ds}_{m}_verbal.csv"
        if os.path.exists(vf):
            vb = pd.read_csv(vf, keep_default_na=False)
            ok_v = np.array([int(p in g) if p else 0 for p, g in zip(vb.pred, gs)])
            conf_v = pd.to_numeric(vb.verbal_conf, errors="coerce").fillna(-1).values
            sigs["verbalized (k=1, T=0)"] = (conf_v, ok_v, conf_v >= 0)
        for k in (3, 5):
            v = vote_table(d, k); val = v.valid.values
            sigs[f"SC vote share (k={k})"] = (v.share.values, v.ok.values, val)
            sigs[f"MMV (k={k}, covered)"] = (v.share.values, v.ok.values, v["cov"].values)
        for name, (c, y, msk) in sigs.items():
            idx_all = np.where(msk)[0]
            f = lambda idx, fn: fn(c[idx_all[idx]], y[idx_all[idx]])
            row = dict(dataset=ds, model=m, signal=name, n_scored=len(idx_all), acc=round(100 * y[idx_all].mean(), 2),
                       ece=round(ece(c[idx_all], y[idx_all]), 2), auroc=round(auroc_err(c[idx_all], y[idx_all]), 3),
                       aurc=round(aurc(c[idx_all], y[idx_all]), 2))
            lo, hi = boot_ci(lambda i: f(i, auroc_err), idx_all, a.boot)
            row["auroc_ci"] = f"[{lo:.3f},{hi:.3f}]"
            lo, hi = boot_ci(lambda i: f(i, ece), idx_all, a.boot)
            row["ece_ci"] = f"[{lo:.2f},{hi:.2f}]"
            rows.append(row)
    pd.DataFrame(rows).to_csv(f"{a.out}/T_baselines.csv", index=False)

    # ---- token cost + truncation
    rows, trows = [], []
    for ds, m in combos:
        reps = [r for (d_, m_, r) in R if d_ == ds and m_ == m]
        for k in KS:
            tok = np.mean([np.mean([sum(t[:k]) for t in R[(ds, m, r)].toks]) for r in reps])
            mm = pd.DataFrame([metrics_k(R[(ds, m, r)], k) for r in reps]).mean()
            rows.append(dict(dataset=ds, model=m, k=k, tokens_per_item=round(tok, 1), rel_token_cost=None,
                             correct_per_1k_tokens=round(mm.acc_all * 10 / tok, 3)))
        base = [x for x in rows if x["dataset"] == ds and x["model"] == m and x["k"] == 1][0]["tokens_per_item"]
        for x in rows:
            if x["dataset"] == ds and x["model"] == m: x["rel_token_cost"] = round(x["tokens_per_item"] / base, 2)
        if m == "deepseek":
            for r in reps:
                d = R[(ds, m, r)]; k = 5
                tr = np.array([sum(t[:k]) > 0 for t in d.trunc]); frac_calls = np.mean([np.mean(t[:k]) for t in d.trunc])
                v = vote_table(d, k); abst = ~v["cov"].values
                tab = [[(tr & abst).sum(), (tr & ~abst).sum()], [(~tr & abst).sum(), (~tr & ~abst).sum()]]
                _, p = fisher_exact(tab)
                trows.append(dict(dataset=ds, rep=r, frac_calls_truncated=round(100 * frac_calls, 2),
                                  items_any_trunc_pct=round(100 * tr.mean(), 2),
                                  abstain_rate_if_trunc=round(100 * abst[tr].mean(), 1) if tr.any() else np.nan,
                                  abstain_rate_if_not=round(100 * abst[~tr].mean(), 1), fisher_p=round(p, 4),
                                  mean_tokens=round(np.mean([np.mean(t) for t in d.toks]), 1),
                                  p95_tokens=int(np.percentile([x for t in d.toks for x in t], 95))))
    pd.DataFrame(rows).to_csv(f"{a.out}/T_cost.csv", index=False)
    pd.DataFrame(trows).to_csv(f"{a.out}/T_trunc.csv", index=False)

    # ---- paired McNemar, Holm within (repeat) family
    from scipy.stats import binomtest as bt
    trows = []
    for r in sorted({r for *_, r in R}):
        fam = []
        for ds, m in combos:
            if (ds, m, r) not in R: continue
            d = R[(ds, m, r)]; V = {k: vote_table(d, k) for k in KS}
            for k1, k2 in ((1, 3), (1, 5), (3, 5)):
                for mode in ("abstain_as_error", "covered_both"):
                    if mode == "abstain_as_error":
                        x = (V[k1]["cov"] & (V[k1].ok == 1)).values; y = (V[k2]["cov"] & (V[k2].ok == 1)).values
                    else:
                        both = (V[k1]["cov"] & V[k2]["cov"]).values; x = (V[k1].ok.values == 1)[both]; y = (V[k2].ok.values == 1)[both]
                    b = int((x & ~y).sum()); c = int((~x & y).sum())
                    p = bt(b, b + c, 0.5).pvalue if b + c else 1.0
                    fam.append(dict(rep=r, dataset=ds, model=m, cmp=f"k={k1} vs k={k2}", mode=mode, b=b, c=c, p=p))
        f = pd.DataFrame(fam)
        for mode, g in f.groupby("mode"):  # Holm
            o = g.sort_values("p"); mm = len(o); adj, prev = [], 0
            for i, p in enumerate(o.p): prev = max(prev, min(1, (mm - i) * p)); adj.append(prev)
            f.loc[o.index, "p_holm"] = adj
        trows.append(f)
    pd.concat(trows).round(5).to_csv(f"{a.out}/T_tests.csv", index=False)

    # ---- conformal / risk-controlled selective classification (200 random 50/50 splits, rep 1)
    rows = []
    rng = np.random.default_rng(0)
    for ds, m in combos:
        if (ds, m, 1) not in R: continue
        d = R[(ds, m, 1)]
        labels = sorted(set(d.gold) | {c for cs in d.calls for c in cs if c})
        gs = [gold_set(r) for r in d.itertuples()]
        for k in (3, 5):
            v = vote_table(d, k); share = v.share.values; err = (1 - v.ok.values) + (~v.valid.values).astype(int) * 0
            err = np.where(v.valid.values, 1 - v.ok.values, 1)
            vc = [dict(pd.Series([c for c in cs[:k] if c]).value_counts() / k) if any(cs[:k]) else {} for cs in d.calls]
            ylab = [r.gold for r in d.itertuples()]
            for alpha in (0.05, 0.10, 0.20):
                cv, rk, cc, ss = [], [], [], []
                for _ in range(200):
                    p = rng.permutation(len(d)); cal, te = p[: len(p) // 2], p[len(p) // 2:]
                    t = ltt_threshold(share[cal], err[cal], alpha)
                    sel = share[te] >= t
                    cv.append(sel.mean()); rk.append(err[te][sel].mean() if sel.any() else 0.0)
                    if ds != "goemotions":
                        c_, s_ = conformal_sets([vc[i] for i in cal], [ylab[i] for i in cal], [vc[i] for i in te], [ylab[i] for i in te], labels, alpha)
                        cc.append(c_); ss.append(s_)
                strict_cov = float((v["cov"]).mean()); strict_risk = float(err[v["cov"].values].mean()) if v["cov"].any() else np.nan
                rows.append(dict(dataset=ds, model=m, k=k, alpha=alpha, ltt_cov=round(100 * np.mean(cv), 1),
                                 ltt_test_risk=round(100 * np.mean(rk), 2), ltt_viol_rate=round(100 * np.mean(np.array(rk) > alpha), 1),
                                 mmv_cov=round(100 * strict_cov, 1), mmv_risk=round(100 * strict_risk, 2),
                                 conformal_set_cov=round(100 * np.mean(cc), 1) if cc else np.nan,
                                 conformal_set_size=round(np.mean(ss), 2) if ss else np.nan))
    pd.DataFrame(rows).to_csv(f"{a.out}/T_conformal.csv", index=False)

    # ---- GoEmotions evaluation-protocol sensitivity
    rows = []
    for m in sorted({m_ for (d_, m_, _) in R if d_ == "goemotions"}):
        reps = [r for (d_, m_, r) in R if d_ == "goemotions" and m_ == m]
        for k in KS:
            for name, sel, prim in (("any-gold (paper)", None, False), ("primary-gold", None, True), ("single-annotation items", 1, False)):
                vals = []
                for r in reps:
                    d = R[("goemotions", m, r)]
                    if sel: d = d[d.n_gold == 1].reset_index(drop=True)
                    vals.append(metrics_k(d, k, prim))
                mm = pd.DataFrame(vals).mean()
                rows.append(dict(model=m, k=k, protocol=name, n=int(mm.n), cov=round(mm["cov"], 1), acc_cov=round(mm.acc_cov, 2),
                                 acc_all=round(mm.acc_all, 2), mmv_ece=round(mm.mmv_ece, 2), sc_auroc=round(mm.sc_auroc, 3)))
    if rows: pd.DataFrame(rows).to_csv(f"{a.out}/T_goemo.csv", index=False)
    # ---- temperature sweep on the SAME first-2000 items (T=0.7 taken from the main run, rep 1)
    rows = []
    for ds, m in combos:
        for T, d in [(float(re.search(r"_T([\d.]+)$", os.path.dirname(f)).group(1)), None) for f in glob.glob(f"{os.path.dirname(a.runs.rstrip('/'))}/unified_T*/{ds}_{m}_rep1.csv")] + [(0.7, None)]:
            f = f"{a.runs}/{ds}_{m}_rep1.csv" if T == 0.7 else f"{os.path.dirname(a.runs.rstrip('/'))}/unified_T{T}/{ds}_{m}_rep1.csv"
            if not os.path.exists(f): continue
            d = pd.read_csv(f, keep_default_na=False)
            for c in ("calls", "toks", "trunc", "p_answer_seq"): d[c] = d[c].apply(json.loads)
            d = d.iloc[:2000].reset_index(drop=True); mt = metrics_k(d, 5)
            ent = np.mean([ -sum(p*np.log(p) for p in (np.array(list(pd.Series([c for c in cs if c]).value_counts()))/max(1,sum(1 for c in cs if c))) if p>0) for cs in d.calls])
            rows.append(dict(dataset=ds, model=m, T=T, n=len(d), cov=round(mt["cov"], 1), acc_cov=round(mt.acc_cov if False else mt["acc_cov"], 1),
                             acc_all=round(mt["acc_all"], 1), mmv_ece=round(mt["mmv_ece"], 1), sc_auroc=round(mt["sc_auroc"], 3), vote_entropy_nats=round(ent, 3)))
    if rows: pd.DataFrame(rows).sort_values(["dataset", "model", "T"]).to_csv(f"{a.out}/T_temperature.csv", index=False)
    print("tables written to", a.out)


if __name__ == "__main__":
    main()
