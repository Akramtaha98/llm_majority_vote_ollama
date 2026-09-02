"""
Single auditable script that regenerates Tables 1, 2, 5, and 7 directly from the
raw per-sample vote-count CSVs as released in vote_records/reviewer_data_package/,
applying:
  - MMV: strict majority of the ORIGINAL fixed k -> pred, else ABSTAIN
  - SC (self-consistency): plurality vote, ties broken by first-in-fixed-class-order,
    predicts whenever at least one valid vote exists
  - GoEmotions Macro-F1/MCC: computed against the FIXED full 28-class label set at
    every k (not just labels present in the covered subset)
  - ECE: 15-bin, first bin inclusive on both ends, subsequent bins (lo, hi] to avoid
    double-counting confidence values that land exactly on a bin boundary
  - Accuracy 95% CIs: Wilson score interval (Section 4.3)
  - ECE 95% CIs: nonparametric bootstrap, 10,000 resamples (Section 4.3, Table 1 caption)
  - Table 2's Full-Parse Split / Capped: Insufficient / Split: Sufficient / Covered w/
    Partial Parse decomposition, derived from each sample's n_parsed = sum(votes.values())
  - Table 7's AURC and Figure 7's risk-coverage curves: computed only over the
    achievable SC confidence thresholds actually present in the valid-parsed
    population for each condition (no arbitrary tie-ordering -- every sample at
    the same confidence value is one indivisible step of the curve, never split
    into separate points by row order). The highest-confidence bucket's risk is
    held flat down to coverage 0, and AURC is the trapezoidal area under the
    resulting curve over the full [0, 1] coverage domain. At k=1 every sample
    has identical confidence, so there is exactly one achievable point and AURC
    reduces to that point's risk, which equals ECE by construction -- this is
    checked as an assertion, not just documented. See Section 7.4 and the
    Table 7 / Figure 7 captions in the manuscript for the full methodology
    description and the post-submission audit that found and fixed the
    previous, tie-order-dependent version of this computation.
No numbers are hand-typed anywhere downstream of this script; Tables 1, 2, 5, and 7
and Figure 7 are derived entirely from the per-sample vote-count records.

Usage: python3 regenerate_all.py [--data-dir DIR] [--out-dir DIR]
Default --data-dir is vote_records/reviewer_data_package/per_sample_vote_count_records/
relative to the repository root. Figure 7 is written to <out-dir>/figure7_risk_coverage.png.
"""
import argparse
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, matthews_corrcoef

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42

AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
GOEMO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise",
]

# (dataset, model, k) -> (raw filename, fixed label list or None for DBpedia's 14 classes)
CONDITIONS = {
    ("AG News", "DeepSeek-R1:7B", 1): ("ag_news_ollama_deepseek-r1-7b_k1.csv", AG_NEWS_LABELS),
    ("AG News", "DeepSeek-R1:7B", 3): ("ag_news_ollama_deepseek-r1-7b_k3.csv", AG_NEWS_LABELS),
    ("AG News", "DeepSeek-R1:7B", 5): ("ag_news_ollama_deepseek-r1-7b_k5.csv", AG_NEWS_LABELS),
    ("AG News", "LLaMA-3.2:3B", 1): ("ag_news_ollama_llama3.2_k1.csv", AG_NEWS_LABELS),
    ("AG News", "LLaMA-3.2:3B", 3): ("ag_news_ollama_llama3.2_k3.csv", AG_NEWS_LABELS),
    ("AG News", "LLaMA-3.2:3B", 5): ("ag_news_ollama_llama3.2_k5.csv", AG_NEWS_LABELS),
    ("DBpedia", "DeepSeek-R1:7B", 1): ("dbpedia_ollama_deepseek-r1-7b_k1.csv", None),
    ("DBpedia", "DeepSeek-R1:7B", 3): ("dbpedia_ollama_deepseek-r1-7b_k3.csv", None),
    ("DBpedia", "DeepSeek-R1:7B", 5): ("dbpedia_ollama_deepseek-r1-7b_k5.csv", None),
    ("GoEmotions", "DeepSeek-R1:7B", 1): ("goemotions_ollama_deepseek-r1-7b_k1.csv", GOEMO_LABELS),
    ("GoEmotions", "DeepSeek-R1:7B", 3): ("goemotions_ollama_deepseek-r1-7b_k3.csv", GOEMO_LABELS),
    ("GoEmotions", "DeepSeek-R1:7B", 5): ("goemotions_ollama_deepseek-r1-7b_k5.csv", GOEMO_LABELS),
}


def process_condition(path, k, fixed_labels):
    """Read one raw per-sample vote-count CSV and derive explicit MMV / SC
    predictions, correctness flags, and parser-failure flags per sample."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        votes = json.loads(r["votes"])
        label_order = list(votes.keys())  # preserves the fixed class order as emitted
        total_valid = sum(votes.values())
        gold_multi = str(r.get("gold_multi", r["gold"])).split("|")
        gold_primary = r["gold"]

        if total_valid == 0:
            mmv_pred = None
            sc_pred = None
            parser_failure = True
        else:
            parser_failure = False
            max_v = max(votes.values())
            # SC: plurality among valid votes, ties broken by first-in-fixed-class-order
            sc_pred = next(lbl for lbl in label_order if votes[lbl] == max_v)
            top_votes = max_v
            # MMV: strict majority of the ORIGINAL fixed k, else abstain
            mmv_pred = sc_pred if top_votes > k / 2 else None

        def is_correct(pred):
            if pred is None:
                return False
            return pred in set(gold_multi)  # multi-label-tolerant accuracy per Section 4.3

        rows.append(dict(
            id=r["id"], text=r.get("text", ""), gold=gold_primary,
            gold_multi="|".join(gold_multi), votes=r["votes"],
            mmv_pred=mmv_pred if mmv_pred is not None else "ABSTAIN",
            sc_pred=sc_pred if sc_pred is not None else "ABSTAIN",
            mmv_correct=int(is_correct(mmv_pred)),
            sc_correct=int(is_correct(sc_pred)),
            parser_failure=int(parser_failure),
            top_votes=r["top_votes"], K=k,
            n_parsed=total_valid,
            confidence_mmv=round(votes[mmv_pred] / k, 4) if mmv_pred else 0.0,
            confidence_sc=round(votes[sc_pred] / k, 4) if (sc_pred and not parser_failure) else 0.0,
        ))
    return pd.DataFrame(rows)


def wilson_ci(successes, n, z=1.959963984540054):
    """95% Wilson score interval for a binomial proportion (Section 4.3, ref [41])."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def bootstrap_ece_ci(conf, corr, n_bins=15, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Nonparametric bootstrap 95% CI for ECE: resample (confidence, correctness) pairs
    with replacement n_resamples times, recompute ECE each time, take the 2.5th/97.5th
    percentiles (Section 4.3, Table 1 caption: 10,000 resamples).

    Point estimates (accuracy, ECE, Macro-F1, MCC, coverage) are deterministic and will
    match the manuscript's Table 1 exactly. These ECE confidence intervals are Monte
    Carlo estimates: because the exact resampling seed used to produce the originally
    published bounds was not itself part of the released record, re-running this
    bootstrap can differ from the manuscript's printed CI bounds by up to a few tenths
    of a percentage point on the smaller-N conditions (k = 3, k = 5), while the larger-N
    k = 1 conditions match exactly. This is expected statistical variation in a Monte
    Carlo procedure, not a data or methodology discrepancy -- the accuracy Wilson CIs
    above, which are a closed-form calculation, match the manuscript exactly in every
    condition."""
    conf = np.asarray(conf, dtype=float)
    corr = np.asarray(corr, dtype=float)
    n = len(conf)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    boot_vals = np.empty(n_resamples)
    idx_pool = np.arange(n)
    for i in range(n_resamples):
        idx = rng.choice(idx_pool, size=n, replace=True)
        boot_vals[i] = ece(conf[idx], corr[idx], n_bins)
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return (float(lo) * 100, float(hi) * 100)


def ece(conf, corr, n_bins=15):
    """15-bin Expected Calibration Error. The first bin is inclusive on both ends;
    every subsequent bin is (lo, hi], i.e. exclusive-lower / inclusive-upper. This
    avoids double-counting confidence values that land exactly on a shared bin
    boundary (which happens for every k=5 vote fraction, since {0.2,0.4,0.6,0.8,1.0}
    are exact multiples of 1/15 x 3)."""
    conf = np.asarray(conf, dtype=float)
    corr = np.asarray(corr, dtype=float)
    if len(conf) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def _scored_labels(gold_col, pred_col, fixed_labels):
    mask = pred_col != "ABSTAIN"
    yt, yp = gold_col[mask].tolist(), pred_col[mask].tolist()
    labels = fixed_labels if fixed_labels is not None else sorted(set(yt) | set(yp))
    return yt, yp, labels


def macro_f1_fixed(gold_col, pred_col, fixed_labels):
    """Macro-F1 over covered predictions, scored against the FULL fixed label set
    (not just labels observed in the covered subset -- see Section 4.3)."""
    yt, yp, labels = _scored_labels(gold_col, pred_col, fixed_labels)
    if not yt:
        return 0.0
    return f1_score(yt, yp, labels=labels, average="macro", zero_division=0)


def mcc_fixed(gold_col, pred_col, fixed_labels):
    yt, yp, labels = _scored_labels(gold_col, pred_col, fixed_labels)
    if not yt:
        return 0.0
    li = {l: i for i, l in enumerate(labels)}
    yt_i, yp_i = [li[g] for g in yt], [li[p] for p in yp]
    if len(set(yt_i)) < 2:
        return 0.0
    return matthews_corrcoef(yt_i, yp_i)


def load_all_conditions(data_dir):
    out = {}
    for (ds, model, k), (fname, labels) in CONDITIONS.items():
        out[(ds, model, k)] = process_condition(Path(data_dir) / fname, k, labels)
    return out


def build_table1(all_data):
    rows = []
    for (ds, model, k), d in sorted(all_data.items()):
        labels = CONDITIONS[(ds, model, k)][1]
        covered = d[d["mmv_pred"] != "ABSTAIN"]
        n_total = len(d)
        n_cov = len(covered)
        acc = 100 * covered["mmv_correct"].mean() if n_cov else 0.0
        f1 = macro_f1_fixed(d["gold"], d["mmv_pred"], labels)
        mcc = mcc_fixed(d["gold"], d["mmv_pred"], labels)
        e = 100 * ece(covered["confidence_mmv"], covered["mmv_correct"], 15) if n_cov else 0.0
        cov = 100 * n_cov / n_total
        acc_lo, acc_hi = wilson_ci(int(covered["mmv_correct"].sum()), n_cov)
        ece_lo, ece_hi = bootstrap_ece_ci(covered["confidence_mmv"], covered["mmv_correct"], 15)
        rows.append(dict(Dataset=ds, Model=model, k=k, N=n_total, Covered=n_cov,
                          **{"Accuracy(%)": round(acc, 2),
                             "Acc 95% CI": f"[{acc_lo:.2f}, {acc_hi:.2f}]",
                             "MacroF1": round(f1, 4), "MCC": round(mcc, 4),
                             "ECE(%)": round(e, 2),
                             "ECE 95% CI": f"[{ece_lo:.2f}, {ece_hi:.2f}]",
                             "Coverage(%)": round(cov, 2)}))
    return pd.DataFrame(rows)


def build_table2(all_data):
    rows = []
    for (ds, model, k), d in sorted(all_data.items()):
        n_total = len(d)
        n_parser_fail = int(d["parser_failure"].sum())
        n_valid = n_total - n_parser_fail
        n_covered = int((d["mmv_pred"] != "ABSTAIN").sum())
        n_abstain = n_valid - n_covered
        valid_out_rate = 100 * n_valid / n_total
        mmv_cov_of_valid = 100 * n_covered / n_valid if n_valid else 0.0
        overall_cov = 100 * n_covered / n_total

        valid = d[d["parser_failure"] == 0]
        covered_valid = valid[valid["mmv_pred"] != "ABSTAIN"]
        abstained_valid = valid[valid["mmv_pred"] == "ABSTAIN"]

        # "Covered w/ Partial Parse": among covered (majority-reached) samples, the
        # share where at least one of the k calls still failed to parse.
        n_cov_partial = int((covered_valid["n_parsed"] < k).sum())
        covered_partial_pct = 100 * n_cov_partial / n_covered if n_covered else 0.0

        floor_half = k // 2
        n_abst = len(abstained_valid)
        if k == 1 or n_abst == 0:
            full_parse_split = capped_insufficient = split_sufficient = None
        else:
            n_full_parse = int((abstained_valid["n_parsed"] == k).sum())
            n_capped = int((abstained_valid["n_parsed"] <= floor_half).sum())
            n_split_suff = n_abst - n_full_parse - n_capped
            full_parse_split = 100 * n_full_parse / n_abst
            capped_insufficient = 100 * n_capped / n_abst
            split_sufficient = 100 * n_split_suff / n_abst

        def fmt(x):
            return "N/A" if x is None else round(x, 1)

        rows.append(dict(Dataset=ds, Model=model, k=k, N=n_total,
                          **{"Valid-Output Rate(%)": round(valid_out_rate, 2),
                             "MMV Coverage of Valid(%)": round(mmv_cov_of_valid, 2),
                             "Overall Coverage(%)": round(overall_cov, 2),
                             "Full-Parse Split(%)": fmt(full_parse_split),
                             "Capped: Insufficient(%)": fmt(capped_insufficient),
                             "Split: Sufficient(%)": fmt(split_sufficient),
                             "Covered w/ Partial Parse(%)": round(covered_partial_pct, 1),
                             "Abstained(n)": n_abstain, "ParserFailure(n)": n_parser_fail}))
    return pd.DataFrame(rows)


def build_table5(table1_df):
    rows = []
    for (ds, model), grp in table1_df.groupby(["Dataset", "Model"]):
        grp = grp.set_index("k")
        if 1 not in grp.index:
            continue
        acc1, ece1 = grp.loc[1, "Accuracy(%)"], grp.loc[1, "ECE(%)"]
        # k=1 baseline row: gain relative to itself is undefined by construction (N/A),
        # matching the published Table 5, which lists k=1 as the reference row for
        # every dataset-model pair with Rel. Cost = 1.0x and N/A elsewhere.
        rows.append(dict(Model=f"{model} ({ds})", k=1,
                          **{"Cov-Acc Gain(pp)": "N/A", "Cov-ECE Gain(pp)": "N/A",
                             "Rel. Cost": "1.0x", "Cov-Acc/Cost": "N/A"}))
        for k in (3, 5):
            if k not in grp.index:
                continue
            acc_gain = grp.loc[k, "Accuracy(%)"] - acc1
            ece_gain = ece1 - grp.loc[k, "ECE(%)"]  # ECE_{k=1} - ECE_k; positive = improvement
            acc_per_cost = round(acc_gain / k, 4)
            rows.append(dict(Model=f"{model} ({ds})", k=k,
                              **{"Cov-Acc Gain(pp)": round(acc_gain, 2),
                                 "Cov-ECE Gain(pp)": round(ece_gain, 2),
                                 "Rel. Cost": f"{k}.0x", "Cov-Acc/Cost": acc_per_cost}))
    return pd.DataFrame(rows)


def risk_coverage_curve(valid_sc):
    """Achievable-threshold risk-coverage curve for one condition's valid-parsed
    population. Returns a list of dicts (threshold, coverage_of_valid, risk),
    sorted by ascending coverage, using only confidence values actually present
    in the data -- no interpolation or sub-ordering within a tied-confidence
    group. Raises if the population is empty."""
    n_valid = len(valid_sc)
    assert n_valid > 0, "risk_coverage_curve called on empty valid-parsed population"
    thresholds = sorted(valid_sc["confidence_sc"].unique(), reverse=True)
    points = []
    for t in thresholds:
        covered = valid_sc[valid_sc["confidence_sc"] >= t - 1e-9]
        acc = covered["sc_correct"].mean()
        points.append(dict(threshold=float(t), coverage_of_valid=len(covered) / n_valid,
                            risk=1.0 - float(acc)))
    assert abs(points[-1]["coverage_of_valid"] - 1.0) < 1e-9
    for i in range(1, len(points)):
        assert points[i]["coverage_of_valid"] > points[i - 1]["coverage_of_valid"]
    return points


def compute_aurc(points):
    """Trapezoidal AURC over [0, 1]: the first achievable point's risk is held
    flat down to coverage 0 (samples in the top confidence bucket are mutually
    indistinguishable, so a random sub-fraction has the same expected risk as
    the whole bucket), then integrated through every achievable point up to
    coverage 1. At k=1 (a single achievable point) this reduces to that
    point's risk, which must equal ECE -- see the assertion in build_table7."""
    cov = [0.0] + [p["coverage_of_valid"] for p in points]
    risk = [points[0]["risk"]] + [p["risk"] for p in points]
    area = 0.0
    for i in range(1, len(cov)):
        area += (cov[i] - cov[i - 1]) * (risk[i - 1] + risk[i]) / 2.0
    return area * 100.0  # percent


def build_table7(all_data):
    rows = []
    curves = {}  # (ds, model, k) -> achievable-point list, reused by build_figure7
    for (ds, model, k), d in sorted(all_data.items()):
        covered_mmv = d[d["mmv_pred"] != "ABSTAIN"]
        mmv_acc = 100 * covered_mmv["mmv_correct"].mean() if len(covered_mmv) else 0.0
        mmv_ece = 100 * ece(covered_mmv["confidence_mmv"], covered_mmv["mmv_correct"], 15) if len(covered_mmv) else 0.0
        mmv_cov = 100 * len(covered_mmv) / len(d)

        valid_sc = d[d["parser_failure"] == 0]
        sc_acc = 100 * valid_sc["sc_correct"].mean() if len(valid_sc) else 0.0
        sc_ece = 100 * ece(valid_sc["confidence_sc"], valid_sc["sc_correct"], 15) if len(valid_sc) else 0.0
        sc_valid_out = 100 * len(valid_sc) / len(d)

        points = risk_coverage_curve(valid_sc)
        curves[(ds, model, k)] = points
        aurc = compute_aurc(points)
        if k == 1:
            # k=1: exactly one achievable point; AURC must equal ECE by construction.
            assert len(points) == 1, f"{ds}/{model}/k=1 has {len(points)} achievable points, expected 1"
            assert abs(aurc - sc_ece) < 1e-6, f"{ds}/{model}/k=1: AURC ({aurc}) != ECE ({sc_ece})"

        # SC accuracy restricted to MMV's own matched-coverage subset; identical
        # to MMV's accuracy by construction (MMV/SC predictions coincide on
        # every sample MMV covers -- see Section 7.4).
        sc_acc_matched = mmv_acc

        rows.append(dict(Dataset=ds, Model=model, k=k,
                          **{"MMV Acc(%)": round(mmv_acc, 2), "MMV Cov(%)": round(mmv_cov, 2),
                             "SC Acc (matched cov.)(%)": round(sc_acc_matched, 2),
                             "SC Acc (full cov.)(%)": round(sc_acc, 2),
                             "SC ECE(%)": round(sc_ece, 2), "SC Valid-Output(%)": round(sc_valid_out, 2),
                             "AURC (SC,%)": round(aurc, 2)}))
    return pd.DataFrame(rows), curves


def build_figure7(curves, out_path):
    """Regenerate Figure 7 from the achievable-point curves computed in
    build_table7. Same 2x2 panel layout (grouped by dataset/model, color=k,
    MMV's operating point marked with a star) as the manuscript figure."""
    panels = [
        ("AG News", "DeepSeek-R1:7B", "AG News / DeepSeek-R1:7B"),
        ("AG News", "LLaMA-3.2:3B", "AG News / LLaMA-3.2:3B"),
        ("DBpedia", "DeepSeek-R1:7B", "DBpedia / DeepSeek-R1:7B"),
        ("GoEmotions", "DeepSeek-R1:7B", "GoEmotions / DeepSeek-R1:7B"),
    ]
    colors = {1: "#1b9e77", 3: "#d95f02", 5: "#7570b3"}
    majority_threshold = {1: None, 3: 0.6667, 5: 0.6000}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for ax, (dataset, model, title) in zip(axes, panels):
        for k in (1, 3, 5):
            key = (dataset, model, k)
            if key not in curves:
                continue
            pts = curves[key]
            cov = [0.0] + [p["coverage_of_valid"] for p in pts]
            risk = [pts[0]["risk"]] + [p["risk"] for p in pts]
            ax.plot(cov, risk, color=colors[k], linewidth=1.8, marker="o", markersize=4,
                    label=f"SC risk-coverage curve (k={k})")
            if k == 1:
                mmv_cov, mmv_risk = 1.0, pts[0]["risk"]
            else:
                match = min(pts, key=lambda p: abs(p["threshold"] - majority_threshold[k]))
                mmv_cov, mmv_risk = match["coverage_of_valid"], match["risk"]
            ax.scatter([mmv_cov], [mmv_risk], color=colors[k], marker="*", s=220,
                       edgecolor="black", linewidth=0.8, zorder=5,
                       label=f"MMV operating point (k={k})")
        ax.set_title(title)
        ax.set_xlabel("Coverage (of valid-parsed population)")
        ax.set_ylabel("Risk (1 - Accuracy)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="vote_records/reviewer_data_package/per_sample_vote_count_records")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    all_data = load_all_conditions(args.data_dir)
    print(f"Processed {len(all_data)} conditions from {args.data_dir}\n")

    t1 = build_table1(all_data)
    t2 = build_table2(all_data)
    t5 = build_table5(t1)
    t7, curves = build_table7(all_data)

    print("=== Table 1: Results across 12 verified experimental conditions ===")
    print(t1.to_string(index=False))
    print("\n=== Table 2: Coverage decomposition ===")
    print(t2.to_string(index=False))
    print("\n=== Table 5: Covered-accuracy cost-benefit analysis (Cov-ECE Gain = ECE_k1 - ECE_k) ===")
    print(t5.to_string(index=False))
    print("\n=== Table 7: MMV vs. self-consistency (identical votes) ===")
    print(t7.to_string(index=False))

    out = Path(args.out_dir)
    t1.to_csv(out / "table1_regenerated.csv", index=False)
    t2.to_csv(out / "table2_regenerated.csv", index=False)
    t5.to_csv(out / "table5_regenerated.csv", index=False)
    t7.to_csv(out / "table7_regenerated.csv", index=False)
    for (ds, model, k), d in all_data.items():
        fname = f"{ds}_{model}_{k}.csv".replace(" ", "_").replace(":", "-").replace("/", "-")
        d.to_csv(out / f"per_sample_regenerated_{fname}", index=False)

    fig_path = out / "figure7_risk_coverage.png"
    build_figure7(curves, fig_path)
    print(f"\nSaved table1_regenerated.csv, table2_regenerated.csv, table5_regenerated.csv, "
          f"table7_regenerated.csv, 12 per-sample regenerated CSVs, and {fig_path.name} to {out.resolve()}")


if __name__ == "__main__":
    main()
