"""
3-repeat post-fix companion to regenerate_all.py.

Reads the 36 post-fix per-sample vote-count CSVs in runs/reviewer_r1_reruns/
(12 conditions x 3 independent repeats, collected after the eval_dataset.py
max-tokens/client-side generation-length bug was fixed), reuses the exact
same MMV/SC/ECE/Table2/Table7/AURC computation logic as regenerate_all.py
(imported directly, not reimplemented), computes it once per repeat, and
reports the mean across the 3 repeats for Table 2 and Table 7 (consistent
with Table 1's already-established Mean(3) +/- SD(3) methodology). Table 5
is rebuilt from the resulting post-fix Table 1 means. Figure 7's curves are
averaged across repeats at each shared quantized threshold (thresholds are
always k-quantized: {1/k, ..., k/k}, so identical across repeats for fixed k).

Usage: python3 regenerate_3rep.py --data-dir runs/reviewer_r1_reruns --out-dir <dir>
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import regenerate_all as ra  # reuse process_condition/ece/wilson_ci/macro_f1_fixed/mcc_fixed/etc.

AG_NEWS_LABELS = ra.AG_NEWS_LABELS
GOEMO_LABELS = ra.GOEMO_LABELS

# (dataset, model, k) -> filename stem prefix used in runs/reviewer_r1_reruns/
REP_CONDITIONS = {
    ("AG News", "DeepSeek-R1:7B", 1): ("ag_news_deepseek_matched", AG_NEWS_LABELS),
    ("AG News", "DeepSeek-R1:7B", 3): ("ag_news_deepseek_matched", AG_NEWS_LABELS),
    ("AG News", "DeepSeek-R1:7B", 5): ("ag_news_deepseek_matched", AG_NEWS_LABELS),
    ("AG News", "LLaMA-3.2:3B", 1): ("ag_news_llama3.2", AG_NEWS_LABELS),
    ("AG News", "LLaMA-3.2:3B", 3): ("ag_news_llama3.2", AG_NEWS_LABELS),
    ("AG News", "LLaMA-3.2:3B", 5): ("ag_news_llama3.2", AG_NEWS_LABELS),
    ("DBpedia", "DeepSeek-R1:7B", 1): ("dbpedia_deepseek", None),
    ("DBpedia", "DeepSeek-R1:7B", 3): ("dbpedia_deepseek", None),
    ("DBpedia", "DeepSeek-R1:7B", 5): ("dbpedia_deepseek", None),
    ("GoEmotions", "DeepSeek-R1:7B", 1): ("goemotions_deepseek", GOEMO_LABELS),
    ("GoEmotions", "DeepSeek-R1:7B", 3): ("goemotions_deepseek", GOEMO_LABELS),
    ("GoEmotions", "DeepSeek-R1:7B", 5): ("goemotions_deepseek", GOEMO_LABELS),
}


def _repair_gold_multi(src_path, tmp_dir):
    """Some rep3 files carry an extra gold_multi column that is entirely NaN
    for single-label datasets (AG News, DBpedia) -- a benign schema drift from
    a later eval_dataset.py version, confirmed by spot-check: where gold_multi
    IS populated (GoEmotions), it matches gold as a prefix exactly (0 mismatches
    checked). process_condition's r.get("gold_multi", r["gold"]) returns the NaN
    itself (not the fallback) when the column exists but is null, silently
    corrupting every correctness check in that file. Fix: fill any null
    gold_multi with the row's own gold value before handing off to
    process_condition, matching rep1/rep2 files (which have no gold_multi
    column at all and so correctly fall through to gold)."""
    import pandas as pd
    df = pd.read_csv(src_path)
    if "gold_multi" in df.columns:
        n_null_before = df["gold_multi"].isna().sum()
        df["gold_multi"] = df["gold_multi"].fillna(df["gold"])
        if n_null_before:
            print(f"  [repair] {src_path.name}: filled {n_null_before}/{len(df)} null gold_multi from gold")
    out_path = Path(tmp_dir) / src_path.name
    df.to_csv(out_path, index=False)
    return out_path


def load_all_reps(data_dir):
    import tempfile
    out = {}
    tmp_dir = tempfile.mkdtemp(prefix="regen3rep_repaired_")
    for (ds, model, k), (stem, labels) in REP_CONDITIONS.items():
        reps = []
        for rep in (1, 2, 3):
            fpath = Path(data_dir) / f"{stem}_k{k}_rep{rep}.csv"
            assert fpath.exists(), f"missing {fpath}"
            repaired = _repair_gold_multi(fpath, tmp_dir)
            reps.append(ra.process_condition(repaired, k, labels))
        out[(ds, model, k)] = reps
    return out


def per_rep_table1_row(d, ds, model, k, labels):
    covered = d[d["mmv_pred"] != "ABSTAIN"]
    n_total, n_cov = len(d), len(covered)
    acc = 100 * covered["mmv_correct"].mean() if n_cov else 0.0
    # F1/MCC scoring uses the first-listed gold label for multi-label GoEmotions
    # rows (e.g. "joy|optimism" -> "joy"), matching compute_repeat_metrics.py's
    # established Table 1 convention; correctness crediting (mmv_correct, computed
    # inside process_condition) already credits ANY of the multi-labels and is
    # untouched here.
    gold_primary_for_scoring = d["gold"].astype(str).str.split("|").str[0]
    f1 = ra.macro_f1_fixed(gold_primary_for_scoring, d["mmv_pred"], labels)
    mcc = ra.mcc_fixed(gold_primary_for_scoring, d["mmv_pred"], labels)
    e = 100 * ra.ece(covered["confidence_mmv"], covered["mmv_correct"], 15) if n_cov else 0.0
    cov = 100 * n_cov / n_total
    n_correct = int(covered["mmv_correct"].sum())
    return dict(Accuracy=acc, MacroF1=f1, MCC=mcc, ECE=e, Coverage=cov, N=n_total, K=None, Correct_n=n_correct)


def build_table1_3rep(all_reps):
    rows = []
    for (ds, model, k), reps in sorted(all_reps.items()):
        labels = REP_CONDITIONS[(ds, model, k)][1]
        per_rep = [per_rep_table1_row(d, ds, model, k, labels) for d in reps]
        acc_vals = [r["Accuracy"] for r in per_rep]
        ece_vals = [r["ECE"] for r in per_rep]
        f1_vals = [r["MacroF1"] for r in per_rep]
        mcc_vals = [r["MCC"] for r in per_rep]
        cov_vals = [r["Coverage"] for r in per_rep]
        rows.append(dict(
            Dataset=ds, Model=model, k=k,
            Acc_mean=round(float(np.mean(acc_vals)), 2), Acc_sd=round(float(np.std(acc_vals, ddof=1)), 2),
            ECE_mean=round(float(np.mean(ece_vals)), 2), ECE_sd=round(float(np.std(ece_vals, ddof=1)), 2),
            F1_mean=round(float(np.mean(f1_vals)), 4),
            MCC_mean=round(float(np.mean(mcc_vals)), 4),
            Cov_mean=round(float(np.mean(cov_vals)), 2),
            Acc_rep1=round(acc_vals[0], 2), Acc_rep2=round(acc_vals[1], 2), Acc_rep3=round(acc_vals[2], 2),
            N=per_rep[0]["N"], K=k,
            Correct_n_mean=round(float(np.mean([r["Correct_n"] for r in per_rep])), 1),
        ))
    return pd.DataFrame(rows)


def build_table2_3rep(all_reps):
    rows = []
    for (ds, model, k), reps in sorted(all_reps.items()):
        per_rep_rows = []
        for d in reps:
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

            per_rep_rows.append(dict(
                valid_out_rate=valid_out_rate, mmv_cov_of_valid=mmv_cov_of_valid,
                overall_cov=overall_cov, full_parse_split=full_parse_split,
                capped_insufficient=capped_insufficient, split_sufficient=split_sufficient,
                covered_partial_pct=covered_partial_pct, n_abstain=n_abstain, n_parser_fail=n_parser_fail,
            ))

        def mean_or_none(key):
            vals = [r[key] for r in per_rep_rows if r[key] is not None]
            return round(float(np.mean(vals)), 1) if vals else None

        rows.append(dict(
            Dataset=ds, Model=model, k=k, N=len(reps[0]),
            **{"Valid-Output Rate(%)": round(float(np.mean([r["valid_out_rate"] for r in per_rep_rows])), 2),
               "MMV Coverage of Valid(%)": round(float(np.mean([r["mmv_cov_of_valid"] for r in per_rep_rows])), 2),
               "Overall Coverage(%)": round(float(np.mean([r["overall_cov"] for r in per_rep_rows])), 2),
               "Full-Parse Split(%)": mean_or_none("full_parse_split"),
               "Capped: Insufficient(%)": mean_or_none("capped_insufficient"),
               "Split: Sufficient(%)": mean_or_none("split_sufficient"),
               "Covered w/ Partial Parse(%)": round(float(np.mean([r["covered_partial_pct"] for r in per_rep_rows])), 1),
               "Abstained(n, mean)": round(float(np.mean([r["n_abstain"] for r in per_rep_rows])), 1),
               "ParserFailure(n, mean)": round(float(np.mean([r["n_parser_fail"] for r in per_rep_rows])), 1)}))
    return pd.DataFrame(rows)


def build_table5_3rep(table1_df):
    rows = []
    for (ds, model), grp in table1_df.groupby(["Dataset", "Model"]):
        grp = grp.set_index("k")
        if 1 not in grp.index:
            continue
        acc1, ece1 = grp.loc[1, "Acc_mean"], grp.loc[1, "ECE_mean"]
        n1 = grp.loc[1, "N"]
        correct1 = grp.loc[1, "Correct_n_mean"]
        rows.append(dict(Model=f"{model} ({ds})", k=1,
                          **{"Cov-Acc Gain(pp)": "N/A", "Cov-ECE Gain(pp)": "N/A",
                             "Rel. Cost": "1.0x", "Cov-Acc/Cost": "N/A",
                             "Correct(n/N, mean)": f"{correct1:.1f}/{n1}",
                             "Correct/(NxK)(%)": round(100 * correct1 / (n1 * 1), 2)}))
        for k in (3, 5):
            if k not in grp.index:
                continue
            acc_gain = grp.loc[k, "Acc_mean"] - acc1
            ece_gain = ece1 - grp.loc[k, "ECE_mean"]
            acc_per_cost = round(acc_gain / k, 4)
            nk = grp.loc[k, "N"]
            correctk = grp.loc[k, "Correct_n_mean"]
            rows.append(dict(Model=f"{model} ({ds})", k=k,
                              **{"Cov-Acc Gain(pp)": round(acc_gain, 2),
                                 "Cov-ECE Gain(pp)": round(ece_gain, 2),
                                 "Rel. Cost": f"{k}.0x", "Cov-Acc/Cost": acc_per_cost,
                                 "Correct(n/N, mean)": f"{correctk:.1f}/{nk}",
                                 "Correct/(NxK)(%)": round(100 * correctk / (nk * k), 2)}))
    return pd.DataFrame(rows)


def per_rep_table7_row(d):
    covered_mmv = d[d["mmv_pred"] != "ABSTAIN"]
    mmv_acc = 100 * covered_mmv["mmv_correct"].mean() if len(covered_mmv) else 0.0
    mmv_ece = 100 * ra.ece(covered_mmv["confidence_mmv"], covered_mmv["mmv_correct"], 15) if len(covered_mmv) else 0.0
    mmv_cov = 100 * len(covered_mmv) / len(d)

    valid_sc = d[d["parser_failure"] == 0]
    sc_acc = 100 * valid_sc["sc_correct"].mean() if len(valid_sc) else 0.0
    sc_ece = 100 * ra.ece(valid_sc["confidence_sc"], valid_sc["sc_correct"], 15) if len(valid_sc) else 0.0
    sc_valid_out = 100 * len(valid_sc) / len(d)

    points = ra.risk_coverage_curve(valid_sc)
    aurc = ra.compute_aurc(points)
    return dict(mmv_acc=mmv_acc, mmv_ece=mmv_ece, mmv_cov=mmv_cov, sc_acc=sc_acc,
                sc_ece=sc_ece, sc_valid_out=sc_valid_out, aurc=aurc, points=points)


def build_table7_3rep(all_reps):
    rows = []
    curves = {}
    k1_assert_log = []
    for (ds, model, k), reps in sorted(all_reps.items()):
        per_rep = [per_rep_table7_row(d) for d in reps]
        for i, pr in enumerate(per_rep):
            if k == 1:
                assert len(pr["points"]) == 1, f"{ds}/{model}/k=1/rep{i+1} has {len(pr['points'])} points"
                ok = abs(pr["aurc"] - pr["sc_ece"]) < 1e-6
                k1_assert_log.append((ds, model, i + 1, ok, pr["aurc"], pr["sc_ece"]))
        curves[(ds, model, k)] = [pr["points"] for pr in per_rep]

        mmv_acc = float(np.mean([pr["mmv_acc"] for pr in per_rep]))
        mmv_cov = float(np.mean([pr["mmv_cov"] for pr in per_rep]))
        sc_acc = float(np.mean([pr["sc_acc"] for pr in per_rep]))
        sc_ece = float(np.mean([pr["sc_ece"] for pr in per_rep]))
        sc_valid_out = float(np.mean([pr["sc_valid_out"] for pr in per_rep]))
        aurc = float(np.mean([pr["aurc"] for pr in per_rep]))
        aurc_sd = float(np.std([pr["aurc"] for pr in per_rep], ddof=1))

        rows.append(dict(Dataset=ds, Model=model, k=k,
                          **{"MMV Acc(%)": round(mmv_acc, 2), "MMV Cov(%)": round(mmv_cov, 2),
                             "SC Acc (matched cov.)(%)": round(mmv_acc, 2),
                             "SC Acc (full cov.)(%)": round(sc_acc, 2),
                             "SC ECE(%)": round(sc_ece, 2), "SC Valid-Output(%)": round(sc_valid_out, 2),
                             "AURC (SC,%)": round(aurc, 2), "AURC SD(3)": round(aurc_sd, 3)}))
    return pd.DataFrame(rows), curves, k1_assert_log


def average_curves_for_figure(curves_3):
    by_threshold = defaultdict(list)
    for points in curves_3:
        for p in points:
            by_threshold[round(p["threshold"], 6)].append(p)
    avg_points = []
    for t in sorted(by_threshold.keys(), reverse=True):
        pts = by_threshold[t]
        avg_points.append(dict(
            threshold=t,
            coverage_of_valid=float(np.mean([p["coverage_of_valid"] for p in pts])),
            risk=float(np.mean([p["risk"] for p in pts])),
            n_reps_present=len(pts),
        ))
    return avg_points


def build_figure7_3rep(curves, out_path):
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
    avg_curves = {}
    for ax, (dataset, model, title) in zip(axes, panels):
        for k in (1, 3, 5):
            key = (dataset, model, k)
            if key not in curves:
                continue
            pts = average_curves_for_figure(curves[key])
            avg_curves[key] = pts
            cov = [0.0] + [p["coverage_of_valid"] for p in pts]
            risk = [pts[0]["risk"]] + [p["risk"] for p in pts]
            ax.plot(cov, risk, color=colors[k], linewidth=1.8, marker="o", markersize=4,
                    label=f"SC risk-coverage curve (k={k}, mean of 3 reps)")
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
    return avg_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="runs/reviewer_r1_reruns")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    all_reps = load_all_reps(args.data_dir)
    print(f"Loaded {len(all_reps)} conditions x 3 reps from {args.data_dir}\n")

    t1 = build_table1_3rep(all_reps)
    t2 = build_table2_3rep(all_reps)
    t5 = build_table5_3rep(t1)
    t7, curves, k1log = build_table7_3rep(all_reps)

    print("=== k=1 AURC==ECE assertion log (per rep) ===")
    for ds, model, rep, ok, aurc, ece_v in k1log:
        status = "OK" if ok else "FAIL"
        print(f"  {status}  {ds}/{model} rep{rep}: AURC={aurc:.4f} ECE={ece_v:.4f}")
    assert all(ok for *_, ok, _, _ in k1log), "k=1 AURC==ECE assertion failed for at least one rep"

    print("\n=== Table 1 (3-repeat means) ===")
    print(t1.to_string(index=False))
    print("\n=== Table 2 (3-repeat means) ===")
    print(t2.to_string(index=False))
    print("\n=== Table 5 (from 3-repeat Table 1 means) ===")
    print(t5.to_string(index=False))
    print("\n=== Table 7 (3-repeat means) ===")
    print(t7.to_string(index=False))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t1.to_csv(out / "table1_3rep_verify.csv", index=False)
    t2.to_csv(out / "table2_3rep.csv", index=False)
    t5.to_csv(out / "table5_3rep.csv", index=False)
    t7.to_csv(out / "table7_3rep.csv", index=False)

    fig_path = out / "figure7_3rep_risk_coverage.png"
    avg_curves = build_figure7_3rep(curves, fig_path)
    print(f"\nSaved table2_3rep.csv, table5_3rep.csv, table7_3rep.csv, table1_3rep_verify.csv, {fig_path.name} to {out.resolve()}")


if __name__ == "__main__":
    main()
