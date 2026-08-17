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
No numbers are hand-typed anywhere downstream of this script; Tables 1, 2, 5, and 7
are derived entirely from the per-sample vote-count records.

Usage: python3 regenerate_all.py [--data-dir DIR]
Default --data-dir is vote_records/reviewer_data_package/per_sample_vote_count_records/
relative to the repository root.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, matthews_corrcoef

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
            confidence_mmv=round(votes[mmv_pred] / k, 4) if mmv_pred else 0.0,
            confidence_sc=round(votes[sc_pred] / k, 4) if (sc_pred and not parser_failure) else 0.0,
        ))
    return pd.DataFrame(rows)


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
        acc = 100 * covered["mmv_correct"].mean() if len(covered) else 0.0
        f1 = macro_f1_fixed(d["gold"], d["mmv_pred"], labels)
        mcc = mcc_fixed(d["gold"], d["mmv_pred"], labels)
        e = 100 * ece(covered["confidence_mmv"], covered["mmv_correct"], 15) if len(covered) else 0.0
        cov = 100 * len(covered) / n_total
        rows.append(dict(Dataset=ds, Model=model, k=k, N=n_total, Covered=len(covered),
                          **{"Accuracy(%)": round(acc, 2), "MacroF1": round(f1, 4),
                             "MCC": round(mcc, 4), "ECE(%)": round(e, 2),
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
        rows.append(dict(Dataset=ds, Model=model, k=k, N=n_total,
                          **{"Valid-Output Rate(%)": round(valid_out_rate, 2),
                             "MMV Coverage of Valid(%)": round(mmv_cov_of_valid, 2),
                             "Overall Coverage(%)": round(overall_cov, 2),
                             "Abstained(n)": n_abstain, "ParserFailure(n)": n_parser_fail}))
    return pd.DataFrame(rows)


def build_table5(table1_df):
    rows = []
    for (ds, model), grp in table1_df.groupby(["Dataset", "Model"]):
        grp = grp.set_index("k")
        if 1 not in grp.index:
            continue
        acc1, ece1 = grp.loc[1, "Accuracy(%)"], grp.loc[1, "ECE(%)"]
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


def build_table7(all_data):
    rows = []
    for (ds, model, k), d in sorted(all_data.items()):
        covered_mmv = d[d["mmv_pred"] != "ABSTAIN"]
        mmv_acc = 100 * covered_mmv["mmv_correct"].mean() if len(covered_mmv) else 0.0
        mmv_ece = 100 * ece(covered_mmv["confidence_mmv"], covered_mmv["mmv_correct"], 15) if len(covered_mmv) else 0.0
        mmv_cov = 100 * len(covered_mmv) / len(d)

        valid_sc = d[d["parser_failure"] == 0]
        sc_acc = 100 * valid_sc["sc_correct"].mean() if len(valid_sc) else 0.0
        sc_ece = 100 * ece(valid_sc["confidence_sc"], valid_sc["sc_correct"], 15) if len(valid_sc) else 0.0
        sc_valid_out = 100 * len(valid_sc) / len(d)

        rows.append(dict(Dataset=ds, Model=model, k=k,
                          **{"MMV Acc(%)": round(mmv_acc, 2), "MMV ECE(%)": round(mmv_ece, 2),
                             "MMV Cov(%)": round(mmv_cov, 2), "SC Acc(%)": round(sc_acc, 2),
                             "SC ECE(%)": round(sc_ece, 2), "SC Valid-Output(%)": round(sc_valid_out, 2),
                             "SC Vote Cov(%)": 100.00}))
    return pd.DataFrame(rows)


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
    t7 = build_table7(all_data)

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
    print(f"\nSaved table1_regenerated.csv, table2_regenerated.csv, table5_regenerated.csv, "
          f"table7_regenerated.csv, and 12 per-sample regenerated CSVs to {out.resolve()}")


if __name__ == "__main__":
    main()
