"""
compute_results.py — Aggregate all run CSVs into Tables 3 and 4 for the paper.

Usage:
    cd /path/to/llm_majority_vote_ollama
    python scripts/compute_results.py

Outputs:
    - Prints Table 3 (full metrics) and Table 4 (vs SOTA) to terminal
    - Saves  results_table3.csv  and  results_table4.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

N_BINS = 15

DATASET_NICE = {"ag_news": "AG News", "dbpedia": "DBpedia", "goemotions": "GoEmotions"}
MODEL_NICE   = {"deepseek": "DeepSeek-R1:7B", "llama": "LLaMA-3.2"}

SOTA = {
    "ag_news":    {"model": "Fine-tuned BERT [23]",    "metric": "Accuracy (%)", "score": 93.8},
    "dbpedia":    {"model": "XLNet [24]",              "metric": "Accuracy (%)", "score": 98.9},
    "goemotions": {"model": "Fine-tuned RoBERTa [26]", "metric": "Macro-F1",    "score": 0.49},
}


def detect_run(fname: str):
    f = fname.lower()
    if   "ag_news"    in f and "deepseek" in f: return "ag_news",    "deepseek"
    elif "ag_news"    in f and "llama"    in f: return "ag_news",    "llama"
    elif "dbpedia"    in f and "deepseek" in f: return "dbpedia",    "deepseek"
    elif "dbpedia"    in f and "llama"    in f: return "dbpedia",    "llama"
    elif "goemotions" in f and "deepseek" in f: return "goemotions", "deepseek"
    elif "goemotions" in f and "llama"    in f: return "goemotions", "llama"
    return None, None

def detect_k(fname: str):
    for k in [5, 3, 1]:
        if f"_k{k}" in fname or f"_k{k}." in fname:
            return k
    return None

def _ece(confidences, corrects, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for b_lo, b_hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= b_lo) & (confidences <= b_hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(corrects[mask].mean() - confidences[mask].mean())
    return float(ece)

def get_metrics(df: pd.DataFrame):
    df = df.dropna(subset=["gold"])
    covered  = df[df["pred"] != "ABSTAIN"].copy()
    total    = len(df)
    n_cov    = len(covered)
    coverage = n_cov / total if total else 0.0
    if n_cov == 0:
        return dict(acc=0, f1=0, mcc=0, ece=0, coverage=coverage, n=total, n_cov=0)
    y_true = covered["gold"].astype(str).tolist()
    y_pred = covered["pred"].astype(str).tolist()
    acc    = covered["correct"].mean() * 100
    labels = sorted(set(y_true + y_pred))
    mf1    = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    lbl_idx = {l: i for i, l in enumerate(labels)}
    yt_idx  = [lbl_idx[l] for l in y_true]
    yp_idx  = [lbl_idx[l] for l in y_pred]
    mcc_v   = matthews_corrcoef(yt_idx, yp_idx) if len(set(yt_idx)) > 1 else 0.0
    confs  = covered["confidence"].values
    corrs  = covered["correct"].astype(float).values
    ece    = _ece(confs, corrs, N_BINS) * 100
    return dict(acc=acc, f1=mf1, mcc=mcc_v, ece=ece,
                coverage=coverage * 100, n=total, n_cov=n_cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows3 = []

    for csv in sorted(runs_dir.glob("*.csv")):
        ds, model = detect_run(csv.name)
        k = detect_k(csv.name)
        if ds is None or k is None:
            continue
        df = pd.read_csv(csv)
        if (df["pred"] == "ABSTAIN").all():
            print(f"SKIPPING {csv.name} — 100% ABSTAIN (re-run after parser fix)")
            continue
        m = get_metrics(df)
        rows3.append({
            "Dataset":      DATASET_NICE[ds],
            "Model":        MODEL_NICE[model],
            "k":            k,
            "Accuracy (%)": f"{m['acc']:.2f}",
            "Macro-F1":     f"{m['f1']:.4f}",
            "MCC":          f"{m['mcc']:.4f}",
            "ECE (%)":      f"{m['ece']:.2f}",
            "Coverage (%)": f"{m['coverage']:.2f}",
            "N":            m["n"],
            "_ds": ds, "_model": model, "_k": k,
            "_acc_raw": m["acc"], "_f1_raw": m["f1"],
        })

    if not rows3:
        print("No valid run CSVs found.")
        return

    df3 = pd.DataFrame(rows3).sort_values(["Dataset", "Model", "k"])
    display = ["Dataset","Model","k","Accuracy (%)","Macro-F1","MCC","ECE (%)","Coverage (%)","N"]
    sep = "=" * 95
    print(f"\n{sep}\n  TABLE 3 — Main Results\n{sep}")
    print(df3[display].to_string(index=False))
    df3[display].to_csv("results_table3.csv", index=False)
    print(f"\n  Saved: results_table3.csv")

    print(f"\n{sep}\n  TABLE 4 — MMV k=5 vs SOTA\n{sep}")
    k5 = df3[df3["_k"] == 5]
    rows4 = []
    for ds_key, sota in SOTA.items():
        is_f1 = "F1" in sota["metric"]
        for model_key, model_nice in MODEL_NICE.items():
            row = k5[(k5["_ds"] == ds_key) & (k5["_model"] == model_key)]
            if row.empty:
                mmv_val, gap = "—", "—"
            else:
                raw = row["_f1_raw"].values[0] if is_f1 else row["_acc_raw"].values[0]
                mmv_val = f"{raw:.4f}" if is_f1 else f"{raw:.2f}%"
                gap = f"{sota['score'] - raw:+.2f}"
            rows4.append({
                "Dataset": DATASET_NICE[ds_key], "SOTA": sota["model"],
                f"SOTA Score": sota["score"], "MMV Model": model_nice,
                "MMV (k=5)": mmv_val, "Gap": gap,
            })
    df4 = pd.DataFrame(rows4)
    print(df4.to_string(index=False))
    df4.to_csv("results_table4.csv", index=False)
    print(f"\n  Saved: results_table4.csv")

    have = set((r["_ds"], r["_model"], r["_k"]) for r in rows3)
    missing = [(ds,m,k) for ds in ["ag_news","dbpedia","goemotions"]
               for m in ["deepseek","llama"] for k in [1,3,5] if (ds,m,k) not in have]
    print(f"\n{sep}")
    if missing:
        print(f"  MISSING ({len(missing)}/18) — run: python scripts/run_parallel.py")
        for ds,m,k in missing:
            print(f"    x  {DATASET_NICE[ds]} + {MODEL_NICE[m]} + k={k}")
    else:
        print("  ALL 18 RUNS COMPLETE")
    print(sep)

if __name__ == "__main__":
    main()
