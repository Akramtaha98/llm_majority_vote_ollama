"""
make_plots.py — Publication-quality figures for the MMV paper.
Generates Figure 1 (performance vs k), Figure 2 (ECE vs k), Figure 3 (confusion matrices).

Usage:
    python scripts/make_plots.py --runs_dir runs/ --out_dir figures/
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, ConfusionMatrixDisplay

N_BINS = 15
DATASET_NICE = {"ag_news": "AG News", "dbpedia": "DBpedia", "goemotions": "GoEmotions"}
MODEL_NICE   = {"deepseek": "DeepSeek-R1:7B", "llama": "LLaMA-3.2"}
MODEL_COLOR  = {"deepseek": "#2563EB", "llama": "#DC2626"}
MODEL_MARKER = {"deepseek": "o", "llama": "s"}
K_VALUES     = [1, 3, 5]


def detect_run(fname):
    f = fname.lower()
    if   "ag_news"    in f and "deepseek" in f: return "ag_news",    "deepseek"
    elif "ag_news"    in f and "llama"    in f: return "ag_news",    "llama"
    elif "dbpedia"    in f and "deepseek" in f: return "dbpedia",    "deepseek"
    elif "dbpedia"    in f and "llama"    in f: return "dbpedia",    "llama"
    elif "goemotions" in f and "deepseek" in f: return "goemotions", "deepseek"
    elif "goemotions" in f and "llama"    in f: return "goemotions", "llama"
    return None, None

def detect_k(fname):
    for k in [5, 3, 1]:
        if f"_k{k}" in fname: return k
    return None

def load_runs(runs_dir):
    data = {}
    for csv in sorted(runs_dir.glob("*.csv")):
        ds, model = detect_run(csv.name)
        k = detect_k(csv.name)
        if ds is None or k is None: continue
        df = pd.read_csv(csv)
        if (df["pred"] == "ABSTAIN").all(): continue
        data.setdefault(ds, {}).setdefault(model, {})[k] = df
    return data

def _ece(conf, corr, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf <= hi)
        if m.sum() == 0: continue
        ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return ece

def get_metrics(df):
    df = df.dropna(subset=["gold"])
    cov = df[df["pred"] != "ABSTAIN"]
    coverage = len(cov) / len(df) if len(df) else 0
    if len(cov) == 0:
        return dict(accuracy=0, macro_f1=0, mcc=0, ece=0, coverage=0)
    yt = cov["gold"].astype(str).tolist()
    yp = cov["pred"].astype(str).tolist()
    labels = sorted(set(yt + yp))
    mf1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    li = {l: i for i, l in enumerate(labels)}
    mcc_v = matthews_corrcoef([li[l] for l in yt], [li[l] for l in yp]) if len(set(yt)) > 1 else 0
    conf = cov["confidence"].values
    corr = cov["correct"].astype(float).values
    return dict(accuracy=cov["correct"].mean(), macro_f1=mf1, mcc=mcc_v,
                ece=_ece(conf, corr, N_BINS), coverage=coverage)


def fig1_performance(data, out_path):
    datasets = ["ag_news", "dbpedia", "goemotions"]
    metrics  = [("accuracy", "Accuracy (%)"), ("macro_f1", "Macro-F1"), ("mcc", "MCC")]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey="row")
    for di, ds in enumerate(datasets):
        for mi, (met, mlabel) in enumerate(metrics):
            ax = axes[di][mi]
            for model in ["deepseek", "llama"]:
                xs, ys = [], []
                for k in K_VALUES:
                    df_run = data.get(ds, {}).get(model, {}).get(k)
                    if df_run is None: continue
                    v = get_metrics(df_run)[met]
                    xs.append(k)
                    ys.append(v * 100 if met != "mcc" else v)
                if xs:
                    ax.plot(xs, ys, color=MODEL_COLOR[model], marker=MODEL_MARKER[model],
                            label=MODEL_NICE[model], lw=2, ms=7)
            ax.set_xticks(K_VALUES)
            ax.set_xlabel("k", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(axis="y", alpha=0.3, ls="--")
            ax.spines[["top","right"]].set_visible(False)
            if mi == 0: ax.set_ylabel(DATASET_NICE[ds], fontsize=11, fontweight="bold")
            if di == 0: ax.set_title(mlabel, fontsize=11, fontweight="bold")
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=2, fontsize=10,
                   bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Figure 1: Performance vs. Ensemble Size", fontsize=13,
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig2_ece(data, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for di, ds in enumerate(["ag_news", "dbpedia", "goemotions"]):
        ax = axes[di]
        for model in ["deepseek", "llama"]:
            xs, ys = [], []
            for k in K_VALUES:
                df_run = data.get(ds, {}).get(model, {}).get(k)
                if df_run is None: continue
                xs.append(k)
                ys.append(get_metrics(df_run)["ece"] * 100)
            if xs:
                ax.plot(xs, ys, color=MODEL_COLOR[model], marker=MODEL_MARKER[model],
                        label=MODEL_NICE[model], lw=2, ms=7)
        ax.set_title(DATASET_NICE[ds], fontsize=11, fontweight="bold")
        ax.set_xticks(K_VALUES)
        ax.set_xlabel("k", fontsize=10)
        if di == 0: ax.set_ylabel("ECE (%)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3, ls="--")
        ax.spines[["top","right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=2, fontsize=10,
                   bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Figure 2: Expected Calibration Error vs. Ensemble Size",
                 fontsize=12, fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig3_confusion(data, out_path):
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, k in zip(axes, [1, 5]):
        df_run = data.get("ag_news", {}).get("llama", {}).get(k)
        if df_run is None:
            ax.set_title(f"k={k} — data not available"); ax.axis("off"); continue
        cov = df_run[df_run["pred"] != "ABSTAIN"]
        acc = cov["correct"].mean() * 100 if len(cov) else 0
        cm = confusion_matrix(cov["gold"], cov["pred"], labels=labels)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
        ax.set_title(f"LLaMA-3.2 — k={k}  |  Acc = {acc:.1f}%",
                     fontsize=11, fontweight="bold")
    fig.suptitle("Figure 3: Confusion Matrices — k=1 vs k=5 (AG News)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir",  default="figures")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    data = load_runs(runs_dir)
    if not data:
        print(f"No run CSVs found in {runs_dir}. Run experiments first.")
        return
    fig1_performance(data, out_dir / "fig1_performance_vs_k.pdf")
    fig2_ece(        data, out_dir / "fig2_ece_vs_k.pdf")
    fig3_confusion(  data, out_dir / "fig3_confusion_matrices.pdf")
    print(f"\nAll figures saved to {out_dir}/")

if __name__ == "__main__":
    main()
