#!/usr/bin/env python3
"""Recompute coverage, accuracy, ECE, Macro-F1, and MCC per repeat, directly from the
raw per-sample repeat-run records in runs/reviewer_r1_reruns/, plus the original
per-sample vote-count records (for GoEmotions multi-label gold crediting).

Reproduces Table 8's Rep 1 / Rep 2 accuracy columns exactly (cross-checked against the
manuscript) and produces Table 9's coverage / ECE / Macro-F1 / MCC columns in full.

Usage:
    python3 scripts/compute_repeat_metrics.py \
        --reruns-dir runs/reviewer_r1_reruns \
        --vote-dir vote_records/reviewer_data_package/per_sample_vote_count_records
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

GOEMO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise",
]
AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
DBPEDIA_LABELS = [
    "Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
    "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal",
    "Plant", "Album", "Film", "WrittenWork",
]

CONDITIONS = [
    # (dataset, model_label, k, file_prefix, labels, multi_label_dataset)
    ("AG News", "LLaMA-3.2:3B", 1, "ag_news_llama3.2_k1", AGNEWS_LABELS, False),
    ("AG News", "LLaMA-3.2:3B", 3, "ag_news_llama3.2_k3", AGNEWS_LABELS, False),
    ("AG News", "LLaMA-3.2:3B", 5, "ag_news_llama3.2_k5", AGNEWS_LABELS, False),
    ("AG News", "DeepSeek-R1:7B (matched, n=300)", 1, "ag_news_deepseek_matched_k1", AGNEWS_LABELS, False),
    ("AG News", "DeepSeek-R1:7B (matched, n=300)", 3, "ag_news_deepseek_matched_k3", AGNEWS_LABELS, False),
    ("AG News", "DeepSeek-R1:7B (matched, n=300)", 5, "ag_news_deepseek_matched_k5", AGNEWS_LABELS, False),
    ("DBpedia", "DeepSeek-R1:7B", 1, "dbpedia_deepseek_k1", DBPEDIA_LABELS, False),
    ("DBpedia", "DeepSeek-R1:7B", 3, "dbpedia_deepseek_k3", DBPEDIA_LABELS, False),
    ("DBpedia", "DeepSeek-R1:7B", 5, "dbpedia_deepseek_k5", DBPEDIA_LABELS, False),
    ("GoEmotions", "DeepSeek-R1:7B", 1, "goemotions_deepseek_k1", GOEMO_LABELS, True),
    ("GoEmotions", "DeepSeek-R1:7B", 3, "goemotions_deepseek_k3", GOEMO_LABELS, True),
    ("GoEmotions", "DeepSeek-R1:7B", 5, "goemotions_deepseek_k5", GOEMO_LABELS, True),
]


def ece(conf, correct, n_bins: int = 15) -> float:
    """15-bin expected calibration error, matching src/llm_vote/metrics.py's convention:
    first bin is [0, hi] inclusive on both ends, subsequent bins are (lo, hi]."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = len(conf)
    if total == 0:
        return float("nan")
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi) if lo == 0.0 else (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        e += (mask.sum() / total) * abs(correct[mask].mean() - conf[mask].mean())
    return e * 100


def mmv_from_votes(row) -> tuple[str | None, float]:
    """Strict-majority MMV rule applied to a raw votes JSON column: predict the
    plurality label only if its vote count exceeds k/2, else abstain."""
    votes = json.loads(row["votes"])
    label_order = list(votes.keys())
    k = row["K"]
    if not votes:
        return None, 0.0
    max_v = max(votes.values())
    sc_pred = next(l for l in label_order if votes[l] == max_v)
    if max_v > k / 2:
        return sc_pred, max_v / k
    return None, 0.0


def process_condition(path: str, labels: list[str], gold_multi_map: dict | None) -> dict:
    df = pd.read_csv(path)
    preds, confs, corrects = [], [], []
    y_true_f1, y_pred_f1 = [], []
    for _, r in df.iterrows():
        pred, conf = mmv_from_votes(r)
        if gold_multi_map is not None:
            gold_set = set(str(gold_multi_map.get(r["id"], r["gold"])).split("|"))
        else:
            gold_set = {r["gold"]}
        correct = int(pred in gold_set) if pred is not None else 0
        preds.append(pred)
        confs.append(conf)
        corrects.append(correct)
        if pred is not None:
            y_pred_f1.append(pred)
            y_true_f1.append(r["gold"])  # first-listed gold, matching Table 1's convention
    covered = [i for i, p in enumerate(preds) if p is not None]
    n_total, n_cov = len(df), len(covered)
    return dict(
        n_total=n_total,
        n_cov=n_cov,
        coverage=100 * n_cov / n_total if n_total else float("nan"),
        accuracy=100 * sum(corrects[i] for i in covered) / n_cov if n_cov else float("nan"),
        ece=ece([confs[i] for i in covered], [corrects[i] for i in covered]) if n_cov else float("nan"),
        macro_f1=f1_score(y_true_f1, y_pred_f1, labels=labels, average="macro", zero_division=0) if n_cov else float("nan"),
        mcc=matthews_corrcoef(y_true_f1, y_pred_f1) if n_cov > 1 else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reruns-dir", default="runs/reviewer_r1_reruns")
    ap.add_argument("--vote-dir", default="vote_records/reviewer_data_package/per_sample_vote_count_records")
    ap.add_argument("--csv-out", default=None, help="optional path to write results as CSV")
    args = ap.parse_args()

    # GoEmotions multi-label gold map, keyed by sample id, recovered from the original
    # per-sample vote-count records (the rerun files only carry the first-listed gold).
    goemo_gm_map = {}
    for k in (1, 3, 5):
        gp = os.path.join(args.vote_dir, f"goemotions_ollama_deepseek-r1-7b_k{k}.csv")
        if os.path.exists(gp):
            dfo = pd.read_csv(gp)
            for _, r in dfo.iterrows():
                goemo_gm_map[r["id"]] = r["gold_multi"]

    rows = []
    header = f"{'Dataset':11s} {'Model':30s} {'k':2s} {'Rep':4s} {'Cov%':6s} {'Acc%':6s} {'ECE%':6s} {'F1':6s} {'MCC':6s}"
    print(header)
    for dataset, model, k, prefix, labels, is_multi in CONDITIONS:
        gm = goemo_gm_map if is_multi else None
        for rep in (1, 2):
            path = os.path.join(args.reruns_dir, f"{prefix}_rep{rep}.csv")
            res = process_condition(path, labels, gm)
            rows.append(dict(dataset=dataset, model=model, k=k, rep=rep, **res))
            print(f"{dataset:11s} {model:30s} {k:<2d} rep{rep:<3d} {res['coverage']:5.1f} "
                  f"{res['accuracy']:6.2f} {res['ece']:6.2f} {res['macro_f1']:6.3f} {res['mcc']:6.3f}")

    if args.csv_out:
        pd.DataFrame(rows).to_csv(args.csv_out, index=False)
        print(f"\nWrote {args.csv_out}")


if __name__ == "__main__":
    main()
