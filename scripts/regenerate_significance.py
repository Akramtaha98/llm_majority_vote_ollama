"""
scripts/regenerate_significance.py -- Auditable reproduction of the paper's
statistical-significance testing (Section 4.3 / 7.4) and the GoEmotions
any-listed-gold-label sensitivity analysis (Section 4.1), directly from the
released per-sample vote-count CSVs in
vote_records/reviewer_data_package/per_sample_vote_count_records/.

This extends scripts/regenerate_all.py (which reproduces Tables 1, 2, 5, and 7)
to also reproduce:
  1. All 10 valid within-condition McNemar exact tests described in Section 4.3,
     including the pre-specified primary comparison (GoEmotions DeepSeek-R1:7B,
     k = 1 vs. k = 5) and the Bonferroni-corrected significance threshold used
     in Section 7.4.
  2. The GoEmotions any-listed-gold-label sensitivity analysis reported in
     Section 4.1 (macro-F1 / MCC computed against whichever gold label the
     prediction matches, not only the first-listed one).

No numbers are hand-typed anywhere downstream of this script for these two
analyses; both are derived entirely from the per-sample vote-count records,
using the identical MMV prediction rule implemented in regenerate_all.py
(strict majority of the ORIGINAL fixed k, else ABSTAIN).

Usage: python3 regenerate_significance.py [--data-dir DIR]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar

GOEMO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise",
]

MCNEMAR_PAIRS = [
    ("AG News LLaMA-3.2:3B", 1, 3, "ag_news_ollama_llama3.2_k1.csv", "ag_news_ollama_llama3.2_k3.csv"),
    ("AG News LLaMA-3.2:3B", 1, 5, "ag_news_ollama_llama3.2_k1.csv", "ag_news_ollama_llama3.2_k5.csv"),
    ("AG News LLaMA-3.2:3B", 3, 5, "ag_news_ollama_llama3.2_k3.csv", "ag_news_ollama_llama3.2_k5.csv"),
    ("DBpedia DeepSeek-R1:7B", 1, 3, "dbpedia_ollama_deepseek-r1-7b_k1.csv", "dbpedia_ollama_deepseek-r1-7b_k3.csv"),
    ("DBpedia DeepSeek-R1:7B", 1, 5, "dbpedia_ollama_deepseek-r1-7b_k1.csv", "dbpedia_ollama_deepseek-r1-7b_k5.csv"),
    ("DBpedia DeepSeek-R1:7B", 3, 5, "dbpedia_ollama_deepseek-r1-7b_k3.csv", "dbpedia_ollama_deepseek-r1-7b_k5.csv"),
    ("GoEmotions DeepSeek-R1:7B", 1, 3, "goemotions_ollama_deepseek-r1-7b_k1.csv", "goemotions_ollama_deepseek-r1-7b_k3.csv"),
    ("GoEmotions DeepSeek-R1:7B", 1, 5, "goemotions_ollama_deepseek-r1-7b_k1.csv", "goemotions_ollama_deepseek-r1-7b_k5.csv"),  # PRIMARY
    ("GoEmotions DeepSeek-R1:7B", 3, 5, "goemotions_ollama_deepseek-r1-7b_k3.csv", "goemotions_ollama_deepseek-r1-7b_k5.csv"),
    ("AG News DeepSeek-R1:7B", 3, 5, "ag_news_ollama_deepseek-r1-7b_k3.csv", "ag_news_ollama_deepseek-r1-7b_k5.csv"),
    # NOTE: AG News DeepSeek-R1:7B k = 1 is intentionally excluded from k = 1 vs. k = 3/5
    # comparisons -- its k = 1 baseline (n = 1,000) is not nested with the k = 3/5 sample
    # (n = 300), so no valid paired McNemar test exists for those pairs (Section 4.3, 7.5).
]

N_MCNEMAR_TESTS = len(MCNEMAR_PAIRS)
BONFERRONI_ALPHA = 0.05 / N_MCNEMAR_TESTS


def mmv_predictions(path: Path) -> pd.DataFrame:
    """Derive MMV predictions/correctness per sample, identical rule to
    regenerate_all.py: strict majority of the original fixed k, else ABSTAIN."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        votes = json.loads(r["votes"])
        label_order = list(votes.keys())
        total_valid = sum(votes.values())
        gold_multi = str(r.get("gold_multi", r["gold"])).split("|")
        gold_primary = r["gold"]
        k = r["K"]
        if total_valid == 0:
            mmv_pred = None
        else:
            max_v = max(votes.values())
            sc_pred = next(lbl for lbl in label_order if votes[lbl] == max_v)
            mmv_pred = sc_pred if max_v > k / 2 else None
        correct = int(mmv_pred in set(gold_multi)) if mmv_pred is not None else 0
        rows.append(dict(id=r["id"], mmv_pred=mmv_pred, correct=correct,
                          gold_primary=gold_primary, gold_multi=gold_multi))
    return pd.DataFrame(rows).set_index("id")


def run_mcnemar(data_dir: Path):
    print("=" * 78)
    print(f"McNemar exact tests (Section 4.3) -- {N_MCNEMAR_TESTS} valid within-condition "
          f"comparisons; Bonferroni alpha' = 0.05/{N_MCNEMAR_TESTS} = {BONFERRONI_ALPHA:.4f}")
    print("=" * 78)
    for dataset, k_a, k_b, file_a, file_b in MCNEMAR_PAIRS:
        da = mmv_predictions(data_dir / file_a)
        db = mmv_predictions(data_dir / file_b)
        both = da.join(db, lsuffix="_a", rsuffix="_b", how="inner")
        covered = both[both["mmv_pred_a"].notna() & both["mmv_pred_b"].notna()]
        n = len(covered)
        a = int(((covered["correct_a"] == 1) & (covered["correct_b"] == 1)).sum())
        b = int(((covered["correct_a"] == 1) & (covered["correct_b"] == 0)).sum())
        c = int(((covered["correct_a"] == 0) & (covered["correct_b"] == 1)).sum())
        d = int(((covered["correct_a"] == 0) & (covered["correct_b"] == 0)).sum())
        p = mcnemar([[a, b], [c, d]], exact=True).pvalue
        flag = "  [PRIMARY, pre-specified]" if (dataset, k_a, k_b) == ("GoEmotions DeepSeek-R1:7B", 1, 5) else ""
        sig = "sig (uncorrected, p<0.05)" if p < 0.05 else "n.s."
        sig_bonf = "sig (Bonferroni)" if p < BONFERRONI_ALPHA else "n.s. (Bonferroni)"
        print(f"{dataset} k={k_a} vs k={k_b}: n={n}, p={p:.4f} -- {sig}, {sig_bonf}{flag}")


def run_goemotions_sensitivity(data_dir: Path):
    print()
    print("=" * 78)
    print("GoEmotions any-listed-gold-label sensitivity analysis (Section 4.1)")
    print("=" * 78)
    for k, fname in [(1, "goemotions_ollama_deepseek-r1-7b_k1.csv"),
                      (3, "goemotions_ollama_deepseek-r1-7b_k3.csv"),
                      (5, "goemotions_ollama_deepseek-r1-7b_k5.csv")]:
        df = mmv_predictions(data_dir / fname)
        covered = df[df["mmv_pred"].notna()].copy()
        y_pred = covered["mmv_pred"]

        y_true_first = covered["gold_primary"]
        f1_first = f1_score(y_true_first, y_pred, labels=GOEMO_LABELS, average="macro", zero_division=0)
        mcc_first = matthews_corrcoef(y_true_first, y_pred)

        def eff_true(row):
            return row["mmv_pred"] if row["mmv_pred"] in row["gold_multi"] else row["gold_primary"]
        y_true_any = covered.apply(eff_true, axis=1)
        f1_any = f1_score(y_true_any, y_pred, labels=GOEMO_LABELS, average="macro", zero_division=0)
        mcc_any = matthews_corrcoef(y_true_any, y_pred)

        affected = int(covered.apply(
            lambda r: r["mmv_pred"] in r["gold_multi"] and r["mmv_pred"] != r["gold_primary"], axis=1
        ).sum())
        pct = 100 * affected / len(covered) if len(covered) else 0.0

        print(f"k={k}: n_covered={len(covered)}, "
              f"macro-F1 first={f1_first:.4f} any={f1_any:.4f} (delta {f1_any - f1_first:+.4f}), "
              f"MCC first={mcc_first:.4f} any={mcc_any:.4f} (delta {mcc_any - mcc_first:+.4f}), "
              f"affected={affected} ({pct:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="vote_records/reviewer_data_package/per_sample_vote_count_records",
                     help="Directory containing the released per-sample vote-count CSVs")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    run_mcnemar(data_dir)
    run_goemotions_sensitivity(data_dir)


if __name__ == "__main__":
    main()
