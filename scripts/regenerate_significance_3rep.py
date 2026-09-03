"""
Post-fix, 3-repeat re-derivation of the paper's McNemar significance tests
(Section 4.3 / 7.5), extending regenerate_significance.py (which only
reproduces the original pre-fix single-run p-values) to the post-fix
runs/reviewer_r1_reruns/ data.

Design: all four dataset-model pairs are now sample-matched/nested across
k = 1, 3, 5 (AG News DeepSeek-R1:7B's k = 1 baseline was rematched to the
k = 3/5 n = 300 sample in this revision), giving 4 pairs x 3 within-condition
comparisons (k1 vs k3, k1 vs k5, k3 vs k5) = 12 valid paired comparisons.
Each of the 3 independent post-fix repeats provides its own valid paired
sample (same items, independent generation draws), so we run McNemar's exact
test separately per repeat rather than pooling repeats into one contingency
table (pooling would treat 3 non-independent draws of the same 300/1000 items
as 900/3000 independent observations, inflating power spuriously).
"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import regenerate_3rep as r3
from statsmodels.stats.contingency_tables import mcnemar

N_TESTS = 12
BONF_ALPHA = 0.05 / N_TESTS

COMPARISONS = [
    ("AG News", "LLaMA-3.2:3B", 1, 3),
    ("AG News", "LLaMA-3.2:3B", 1, 5),
    ("AG News", "LLaMA-3.2:3B", 3, 5),
    ("AG News", "DeepSeek-R1:7B", 1, 3),
    ("AG News", "DeepSeek-R1:7B", 1, 5),
    ("AG News", "DeepSeek-R1:7B", 3, 5),
    ("DBpedia", "DeepSeek-R1:7B", 1, 3),
    ("DBpedia", "DeepSeek-R1:7B", 1, 5),
    ("DBpedia", "DeepSeek-R1:7B", 3, 5),
    ("GoEmotions", "DeepSeek-R1:7B", 1, 3),
    ("GoEmotions", "DeepSeek-R1:7B", 1, 5),  # PRIMARY
    ("GoEmotions", "DeepSeek-R1:7B", 3, 5),
]

def mcnemar_pair(d_a, d_b):
    a = d_a.set_index("id")[["mmv_pred", "mmv_correct"]]
    b = d_b.set_index("id")[["mmv_pred", "mmv_correct"]]
    both = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    covered = both[(both["mmv_pred_a"] != "ABSTAIN") & (both["mmv_pred_b"] != "ABSTAIN")]
    n = len(covered)
    aa = int(((covered["mmv_correct_a"] == 1) & (covered["mmv_correct_b"] == 1)).sum())
    bb = int(((covered["mmv_correct_a"] == 1) & (covered["mmv_correct_b"] == 0)).sum())
    cc = int(((covered["mmv_correct_a"] == 0) & (covered["mmv_correct_b"] == 1)).sum())
    dd = int(((covered["mmv_correct_a"] == 0) & (covered["mmv_correct_b"] == 0)).sum())
    p = mcnemar([[aa, bb], [cc, dd]], exact=True).pvalue
    return n, p, bb, cc

def main():
    all_reps = r3.load_all_reps("runs/reviewer_r1_reruns")
    print(f"{'Comparison':45s} {'rep':4s} {'n':5s} {'p':8s} {'b(a-only)':10s} {'c(b-only)':10s} sig? sig(bonf)?")
    print("=" * 100)
    for ds, model, ka, kb in COMPARISONS:
        reps_a = all_reps[(ds, model, ka)]
        reps_b = all_reps[(ds, model, kb)]
        label = f"{ds} {model} k={ka} vs k={kb}"
        flag = "  [PRIMARY]" if (ds, model, ka, kb) == ("GoEmotions", "DeepSeek-R1:7B", 1, 5) else ""
        for rep in range(3):
            n, p, b, c = mcnemar_pair(reps_a[rep], reps_b[rep])
            sig = "sig" if p < 0.05 else "n.s."
            sigb = "sig" if p < BONF_ALPHA else "n.s."
            print(f"{label:45s} {rep+1:<4d} {n:<5d} {p:<8.4f} {b:<10d} {c:<10d} {sig:5s}{sigb:8s}{flag}")
        print()

if __name__ == "__main__":
    main()
