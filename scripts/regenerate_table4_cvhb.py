"""
Post-fix regeneration of Table 4 (5-fold cross-validated histogram binning
(CV-HB) vs. raw MMV ECE on AG News), computed from scratch against the
post-fix per-sample records since the original run's CV-fold random state
was not preserved (Data Availability Statement). Pools the three post-fix
repeats' covered samples per condition, consistent with Figure 4's reliability
diagrams. Uses a newly documented seed (SEED=42) for the 5-fold split.

Fold grouping (fixed post-review): the pooled population contains each
underlying sample id up to three times (once per independent repeat). The
original implementation split this pooled array with plain StratifiedKFold,
which does not group by sample id, so a given sample's repeat-copies could
land in different folds -- e.g., one repeat's copy of sample id=7 in the
training fold and another repeat's copy of the same id=7 in the held-out
fold. Because a sample's repeats are correlated (same underlying text and
gold label, only the stochastic LLM votes differ), this is a real
train/test leakage risk, flagged during peer review. The fold split now
uses StratifiedGroupKFold, grouped by sample id, so all repeat-copies of a
given input always fall in the same fold together. This changed the
reported CV-HB ECE for three of six condition/k cells (see Data
Availability Statement and Section 7.2 for before/after values); the
qualitative conclusion, that CV-HB substantially reduces ECE relative to
raw MMV confidence, is unchanged.

Histogram-binning calibrator: 10 fixed-width bins over [0,1] fit on the
training folds' (confidence, correctness) pairs -- each bin's calibrated
output is the training folds' empirical accuracy within that bin, matching
the standard histogram-binning post-hoc calibration method. Evaluated with
the paper's standard 15-bin ECE (regenerate_all.ece) on the held-out folds'
pooled (calibrated_confidence, correctness) pairs.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import regenerate_3rep as r3
import regenerate_all as ra
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_SPLITS = 5
N_HB_BINS = 10


def cv_hb_calibrate(conf, correct, groups, seed=SEED, n_splits=N_SPLITS, n_hb_bins=N_HB_BINS):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    groups = np.asarray(groups)
    calibrated = np.full_like(conf, np.nan)
    bin_edges = np.linspace(0, 1, n_hb_bins + 1)
    # Grouped by sample id: all repeat-copies of a given input stay in the
    # same fold together, so no repeat of a sample can be used to calibrate
    # a held-out repeat of that same sample (see module docstring).
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(conf, correct, groups):
        train_conf, train_correct = conf[train_idx], correct[train_idx]
        train_bin = np.clip(np.digitize(train_conf, bin_edges) - 1, 0, n_hb_bins - 1)
        bin_acc = {}
        for b in range(n_hb_bins):
            m = train_bin == b
            if m.sum() > 0:
                bin_acc[b] = train_correct[m].mean()
        overall_train_acc = train_correct.mean()
        test_conf = conf[test_idx]
        test_bin = np.clip(np.digitize(test_conf, bin_edges) - 1, 0, n_hb_bins - 1)
        for i, idx in enumerate(test_idx):
            calibrated[idx] = bin_acc.get(test_bin[i], overall_train_acc)
    return calibrated


def pooled_covered(all_reps, ds, model, k):
    dfs = all_reps[(ds, model, k)]
    covered = [d[d["mmv_pred"] != "ABSTAIN"] for d in dfs]
    conf = np.concatenate([c["confidence_mmv"].to_numpy() for c in covered])
    correct = np.concatenate([c["mmv_correct"].to_numpy() for c in covered])
    ids = np.concatenate([c["id"].to_numpy() for c in covered])
    return conf, correct, ids


def main():
    all_reps = r3.load_all_reps("runs/three_repeat")
    rows = []
    for model in ("LLaMA-3.2:3B", "DeepSeek-R1:7B"):
        for k in (1, 3, 5):
            conf, correct, ids = pooled_covered(all_reps, "AG News", model, k)
            raw_ece = 100 * ra.ece(conf, correct, 15)
            n_conf_levels = len(np.unique(conf))
            cal = cv_hb_calibrate(conf, correct, ids)
            cvhb_ece = 100 * ra.ece(cal, correct, 15)
            rows.append((model, k, len(conf), n_conf_levels, raw_ece, cvhb_ece))
            print(f"{model:15s} k={k}  n={len(conf):5d}  conf_levels={n_conf_levels}  "
                  f"Raw ECE={raw_ece:6.2f}%  CV-HB ECE={cvhb_ece:6.2f}%")
    return rows


if __name__ == "__main__":
    main()
