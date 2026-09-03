#!/usr/bin/env bash
# reproduce_paper.sh -- single top-level driver that regenerates every table and
# figure in the current (post-fix, 3-repeat) manuscript from the released raw
# per-sample data, in the correct order, and reports which script produced what.
#
# This does NOT re-run any LLM calls -- it recomputes every reported statistic
# from the already-released per-sample vote-count records and repeat-run CSVs
# under vote_records/ and runs/reviewer_r1_reruns/. To regenerate those raw
# per-sample records themselves (i.e. to re-run the model), see the
# "Reproducing the paper's results" section of the top-level README.
#
# Usage: bash scripts/reproduce_paper.sh [OUT_DIR]
#   OUT_DIR defaults to outputs/regenerated (created if missing).
#
# Must be run from the repository root (it cd's there itself, so it also works
# if invoked from elsewhere via `bash /path/to/scripts/reproduce_paper.sh`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

OUT_DIR="${1:-outputs/regenerated}"
mkdir -p "$OUT_DIR"
mkdir -p out_figs  # regenerate_figure6.py writes here unconditionally

echo "=== Table/figure -> script mapping (this run) ==="
cat <<'EOF'
  Table 1  (Mean(3) +/- SD(3) accuracy/ECE/F1/MCC/coverage, 12 conditions) -> regenerate_3rep.py
  Table 2  (parser-failure / abstention decomposition)                    -> regenerate_3rep.py
  Table 4  (cross-validated histogram-binning calibration)                -> regenerate_table4_cvhb.py
  Table 5  (covered-accuracy cost-benefit analysis)                       -> regenerate_3rep.py
  Table 7  (MMV vs. self-consistency, matched-coverage / AURC)            -> regenerate_3rep.py
  Table 8  (per-repeat accuracy, Rep1/Rep2/Rep3, Mean(3)/Range(3))        -> compute_repeat_metrics.py
  Table 9  (per-repeat coverage/ECE/Macro-F1/MCC, 36 rows)                -> compute_repeat_metrics.py
  Figure 2 (accuracy vs. k)                                               -> make_figures_3rep.py
  Figure 3 (ECE vs. k)                                                    -> make_figures_3rep.py
  Figure 4 (reliability diagrams)                                        -> make_figures_3rep.py
  Figure 5 (vote agreement distribution, AG News DeepSeek-R1:7B)          -> make_figures_3rep.py
  Figure 6 (confusion matrix)                                             -> regenerate_figure6.py
  Figure 7 (risk-coverage curves, SC + MMV operating points)              -> regenerate_3rep.py
  McNemar / Bonferroni significance tests (36 tests, Section 7.5)         -> regenerate_significance_3rep.py
  Table 1's own per-run k=1 AURC==ECE identity check                      -> regenerate_3rep.py (assertion)

  Single-run, pre-fix diagnostic only (not the paper's reported numbers):
  Table 1/2/5/7 from a single un-repeated run, any data-dir                -> regenerate_all.py
EOF
echo

echo "=== [1/6] Table 1, 2, 5, 7 and Figure 7 (regenerate_3rep.py) ==="
python3 scripts/regenerate_3rep.py --data-dir runs/reviewer_r1_reruns --out-dir "$OUT_DIR"
echo

echo "=== [2/6] Figures 2, 3, 4, 5 (make_figures_3rep.py) ==="
python3 scripts/make_figures_3rep.py "$OUT_DIR"
echo

echo "=== [3/6] Table 8, Table 9 (compute_repeat_metrics.py) ==="
python3 scripts/compute_repeat_metrics.py \
  --reruns-dir runs/reviewer_r1_reruns \
  --vote-dir vote_records/reviewer_data_package/per_sample_vote_count_records \
  --csv-out "$OUT_DIR/table9_repeat_metrics.csv"
echo

echo "=== [4/6] Table 4 -- cross-validated histogram binning (regenerate_table4_cvhb.py) ==="
python3 scripts/regenerate_table4_cvhb.py
echo

echo "=== [5/6] Figure 6 -- confusion matrix (regenerate_figure6.py) ==="
python3 scripts/regenerate_figure6.py
echo

echo "=== [6/6] McNemar / Bonferroni significance tests (regenerate_significance_3rep.py) ==="
python3 scripts/regenerate_significance_3rep.py
echo

echo "=== Done. Outputs written to: $OUT_DIR (tables/Figures 2,3,4,5,7) and out_figs/ (Figure 6). ==="
echo "Table 4 and the significance-test script print their results to stdout above;"
echo "redirect this script's output to a file if you want a saved log."
