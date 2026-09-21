#!/usr/bin/env bash
# Category D script 1/3 -- re-collect the two excluded LLaMA-3.2:3B conditions
# (DBpedia, stratified across all 14 classes; GoEmotions, with a gold-label
# diversity check) that Reviewer 3 says must be re-run rather than excluded.
#
# PREREQUISITES (already applied to the repo as of this delivery -- both
# src/llm_vote/datasets.py (load_dbpedia_stratified function, added right
# after load_dbpedia) and scripts/eval_dataset.py (--stratified flag, wired
# into the dbpedia dispatch branch) already contain this code; nothing to
# paste in manually. Only remaining prerequisite:
#   1. Confirm Ollama is running locally with llama3.2:3b-instruct pulled:
#        ollama pull llama3.2:3b-instruct
#
# WHAT THIS SCRIPT DOES:
#   - DBpedia: draws ONE stratified 300-sample set (seed=42, ~21-22 per class)
#     and reuses the identical sample across all 3 repeats and all 3 k values,
#     consistent with this paper's existing methodology of holding the input
#     sample fixed per condition while letting only generation stochasticity
#     vary across repeats (Section 4.4). Runs k = 1, 3, 5, x 3 repeats = 9 runs.
#   - GoEmotions: reuses the existing load_goemotions loader (which is known
#     to work correctly -- the parallel DeepSeek-R1:7B run using this same
#     loader produced a genuinely diverse, correct gold-label distribution).
#     The original LLaMA failure was a data-retention bug downstream of
#     collection, not a loader bug, so no sampling change is needed here --
#     only the post-hoc validation check below, which is new.
#   - After each GoEmotions run, validates that at least 10 distinct primary
#     gold labels appear in the output CSV before accepting the run. If this
#     check fails, the run is almost certainly hitting the same bug as
#     before and must be investigated rather than used.
#
# ESTIMATED COST: 18 total (dataset x k x repeat) runs x up to 300 calls each
# (k=5 = 1500 calls per repeat for that k) -- budget real wall-clock time on
# a single local GPU; this is comparable in scale to the original 3-repeat
# reviewer_r1_reruns collection already in this repo.

set -euo pipefail
cd "$(dirname "$0")/.."  # run from repo root

OUT_DIR="runs/reviewer3_llama_reruns"
mkdir -p "$OUT_DIR"

MODEL="llama3.2:3b-instruct"
SEED=42

echo "=== DBpedia (stratified, LLaMA-3.2:3B) ==="
for K in 1 3 5; do
  for REP in 1 2 3; do
    OUT="$OUT_DIR/dbpedia_llama3.2_stratified_k${K}_rep${REP}.csv"
    if [ -f "$OUT" ]; then
      echo "skip (exists): $OUT"
      continue
    fi
    echo "--- DBpedia k=$K rep=$REP ---"
    python3 scripts/eval_dataset.py \
      --provider ollama --model "$MODEL" \
      --dataset dbpedia --stratified \
      --k "$K" --max-samples 300 --seed "$SEED" \
      --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 --num-ctx 4096 \
      --preds "$OUT"
  done
done

echo
echo "=== GoEmotions (LLaMA-3.2:3B, with gold-label diversity check) ==="
for K in 1 3 5; do
  for REP in 1 2 3; do
    OUT="$OUT_DIR/goemotions_llama3.2_k${K}_rep${REP}.csv"
    if [ -f "$OUT" ]; then
      echo "skip (exists): $OUT"
      continue
    fi
    echo "--- GoEmotions k=$K rep=$REP ---"
    python3 scripts/eval_dataset.py \
      --provider ollama --model "$MODEL" \
      --dataset goemotions \
      --k "$K" --max-samples 300 --seed "$SEED" \
      --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 --num-ctx 4096 \
      --preds "$OUT"

    N_DISTINCT=$(python3 -c "
import pandas as pd
d = pd.read_csv('$OUT')
gold1 = d['gold'].astype(str).str.split('|').str[0]
n = gold1.nunique()
print(n)
")
    echo "    distinct primary gold labels in this run: $N_DISTINCT"
    if [ "$N_DISTINCT" -lt 10 ]; then
      echo "    !!! VALIDATION FAILED: fewer than 10 distinct gold labels."
      echo "    !!! This looks like the same data-retention bug as the original"
      echo "    !!! exclusion (Section 7.6). Do NOT use this file -- investigate"
      echo "    !!! the eval_dataset.py -> CSV write path before re-running."
      mv "$OUT" "${OUT}.SUSPECT"
    fi
  done
done

echo
echo "Done. New per-sample CSVs are in $OUT_DIR/"
echo "Next: run scripts/regenerate_3rep.py-style aggregation against this"
echo "directory to compute Accuracy/MacroF1/MCC/ECE/Coverage per condition,"
echo "then decide with Akram whether to fold these two conditions into Table 1"
echo "as a genuine 18-condition table, or keep them as a separate appendix"
echo "table if results look inconsistent with the 12 already-verified conditions."
