#!/usr/bin/env bash
# run_all.sh — Sequential fallback runner for all 18 MMV experiments.
# Recommended: use run_parallel.py instead (faster).
#
# Usage:
#   cd /path/to/llm_majority_vote_ollama
#   pip install -e .
#   ollama serve  # in separate terminal
#   bash scripts/run_all.sh

set -euo pipefail

N=200
SEED=42
TEMP=0.7
MIN_ROWS=100

mkdir -p runs

run_exp() {
    local MODEL="$1"
    local TAG="$2"
    local DATASET="$3"
    local K="$4"
    local OUTFILE="runs/${DATASET}_ollama_${TAG}_k${K}.csv"

    if [ -f "$OUTFILE" ]; then
        ROWS=$(python -c "import pandas as pd; df=pd.read_csv('$OUTFILE'); print(len(df))" 2>/dev/null || echo 0)
        ALL_ABS=$(python -c "import pandas as pd; df=pd.read_csv('$OUTFILE'); print(int((df['pred']=='ABSTAIN').all()))" 2>/dev/null || echo 1)
        if [ "$ROWS" -ge "$MIN_ROWS" ] && [ "$ALL_ABS" -eq 0 ]; then
            echo "  SKIP  $OUTFILE  ($ROWS rows)"
            return
        else
            echo "  RERUN $OUTFILE  (rows=$ROWS, all-abstain=$ALL_ABS)"
            rm -f "$OUTFILE"
        fi
    fi

    echo "  RUN   $DATASET / $MODEL / k=$K ..."
    python scripts/eval_dataset.py \
        --provider ollama \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --k "$K" \
        --max-samples "$N" \
        --seed "$SEED" \
        --temperature "$TEMP" \
        --preds "$OUTFILE"
}

echo "========================================================"
echo "  Model: deepseek-r1:7b"
echo "========================================================"
for DS in ag_news dbpedia goemotions; do
    for K in 1 3 5; do
        run_exp "deepseek-r1:7b" "deepseek-r1-7b" "$DS" "$K"
    done
done

echo ""
echo "========================================================"
echo "  Model: llama3.2"
echo "========================================================"
for DS in ag_news dbpedia goemotions; do
    for K in 1 3 5; do
        run_exp "llama3.2" "llama3.2" "$DS" "$K"
    done
done

echo ""
echo "========================================================"
echo "  All done."
echo "  Next: python scripts/compute_results.py"
echo "        python scripts/make_plots.py"
echo "========================================================"
