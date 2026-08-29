set -euo pipefail

REPS="${REPS:-2}"
SEED=42
TEMP=0.7
MAXTOK=8
OUTDIR="runs/reviewer_r1_reruns"
mkdir -p "$OUTDIR"

run_exp() {
    local MODEL="$1" DATASET="$2" K="$3" N="$4" SHUFFLE_FLAG="$5" TAG="$6" REP="$7"
    local OUTFILE="${OUTDIR}/${TAG}_k${K}_rep${REP}.csv"
    if [ -f "$OUTFILE" ]; then
        echo "  SKIP  $OUTFILE (already exists)"
        return
    fi
    echo "  RUN   $TAG / k=$K / rep=$REP  ->  $OUTFILE"
    python -m scripts.eval_dataset \
        --provider ollama --model "$MODEL" --dataset "$DATASET" \
        --k "$K" --max-samples "$N" --seed "$SEED" \
        --temperature "$TEMP" --max-tokens "$MAXTOK" \
        $SHUFFLE_FLAG --preds "$OUTFILE"
}

run_group_ag_news_deepseek() {
    local REP="$1"
    run_exp "deepseek-r1:7b" "ag_news" 1 300 "" "ag_news_deepseek_matched" "$REP"
    run_exp "deepseek-r1:7b" "ag_news" 3 300 "" "ag_news_deepseek_matched" "$REP"
    run_exp "deepseek-r1:7b" "ag_news" 5 300 "" "ag_news_deepseek_matched" "$REP"
}

run_group_dbpedia() {
    local REP="$1"
    run_exp "deepseek-r1:7b" "dbpedia" 1 300 "" "dbpedia_deepseek" "$REP"
    run_exp "deepseek-r1:7b" "dbpedia" 3 300 "" "dbpedia_deepseek" "$REP"
    run_exp "deepseek-r1:7b" "dbpedia" 5 300 "" "dbpedia_deepseek" "$REP"
}

run_group_goemotions() {
    local REP="$1"
    run_exp "deepseek-r1:7b" "goemotions" 1 300 "" "goemotions_deepseek" "$REP"
    run_exp "deepseek-r1:7b" "goemotions" 3 300 "" "goemotions_deepseek" "$REP"
    run_exp "deepseek-r1:7b" "goemotions" 5 300 "" "goemotions_deepseek" "$REP"
}

run_group_ag_news_llama() {
    local REP="$1"
    run_exp "llama3.2" "ag_news" 1 1000 "--no-shuffle" "ag_news_llama3.2" "$REP"
    run_exp "llama3.2" "ag_news" 3 1000 "--no-shuffle" "ag_news_llama3.2" "$REP"
    run_exp "llama3.2" "ag_news" 5 1000 "--no-shuffle" "ag_news_llama3.2" "$REP"
}

for REP in $(seq 1 "$REPS"); do
    echo "============================================================"
    echo "  Repeat $REP / $REPS  (4 groups running concurrently)"
    echo "============================================================"

    run_group_ag_news_llama "$REP" &
    PID_LLAMA=$!
    run_group_ag_news_deepseek "$REP" &
    PID_AGNEWS=$!
    run_group_dbpedia "$REP" &
    PID_DBPEDIA=$!
    run_group_goemotions "$REP" &
    PID_GOEMOTIONS=$!

    wait "$PID_LLAMA" "$PID_AGNEWS" "$PID_DBPEDIA" "$PID_GOEMOTIONS"
    echo "  Repeat $REP complete."
done

echo ""
echo "============================================================"
echo "  All reruns complete. Results in: $OUTDIR"
echo "  Next: tar czf reviewer_r1_reruns.tar.gz $OUTDIR"
echo "============================================================"
