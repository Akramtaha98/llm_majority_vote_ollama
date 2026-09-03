set -euo pipefail

REPS="${REPS:-2}"
SEED=42
TEMP=0.7
OUTDIR="runs/reviewer_r1_reruns"
TIMING_LOG="${OUTDIR}/timing.log"
mkdir -p "$OUTDIR"

fmt_hms() {
    local T="$1"
    printf '%02dh:%02dm:%02ds' $((T/3600)) $(((T%3600)/60)) $((T%60))
}

run_exp() {
    local MODEL="$1" DATASET="$2" K="$3" N="$4" SHUFFLE_FLAG="$5" TAG="$6" REP="$7"
    local OUTFILE="${OUTDIR}/${TAG}_k${K}_rep${REP}.csv"
    if [ -f "$OUTFILE" ]; then
        echo "  SKIP  $OUTFILE (already exists)"
        return
    fi
    local START_TS START_HUMAN ELAPSED
    START_TS=$(date +%s)
    START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
    echo "  RUN   $TAG / k=$K / rep=$REP  ->  $OUTFILE   [started $START_HUMAN]"
    python -m scripts.eval_dataset \
        --provider ollama --model "$MODEL" --dataset "$DATASET" \
        --k "$K" --max-samples "$N" --seed "$SEED" \
        --temperature "$TEMP" \
        $SHUFFLE_FLAG --save-raw-outputs --preds "$OUTFILE"
    ELAPSED=$(( $(date +%s) - START_TS ))
    local LINE="  DONE  $TAG / k=$K / rep=$REP  ->  $OUTFILE   [took $(fmt_hms $ELAPSED)]"
    echo "$LINE"
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $TAG k=$K rep=$REP  ${ELAPSED}s  $(fmt_hms $ELAPSED)" >> "$TIMING_LOG"
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

SCRIPT_START_TS=$(date +%s)

for REP in $(seq 1 "$REPS"); do
    REP_START_TS=$(date +%s)
    echo "============================================================"
    echo "  Repeat $REP / $REPS  (4 groups running concurrently)   [started $(date '+%Y-%m-%d %H:%M:%S')]"
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
    REP_ELAPSED=$(( $(date +%s) - REP_START_TS ))
    echo "  Repeat $REP complete.   [took $(fmt_hms $REP_ELAPSED)]"
    echo "$(date '+%Y-%m-%d %H:%M:%S')  REPEAT_${REP}_TOTAL  ${REP_ELAPSED}s  $(fmt_hms $REP_ELAPSED)" >> "$TIMING_LOG"
done

TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START_TS ))
echo ""
echo "============================================================"
echo "  All reruns complete. Results in: $OUTDIR"
echo "  Total wall time this invocation: $(fmt_hms $TOTAL_ELAPSED)"
echo "  Per-run timing log: $TIMING_LOG"
echo "  Next: tar czf reviewer_r1_reruns.tar.gz $OUTDIR"
echo "============================================================"
