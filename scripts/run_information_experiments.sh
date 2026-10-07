#!/usr/bin/env bash
# GPU experiments requested for the Information submission. Run from repo root.
# Each block is resumable (skips existing outputs). Expected GPU time is large;
# run blocks separately:  bash scripts/run_information_experiments.sh verbal|temp|matched|scale
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python3; [ -x .venv/bin/python3 ] && PY=.venv/bin/python3
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
MODELS=("deepseek-r1:7b:deepseek" "llama3.2:llama")
run() { # out model dataset k n T extra...
  local out=$1 model=$2 ds=$3 k=$4 n=$5 T=$6; shift 6
  [ -f "$out" ] && { echo "skip $out"; return; }
  "$PY" scripts/eval_dataset.py --provider ollama --model "$model" --dataset "$ds" --k "$k" \
    --max-samples "$n" --seed 42 --temperature "$T" --top-p 0.9 --top-k 40 --repeat-penalty 1.1 \
    --num-ctx 4096 --save-raw-outputs --preds "$out" "$@"
}
case "${1:-}" in
verbal)   # single-call verbalized-confidence baseline: 3 datasets x 2 models, n=300
  mkdir -p runs/verbalized
  for ds in ag_news dbpedia goemotions; do
    for m in "deepseek-r1:7b:deepseek" "llama3.2:llama"; do
      model="${m%:*}"; tag="${m##*:}"; out="runs/verbalized/${ds}_${tag}.csv"
      [ -f "$out" ] && continue
      mmv="runs/three_repeat/${ds}_${tag}_k5_rep1.csv"; [ "$ds" = ag_news ] && [ "$tag" = deepseek ] && mmv="runs/three_repeat/ag_news_deepseek_matched_k5_rep1.csv"
      args=(--model "$model" --dataset "$ds" --max-samples 300 --seed 42 --out "$out")
      [ -f "$mmv" ] && args+=(--mmv_csv "$mmv")
      "$PY" scripts/verbalized_confidence.py "${args[@]}"
    done
  done ;;
temp)     # extended temperature study: 3 datasets x 2 models x T in {0,0.3,0.7,1.0}, k=5, n=300
  mkdir -p runs/temperature_extended
  for ds in ${TEMP_DS:-ag_news dbpedia goemotions}; do
    for m in "${MODELS[@]}"; do
      model="${m%:*}"; tag="${m##*:}"
      [ -n "${TEMP_TAG:-}" ] && [ "$TEMP_TAG" != "$tag" ] && continue
      for T in 0.0 0.3 0.7 1.0; do
        run "runs/temperature_extended/${ds}_${tag}_k5_temp${T}.csv" "$model" "$ds" 5 300 "$T"
      done
    done
  done ;;
matched)  # same shuffled n=300 AG News sample for BOTH models at k=1,3,5, 3 repeats
  mkdir -p runs/matched_agnews
  for rep in 1 2 3; do for k in 1 3 5; do for m in "${MODELS[@]}"; do
    model="${m%:*}"; tag="${m##*:}"
    run "runs/matched_agnews/ag_news_${tag}_k${k}_rep${rep}.csv" "$model" ag_news "$k" 300 0.7
  done; done; done ;;
scale)    # larger GoEmotions sample for DeepSeek (n=1000), k=1,3,5, 3 repeats
  mkdir -p runs/scale_goemotions
  for rep in 1 2 3; do for k in 1 3 5; do
    run "runs/scale_goemotions/goemotions_deepseek_n1000_k${k}_rep${rep}.csv" deepseek-r1:7b goemotions "$k" 1000 0.7
  done; done ;;
*) echo "usage: $0 verbal|temp|matched|scale"; exit 1 ;;
esac
echo "done: $1"
