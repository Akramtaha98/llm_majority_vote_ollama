#!/usr/bin/env bash
# usage (from repo root): bash revision/worker.sh <gpu_id> [revision/jobs.txt]
# jobs line: <tag> <dataset> <n> <rep> [T|verbal]   (T default 0.7). Atomic mkdir locks => 4 workers share one queue.
set -uo pipefail
GPU=${1:?gpu id}; JOBS=${2:-revision/jobs.txt}; cd "$(dirname "$0")/.."
source .venv/bin/activate
export LLM_VOTE_REPO=$PWD VLLM_LOGGING_LEVEL=WARNING
mkdir -p runs logs locks
while read -r tag ds n rep opt; do
  [ -z "$tag" ] && continue
  case ${opt:-0.7} in
    verbal) out=runs/unified; flag="--verbal"; name=${ds}_${tag}_verbal; resf=${out}/${name}.csv;;
    *)      T=${opt:-0.7}; if [ "$T" = 0.7 ]; then out=runs/unified; else out=runs/unified_T$T; fi
            flag="--temperature $T"; [ "$rep" = 1 ] && [ "$T" = 0.7 ] && flag="$flag --score-direct"
            name=${ds}_${tag}_rep${rep}; resf=${out}/${name}.csv;;
  esac
  lock=locks/${name}_${opt:-0.7}; mkdir "$lock" 2>/dev/null || continue
  [ -f "$resf" ] && continue
  case $tag in deepseek) M=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B;; llama) M=unsloth/Llama-3.2-3B-Instruct;; esac
  echo "[gpu$GPU] $(date +%T) start $name ${opt:-0.7}"
  CUDA_VISIBLE_DEVICES=$GPU python revision/run_unified_vllm.py --model $M --tag $tag --dataset $ds --n $n --rep $rep $flag --rep-penalty ${REP_PENALTY:-1.0} --out $out \
      > logs/${name}_${opt:-0.7}.log 2>&1 \
    && echo "[gpu$GPU] $(date +%T) done  $name" || { echo "[gpu$GPU] FAILED $name (see logs/)"; rmdir "$lock"; }
done < "$JOBS"
echo "[gpu$GPU] worker finished $JOBS"
