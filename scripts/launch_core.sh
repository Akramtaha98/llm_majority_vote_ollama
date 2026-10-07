#!/usr/bin/env bash
# Core Information experiments: `verbal` + extended `temp`, in parallel, on one GPU.
# Usage (repo root):  bash scripts/launch_core.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export OLLAMA_NUM_PARALLEL=8 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_KEEP_ALIVE=24h
if ! curl -s localhost:11434 >/dev/null 2>&1; then
  nohup ollama serve > logs/ollama_serve.log 2>&1 &
  sleep 8
fi
ollama pull deepseek-r1:7b && ollama pull llama3.2
nohup bash scripts/run_information_experiments.sh verbal > logs/verbal_run.log 2>&1 &
for ds in ag_news dbpedia goemotions; do
  for tag in deepseek llama; do
    TEMP_DS=$ds TEMP_TAG=$tag nohup bash scripts/run_information_experiments.sh temp \
      > logs/temp_run_${ds}_${tag}.log 2>&1 &
  done
done
sleep 2; echo "Launched. Monitor with:  bash scripts/monitor_core.sh"
