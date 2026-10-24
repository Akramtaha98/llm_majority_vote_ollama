#!/usr/bin/env bash
# Larger-model check (2,000 items per dataset, 1 repeat). Run from the repo root after revision/setup.sh.
#   GPU 2        : DeepSeek-R1-Distill-Qwen-32B (tag deepseek, reasoning model)
#   GPUs 0 and 1 : Llama-3.3-70B-Instruct (tag large, tensor parallel 2)
# Weights: if downloads fail with "CAS Client Error", export HF_HUB_DISABLE_XET=1 and pre-download with
#   python -c "from huggingface_hub import snapshot_download as s; s('unsloth/Llama-3.3-70B-Instruct')"
set -uo pipefail; cd "$(dirname "$0")/.."
source .venv/bin/activate; export LLM_VOTE_REPO=$PWD HF_HUB_DISABLE_XET=1; mkdir -p logs
( for ds in ag_news dbpedia goemotions; do
    CUDA_VISIBLE_DEVICES=2 python revision/run_unified_vllm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tag deepseek \
      --dataset $ds --n 2000 --rep 1 --score-direct --rep-penalty 1.0 --out runs/unified_big_r1_32b
  done ) > logs/big32b.log 2>&1 &
( for ds in ag_news dbpedia goemotions; do
    CUDA_VISIBLE_DEVICES=0,1 python revision/run_unified_vllm.py --model unsloth/Llama-3.3-70B-Instruct --tag large --tp 2 \
      --dataset $ds --n 2000 --rep 1 --score-direct --rep-penalty 1.0 --out runs/unified_big_llama70b
  done ) > logs/big70b.log 2>&1 &
wait
# analysis:
#   python revision/analyze_unified.py --runs runs/unified_big_r1_32b   --out results/tables_32b --boot 1000
#   python revision/analyze_unified.py --runs runs/unified_big_llama70b --out results/tables_70b --boot 1000
