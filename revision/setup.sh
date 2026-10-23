#!/usr/bin/env bash
# from repo root, after git pull: bash revision/setup.sh   (Llama-3.2 is gated: accept licence + `huggingface-cli login` first)
set -euo pipefail; cd "$(dirname "$0")/.."
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e . vllm scipy huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
echo setup done
