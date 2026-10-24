#!/usr/bin/env bash
# from repo root, after git pull: bash revision/setup.sh   (the Llama weights come from the public unsloth re-upload, so no gated-licence login is needed; set HF_HUB_DISABLE_XET=1 if downloads fail with CAS errors)
set -euo pipefail; cd "$(dirname "$0")/.."
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e . vllm==0.32.0 scipy huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
huggingface-cli download unsloth/Llama-3.2-3B-Instruct
echo setup done
