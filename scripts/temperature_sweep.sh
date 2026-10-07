#!/usr/bin/env bash
# Category D script 2/3 -- temperature sweep, addressing Reviewer 3's point
# that all results use a single fixed temperature (0.7) and asking whether
# MMV's benefit depends on that specific sampling setting.
#
# No new eval code is needed: scripts/eval_dataset.py already accepts
# --temperature as a passthrough to the Ollama client (confirmed in
# src/llm_vote/ollama_client.py, which just forwards the float to the API).
# This script only drives that existing CLI across a small temperature grid.
#
# SCOPE: run on ONE representative condition per model to keep GPU time
# bounded (a full 12-condition x N-temperature x 3-repeat grid would be
# enormous). Recommended: AG News (fastest dataset, both models already
# have clean, verified baseline results at T=0.7, k=5) at k=5.
#   - DeepSeek-R1:7B, AG News, k=5
#   - LLaMA-3.2:3B,  AG News, k=5
# Each at T in {0.0 (greedy), 0.3, 0.7 (existing baseline value), 1.0}.
# 1 repeat per temperature (not 3) since the point is to characterize the
# effect of temperature on MMV's aggregate behavior (coverage, abstention
# rate, ECE, vote diversity), not to re-establish repeat-level variance,
# which is already characterized at T=0.7 by the existing three_repeat
# data. If the results suggest something surprising, re-run with repeats.
#
# METRICS REPORTED per (model, temperature): accuracy, macro-F1, MCC,
# coverage, ECE, mean vote diversity (entropy of vote distribution across
# the k samples), and total wall-clock/call cost -- directly answering
# "does temperature change MMV's core behavior."

set -euo pipefail
cd "$(dirname "$0")/.."  # run from repo root

# Use the repo's own .venv interpreter by explicit path when present, instead
# of relying on `python3` resolving correctly through PATH/shell-hash state.
# This avoids a class of bug where `which python3` reports the venv correctly
# in an interactive shell, but a freshly-spawned script process (or a stale
# shell command hash) ends up invoking a different `python3` that doesn't
# have this repo installed (`pip install -e .`), causing a spurious
# "ModuleNotFoundError: No module named 'llm_vote'".
if [ -x ".venv/bin/python3" ]; then
  PYTHON=".venv/bin/python3"
else
  PYTHON="python3"
fi
# Belt-and-suspenders: also put src/ directly on PYTHONPATH. On some Python
# builds the editable-install (.pth-based) meta path finder that `pip install
# -e .` registers behaves differently for `python3 -c "import ..."` than for
# `python3 some_script.py` (observed: the former succeeds, the latter still
# raises ModuleNotFoundError for the same interpreter). Setting PYTHONPATH
# makes llm_vote importable via plain path lookup regardless of whether that
# finder is active, so this works even if the root cause above is never
# fully explained.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Using interpreter: $PYTHON ($("$PYTHON" -c 'import sys; print(sys.executable)'))"
echo "PYTHONPATH=$PYTHONPATH"

OUT_DIR="runs/temperature_sweep"
mkdir -p "$OUT_DIR"

SEED=42
K=5
MAX_SAMPLES=300
TEMPS=(0.0 0.3 0.7 1.0)

# Parallel arrays instead of an associative array: macOS ships bash 3.2
# (Apple stopped bundling GPLv3 bash), which has no `declare -A` support and
# fails with a confusing "unbound variable" error if you try. This form works
# on both bash 3.2 (macOS default `bash`) and bash 4+ (Homebrew bash, Linux).
MODEL_TAGS=(deepseek llama3.2)
MODEL_NAMES=(deepseek-r1:7b llama3.2)

for i in "${!MODEL_TAGS[@]}"; do
  TAG="${MODEL_TAGS[$i]}"
  MODEL="${MODEL_NAMES[$i]}"
  echo "=== AG News, $MODEL, k=$K, temperature sweep ==="
  for T in "${TEMPS[@]}"; do
    OUT="$OUT_DIR/ag_news_${TAG}_k${K}_temp${T}.csv"
    if [ -f "$OUT" ]; then
      echo "skip (exists): $OUT"
      continue
    fi
    echo "--- temperature=$T ---"
    "$PYTHON" scripts/eval_dataset.py \
      --provider ollama --model "$MODEL" \
      --dataset ag_news \
      --k "$K" --max-samples "$MAX_SAMPLES" --seed "$SEED" \
      --temperature "$T" --top-p 0.9 --top-k 40 --repeat-penalty 1.1 --num-ctx 4096 \
      --preds "$OUT"
  done
done

echo
echo "=== Aggregating results ==="
"$PYTHON" - "$OUT_DIR" <<'PYEOF'
import sys, glob, json, math
import pandas as pd

out_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(f"{out_dir}/*.csv")):
    d = pd.read_csv(path)
    # parse condition from filename: ag_news_<tag>_k<k>_temp<t>.csv
    base = path.split("/")[-1].replace(".csv", "")
    parts = base.split("_")
    tag = parts[2]
    k_str = [p for p in parts if p.startswith("k")][0]
    temp_str = [p for p in parts if p.startswith("temp")][0]
    temperature = float(temp_str.replace("temp", ""))

    n = len(d)
    covered = d[d["pred"].notna() & (d["pred"] != "")]
    coverage = len(covered) / n if n else float("nan")
    acc = (covered["pred"] == covered["gold"]).mean() if len(covered) else float("nan")

    # vote diversity: normalized entropy of the vote-count distribution per row
    def row_entropy(votes_json):
        try:
            votes = json.loads(votes_json.replace("'", '"'))
        except Exception:
            return float("nan")
        total = sum(votes.values())
        if total == 0:
            return float("nan")
        h = -sum((c / total) * math.log2(c / total) for c in votes.values() if c > 0)
        h_max = math.log2(len(votes)) if len(votes) > 1 else 1.0
        return h / h_max if h_max > 0 else 0.0

    if "votes" in d.columns:
        mean_entropy = d["votes"].apply(row_entropy).mean()
    else:
        mean_entropy = float("nan")

    rows.append({
        "model": tag,
        "temperature": temperature,
        "n": n,
        "coverage_%": round(coverage * 100, 2) if coverage == coverage else None,
        "accuracy_%": round(acc * 100, 2) if acc == acc else None,
        "mean_vote_entropy_norm": round(mean_entropy, 4) if mean_entropy == mean_entropy else None,
    })

summary = pd.DataFrame(rows).sort_values(["model", "temperature"])
print(summary.to_string(index=False))
summary.to_csv(f"{out_dir}/summary.csv", index=False)
print(f"\nSaved summary to {out_dir}/summary.csv")
PYEOF

echo
echo "Done. Review $OUT_DIR/summary.csv with Akram before deciding how (or"
echo "whether) to fold this into the manuscript -- e.g. as a new short"
echo "robustness subsection in Section 7, or as a rebuttal-only appendix if"
echo "results simply confirm the effect is small and don't need to enter"
echo "the main text."
