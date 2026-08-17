"""
run_parallel.py — Run all 18 MMV experiments with parallel workers.

Strategy
--------
Ollama keeps ONE model loaded in GPU memory at a time.  Switching models
flushes the GPU cache and reloads weights, which wastes minutes.  So we
group experiments by model and run each model-group sequentially.

Within a model-group, all (dataset × k) combinations run in parallel using
a thread-pool.  Because each experiment makes many sequential HTTP calls to
Ollama, threads spend most of their time waiting for the GPU; the OS schedules
another thread's HTTP overhead during that wait, giving real speed-up even
on one GPU.

Recommended workers per model-group: 3  (safe for a single A100 40 GB)

Usage
-----
    cd /path/to/llm_majority_vote_ollama
    pip install -e .
    ollama serve                        # in another terminal
    ollama pull deepseek-r1:7b
    ollama pull llama3.2

    # Run all 18 experiments (3 parallel workers per model)
    python scripts/run_parallel.py

    # Custom options
    python scripts/run_parallel.py --workers 4 --samples 500 --k 1 3 5

    # After completion
    python scripts/compute_results.py
    python scripts/make_plots.py
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── Experiment matrix ────────────────────────────────────────────────────────

EXPERIMENTS = {
    "deepseek-r1:7b": [
        ("ag_news",    "deepseek-r1-7b"),
        ("dbpedia",    "deepseek-r1-7b"),
        ("goemotions", "deepseek-r1-7b"),
    ],
    "llama3.2": [
        ("ag_news",    "llama3.2"),
        ("dbpedia",    "llama3.2"),
        ("goemotions", "llama3.2"),
    ],
}

K_VALUES = [1, 3, 5]


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_complete(csv_path: Path, min_rows: int = 100) -> bool:
    """Return True if the CSV exists, has enough rows, and is not all-ABSTAIN."""
    if not csv_path.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if len(df) < min_rows:
            return False
        if (df["pred"] == "ABSTAIN").all():
            return False
        return True
    except Exception:
        return False


def run_one(model: str, dataset: str, tag: str, k: int,
            samples: int, seed: int, temperature: float) -> tuple[str, bool, str]:
    """
    Run a single eval_dataset.py experiment as a subprocess.
    Returns (label, success, output_text).
    """
    out_csv = Path("runs") / f"{dataset}_ollama_{tag}_k{k}.csv"
    label   = f"{dataset}+{tag}+k={k}"

    if is_complete(out_csv):
        return label, True, f"  ✓ SKIP (already complete): {out_csv}"

    cmd = [
        sys.executable, "scripts/eval_dataset.py",
        "--provider",     "ollama",
        "--model",        model,
        "--dataset",      dataset,
        "--k",            str(k),
        "--max-samples",  str(samples),
        "--seed",         str(seed),
        "--temperature",  str(temperature),
        "--max-tokens",   "50",
        "--preds",        str(out_csv),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,   # 2-hour hard cap per experiment
        )
        if result.returncode == 0:
            # Extract summary line from stdout
            lines = [l for l in result.stdout.splitlines() if "%" in l or "Coverage" in l]
            summary = " | ".join(lines[-3:]) if lines else "done"
            return label, True, f"  ✓ {label}: {summary}"
        else:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            return label, False, f"  ✗ {label}: FAILED — {err}"
    except subprocess.TimeoutExpired:
        return label, False, f"  ✗ {label}: TIMEOUT after 2 h"
    except Exception as e:
        return label, False, f"  ✗ {label}: ERROR — {e}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Run all MMV experiments in parallel (grouped by model)."
    )
    ap.add_argument("--workers",     type=int,   default=3,
                    help="Parallel workers per model-group (default: 3)")
    ap.add_argument("--samples",     type=int,   default=200,
                    help="Samples per experiment (default: 200; use 1000 for paper)")
    ap.add_argument("--k",           type=int,   nargs="+", default=K_VALUES,
                    help="Ensemble sizes to evaluate (default: 1 3 5)")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    Path("runs").mkdir(exist_ok=True)

    total   = len(EXPERIMENTS) * 3 * len(args.k)   # models × datasets × k
    done    = 0
    failed  = []

    banner = (
        "═" * 62 + "\n"
        f"  MMV Parallel Runner\n"
        f"  Workers/model : {args.workers}\n"
        f"  Samples/run   : {args.samples}\n"
        f"  k values      : {args.k}\n"
        f"  Total runs    : {total}\n"
        + "═" * 62
    )
    print(banner)

    for model, combos in EXPERIMENTS.items():
        print(f"\n{'─'*62}")
        print(f"  Model: {model}  ({len(combos) * len(args.k)} experiments)")
        print(f"{'─'*62}")

        futures_map = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for dataset, tag in combos:
                for k in args.k:
                    future = pool.submit(
                        run_one, model, dataset, tag, k,
                        args.samples, args.seed, args.temperature
                    )
                    futures_map[future] = (dataset, tag, k)

            for future in as_completed(futures_map):
                label, success, msg = future.result()
                print(msg)
                done += 1
                if not success:
                    failed.append(label)

    print(f"\n{'═'*62}")
    print(f"  Completed: {done}/{total}")
    if failed:
        print(f"  Failed runs ({len(failed)}):")
        for f in failed:
            print(f"    ✗ {f}")
    else:
        print("  All runs succeeded ✓")

    print(f"\n  Next steps:")
    print(f"    python scripts/compute_results.py   # Tables 3 & 4")
    print(f"    python scripts/make_plots.py         # Figures 1-3")
    print("═" * 62)


if __name__ == "__main__":
    main()
