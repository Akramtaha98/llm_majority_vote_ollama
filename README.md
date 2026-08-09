# Minimal Majority Vote (MMV) for LLM-Based Text Classification

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research--code-orange)

> Zero-shot text classification with calibrated confidence — no training, no logits, no held-out calibration set required.

Ask the LLM the same question **k independent times**, take the hard majority label, and use `conf = top_votes / k` as a per-instance uncertainty signal. Abstain when no strict majority is reached. Simple idea, measurable accuracy and calibration gains over a single zero-shot call.

This repository is the reference implementation for *Minimal Majority Vote Ensembles for Robust LLM-Based Text Classification* (Taher, Salman, Raheem & Humaidi).

---

## Table of Contents

- [Key Results](#key-results)
- [Why MMV?](#why-mmv)
- [Quickstart](#quickstart)
- [How It Works](#how-it-works)
- [Supported Datasets](#supported-datasets)
- [Output Format](#output-format)
- [Reproduce Paper Results](#reproduce-paper-results)
- [A Note on Excluded Conditions](#a-note-on-excluded-conditions)
- [Repo Structure](#repo-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Key Results

Verified results across the **12 audited conditions** reported in the paper (4 dataset–model pairs × k ∈ {1, 3, 5}).

| Dataset | Model | k=1 Acc | Best Acc | ECE (k=1) | Best ECE |
|---------|-------|---------|----------|-----------|----------|
| AG News | DeepSeek-R1:7B | 77.74% | **84.15% (k=5)** | 22.26% | **10.28% (k=5)** |
| AG News | LLaMA-3.2:3B | 62.25% | **65.31% (k=3)** | 37.75% | **24.68% (k=5)** |
| DBpedia | DeepSeek-R1:7B | 93.36% | **95.56% (k=3)** | 6.64% | **4.63% (k=5)** |
| GoEmotions | DeepSeek-R1:7B | 26.19% | **38.12% (k=5)** | 73.81% | **34.62% (k=5)** |

95% CIs (bootstrap, 2000 resamples) on all ECE values are reported in the paper's Table 1. **Recommended default: k = 5** — best or near-best accuracy and calibration across all retained conditions; k = 3 is a lower-cost alternative with a modest accuracy trade-off.

---

## Why MMV?

Standard post-hoc calibration (temperature scaling, Platt scaling, histogram binning) **fails** for zero-shot LLM classifiers: every k=1 prediction carries `confidence = 1.0` by construction, so there is nothing to calibrate. Cross-validated histogram binning achieves ECE ≈ 0% on k=1 data only by assigning every instance the same probability (the dataset accuracy) — eliminating per-instance uncertainty entirely.

MMV fixes this by generating **genuinely discriminative** confidence values through the vote mechanism, enabling selective prediction and abstention without any labeled calibration data.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/Akramtaha98/llm_majority_vote_ollama.git
cd llm_majority_vote_ollama
pip install -e .
```

### 2. Run a local model with Ollama (free)

```bash
# macOS
brew install ollama
ollama serve          # keep this terminal open

# pull models
ollama pull llama3.2
ollama pull deepseek-r1:7b
```

### 3. First experiment

```bash
# AG News, DeepSeek-R1:7B, k=5 (recommended)
python -m scripts.eval_dataset \
  --provider ollama --model deepseek-r1:7b \
  --dataset ag_news --k 5 --max-samples 300

# AG News, LLaMA-3.2:3B, k=3
python -m scripts.eval_dataset \
  --provider ollama --model llama3.2 \
  --dataset ag_news --k 3 --max-samples 1000
```

Results (CSV + printed metrics) are saved to `runs/`.

---

## How It Works

```
for each text x:
    ask LLM k independent times → labels y1, y2, ..., yk
    winner = most_common(y1..yk)
    if votes_for(winner) > k / 2:
        predict winner,  conf = votes_for(winner) / k
    else:
        ABSTAIN   # no strict majority reached
```

`conf` takes discrete values, e.g. `{2/3, 3/3}` for k=3 and `{3/5, 4/5, 5/5}` for k=5. The Condorcet Jury Theorem guarantees majority-vote accuracy increases with k when individual-call accuracy exceeds 50%.

For **DeepSeek-R1:7B**, `<think>…</think>` reasoning traces are stripped with `re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)` before label extraction.

---

## Supported Datasets

| Dataset | Classes | Notes |
|---------|---------|-------|
| `ag_news` | 4 (World, Sports, Business, Sci/Tech) | English news topics |
| `dbpedia` | 14 (entity types) | English knowledge base |
| `goemotions` | 27 (fine-grained emotions) | Simplified to single-label by top response |

```bash
# DBpedia
python -m scripts.eval_dataset \
  --provider ollama --model deepseek-r1:7b \
  --dataset dbpedia --k 3 --max-samples 300

# GoEmotions
python -m scripts.eval_dataset \
  --provider ollama --model deepseek-r1:7b \
  --dataset goemotions --k 5 --max-samples 300
```

---

## Output Format

Each run writes a CSV to `runs/`, e.g. `runs/ag_news_ollama_deepseek-r1-7b_k5.csv`:

| Column | Description |
|--------|-------------|
| `id` | Sample identifier |
| `gold` | True label |
| `pred` | Majority-vote label (or `ABSTAIN`) |
| `text` | Input text |
| `votes` | JSON dict of per-label vote counts across the k calls |
| `top_votes` | Vote count for the winning label |
| `K` | Number of independent calls requested |
| `confidence` | `top_votes / K` (0–1) |
| `correct` | 1 if `pred == gold`, 0 otherwise |

Printed metrics: **Accuracy** (on covered/non-abstained items), **Macro-F1**, **MCC**, **ECE** (15-bin, with bootstrap CIs), **Coverage**.

> **Low coverage?** Your model is outputting free text instead of exact labels.
> Fix: add `"Respond with only one word from: {label_list}. No explanation."` to the prompt.

---

## Reproduce Paper Results

```bash
# AG News, DeepSeek-R1:7B and LLaMA-3.2, k ∈ {1, 3, 5}
for MODEL in deepseek-r1:7b llama3.2; do
  for K in 1 3 5; do
    python -m scripts.eval_dataset \
      --provider ollama --model $MODEL \
      --dataset ag_news --k $K --max-samples 1000
  done
done

# DBpedia and GoEmotions, DeepSeek-R1:7B only, k ∈ {1, 3, 5}
for DATASET in dbpedia goemotions; do
  for K in 1 3 5; do
    python -m scripts.eval_dataset \
      --provider ollama --model deepseek-r1:7b \
      --dataset $DATASET --k $K --max-samples 300
  done
done
```

Inference hyperparameters used in the paper: `temperature=0.7`, `top_p=0.9`, `top_k=40`, `context_window=4096`, `repeat_penalty=1.1`.

---

## A Note on Excluded Conditions

The paper originally ran 18 conditions (6 dataset–model pairs × k ∈ {1, 3, 5}). Two dataset–model pairs (6 conditions) were excluded after a post-hoc data-integrity audit and are **not** part of the verified results above:

- **DBpedia × LLaMA-3.2** — the sampled text was drawn entirely from a single class (`Company`), making the split non-representative.
- **GoEmotions × LLaMA-3.2** — the only available raw run file had every one of its 1,000 rows labeled `neutral` (vs. a properly diverse ~26–28-label distribution in the parallel DeepSeek-R1:7B run), indicating file corruption rather than a usable, if imperfect, sample.

Per-sample vote-count records for all 12 retained conditions are available on request for verification; see the paper's Data Availability Statement.

---

## Repo Structure

```
llm_majority_vote_ollama/
├── scripts/
│   ├── eval_dataset.py      # main experiment runner
│   ├── compute_results.py   # aggregate CSVs into summary tables
│   ├── run_parallel.py      # parallelized batch runner
│   └── make_plots.py        # generate figures from CSVs
├── src/llm_vote/
│   ├── voter.py              # MMV logic (majority vote + abstention)
│   ├── metrics.py            # accuracy, macro-F1, MCC, ECE (15-bin)
│   ├── datasets.py           # AG News, DBpedia, GoEmotions loaders
│   ├── ollama_client.py      # local models via Ollama
│   ├── openai_client.py      # OpenAI API (optional)
│   ├── prompting.py          # zero-shot prompts (edit here to tune)
│   └── utils.py
├── data/                     # custom CSV datasets
└── runs/                     # experiment outputs (CSV + logs)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused … 11434` | Run `ollama serve` in a separate terminal |
| Coverage = 0% | Model is outputting explanations, not labels — tighten the prompt |
| Slow runs | Reduce `--max-samples`, lower `k`, or use a smaller model |
| DeepSeek outputs garbled labels | Confirm `<think>` stripping is active in `ollama_client.py` |

---

## Citation

```bibtex
@article{taher_mmv_2026,
  title   = {Minimal Majority Vote Ensembles for Robust LLM-Based Text Classification},
  author  = {Taher, Omar N. M. and Salman, Aymen D. and Raheem, Firas A. and Humaidi, Amjad J.},
  year    = {2026},
  url     = {https://github.com/Akramtaha98/llm_majority_vote_ollama}
}
```

---

## License

MIT
