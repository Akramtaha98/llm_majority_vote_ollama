# Minimal Majority Vote (MMV)

### Calibrated zero-shot LLM classification — no training, no logits, no calibration set

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-research--code-orange)](#)
[![Ollama](https://img.shields.io/badge/runs%20on-Ollama-black?logo=ollama&logoColor=white)](https://ollama.com)

Ask the LLM the same question **k independent times**, take the hard majority label, and use `conf = top_votes / k` as a per-instance uncertainty signal. Abstain when no strict majority is reached. No fine-tuning, no logit access, no held-out calibration data just repeated zero-shot calls and a vote count.

This repository is the reference implementation for *Minimal Majority Vote Ensembles for Robust LLM-Based Text Classification*.

---

## Contents

- [Key results](#key-results)
- [Why MMV](#why-mmv)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Supported datasets](#supported-datasets)
- [Output format](#output-format)
- [Reproducing the paper's results](#reproducing-the-papers-results)
- [Excluded conditions](#excluded-conditions)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Key results

Verified results across the **12 audited conditions** reported in the paper (4 dataset–model pairs × k ∈ {1, 3, 5}).

| Dataset | Model | k=1 Acc | Best Acc | k=1 ECE | Best ECE |
|---|---|---|---|---|---|
| AG News | DeepSeek-R1:7B | 77.74% | **84.15%** (k=5) | 22.26% | **10.28%** (k=5) |
| AG News | LLaMA-3.2:3B | 62.25% | **65.31%** (k=3) | 37.75% | **24.68%** (k=5) |
| DBpedia | DeepSeek-R1:7B | 93.36% | **95.56%** (k=3) | 6.64% | **4.63%** (k=5) |
| GoEmotions | DeepSeek-R1:7B | 26.19% | **38.12%** (k=5) | 73.81% | **34.62%** (k=5) |

95% bootstrap CIs for every ECE value are reported in the paper's Table 1. The cost-effective ensemble size is task-dependent, not universal: **k = 3** is a strong default on near-ceiling, well-separated tasks (AG News, DBpedia), while harder, higher-cardinality tasks like GoEmotions keep gaining accuracy through **k = 5** — see the paper's Cost-Benefit Analysis (Section 7.3) for the full picture, including why that GoEmotions gain is coverage-conditioned rather than a free lunch.

---

## Why MMV

Standard post-hoc calibration (temperature scaling, Platt scaling, histogram binning) has nothing to work with on a zero-shot LLM classifier: every `k = 1` prediction carries `confidence = 1.0` by construction, so there is no per-instance variation to calibrate. Cross-validated histogram binning can drive ECE to near-zero on `k = 1` outputs, but only by collapsing every prediction to the dataset's overall accuracy — which destroys per-instance discrimination and makes abstention or selective prediction impossible.

MMV sidesteps this by generating **genuinely discriminative** confidence values through the vote mechanism itself: `conf ∈ {2/3, 3/3}` at k=3, `conf ∈ {3/5, 4/5, 5/5}` at k=5. That's enough signal to abstain on the cases the model is least sure about, without any labeled calibration data.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/Akramtaha98/llm_majority_vote_ollama.git
cd llm_majority_vote_ollama
pip install -e .
```

### 2. Pull a local model (free, via Ollama)

```bash
brew install ollama          # macOS
ollama serve                 # keep this terminal open

ollama pull llama3.2         # LLaMA 3.2:3B
ollama pull deepseek-r1:7b   # DeepSeek-R1-Distill-Qwen-7B
```

### 3. Run your first experiment

```bash
# AG News, DeepSeek-R1:7B, k=5 (best accuracy + calibration in this study)
python -m scripts.eval_dataset \
  --provider ollama --model deepseek-r1:7b \
  --dataset ag_news --k 5 --max-samples 300

# AG News, LLaMA-3.2:3B, k=3
python -m scripts.eval_dataset \
  --provider ollama --model llama3.2 \
  --dataset ag_news --k 3 --max-samples 1000
```

Results (CSV + printed metrics) land in `runs/`.

---

## How it works

```text
for each text x:
    ask the LLM k independent times  →  labels y1, y2, ..., yk
    winner = most_common(y1..yk)
    if votes_for(winner) > k / 2:
        predict winner,  conf = votes_for(winner) / k
    else:
        ABSTAIN   # no strict majority reached
```

The strict-majority threshold uses the **original, fixed k** as its denominator — not the number of calls that happened to parse into a valid label. If too few calls parse to mathematically reach a majority of the original k, the sample abstains rather than being judged against a smaller denominator.

For **DeepSeek-R1:7B**, `<think>…</think>` reasoning traces are stripped with `re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)` before label extraction, and the remaining text is normalized against the task's fixed label set.

---

## Supported datasets

| Flag | Classes | Notes |
|---|---|---|
| `ag_news` | 4 (World, Sports, Business, Sci/Tech) | English news topics |
| `dbpedia` | 14 (entity types) | English knowledge base |
| `goemotions` | 27 (fine-grained emotions) + neutral | Natively multi-label; MMV scores accuracy against any listed gold label, Macro-F1/MCC against the first-listed label (see paper Section 4.3) |
| `labr` | 2 (binary sentiment) | Arabic; supported by the codebase, not part of the paper's reported study |

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

## Output format

Each run writes a CSV to `runs/`, e.g. `runs/ag_news_ollama_deepseek-r1-7b_k5.csv`:

| Column | Description |
|---|---|
| `id` | Sample identifier |
| `gold` | True label (first-listed label for multi-label datasets) |
| `gold_multi` | Full pipe-separated gold label set (GoEmotions only) |
| `pred` | Majority-vote label, or `ABSTAIN` |
| `text` | Input text |
| `votes` | JSON dict of per-label vote counts across the k calls |
| `top_votes` | Vote count for the winning label |
| `K` | Number of independent calls requested |
| `confidence` | `top_votes / K`, in [0, 1] |
| `correct` | 1 if the prediction matches gold, 0 otherwise |

Printed metrics: **accuracy** (on covered/non-abstained items), **Macro-F1**, **MCC**, **ECE** (15-bin, with bootstrap CIs), **coverage**.

> **Coverage stuck near 0%?** The model is outputting free text instead of exact labels. Tighten the prompt: *"Respond with only one word from: {label_list}. No explanation."*

---

## Reproducing the paper's results

```bash
# AG News: DeepSeek-R1:7B and LLaMA-3.2:3B, k ∈ {1, 3, 5}
for MODEL in deepseek-r1:7b llama3.2; do
  for K in 1 3 5; do
    python -m scripts.eval_dataset \
      --provider ollama --model "$MODEL" \
      --dataset ag_news --k "$K" --max-samples 1000
  done
done

# DBpedia and GoEmotions: DeepSeek-R1:7B only, k ∈ {1, 3, 5}
for DATASET in dbpedia goemotions; do
  for K in 1 3 5; do
    python -m scripts.eval_dataset \
      --provider ollama --model deepseek-r1:7b \
      --dataset "$DATASET" --k "$K" --max-samples 300
  done
done
```

Inference hyperparameters used in the paper: `temperature=0.7`, `top_p=0.9`, `top_k=40`, `repeat_penalty=1.1`, `context_window=4096`. No generation seed is passed to any per-call request; a fixed seed (=42) is used exactly once per dataset, before any model calls, purely to shuffle the sample.

Per-sample vote-count records for all 12 verified conditions — the exact data behind Tables 1 and 2 in the paper — are in [`vote_records/reviewer_data_package/`](vote_records/reviewer_data_package/).

---

## Excluded conditions

The paper originally scoped 18 conditions (6 dataset–model pairs × k ∈ {1, 3, 5}). Two dataset–model pairs (6 conditions) were excluded after a post-hoc data-integrity audit and are **not** part of the verified results above:

- **DBpedia × LLaMA-3.2** — the retained sample was drawn entirely from a single class (`Company`), making it non-representative.
- **GoEmotions × LLaMA-3.2** — the retained per-sample record had every one of its 1,000 rows labeled `neutral`, inconsistent with GoEmotions' genuine ~26–28-label distribution.

Both are flagged as open items for re-collection rather than reported with unverifiable numbers.

---

## Repository layout

```text
llm_majority_vote_ollama/
├── scripts/
│   ├── eval_dataset.py        # main experiment runner
│   ├── compute_results.py     # aggregate CSVs into summary tables
│   ├── run_parallel.py        # parallelized batch runner
│   └── make_plots.py          # generate figures from CSVs
├── src/llm_vote/
│   ├── voter.py                # MMV logic: majority vote + abstention
│   ├── metrics.py              # accuracy, macro-F1, MCC, ECE (15-bin)
│   ├── datasets.py             # AG News, DBpedia, GoEmotions, LABR loaders
│   ├── ollama_client.py        # local models via Ollama
│   ├── openai_client.py        # OpenAI API (optional)
│   ├── prompting.py            # zero-shot prompts — edit here to tune
│   └── utils.py
├── data/                        # custom CSV datasets
├── runs/                        # experiment outputs (CSV + logs)
└── vote_records/                # per-sample vote-count records for the paper's 12 verified conditions
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused … 11434` | Run `ollama serve` in a separate terminal |
| Coverage = 0% | Model is writing explanations, not labels — tighten the prompt |
| Slow runs | Lower `--max-samples`, reduce `k`, or use a smaller model |
| DeepSeek outputs look garbled | Confirm `<think>` stripping is active in `ollama_client.py` |

---

## Citation

This code accompanies *Minimal Majority Vote Ensembles for Robust LLM-Based Text Classification*, currently under review. A formal citation with venue and DOI will be added here once the paper is published; in the meantime, please cite the repository and the manuscript authors directly:

```bibtex
@misc{mmv_2026,
  title  = {Minimal Majority Vote Ensembles for Robust LLM-Based Text Classification},
  author = {Taher, Omar N. M. and Jeiad, Hassan A. and Noaman, Noaman M. and
            Raheem, Firas A. and Salman, Aymen D. and Humaidi, Amjad J.},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/Akramtaha98/llm_majority_vote_ollama}
}
```

---

## License

MIT
