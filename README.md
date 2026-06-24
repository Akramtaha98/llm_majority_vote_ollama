# Minimal Majority Vote (MMV) for LLM Text Classification

> Zero-shot text classification with calibrated confidence — no training, no logits, no calibration set required.

Ask the LLM the same question **k times**, take the majority label, and use `conf = top_votes / k` as a per-instance uncertainty signal. Simple idea, measurable gains.

---

## Key Results

| Dataset | Model | k=1 Acc | Best Acc | ECE (k=1) | ECE (best k) |
|---------|-------|---------|----------|-----------|--------------|
| AG News | LLaMA 3.2:3B | 62.25% | 64.40% (k=3) | 37.75% | 24.46% (k=5) |
| AG News | DeepSeek-R1:7B | 77.74% | **81.00% (k=3)** | 21.50% | **10.80% (k=5)** |
| DBpedia | DeepSeek-R1:7B | 93.36% | 94.33% (k=3) | 6.33% | 4.93% (k=5) |
| GoEmotions | DeepSeek-R1:7B | 26.19% | 32.00% (k=5)* | 62.00% | 23.20% (k=5) |

\* p = 0.029, McNemar's test. Bootstrap 95% CIs confirm all ECE improvements.

**Recommended default: k = 3** (best accuracy/cost ratio across all conditions).

---

## Why MMV?

Standard post-hoc calibration (temperature scaling, Platt scaling, histogram binning) **fails** for zero-shot LLM classifiers: every k=1 prediction carries `confidence = 1.0` by construction, so there is nothing to calibrate. Cross-validated histogram binning achieves ECE ≈ 0% on k=1 data only by assigning every instance the same probability (the dataset accuracy) — eliminating per-instance uncertainty entirely.

MMV fixes this by generating **genuinely discriminative** confidence values through the vote mechanism, enabling selective prediction and abstention without any labeled data.

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
# AG News, DeepSeek-R1:7B, k=3 (recommended)
python -m scripts.eval_dataset \
  --provider ollama --model deepseek-r1:7b \
  --dataset ag_news --k 3 --max-samples 300

# LLaMA 3.2:3B, k=5
python -m scripts.eval_dataset \
  --provider ollama --model llama3.2 \
  --dataset ag_news --k 5 --max-samples 1000
```

Results (CSV + metrics) are saved to `runs/`.

---

## How It Works

```
for each text x:
    ask LLM k independent times → labels y1, y2, ..., yk
    winner = most_common(y1..yk)
    if votes_for(winner) > k/2:
        predict winner,  conf = votes_for(winner) / k
    else:
        ABSTAIN  (no strict majority)
```

`conf` takes values in `{2/3, 3/3}` for k=3 and `{3/5, 4/5, 5/5}` for k=5.  
The Condorcet Jury Theorem guarantees majority vote accuracy increases with k when individual accuracy > 50%.

For **DeepSeek-R1:7B**, `<think>…</think>` reasoning traces are stripped with `re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)` before label extraction.

---

## Supported Datasets

| Dataset | Classes | Notes |
|---------|---------|-------|
| `ag_news` | 4 (World, Sports, Business, Sci/Tech) | English news topics |
| `dbpedia` | 14 (entity types) | English knowledge base |
| `goemotions` | 27 (fine-grained emotions) | Simplified to single-label |

```bash
# DBpedia
python -m scripts.eval_dataset \
  --provider ollama --model deepseek-r1:7b \
  --dataset dbpedia --k 3 --max-samples 300

# GoEmotions
python -m scripts.eval_dataset \
  --provider ollama --model llama3.2 \
  --dataset goemotions --k 5 --max-samples 1000
```

---

## Output Format

Each run writes a CSV to `runs/`, e.g. `runs/ag_news_ollama_deepseek-r1-7b_k3.csv`:

| Column | Description |
|--------|-------------|
| `text` | Input text |
| `gold` | True label |
| `pred` | Majority-vote label (or `ABSTAIN`) |
| `confidence` | `top_votes / k` (0–1) |
| `correct` | 1 if pred == gold, 0 otherwise |

Printed metrics: **Accuracy**, **Macro-F1**, **MCC**, **ECE** (15-bin), **Coverage**.

> **Low coverage?** Your model is outputting free text instead of exact labels.  
> Fix: add `"Respond with only one word from: {label_list}. No explanation."` to the prompt.

---

## Reproduce Paper Results

```bash
# Effect of k on AG News (LLaMA)
for K in 1 3 5; do
  python -m scripts.eval_dataset \
    --provider ollama --model llama3.2 \
    --dataset ag_news --k $K --max-samples 1000
done

# DeepSeek full sweep
for DATASET in ag_news dbpedia goemotions; do
  for K in 1 3 5; do
    python -m scripts.eval_dataset \
      --provider ollama --model deepseek-r1:7b \
      --dataset $DATASET --k $K --max-samples 300
  done
done
```

Inference hyperparameters used in the paper: `temperature=0.7`, `top_p=0.9`, `top_k=40`, `context_window=4096`, `repeat_penalty=1.1`.

---

## Repo Structure

```
llm_majority_vote_ollama/
├── scripts/
│   ├── eval_dataset.py      # main experiment runner
│   └── make_plots.py        # generate figures from CSVs
├── src/llm_vote/
│   ├── voter.py             # MMV logic (majority vote + abstention)
│   ├── metrics.py           # accuracy, macro-F1, MCC, ECE (15-bin)
│   ├── datasets.py          # AG News, DBpedia, GoEmotions loaders
│   ├── ollama_client.py     # local models via Ollama
│   ├── openai_client.py     # OpenAI API (optional)
│   └── prompting.py         # zero-shot prompts (edit here to tune)
├── data/                    # custom CSV datasets
└── runs/                    # experiment outputs (CSV + logs)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused … 11434` | Run `ollama serve` in a separate terminal |
| Coverage = 0% | Model outputting explanations, not labels — tighten the prompt |
| Slow runs | Reduce `--max-samples`, lower `k`, or use a smaller model |
| DeepSeek outputs garbled labels | Confirm `<think>` stripping is active in `ollama_client.py` |

---

## Citation

```bibtex
@article{taha2025mmv,
  title   = {Minimal Majority Vote Ensembles for Robust LLM-Based Text Classification},
  author  = {Taha, Akram and Zeyad},
  journal = {Machine Learning and Knowledge Extraction (MDPI MAKE)},
  year    = {2025},
  url     = {https://github.com/Akramtaha98/llm_majority_vote_ollama}
}
```

---

## License

MIT
