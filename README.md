# Minimal Majority Vote (MMV)

**Calibrated zero-shot LLM classification. No training, no logits, no calibration set.**

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

Verified results across the **12 audited conditions** reported in the paper (4 dataset-model pairs x k in {1, 3, 5}), each the mean of three independent post-fix repeats (Table 1).

| Dataset | Model | k=1 Acc | Best Acc | k=1 ECE | Best ECE |
|---|---|---|---|---|---|
| AG News | DeepSeek-R1:7B | 80.39% | **83.05%** (k=5) | 19.61% | **10.57%** (k=5) |
| AG News | LLaMA-3.2:3B | 61.73% | **65.04%** (k=5) | 38.27% | **24.37%** (k=5) |
| DBpedia | DeepSeek-R1:7B | 89.98% | **94.65%** (k=5) | 10.02% | **2.94%** (k=3) |
| GoEmotions | DeepSeek-R1:7B | 26.72% | **39.64%** (k=5) | 73.28% | **34.97%** (k=5) |

Mean +/- SD across three independent post-fix repeats for every accuracy and ECE value is reported in the paper's Table 1 (Section 4.3); the original single-run bootstrap/Wilson-interval CIs remain the basis for the significance testing discussed in Section 7.5, not for Table 1 itself. Note DBpedia's best ECE is at **k = 3**, not k = 5 - ECE falls sharply from k=1 to k=3 then rises very slightly at k=5 (Section 5.2). The cost-effective ensemble size is task-dependent, not universal. **k = 3** is a strong default on near-ceiling, well-separated tasks (AG News, DBpedia); harder, higher-cardinality tasks like GoEmotions keep gaining accuracy through **k = 5**. See the paper's Cost-Benefit Analysis (Section 7.3) for the full picture, including why the GoEmotions gain is coverage-conditioned rather than a free lunch.

---

## Why MMV

Standard post-hoc calibration (temperature scaling, Platt scaling, histogram binning) has nothing to work with on a zero-shot LLM classifier: every `k = 1` prediction carries `confidence = 1.0` by construction, so there's no per-instance variation to calibrate. Cross-validated histogram binning can drive ECE to near-zero on `k = 1` outputs, but only by collapsing every prediction to the dataset's overall accuracy, which destroys per-instance discrimination and makes abstention or selective prediction impossible.

MMV sidesteps this by generating genuinely discriminative confidence values through the vote mechanism itself: `conf ∈ {2/3, 3/3}` at k=3, `conf ∈ {3/5, 4/5, 5/5}` at k=5. That's enough signal to abstain on the cases the model is least sure about, without any labeled calibration data.

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

ollama pull llama3.2         # LLaMA-3.2:3B
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

Results (CSV + printed metrics) land in `runs/`. Don't pass `--max-tokens` unless you have a specific reason to override the generation-length default - `eval_dataset.py` already picks the right cap per model family (see [Troubleshooting](#troubleshooting)).

---

## How it works

```text
for each text x:
    ask the LLM k independent times  ->  labels y1, y2, ..., yk
    winner = most_common(y1..yk)
    if votes_for(winner) > k / 2:
        predict winner,  conf = votes_for(winner) / k
    else:
        ABSTAIN   # no strict majority reached
```

The strict-majority threshold uses the **original, fixed k** as its denominator, not the number of calls that happened to parse into a valid label. If too few calls parse to mathematically reach a majority of the original k, the sample abstains rather than being judged against a smaller denominator.

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

Printed metrics: **accuracy** (on covered/non-abstained items), **Macro-F1**, **MCC**, **ECE** (15-bin), **coverage**. `eval_dataset.py` doesn't itself print confidence intervals; accuracy Wilson-score CIs and ECE bootstrap CIs (10,000 resamples) can be computed from a single run's saved per-sample CSV by `scripts/regenerate_all.py`, but this is no longer how the paper's Table 1 reports uncertainty: Table 1 now uses Mean(3) +/- SD(3) across three independent post-fix repeats per condition, computed by `scripts/regenerate_3rep.py` (see "Reproducing the paper's results" below). The single-run Wilson/bootstrap CIs remain available as a diagnostic for a one-off run outside the paper's own repeat-based methodology.

> **Coverage stuck near 0%?** The model is outputting free text instead of exact labels. Tighten the prompt: *"Respond with only one word from: {label_list}. No explanation."*

---

## Reproducing the paper's results

**One-command reproduction of every table and figure from the released data** (no LLM
calls, seconds to run): `bash scripts/reproduce_paper.sh [OUT_DIR]`. This single driver
runs the six scripts below in order and prints a table/figure -> script mapping before it
starts, so it is no longer necessary to know which of the six scripts produces which
result. Outputs land in `OUT_DIR` (default `outputs/regenerated/`), except Figure 6,
which the underlying script always writes to `out_figs/`.

| Result | Script |
|---|---|
| Table 1, 2, 5, 7; Figure 7 | `scripts/regenerate_3rep.py` |
| Figures 2, 3, 4, 5 | `scripts/make_figures_3rep.py` |
| Table 8, Table 9 | `scripts/compute_repeat_metrics.py` |
| Table 4 (CV-HB calibration) | `scripts/regenerate_table4_cvhb.py` |
| Figure 6 (confusion matrix) | `scripts/regenerate_figure6.py` |
| McNemar / Bonferroni significance tests | `scripts/regenerate_significance_3rep.py` |
| Single-run diagnostic (pre-fix or any one run; not the paper's reported numbers) | `scripts/regenerate_all.py` |

To regenerate the *raw* per-sample records these six scripts read (i.e. to re-run the
model rather than just recompute statistics from already-released data), see the
per-condition commands below.

**Sampling-provenance note** (post-submission audit; manuscript Section 7.6): not every condition below was originally collected with the seed-42 shuffle applied. AG News DeepSeek-R1:7B at k = 3 and k = 5, DBpedia (all k), and GoEmotions (all k) were shuffled as described below. AG News DeepSeek-R1:7B at k = 1 and all three AG News LLaMA-3.2:3B conditions (k = 1, 3, 5) were instead collected from an unshuffled, first-N slice, predating the shuffle's introduction into this pipeline. Pass `--no-shuffle` to `eval_dataset.py` to reproduce those four conditions specifically - the commands below already use the historically correct flag for each condition.

```bash
# AG News, DeepSeek-R1:7B: k=1 uses an independent n=1,000, unshuffled;
# k=3 and k=5 use a separate, shuffled n=300 (these two are nested with each other,
# not with the k=1 sample -- see Section 4.4/7.6).
python -m scripts.eval_dataset \
  --provider ollama --model deepseek-r1:7b \
  --dataset ag_news --k 1 --max-samples 1000 --no-shuffle
for K in 3 5; do
  python -m scripts.eval_dataset \
    --provider ollama --model deepseek-r1:7b \
    --dataset ag_news --k "$K" --max-samples 300
done

# AG News, LLaMA-3.2:3B: all three k share one unshuffled n=1,000 sample.
for K in 1 3 5; do
  python -m scripts.eval_dataset \
    --provider ollama --model llama3.2 \
    --dataset ag_news --k "$K" --max-samples 1000 --no-shuffle
done

# DBpedia and GoEmotions: DeepSeek-R1:7B only, k in {1, 3, 5}, shuffled n=300
for DATASET in dbpedia goemotions; do
  for K in 1 3 5; do
    python -m scripts.eval_dataset \
      --provider ollama --model deepseek-r1:7b \
      --dataset "$DATASET" --k "$K" --max-samples 300
  done
done
```

Inference hyperparameters used in the paper: `temperature=0.7`, `top_p=0.9`, `top_k=40`, `repeat_penalty=1.1`, `context_window=4096`. No generation seed is passed to any per-call request; where used, the fixed seed (=42) is applied exactly once per dataset, before any model calls, purely to shuffle the sample (see the sampling-provenance note above for which conditions this actually applies to).

**Table 1 and everything derived from it are now built from a 3-repeat post-fix dataset, not the original single-run data described above.** Between the original data collection and these repeats we fixed a client-side generation-length bug that had been silently truncating DeepSeek-R1:7B's reasoning trace (see [Troubleshooting](#troubleshooting)); the commands above still describe the original, pre-fix single-run collection procedure and remain useful for understanding the codebase's sampling logic, but they are no longer what Table 1 reports.

The 36 raw per-sample repeat-run records (12 conditions x 3 independent repeats, all collected under the fixed client) live in [`runs/reviewer_r1_reruns/`](runs/reviewer_r1_reruns/); see [`runs/README.md`](runs/README.md). The following scripts regenerate the current, post-fix paper directly from these records, with no hand-typed numbers anywhere downstream:

- `python3 scripts/regenerate_3rep.py` - Tables 1, 2, 5, 7 (reuses `regenerate_all.py`'s own MMV/SC/ECE/AURC logic rather than reimplementing it).
- `python3 scripts/make_figures_3rep.py` - Figures 2, 3, 4, 5.
- `python3 scripts/compute_repeat_metrics.py` - Table 8 (accuracy per repeat) and Table 9 in full (coverage/ECE/Macro-F1/MCC, all 36 rows).
- `python3 scripts/regenerate_table4_cvhb.py` - Table 4 (5-fold cross-validated histogram-binning calibration, seed=42, pooled across the three repeats).
- `python3 scripts/regenerate_figure6.py` - Figure 6 (confusion matrix, pooled across the three repeats).
- `python3 scripts/regenerate_significance_3rep.py` - the paper's McNemar significance tests and Bonferroni-corrected threshold (Section 4.3, 7.5), re-derived separately per repeat (36 tests: 12 within-condition comparisons x 3 repeats) rather than pooled, since the three repeats are non-independent draws over the same matched samples.

Per-sample vote-count records for the original single-run collection - the exact data behind the pre-fix `regenerate_all.py`/`regenerate_significance.py` scripts below - live in [`vote_records/reviewer_data_package/`](vote_records/reviewer_data_package/). A pre-scored copy of the same records, with explicit `mmv_pred`, `sc_pred`, `mmv_correct`, `sc_correct`, and `parser_failure` columns, is in [`vote_records/reviewer_data_package/per_sample_vote_count_records_scored/`](vote_records/reviewer_data_package/per_sample_vote_count_records_scored/).

`python3 scripts/regenerate_all.py` and `python3 scripts/regenerate_significance.py` remain in the repository and still reproduce the *original, pre-fix* single-run Tables 1/2/5/7 and the original 10-test McNemar/Bonferroni analysis exactly as originally submitted; they are retained for provenance and are reused internally by the 3-rep scripts above, but they no longer describe what the current manuscript reports.

For a worked example of how released data is audited against a specific reviewer question, see [`AUDIT_APPLE_M3.md`](AUDIT_APPLE_M3.md), which documents the exact search commands and results used to verify that no released record references an "Apple M3" example a reviewer recalled from an earlier manuscript draft.

---

## Excluded conditions

The paper originally scoped 18 conditions (6 dataset-model pairs x k in {1, 3, 5}). Two dataset-model pairs (6 conditions) were excluded after a post-hoc data-integrity audit and are **not** part of the verified results above:

- **DBpedia x LLaMA-3.2** - the retained sample was drawn entirely from a single class (`Company`), making it non-representative.
- **GoEmotions x LLaMA-3.2** - the retained per-sample record had every one of its 1,000 rows labeled `neutral`, inconsistent with GoEmotions' genuine ~26-28-label distribution.

Both are flagged as open items for re-collection rather than reported with unverifiable numbers. See manuscript Section 7.6 (Limitations) and [`vote_records/reviewer_data_package/README.md`](vote_records/reviewer_data_package/README.md) for the full audit note; the raw (excluded) run files are kept in [`runs/`](runs/) for transparency.

---

## Repository layout

```text
llm_majority_vote_ollama/
├── scripts/
│   ├── eval_dataset.py             # main experiment runner
│   ├── compute_results.py          # legacy ad-hoc aggregation (superseded by regenerate_all.py)
│   ├── regenerate_all.py           # original pre-fix: regenerates Tables 1, 2, 5, 7 from raw votes
│   ├── regenerate_significance.py  # original pre-fix: McNemar tests + Bonferroni (10 tests)
│   ├── compute_repeat_metrics.py   # current: Table 8 (accuracy) and Table 9 in full (36 rows)
│   ├── regenerate_3rep.py          # current: Tables 1, 2, 5, 7 from the 3-repeat post-fix data
│   ├── regenerate_significance_3rep.py  # current: McNemar/Bonferroni re-derived per repeat (36 tests)
│   ├── regenerate_table4_cvhb.py   # current: Table 4 (5-fold CV histogram-binning, seed=42)
│   ├── regenerate_figure6.py       # current: Figure 6 (confusion matrix) from post-fix data
│   ├── make_figures_3rep.py        # current: Figures 2, 3, 4, 5 from the 3-repeat post-fix data
│   ├── run_all.sh / run_parallel.py           # batch runners (sequential / parallel)
│   ├── rerun_reviewer_r1_concurrent.sh        # reviewer-requested repeat runs
│   └── make_plots.py               # generate figures from CSVs (original pre-fix collection)
├── src/llm_vote/
│   ├── voter.py             # MMV logic: majority vote + abstention
│   ├── metrics.py           # accuracy, macro-F1, MCC, ECE (15-bin)
│   ├── datasets.py          # AG News, DBpedia, GoEmotions, LABR loaders
│   ├── ollama_client.py     # local models via Ollama
│   ├── openai_client.py     # OpenAI API (optional)
│   ├── prompting.py         # zero-shot prompts - edit here to tune
│   └── utils.py
├── data/                    # custom CSV datasets
├── runs/                    # experiment outputs (CSV + logs); see runs/README.md
├── vote_records/            # per-sample vote-count records for the paper's 12 verified conditions
├── archive/                 # quarantined out-of-scope artifacts (LABR pilot data), kept for transparency
└── AUDIT_APPLE_M3.md        # provenance audit for a reviewer-flagged example (see above)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused ... 11434` | Run `ollama serve` in a separate terminal |
| Coverage = 0% | Model is writing explanations, not labels - tighten the prompt |
| Slow runs | Lower `--max-samples`, reduce `k`, or use a smaller model |
| DeepSeek outputs look garbled, or `<think>` text leaks into predictions | Confirm `<think>` stripping is active in `ollama_client.py`, and that you haven't passed `--max-tokens` with a value too small to let the reasoning trace finish - leave it unset and let the per-model default apply |

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
