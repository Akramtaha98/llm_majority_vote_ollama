# runs/

Per-sample prediction CSVs produced by `scripts/run_all.sh` / `scripts/run_parallel.py`
and `scripts/rerun_reviewer_r1_concurrent.sh`, referenced throughout the paper and its
revisions.

## Top-level files (18)

One file per `{dataset} x {model} x k` condition, named
`{dataset}_ollama_{model}_k{K}.csv` (e.g. `ag_news_ollama_deepseek-r1-7b_k3.csv`):

- Datasets: `ag_news`, `dbpedia`, `goemotions`
- Models: `deepseek-r1-7b`, `llama3.2`
- k: `1`, `3`, `5`

These are the original single-repeat runs behind Tables 1-7 and Figures 2-4, produced
via `scripts/compute_results.py` and `scripts/make_plots.py`.

**Excluded from the paper's verified results** (manuscript Section 7.6, Limitations):
LLaMA-3.2:3B on DBpedia and LLaMA-3.2:3B on GoEmotions. A post-hoc audit found each
unreliable for a different reason -- the DBpedia run sampled from a single class only,
and the GoEmotions run assigned the same gold label to every row. Both files are kept
here for transparency but are not the source of any reported statistic; see
`vote_records/reviewer_data_package/README.md` for the full note.

## `reviewer_r1_reruns/`

24 files (12 verified conditions x 2 independent repeats), produced by
`scripts/rerun_reviewer_r1_concurrent.sh` in response to reviewer feedback. Used to
compute Table 8 (accuracy, Rep1/Rep2) and Table 9 (coverage, ECE, Macro-F1, MCC) via
`scripts/compute_repeat_metrics.py`.

## Related

LABR-related run artifacts have been moved to `archive/unused_labr/`; they are not
part of any reported result.
