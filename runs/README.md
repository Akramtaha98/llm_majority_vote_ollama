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

36 files (12 verified conditions x 3 independent repeats), produced by
`scripts/rerun_reviewer_r1_concurrent.sh` (rep1/rep2) and a third GPU rerun (rep3) in
response to reviewer feedback. All three repeats were collected under the fixed
`num_predict` client (see the top-level README's Reproducing section). Used to compute
Table 1 (Mean(3) +/- SD(3)), Table 2, Table 5, Table 7, and Figures 2-5, 7 via
`scripts/regenerate_3rep.py` / `scripts/make_figures_3rep.py`, and Table 8
(accuracy, Rep1/Rep2/Rep3) and Table 9 (coverage, ECE, Macro-F1, MCC, all 36 rows) via
`scripts/compute_repeat_metrics.py`.

**Schema note: `gold_multi` is not present in every file.** Only the 12 `*_rep3.csv`
files carry a `gold_multi` column (a later `eval_dataset.py` version); the 24 `*_rep1.csv`
/ `*_rep2.csv` files do not have this column at all. For the single-label datasets (AG
News, DBpedia) this is harmless -- `gold_multi` equals `gold` there, and both
`regenerate_3rep.py` and `compute_repeat_metrics.py` fall through to `gold` when the
column is missing. For GoEmotions, which is natively multi-label, falling through to
`gold` (the single first-listed label) instead of the true multi-label set would
under-credit any prediction that matches a secondary gold label but not the first-listed
one; both scripts instead recover the correct multi-label gold for every GoEmotions
rep1/rep2 row from the original per-sample vote-count records in
`vote_records/reviewer_data_package/per_sample_vote_count_records/` (which do carry a
correct `gold_multi` for every row, all k), keyed by sample `id`, exactly matching rep3's
own values where both are available.

## Related

LABR-related run artifacts have been moved to `archive/unused_labr/`; they are not
part of any reported result.
