# Reviewer Data Package -- "Minimal Majority Vote Ensembles for Robust LLM-Based Text Classification"

This package accompanies the manuscript and is provided so reviewers can independently
verify the coverage/parser-failure statistics reported in Table 1 and Table 2, without
waiting for the post-acceptance public archive.

## Contents

- `per_sample_vote_count_records/` -- 12 CSV files, one per (dataset, model, k) condition
  reported in the paper. AG News x {DeepSeek-R1:7B, LLaMA-3.2} x {k=1,3,5}; DBpedia x
  DeepSeek-R1:7B x {k=1,3,5}; GoEmotions x DeepSeek-R1:7B x {k=1,3,5}.
- `FINAL_corrected_results_table.csv` -- the audited summary table (N_total, N_covered,
  N_abstain_no_majority, N_invalid_output, Coverage, Accuracy, Macro-F1, MCC, ECE, and
  bootstrap ECE CIs) for all 18 originally-run conditions, including the 2 excluded from
  the paper (LLaMA-3.2 on DBpedia; LLaMA-3.2 on GoEmotions), for transparency.

## Per-sample CSV columns

`id, gold, [gold_multi,] pred, text, votes, top_votes, K, confidence, correct`

- `votes`: a JSON dict of vote counts per label for that sample, e.g.
  `{"World": 0, "Sports": 0, "Business": 2, "Sci/Tech": 0}`. The number of the k calls
  that parsed successfully for a sample is `sum(votes.values())`.
- `top_votes`, `K`, `confidence`: the winning label's vote count, the ensemble size, and
  `top_votes / K`.
- `correct`: 1 if the returned prediction was scored correct against gold (for
  GoEmotions, matched against the multi-label gold set as described in the manuscript's
  Materials and Methods section), 0 otherwise.

## How to reproduce Table 2's decomposition

For each row: let `n_parsed = sum(votes.values())`. If `n_parsed == 0`, the sample is a
parser failure (excluded from Valid-Output Rate). Otherwise, if `max(votes.values()) >
K/2`, the sample is covered (majority reached); check whether `n_parsed < K` to get the
"Covered w/ Partial Parse" statistic. If `max(votes.values()) <= K/2`, the sample
abstained; classify it as Full-Parse Split (`n_parsed == K`), Capped: Insufficient
(`n_parsed <= floor(K/2)`), or Split: Sufficient (otherwise). This matches Table 2's
caption in the manuscript exactly.

## Note on excluded conditions

LLaMA-3.2 on DBpedia and LLaMA-3.2 on GoEmotions are **not** included in
`per_sample_vote_count_records/` because a post-hoc audit found each unreliable for a
different reason (documented in the manuscript, Section 7.5): the DBpedia run sampled
from a single class only, and the only retained GoEmotions record for LLaMA-3.2 assigns
the same gold label to every row. Both conditions have been excluded from the paper's
results rather than reported with unverifiable numbers. `FINAL_corrected_results_table.csv`
still lists their pre-exclusion summary statistics for transparency about what was found
and why it was excluded.

## Contact

Questions about this data package can be directed to the corresponding author via the
editorial system during this review round.
