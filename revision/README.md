# `revision/`: the evaluation pipeline behind the paper

Everything reported in *Self-Consistency Vote Share as a Selective-Classification Signal for Zero-Shot LLM Text Classifiers* comes from this directory. Run all commands from the repository root.

| File | Purpose |
|---|---|
| `setup.sh` | create `.venv`, install vLLM 0.32.0 and download the two small models |
| `run_unified_vllm.py` | one job = one (model, dataset, repeat): 5 sampled calls per item, per-call logging, optional label-set scoring (`--score-direct`) or verbalized baseline (`--verbal`) |
| `worker.sh`, `launch_all.sh`, `monitor.sh` | 4 GPU workers sharing one queue with atomic `mkdir` locks |
| `jobs.txt` | 18 main jobs (2 models x 3 datasets x 3 repeats) |
| `jobs_extra.txt` | 6 verbalized-confidence jobs and 18 temperature jobs (T = 0.01, 0.3, 1.0; T = 0.7 comes from the main run) |
| `run_larger_models.sh` | 32B and 70B check on the first 2,000 items |
| `analyze_unified.py` | writes every table in the paper and the Supplement (`T_*.csv`) |
| `package_release.sh` | builds the per-call data tarball and copies the result tables to `results/` |
| `make_synthetic.py` | synthetic runs to smoke-test `analyze_unified.py` without a GPU |

## Protocol (what the paper reports)

* **Items:** each dataset's test split shuffled with `random.Random(42)`, first N items, identical for both models and every k. AG News 7,600 (full), DBpedia-14 7,000, GoEmotions (simplified) 5,427 (full). The larger-model check and the temperature sweep use the first 2,000 of the same order.
* **Models:** `unsloth/Llama-3.2-3B-Instruct` (public re-upload of Meta's weights), `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`; larger check: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`, `unsloth/Llama-3.3-70B-Instruct`.
* **Serving:** vLLM 0.32.0, bf16, context 6,144, NVIDIA H200.
* **Sampling:** 5 calls per item, T = 0.7, top-p 0.9, top-k 40, **repetition penalty 1.0**, per-request seed `10007 * repeat + item`; k = 1, 3, 5 are the first 1, 3, 5 calls; 3 repeats (1 for the larger models). Generation limit 4,096 tokens for DeepSeek, 32 otherwise. Every call logs label, generated tokens, truncation flag and answer-token probability.
* **Temperature:** vLLM refuses several samples at exactly T = 0, so the near-greedy setting is run at T = 0.01.
* **Repetition penalty:** vLLM applies it to the whole prompt, which contains the label names. A pilot (AG News, Llama, first 1,000 items, k = 1) gave 49.4% accuracy and 6.8% unparseable outputs at 1.1 against 54.1% and 0.4% at 1.0, so 1.0 is the default everywhere. Pass `--rep-penalty 1.1` to reproduce the sensitivity check.

## Reproduce

```bash
bash revision/setup.sh
source .venv/bin/activate && export LLM_VOTE_REPO=$PWD
# smoke test (one GPU, ~5 min)
CUDA_VISIBLE_DEVICES=0 python revision/run_unified_vllm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tag deepseek \
    --dataset ag_news --n 50 --rep 1 --score-direct --out runs/smoke && rm -rf runs/smoke
bash revision/launch_all.sh            # main + verbalized + temperature jobs on 4 GPUs (about 80 min on 4 x H200 for the main set)
watch bash revision/monitor.sh
python revision/analyze_unified.py --runs runs/unified --out results/tables_main --boot 2000
bash revision/run_larger_models.sh     # optional: 32B and 70B
```

No GPU is needed to test the analysis: `python revision/make_synthetic.py /tmp/synth/unified && python revision/analyze_unified.py --runs /tmp/synth/unified --out /tmp/synth/tables --boot 50`.

## Output tables

| Table | Paper (S = Supplement table) | Content |
|---|---|---|
| `T_main.csv` | Table 1, S1 | coverage, covered accuracy, accuracy with abstentions as errors, SC accuracy, ECE, AUROC, AURC, mean and SD over repeats |
| `T_baselines.csv` | Table 2, Figure 2, S2-S4 | label-set probability, answer-token probability, verbalized confidence and vote share with bootstrap CIs |
| `T_cost.csv`, `T_trunc.csv` | Table 3, S10 | tokens per item, correct answers per 1k tokens, truncation and its association with abstention |
| `T_tests.csv` | Table 4, S9 | paired McNemar tests, Holm-adjusted within repeat |
| `T_conformal.csv` | Table 5, S8 | Learn-then-Test thresholds and split-conformal sets |
| `T_goemo.csv` | S7 | any-gold, first-listed-gold and single-annotation protocols |
| `T_temperature.csv` | S6 | temperature sweep at k = 5 |

Column names: `rule_*` is the fixed strict-majority rule evaluated on answered items; `sc_*` is self-consistency on all items.
