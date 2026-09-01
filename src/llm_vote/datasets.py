from __future__ import annotations
from typing import List, Tuple
import random
from datasets import load_dataset
import pandas as pd

# Dataset-level sampling seed (Section 4.4): when shuffle=True (the default), used
# once per dataset, before any model calls, to deterministically shuffle the full
# test split prior to slicing out max_samples. This is unrelated to per-call
# generation randomness (no generation seed is passed to the Ollama API -- see
# ollama_client.py). NOT every condition in this paper's released records was
# actually collected with shuffle=True -- a post-submission audit (manuscript
# Section 7.6) found that AG News DeepSeek-R1:7B at k = 1 and all AG News
# LLaMA-3.2:3B conditions were collected with shuffle=False (an unshuffled,
# first-N slice), predating this shuffle's introduction into the pipeline. Pass
# shuffle=False explicitly to reproduce those specific historical conditions.
DATASET_SHUFFLE_SEED = 42


def _shuffle_and_slice(items: list, max_samples: int | None, seed: int = DATASET_SHUFFLE_SEED) -> list:
    """Deterministically shuffle `items` with a fixed, dataset-scoped RNG (isolated
    from the global `random` module state) and take the first max_samples. Returns
    the full shuffled list if max_samples is falsy."""
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    if max_samples:
        indices = indices[:max_samples]
    return [items[i] for i in indices]


def _slice_unshuffled(items: list, max_samples: int | None) -> list:
    """First-N slice with no shuffling, preserving the dataset's original order."""
    if max_samples:
        return items[:max_samples]
    return items


def load_ag_news(max_samples: int | None = None, seed: int = DATASET_SHUFFLE_SEED, shuffle: bool = True):
    """A post-submission audit (manuscript Section 7.6) found that the historical
    per-sample records released with this paper were NOT all collected with the
    seed-42 shuffle below applied: AG News DeepSeek-R1:7B at k = 3 and k = 5 were
    drawn from the shuffled test split, but AG News DeepSeek-R1:7B at k = 1 and all
    three AG News LLaMA-3.2:3B conditions (k = 1, 3, 5) were drawn from an
    unshuffled, first-N slice of the same test split. Pass shuffle=False to
    reproduce the unshuffled conditions; the default (shuffle=True) matches the
    two AG News DeepSeek-R1:7B (k = 3, k = 5) conditions and is the recommended
    setting for any new data collection with this codebase going forward."""
    ds = load_dataset("fancyzhx/ag_news", split="test")
    label_names = ["World","Sports","Business","Sci/Tech"]
    rows = [(x["text"], label_names[x["label"]]) for x in ds]
    rows = _shuffle_and_slice(rows, max_samples, seed) if shuffle else _slice_unshuffled(rows, max_samples)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    task = "AG News Topic Classification"
    return texts, labels, task, label_names

def load_dbpedia(max_samples: int | None = None, seed: int = DATASET_SHUFFLE_SEED, shuffle: bool = True):
    ds = load_dataset("fancyzhx/dbpedia_14", split="test")
    label_names = [
        "Company","EducationalInstitution","Artist","Athlete","OfficeHolder",
        "MeanOfTransportation","Building","NaturalPlace","Village","Animal",
        "Plant","Album","Film","WrittenWork"
    ]
    rows = [(f"{x.get('title','')}. {x['content']}".strip(), label_names[x["label"]]) for x in ds]
    rows = _shuffle_and_slice(rows, max_samples, seed) if shuffle else _slice_unshuffled(rows, max_samples)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    task = "DBpedia Ontology Classification"
    return texts, labels, task, label_names

def load_goemotions(max_samples: int | None = None, seed: int = DATASET_SHUFFLE_SEED, shuffle: bool = True):
    """The dataset config used to collect this paper's original records
    ("go_emotions", config "raw", split "test") no longer resolves on the current
    Hugging Face Hub -- the "raw" config now exposes only a "train" split there,
    a change made upstream after this study's data collection, not by us. This
    loader instead uses the "simplified" config's "test" split, which retains the
    same fixed 28-category label set (27 emotions + neutral) and genuine
    multi-label gold annotations. We verified directly (Section 7.6) that this
    substitute, shuffled with seed=42 and sliced to n=300, reproduces this paper's
    released GoEmotions per-sample record exactly, row for row -- so despite the
    upstream config change, the original GoEmotions sample selection remains
    fully reproducible through this loader."""
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    label_names = ds.features["labels"].feature.names
    gold_multi_all = [[label_names[i] for i in x["labels"]] for x in ds]
    rows = [(x["text"], gm) for x, gm in zip(ds, gold_multi_all)]
    rows = _shuffle_and_slice(rows, max_samples, seed) if shuffle else _slice_unshuffled(rows, max_samples)
    texts = [r[0] for r in rows]
    gold_multi = [r[1] for r in rows]
    labels_primary = [g[0] if len(g) > 0 else "neutral" for g in gold_multi]
    task = "GoEmotions Emotion Classification"
    return texts, labels_primary, task, list(label_names), gold_multi

def load_labr_csv(csv_path: str, max_samples: int | None = None, seed: int = DATASET_SHUFFLE_SEED):
    df = pd.read_csv(csv_path)
    rows = list(zip(df["text"].astype(str).tolist(), df["label"].astype(str).tolist()))
    rows = _shuffle_and_slice(rows, max_samples, seed)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    task = "LABR Arabic Sentiment (Positive/Negative)"
    label_names = ["Positive","Negative"]
    return texts, labels, task, label_names
