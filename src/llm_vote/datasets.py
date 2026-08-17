from __future__ import annotations
from typing import List, Tuple
import random
from datasets import load_dataset
import pandas as pd

# Dataset-level sampling seed (Section 4.2): used once per dataset, before any
# model calls, to deterministically shuffle the full test split prior to slicing
# out max_samples. This is unrelated to per-call generation randomness (no
# generation seed is passed to the Ollama API -- see ollama_client.py).
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


def load_ag_news(max_samples: int | None = None, seed: int = DATASET_SHUFFLE_SEED):
    ds = load_dataset("ag_news", split="test")
    label_names = ["World","Sports","Business","Sci/Tech"]
    rows = [(x["text"], label_names[x["label"]]) for x in ds]
    rows = _shuffle_and_slice(rows, max_samples, seed)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    task = "AG News Topic Classification"
    return texts, labels, task, label_names

def load_dbpedia(max_samples: int | None = None, seed: int = DATASET_SHUFFLE_SEED):
    ds = load_dataset("dbpedia_14", split="test")
    label_names = [
        "Company","EducationalInstitution","Artist","Athlete","OfficeHolder",
        "MeanOfTransportation","Building","NaturalPlace","Village","Animal",
        "Plant","Album","Film","WrittenWork"
    ]
    rows = [(f"{x.get('title','')}. {x['content']}".strip(), label_names[x["label"]]) for x in ds]
    rows = _shuffle_and_slice(rows, max_samples, seed)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    task = "DBpedia Ontology Classification"
    return texts, labels, task, label_names

def load_goemotions(max_samples: int | None = None, seed: int = DATASET_SHUFFLE_SEED):
    ds = load_dataset("go_emotions", "raw", split="test")
    label_names = ds.features["labels"].feature.names
    gold_multi_all = [[label_names[i] for i in x["labels"]] for x in ds]
    rows = [(x["text"], gm) for x, gm in zip(ds, gold_multi_all)]
    rows = _shuffle_and_slice(rows, max_samples, seed)
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
