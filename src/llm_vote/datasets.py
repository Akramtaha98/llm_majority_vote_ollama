from __future__ import annotations
from typing import List, Tuple
import random
from datasets import load_dataset
import pandas as pd

# Dataset-level sampling seed. With shuffle=True (the default), the full test split is
# shuffled once with random.Random(DATASET_SHUFFLE_SEED) and the first max_samples items
# are used. Every result in the paper (revision/run_unified_vllm.py) uses this default,
# so both models and all values of k see identical items. The seed is unrelated to
# per-call generation randomness. shuffle=False (first-N slice) exists only to read the
# first-version records in archive/; it is not used by the paper.
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
    """AG News test split (7,600 items), shuffled with `seed` by default, first `max_samples` items."""
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


def load_dbpedia_stratified(max_samples: int | None = None, seed: int = DATASET_SHUFFLE_SEED):
    """Stratified variant of load_dbpedia: draws an explicitly balanced sample
    across all 14 DBpedia classes rather than relying on a global shuffle to
    produce balance by chance. Used only for the first-version records in archive/
    (an early LLaMA-3.2:3B DBpedia collection was drawn from a single class); the
    paper's runs use load_dbpedia with the seed-42 shuffle.
    Deterministic for a fixed seed. Requires max_samples to be set (no
    "full dataset" mode, since stratification needs a target size)."""
    if not max_samples:
        raise ValueError("load_dbpedia_stratified requires an explicit max_samples")
    ds = load_dataset("fancyzhx/dbpedia_14", split="test")
    label_names = [
        "Company","EducationalInstitution","Artist","Athlete","OfficeHolder",
        "MeanOfTransportation","Building","NaturalPlace","Village","Animal",
        "Plant","Album","Film","WrittenWork"
    ]
    n_classes = len(label_names)
    by_class: dict[int, list] = {i: [] for i in range(n_classes)}
    for x in ds:
        by_class[x["label"]].append(f"{x.get('title','')}. {x['content']}".strip())

    rng = random.Random(seed)
    for i in range(n_classes):
        rng.shuffle(by_class[i])

    base_n = max_samples // n_classes
    remainder = max_samples % n_classes
    # distribute the remainder across a fixed, seeded random permutation of
    # classes, so which classes get the "+1" is deterministic but not biased
    # toward class index order.
    class_order = list(range(n_classes))
    rng.shuffle(class_order)
    bonus_classes = set(class_order[:remainder])

    rows = []
    per_class_counts = {}
    for i in range(n_classes):
        n_i = base_n + (1 if i in bonus_classes else 0)
        per_class_counts[label_names[i]] = n_i
        if n_i > len(by_class[i]):
            raise ValueError(
                f"Requested {n_i} samples for class {label_names[i]!r} but only "
                f"{len(by_class[i])} are available in the test split."
            )
        for text in by_class[i][:n_i]:
            rows.append((text, label_names[i]))

    # final shuffle so classes are interleaved in the eval order (matters for
    # any downstream code that assumes no ordering structure), still seeded.
    rng.shuffle(rows)

    print(f"[load_dbpedia_stratified] per-class sample counts: {per_class_counts}")
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
    multi-label gold annotations. The paper's GoEmotions results use the full
    simplified test split (5,427 items) through this loader."""
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
