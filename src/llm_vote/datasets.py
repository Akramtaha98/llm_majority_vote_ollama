from __future__ import annotations
from typing import List, Tuple
from datasets import load_dataset
import pandas as pd
import random
import os

EMOTION_NAMES = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","neutral","optimism","pride",
    "realization","relief","remorse","sadness","surprise",
]

def _shuffle_and_slice(texts, labels, max_samples, seed=42):
    """Shuffle deterministically before slicing to avoid class-ordering bias."""
    paired = list(zip(texts, labels))
    rng = random.Random(seed)
    rng.shuffle(paired)
    if max_samples:
        paired = paired[:max_samples]
    return [p[0] for p in paired], [p[1] for p in paired]

def load_ag_news(max_samples: int | None = None, seed: int = 42):
    ds = load_dataset("ag_news", split="test")
    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    texts  = [x["text"] for x in ds]
    labels = [label_names[x["label"]] for x in ds]
    texts, labels = _shuffle_and_slice(texts, labels, max_samples, seed)
    return texts, labels, "AG News Topic Classification", label_names

def load_dbpedia(max_samples: int | None = None, seed: int = 42):
    ds = load_dataset("dbpedia_14", split="test")
    label_names = [
        "Company","EducationalInstitution","Artist","Athlete","OfficeHolder",
        "MeanOfTransportation","Building","NaturalPlace","Village","Animal",
        "Plant","Album","Film","WrittenWork"
    ]
    texts  = [f"{x.get('title','')}. {x['content']}".strip() for x in ds]
    labels = [label_names[x["label"]] for x in ds]
    texts, labels = _shuffle_and_slice(texts, labels, max_samples, seed)
    return texts, labels, "DBpedia Ontology Classification", label_names

def load_goemotions(max_samples: int | None = None, seed: int = 42,
                    csv_fallback: str = "data/goemotions_test_200.csv"):
    """
    Load GoEmotions using the 'simplified' HF config (integer label lists).
    Falls back to local CSV if HF is unavailable.
    """
    for split_name in ["test", "validation", "train"]:
        try:
            ds = load_dataset("go_emotions", "simplified", split=split_name)
            try:
                label_names = ds.features["labels"].feature.names
            except Exception:
                label_names = EMOTION_NAMES

            texts = [x["text"] for x in ds]
            gold_multi = []
            for x in ds:
                lbls = x.get("labels", [])
                if isinstance(lbls, (list, tuple)) and len(lbls) > 0:
                    gold_multi.append([label_names[i] for i in lbls])
                else:
                    gold_multi.append(["neutral"])

            labels_primary = [g[0] for g in gold_multi]

            combined = list(zip(texts, labels_primary, gold_multi))
            rng = random.Random(seed)
            rng.shuffle(combined)
            if max_samples:
                combined = combined[:max_samples]
            texts          = [c[0] for c in combined]
            labels_primary = [c[1] for c in combined]
            gold_multi     = [c[2] for c in combined]

            return texts, labels_primary, "GoEmotions Emotion Classification", list(label_names), gold_multi
        except Exception:
            continue

    # Fallback: local CSV
    if not os.path.exists(csv_fallback):
        raise FileNotFoundError(
            f"Could not load HF go_emotions and fallback CSV not found: {csv_fallback}"
        )
    df = pd.read_csv(csv_fallback)
    texts = df["text"].astype(str).tolist()
    labels_primary = df["gold"].astype(str).tolist()
    gold_multi = None
    label_names = sorted(set(labels_primary))
    if max_samples:
        texts = texts[:max_samples]
        labels_primary = labels_primary[:max_samples]
    return texts, labels_primary, "GoEmotions Emotion Classification", label_names, gold_multi

def load_labr_csv(csv_path: str, max_samples: int | None = None):
    df = pd.read_csv(csv_path)
    texts  = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    if max_samples:
        texts, labels = texts[:max_samples], labels[:max_samples]
    return texts, labels, "LABR Arabic Sentiment (Positive/Negative)", ["Positive", "Negative"]
