from __future__ import annotations
from typing import List, Tuple
from datasets import load_dataset
import pandas as pd

def load_ag_news(max_samples: int | None = None):
    ds = load_dataset("ag_news", split="test")
    texts = [x["text"] for x in ds]
    label_names = ["World","Sports","Business","Sci/Tech"]
    labels = [label_names[x["label"]] for x in ds]
    if max_samples:
        texts, labels = texts[:max_samples], labels[:max_samples]
    task = "AG News Topic Classification"
    return texts, labels, task, label_names

def load_dbpedia(max_samples: int | None = None):
    ds = load_dataset("dbpedia_14", split="test")
    label_names = [
        "Company","EducationalInstitution","Artist","Athlete","OfficeHolder",
        "MeanOfTransportation","Building","NaturalPlace","Village","Animal",
        "Plant","Album","Film","WrittenWork"
    ]
    texts = [f"{x.get('title','')}. {x['content']}".strip() for x in ds]
    labels = [label_names[x["label"]] for x in ds]
    if max_samples:
        texts, labels = texts[:max_samples], labels[:max_samples]
    task = "DBpedia Ontology Classification"
    return texts, labels, task, label_names

def load_goemotions(max_samples: int | None = None):
    ds = load_dataset("go_emotions", "raw", split="test")
    label_names = ds.features["labels"].feature.names
    texts = [x["text"] for x in ds]
    gold_multi = [[label_names[i] for i in x["labels"]] for x in ds]
    labels_primary = [g[0] if len(g)>0 else "neutral" for g in gold_multi]
    if max_samples:
        texts, labels_primary, gold_multi = texts[:max_samples], labels_primary[:max_samples], gold_multi[:max_samples]
    task = "GoEmotions Emotion Classification"
    return texts, labels_primary, task, list(label_names), gold_multi

def load_labr_csv(csv_path: str, max_samples: int | None = None):
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    if max_samples:
        texts, labels = texts[:max_samples], labels[:max_samples]
    task = "LABR Arabic Sentiment (Positive/Negative)"
    label_names = ["Positive","Negative"]
    return texts, labels, task, label_names
