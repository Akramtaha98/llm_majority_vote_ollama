"""
Single-call verbalized-confidence baseline (no repeated sampling).

For each item, ONE call asks the model for a label and a 0-100 confidence.
Reports accuracy, coverage (parseable outputs), ECE (15 equal-width bins),
AUROC for error detection, and AURC. With --mmv_csv it also computes the same
uncertainty metrics for MMV's vote-share confidence and for the self-consistency
(SC) vote share on a released/new k>1 vote-count CSV, so that single-call and
repeated-sampling uncertainty are compared on identical metrics.

Usage (repo root, after `pip install -e .`):
  python scripts/verbalized_confidence.py --model deepseek-r1:7b --dataset ag_news \
      --max-samples 300 --seed 42 --out runs/verbalized/ag_news_deepseek.csv \
      --mmv_csv runs/three_repeat/ag_news_deepseek_matched_k5_rep1.csv
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np, pandas as pd
from tqdm import tqdm
from llm_vote.datasets import load_ag_news, load_dbpedia, load_goemotions
from llm_vote.ollama_client import OllamaClient

SYSTEM = ("You are a careful text classifier. Return exactly two lines:\n"
          "Label: <one label from the allowed list>\n"
          "Confidence: <integer 0-100, your probability that the label is correct>")

def build(text, labels, task):
    return (f"Task: Single-label classification for {task}.\nAllowed labels: [{', '.join(labels)}]\n"
            f"Text:\n{text}\nRespond in the required two-line format.")

def parse(raw, labels):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    m_l = re.search(r"Label\s*:\s*(.+)", raw)
    m_c = re.search(r"Confidence\s*:\s*(\d+(?:\.\d+)?)", raw)
    if not (m_l and m_c): return None, None
    lab = re.sub(r"\s+", " ", m_l.group(1).strip().strip("[]\"'."))
    if lab not in labels: return None, None
    return lab, min(max(float(m_c.group(1)), 0.0), 100.0) / 100.0

def ece(conf, corr, bins=15):
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.any(): tot += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return tot

def auroc_error(conf, corr):
    """AUROC of (1-conf) as a detector of errors; ties get average rank."""
    conf, corr = np.asarray(conf, float), np.asarray(corr, int)
    err = 1 - corr
    if err.sum() == 0 or err.sum() == len(err): return float("nan")
    s = pd.Series(1 - conf).rank(method="average").to_numpy()
    n1, n0 = err.sum(), len(err) - err.sum()
    return (s[err == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def aurc(conf, corr):
    """Area under risk-coverage curve; tied confidences are pooled into one point."""
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    pts, cov0 = [], 0.0
    vals = np.unique(conf)[::-1]; n = len(conf); area = 0.0; prev_cov, prev_risk = 0.0, None
    for v in vals:
        m = conf >= v; cov = m.mean(); risk = 1 - corr[m].mean()
        if prev_risk is None: area += cov * risk
        else: area += (cov - prev_cov) * (risk + prev_risk) / 2
        prev_cov, prev_risk = cov, risk
    return area

def summarize(name, conf, corr, n_total):
    return {"method": name, "n_total": n_total, "n_scored": len(conf),
            "coverage_parsed_%": round(100 * len(conf) / n_total, 2),
            "accuracy_scored_%": round(100 * float(np.mean(corr)), 2),
            "ECE_%": round(100 * ece(conf, corr), 2),
            "AUROC_error": round(auroc_error(conf, corr), 4),
            "AURC_%": round(100 * aurc(conf, corr), 2)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-r1:7b")
    p.add_argument("--dataset", required=True, choices=["ag_news", "dbpedia", "goemotions"])
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--mmv_csv", default=None, help="vote-count CSV (k>1) for MMV/SC comparison")
    a = p.parse_args()
    gm = None
    if a.dataset == "ag_news": texts, gold, task, labels = load_ag_news(max_samples=a.max_samples, seed=a.seed, shuffle=True)
    elif a.dataset == "dbpedia": texts, gold, task, labels = load_dbpedia(max_samples=a.max_samples, seed=a.seed, shuffle=True)
    else: texts, gold, task, labels, gm = load_goemotions(max_samples=a.max_samples, seed=a.seed, shuffle=True)
    mt = a.max_tokens or (1024 if "deepseek" in a.model.lower() else 64)
    client = OllamaClient(model=a.model, temperature=a.temperature, top_p=0.9, top_k=40,
                          repeat_penalty=1.1, num_ctx=4096, max_tokens=mt)
    rows = []
    for i, t in enumerate(tqdm(texts)):
        raw = client.classify_once(SYSTEM, build(t, labels, task))
        lab, conf = parse(raw, labels)
        ok = None if lab is None else int(lab in set(gm[i]) if gm is not None else lab == gold[i])
        rows.append({"id": i, "gold": gold[i], "pred": lab, "verbal_conf": conf, "correct": ok, "raw": raw})
    df = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    ok = df.dropna(subset=["verbal_conf", "correct"])
    out = [summarize("verbalized_single_call", ok.verbal_conf, ok.correct, len(df))]
    if a.mmv_csv:
        m = pd.read_csv(a.mmv_csv)
        v = m[m["top_votes"] > 0].copy(); v["sc_conf"] = v.top_votes / v.K
        sc_ok = v.assign(correct=v["correct"].fillna(0))
        out.append(summarize("SC_vote_share_all_valid", sc_ok.sc_conf, (sc_ok.pred == sc_ok.gold).astype(int), len(m)))
        c = m[m["pred"] != "ABSTAIN"]
        out.append(summarize("MMV_covered", c.confidence, c.correct, len(m)))
    s = pd.DataFrame(out); print(s.to_string(index=False))
    s.to_csv(a.out.replace(".csv", "_summary.csv"), index=False)

if __name__ == "__main__":
    main()
