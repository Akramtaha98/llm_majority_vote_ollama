from __future__ import annotations
import argparse, json, random, os
from typing import List, Optional, Dict
from tqdm import tqdm
import pandas as pd

from llm_vote.prompting import EN_SYSTEM, AR_SYSTEM, build_prompt
from llm_vote.voter import majority_vote_single
from llm_vote.metrics import accuracy, macro_f1, mcc, expected_calibration_error
from llm_vote.datasets import load_ag_news, load_dbpedia, load_goemotions, load_labr_csv
from llm_vote.utils import votes_to_confidence

# Providers
from llm_vote.ollama_client import OllamaClient
try:
    from llm_vote.openai_client import OpenAIClient
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["ollama","openai"], default="ollama")
    p.add_argument("--model", default="llama3.2:3b-instruct")
    p.add_argument("--dataset", required=True, choices=["ag_news","dbpedia","goemotions","labr"])
    p.add_argument("--labr_csv", default="data/labr_binary_test.csv")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--abstain", type=float, default=0.0)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preds", default=None)
    args = p.parse_args()

    random.seed(args.seed)

    if args.dataset == "ag_news":
        texts, gold, task, label_names = load_ag_news(max_samples=args.max_samples)
        sys_prompt = EN_SYSTEM
    elif args.dataset == "dbpedia":
        texts, gold, task, label_names = load_dbpedia(max_samples=args.max_samples)
        sys_prompt = EN_SYSTEM
    elif args.dataset == "goemotions":
        texts, gold_primary, task, label_names, gold_multi = load_goemotions(max_samples=args.max_samples)
        gold = gold_primary
        sys_prompt = EN_SYSTEM
    else:
        texts, gold, task, label_names = load_labr_csv(args.labr_csv, max_samples=args.max_samples)
        sys_prompt = AR_SYSTEM

    # Choose client
    if args.provider == "ollama":
        client = OllamaClient(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    else:
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI client not available. Install 'openai' and set OPENAI_API_KEY.")
        client = OpenAIClient(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)

    preds: List[Optional[str]] = []
    confs: List[float] = []
    corrects: List[bool] = []
    label_to_idx = {l:i for i,l in enumerate(label_names)}
    rows = []
    K = args.k

    for i, text in enumerate(tqdm(texts, desc="Classifying")):
        user_prompt = build_prompt(text, label_names, task)
        pred, votes = majority_vote_single(
            client, sys_prompt, user_prompt,
            labels=label_names, K=K, abstain_threshold=args.abstain,
            early_stop=args.early_stop
        )
        if pred is None:
            conf, top_label, top = 0.0, "", 0
            is_correct = False
        else:
            conf, top_label, top = votes_to_confidence(votes, K)
            if args.dataset == "goemotions":
                is_correct = pred in set(gold_multi[i]) if gold_multi else (pred == gold[i])
            else:
                is_correct = (pred == gold[i])

        preds.append(pred)
        confs.append(conf)
        corrects.append(is_correct)

        rows.append({
            "id": i,
            # gold = primary label (single string, never blank)
            "gold": gold[i],
            # gold_multi = pipe-separated multi-labels for GoEmotions
            "gold_multi": "|".join(gold_multi[i]) if args.dataset == "goemotions" and gold_multi else gold[i],
            "pred": pred if pred is not None else "ABSTAIN",
            "text": text,
            "votes": json.dumps(votes, ensure_ascii=False),
            "top_votes": top,
            "K": K,
            "confidence": round(conf, 4),
            "correct": int(is_correct),
        })

    acc  = accuracy(gold, preds)
    mf1  = macro_f1(gold, preds, label_names)
    mcc_val = mcc(gold, preds, label_to_idx)
    ans_confs = [c for p,c in zip(preds, confs) if p is not None]
    ans_corrs = [c for p,c in zip(preds, corrects) if p is not None]
    # Paper specifies 15-bin ECE
    ece = expected_calibration_error(ans_confs, ans_corrs, n_bins=15)
    covered = sum(1 for p in preds if p is not None) / len(preds) if preds else 0.0

    print(f"\nProvider : {args.provider}   Model: {args.model}")
    print(f"Dataset  : {args.dataset}   K={K}   Samples={len(preds)}")
    print(f"Accuracy (covered): {acc*100:.2f}%")
    print(f"Macro-F1 (covered): {mf1*100:.2f}%")
    print(f"MCC      (covered): {mcc_val:.4f}")
    print(f"ECE (15-bin)      : {ece*100:.2f}%")
    print(f"Coverage          : {covered*100:.2f}%")

    os.makedirs("runs", exist_ok=True)
    out_csv = args.preds or f"runs/{args.dataset}_{args.provider}_{args.model.replace(':','-').replace('/','-')}_k{K}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

if __name__ == "__main__":
    main()
