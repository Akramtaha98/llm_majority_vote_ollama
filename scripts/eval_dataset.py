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
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repeat-penalty", type=float, default=1.1)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-shuffle", action="store_true",
                    help="Disable the dataset-level shuffle and use an unshuffled, "
                         "first-N slice instead. A post-submission audit (manuscript "
                         "Section 7.5) found that AG News DeepSeek-R1:7B at k = 1 and "
                         "all AG News LLaMA-3.2:3B conditions in this paper's released "
                         "records were originally collected this way, not with the "
                         "shuffle; pass this flag to reproduce those specific conditions.")
    p.add_argument("--preds", default=None)
    args = p.parse_args()

    # `--seed` is the dataset-level shuffle seed (Section 4.2): applied once per
    # dataset, before any model calls, to deterministically shuffle the full test
    # split prior to slicing. It is unrelated to per-call generation randomness --
    # no generation seed is ever passed to the Ollama API (see OllamaClient). Not
    # every condition in this paper's released records was actually collected with
    # this shuffle applied -- see Section 7.5 and --no-shuffle above.
    random.seed(args.seed)
    do_shuffle = not args.no_shuffle

    if args.dataset == "ag_news":
        texts, gold, task, label_names = load_ag_news(max_samples=args.max_samples, seed=args.seed, shuffle=do_shuffle)
        sys_prompt = EN_SYSTEM
    elif args.dataset == "dbpedia":
        texts, gold, task, label_names = load_dbpedia(max_samples=args.max_samples, seed=args.seed, shuffle=do_shuffle)
        sys_prompt = EN_SYSTEM
    elif args.dataset == "goemotions":
        texts, gold_primary, task, label_names, gold_multi = load_goemotions(max_samples=args.max_samples, seed=args.seed, shuffle=do_shuffle)
        gold = gold_primary
        sys_prompt = EN_SYSTEM
    else:
        texts, gold, task, label_names = load_labr_csv(args.labr_csv, max_samples=args.max_samples, seed=args.seed)
        sys_prompt = AR_SYSTEM

    # Choose client
    if args.provider == "ollama":
        client = OllamaClient(
            model=args.model, temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, repeat_penalty=args.repeat_penalty, num_ctx=args.num_ctx,
            max_tokens=args.max_tokens,
        )
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
            labels=label_names, K=K, abstain_threshold=args.abstain, early_stop=args.early_stop
        )
        if pred is None:
            conf, top_label, top = 0.0, "", 0
            is_correct = False
        else:
            conf, top_label, top = votes_to_confidence(votes, K)
            if args.dataset == "goemotions":
                is_correct = pred in set(gold_multi[i])
            else:
                is_correct = (pred == gold[i])

        preds.append(pred)
        confs.append(conf)
        corrects.append(is_correct)

        rows.append({
            "id": i,
            "gold": gold[i] if args.dataset != "goemotions" else "|".join(gold_multi[i]),
            "pred": pred if pred is not None else "ABSTAIN",
            "text": text,
            "votes": json.dumps(votes, ensure_ascii=False),
            "top_votes": top,
            "K": K,
            "confidence": round(conf, 4),
            "correct": int(is_correct),
        })

    acc = accuracy(gold, preds)
    mf1 = macro_f1(gold, preds, label_names)
    mcc_val = mcc(gold, preds, label_to_idx)
    ans_confs = [c for p,c in zip(preds, confs) if p is not None]
    ans_corrs = [c for p,c in zip(preds, corrects) if p is not None]
    ece = expected_calibration_error(ans_confs, ans_corrs, n_bins=15)
    covered = sum(1 for p in preds if p is not None) / len(preds) if preds else 0.0

    print(f"Provider: {args.provider}   Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"K={K}  Abstain<thresh?: {args.abstain}  EarlyStop: {args.early_stop}")
    print(f"Accuracy (answered): {acc*100:.2f}%")
    print(f"Macro-F1 (answered): {mf1*100:.2f}%")
    print(f"MCC (answered): {mcc_val:.3f}")
    print(f"ECE: {ece*100:.2f}%")
    print(f"Coverage (answered%): {covered*100:.2f}%")

    os.makedirs("runs", exist_ok=True)
    out_csv = args.preds or f"runs/{args.dataset}_{args.provider}_{args.model.replace(':','-')}_k{K}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv}")

if __name__ == "__main__":
    main()
