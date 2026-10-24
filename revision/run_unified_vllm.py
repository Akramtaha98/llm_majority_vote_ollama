"""Unified evaluation runner (vLLM). One job = one (model, dataset, repeat).

Protocol (answers R1-C3/C4/C5/C7):
  * SAME items for both models: seed-42 shuffled test split, first N (llm_vote.datasets loaders).
  * K_MAX stochastic calls per item are drawn ONCE; k=1,3,5 are the first k calls (nested), so
    cross-k comparisons are paired and 3x cheaper than separate runs.
  * Repeat r uses per-request seeds derived from r  -> three independent generation draws.
  * Every call logs: parsed label, #generated tokens, truncated flag (finish_reason == "length"
    or unclosed <think>), and, for the answer segment, the sequence probability of the answer tokens.
  * --score-direct (single deterministic pass, run it for rep 1): label-set probability baseline.
    p(label | prompt) for every label via prompt_logprobs, normalised over the label set.
    DeepSeek is scored with an empty think block ("<think>\n\n</think>\n\n") = no-reasoning logprob baseline.

Usage:
  python run_unified_vllm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tag deepseek \
      --dataset ag_news --n 7600 --rep 1 --score-direct --out runs/unified
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.environ.get("LLM_VOTE_REPO", os.environ.get("MMV_REPO", ".")), "src"))
from llm_vote.datasets import load_ag_news, load_dbpedia, load_goemotions  # noqa: E402
from llm_vote.prompting import EN_SYSTEM, build_prompt  # noqa: E402
from llm_vote.voter import normalize_label  # noqa: E402

INSTR = "Instruction: Respond with ONLY one label from the allowed list. No punctuation, no extra words.\n\n"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_call(text: str, labels: list[str]):
    """Same extraction rule as OllamaClient.classify_once + voter (case-sensitive exact match).
    Returns (label|None, truncated_in_think)."""
    if "<think>" in text.lower() and "</think>" not in text.lower():
        return None, True
    if "</think>" in text.lower() and "<think>" not in text.lower():  # template already opened <think>
        text = text.split("</think>", 1)[1]
    txt = THINK_RE.sub("", text).strip()
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    if ":" in first:
        first = first.split(":")[-1].strip()
    norm = normalize_label(first)
    for l in labels:
        if normalize_label(l) == norm:
            return l, False
    return None, False


def load_items(ds: str, n: int, seed: int):
    if ds == "ag_news":
        texts, gold, task, labels = load_ag_news(max_samples=n, seed=seed, shuffle=True); gm = None
    elif ds == "dbpedia":
        texts, gold, task, labels = load_dbpedia(max_samples=n, seed=seed, shuffle=True); gm = None
    else:
        texts, gold, task, labels, gm = load_goemotions(max_samples=n, seed=seed, shuffle=True)
    return texts, gold, task, labels, gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="deepseek | llama | large (large = extra non-reasoning model)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (use 2 for 70B on 2 GPUs)")
    ap.add_argument("--dataset", required=True, choices=["ag_news", "dbpedia", "goemotions"])
    ap.add_argument("--n", type=int, default=0, help="0 = full split")
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--rep-penalty", type=float, default=1.0, help="repetition penalty. vLLM penalises ALL prompt+output tokens (Ollama: last 64 only), so the label names in the prompt would be penalised; 1.0 (off) is used for every reported result, 1.1 only in the S6 sensitivity check")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 4096 reasoning / 32 otherwise")
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--score-direct", action="store_true")
    ap.add_argument("--verbal", action="store_true", help="single greedy call asking for Label + 0-100 Confidence (verbalized baseline) on the SAME items")
    ap.add_argument("--out", default="runs/unified")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=6144)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    out_csv = f"{a.out}/{a.dataset}_{a.tag}_rep{a.rep}.csv"
    if a.verbal: out_csv = f"{a.out}/{a.dataset}_{a.tag}_verbal.csv"
    if os.path.exists(out_csv):
        print("skip", out_csv); return

    from vllm import LLM, SamplingParams
    reasoning = a.tag == "deepseek"
    mt = a.max_tokens or (4096 if reasoning else 32)

    texts, gold, task, labels, gm = load_items(a.dataset, a.n or None, a.data_seed)
    N = len(texts)
    print(f"[{a.tag}/{a.dataset}/rep{a.rep}] N={N} labels={len(labels)} max_tokens={mt}", flush=True)

    llm = LLM(model=a.model, dtype="bfloat16", tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enable_prefix_caching=True, seed=a.rep)
    tok = llm.get_tokenizer()
    think_end_id = None
    if reasoning:
        ids = tok.encode("</think>", add_special_tokens=False)
        think_end_id = ids[0] if len(ids) == 1 else None

    def chat(user_msg: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True)

    if a.verbal:
        VS = ("You are a careful text classifier. Return exactly two lines:\nLabel: <one label from the allowed list>\n"
              "Confidence: <integer 0-100, your probability that the label is correct>")
        vp = [chat(VS + "\n\nTask: Single-label classification for %s.\nAllowed labels: [%s]\nText:\n%s\nRespond in the required two-line format." % (task, ", ".join(labels), t)) for t in texts]
        outs = llm.generate(vp, SamplingParams(n=1, temperature=0.0, max_tokens=mt, seed=0))
        rows = []
        for i, o in enumerate(outs):
            c = o.outputs[0]; raw = re.sub(r"<think>.*?</think>", "", c.text, flags=re.S)
            if "<think>" in c.text and "</think>" not in c.text: raw = ""
            if "</think>" in raw: raw = raw.split("</think>", 1)[1]
            ml = re.search(r"Label\s*:\s*(.+)", raw); mc = re.search(r"Confidence\s*:\s*(\d+(?:\.\d+)?)", raw)
            lab = None; conf = None
            if ml and mc:
                l = re.sub(r"\s+", " ", ml.group(1).strip().strip("[]\"'."))
                if l in labels: lab, conf = l, min(max(float(mc.group(1)), 0), 100) / 100
            rows.append({"id": i, "gold": gold[i], "gold_multi": "|".join(gm[i]) if gm is not None else "", "pred": lab, "verbal_conf": conf,
                         "toks": len(c.token_ids), "trunc": int(c.finish_reason == "length")})
        pd.DataFrame(rows).to_csv(out_csv, index=False); print("saved", out_csv); return

    # identical content to OllamaClient.classify_once: system + instruction + user prompt as one user turn
    prompts = [chat(EN_SYSTEM + "\n\n" + INSTR + build_prompt(t, labels, task)) for t in texts]
    sps = [SamplingParams(n=a.k_max, temperature=max(a.temperature, 0.01), top_p=0.9, top_k=40, repetition_penalty=a.rep_penalty,
                          max_tokens=mt, seed=10_007 * a.rep + i, logprobs=0) for i in range(N)]
    t0 = time.time()
    outs = llm.generate(prompts, sps)
    print(f"generation done in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for i, o in enumerate(outs):
        calls, toks, trunc, pfin, ans = [], [], [], [], []
        for c in o.outputs:
            text = c.text
            lab, think_cut = parse_call(text, labels)
            was_trunc = bool(c.finish_reason == "length") or think_cut
            calls.append(lab); toks.append(len(c.token_ids)); trunc.append(int(was_trunc))
            # sequence probability of the final-answer segment (after </think> for reasoning model)
            start = 0
            if think_end_id is not None and think_end_id in c.token_ids:
                start = c.token_ids.index(think_end_id) + 1
            lp = 0.0
            if c.logprobs:
                for tid, d in zip(c.token_ids[start:], c.logprobs[start:]):
                    if d and tid in d:
                        lp += d[tid].logprob
            pfin.append(round(math.exp(lp), 6) if not think_cut else None)
            ans.append(text[-60:].replace("\n", " "))
        rows.append({"id": i, "gold": gold[i], "gold_multi": "|".join(gm[i]) if gm is not None else "",
                     "n_gold": len(gm[i]) if gm is not None else 1,
                     "calls": json.dumps(calls), "toks": json.dumps(toks), "trunc": json.dumps(trunc),
                     "p_answer_seq": json.dumps(pfin), "ans_tail": json.dumps(ans, ensure_ascii=False)})
    df = pd.DataFrame(rows)

    if a.score_direct:
        # deterministic label-set scoring: log p(label tokens | prompt [+ empty think block])
        base = []
        for p in prompts:
            if reasoning:
                if not p.rstrip().endswith("<think>"):
                    p = p + "<think>\n"
                p = p + "\n</think>\n\n"
            base.append(p)
        plen = [len(tok(p, add_special_tokens=False).input_ids) for p in base]
        sp1 = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        d_pred, d_conf, d_gap = [], [], []
        B = 256
        for s in range(0, N, B):
            reqs, meta = [], []
            for j in range(s, min(N, s + B)):
                for li, l in enumerate(labels):
                    reqs.append(base[j] + l); meta.append((j, li))
            res = llm.generate(reqs, sp1)
            lps = np.full((min(N, s + B) - s, len(labels)), -1e9)
            for (j, li), r in zip(meta, res):
                ids = r.prompt_token_ids; pl = r.prompt_logprobs
                lps[j - s, li] = sum(pl[t][ids[t]].logprob for t in range(plen[j], len(ids)) if pl[t] and ids[t] in pl[t])
            for row in lps:
                pr = np.exp(row - row.max()); pr /= pr.sum()
                o = np.argsort(-pr)
                d_pred.append(labels[o[0]]); d_conf.append(float(pr[o[0]])); d_gap.append(float(pr[o[0]] - pr[o[1]]))
        df["direct_pred"], df["direct_conf"], df["direct_margin"] = d_pred, d_conf, d_gap

    df.to_csv(out_csv, index=False)
    print("saved", out_csv, f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
