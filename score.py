"""Stage 2: score the cached style traces. No generation happens here.

  NLI  : for every adjacent step pair (s_i -> s_{i+1}) in every style trace,
         store P(entail), P(neutral), P(contradict) from each NLI backbone.
         Trust mappings are applied later in analyze.py, so the six-mapping
         ablation costs nothing extra.
  Sim  : mean pairwise cosine of the three style traces (sentence embeddings).

Writes OUT_DIR/nli/<backbone>/<model>/<split>.jsonl and OUT_DIR/sim/<model>/<split>.jsonl.

Usage:  python score.py                 # everything that has been generated
        python score.py --backbones roberta
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
import prompts as P


def _gen_files():
    root = os.path.join(config.OUT_DIR, "gen")
    for m in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for s in config.SPLITS:
            p = os.path.join(root, m, f"{s}.jsonl")
            if os.path.exists(p):
                yield m, s, p


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


@torch.inference_mode()
def nli_probs(tok, model, pairs, order, bs=64):
    out = []
    for b in range(0, len(pairs), bs):
        prem, hyp = zip(*pairs[b:b + bs])
        enc = tok(list(prem), list(hyp), return_tensors="pt", padding=True, truncation=True,
                  max_length=config.NLI_MAX_LEN).to(model.device)
        p = torch.softmax(model(**enc).logits.float(), -1)[:, order]
        out.extend(p.cpu().tolist())
    return out


def run_nli(backbone):
    name = config.NLI_MODELS[backbone]
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float16)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    if model.device.type == "cpu":
        model = model.float()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    order = [lab["entailment"], lab["neutral"], lab["contradiction"]]  # -> [e, n, c]
    for m, s, gp in _gen_files():
        out_dir = os.path.join(config.OUT_DIR, "nli", backbone, m)
        os.makedirs(out_dir, exist_ok=True)
        op = os.path.join(out_dir, f"{s}.jsonl")
        recs = _load(gp)
        if os.path.exists(op) and len(_load(op)) == len(recs):
            print(f"[nli:{backbone}] {m}/{s}: up to date")
            continue
        pairs, index = [], []
        for r in recs:
            for t in r["style"]:
                st = P.split_steps(t["text"], config.MAX_STEPS)
                index.append(len(st) - 1 if len(st) > 1 else 0)
                pairs.extend(zip(st[:-1], st[1:]))
        probs = nli_probs(tok, model, pairs, order) if pairs else []
        k = j = 0
        with open(op, "w") as f:
            for r in recs:
                traces = []
                for _ in r["style"]:
                    ne = index[j]; j += 1
                    traces.append(probs[k:k + ne]); k += ne
                f.write(json.dumps({"id": r["id"], "edges": traces}) + "\n")
        print(f"[nli:{backbone}] {m}/{s}: {len(pairs)} edges")
    del model
    torch.cuda.empty_cache()


def run_sim():
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(config.EMBED_MODEL, device="cuda")
    for m, s, gp in _gen_files():
        out_dir = os.path.join(config.OUT_DIR, "sim", m)
        os.makedirs(out_dir, exist_ok=True)
        recs = _load(gp)
        texts = [t["text"] for r in recs for t in r["style"]]
        v = emb.encode(texts, normalize_embeddings=True, convert_to_tensor=True, batch_size=64)
        v = v.view(len(recs), config.K_STYLE, -1)
        sims = torch.einsum("ikd,ild->ikl", v, v)
        iu = torch.triu_indices(config.K_STYLE, config.K_STYLE, 1)
        cos = sims[:, iu[0], iu[1]].mean(-1).cpu().tolist()
        with open(os.path.join(out_dir, f"{s}.jsonl"), "w") as f:
            for r, c in zip(recs, cos):
                f.write(json.dumps({"id": r["id"], "cosine": c}) + "\n")
        print(f"[sim] {m}/{s}: mean cosine {sum(cos)/len(cos):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=list(config.NLI_MODELS))
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    for b in a.backbones:
        run_nli(b)
    if not a.skip_sim:
        run_sim()


if __name__ == "__main__":
    main()
