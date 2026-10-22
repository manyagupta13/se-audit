"""Stage 1: generate every candidate pool, plus the LLM-judge's picks.

For each (model, split) this writes, under OUT_DIR/gen/<model>/:
  <split>.jsonl        one line per item: 3 style traces + 5 self-consistency samples
  <split>.judge.jsonl  one line per item: the judge's raw output and its pick

Both files are append-only and resumable: rerun the same command after a
Colab disconnect and it continues where it stopped. Nothing downstream ever
regenerates, so every table is computed from the same pool.

Usage:
  python generate.py --model qwen1.5b              # all splits, gen + judge
  python generate.py --model phi2 --splits wol4 ma1
  python generate.py --model qwen3b --stage judge
"""
import argparse
import gc
import json
import os
import random
import time
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
import data
import prompts as P
from io_utils import append_jsonl, read_jsonl, repair_jsonl


def _done_ids(path):
    return {r["id"] for r in read_jsonl(path)}


def _paths(model_key, split):
    d = os.path.join(config.OUT_DIR, "gen", model_key)
    return os.path.join(d, f"{split}.jsonl"), os.path.join(d, f"{split}.judge.jsonl")


def split_complete(model_key, split, stage="all", limit=None):
    """True if this split needs no more work, so the model need not even be loaded."""
    want = {it["id"] for it in data.load(split)[:limit]}
    gen_p, judge_p = _paths(model_key, split)
    gen_ok = want <= _done_ids(gen_p)
    judge_ok = want <= _done_ids(judge_p)
    return {"gen": gen_ok, "judge": judge_ok, "all": gen_ok and judge_ok}[stage]


def load_model(model_key, dtype):
    name = config.MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:  # transformers >= 4.56 uses `dtype`; older versions use `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return tok, model


@torch.inference_mode()
def run_generate(tok, model, model_key, texts, n_return, sample=True, max_new=None):
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=(model_key == "phi2")).to(model.device)
    kw = dict(max_new_tokens=max_new or config.MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id,
              num_return_sequences=n_return)
    if sample:
        kw.update(do_sample=True, temperature=config.TEMPERATURE, top_p=config.TOP_P)
    else:
        kw.update(do_sample=False)
    if model_key == "phi2":
        kw.update(stop_strings=P.PHI2_STOPS, tokenizer=tok)
    out = model.generate(**enc, **kw)
    new = out[:, enc["input_ids"].shape[1]:]
    return [P.clean(t, model_key) for t in tok.batch_decode(new, skip_special_tokens=True)]


def _seed(model_key, split, first_id):
    s = zlib.crc32(f"{config.SEED}|{model_key}|{split}|{first_id}".encode())
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def gen_split(tok, model, model_key, split, limit=None):
    items = data.load(split)[:limit]
    out_dir = os.path.join(config.OUT_DIR, "gen", model_key)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split}.jsonl")
    repair_jsonl(path)          # drop a line cut off by a crash before appending after it
    done = _done_ids(path)
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        print(f"[gen] {model_key}/{split}: already complete")
        return
    chunk = max(1, config.GEN_BATCH[model_key] // config.K_SC)
    t0 = time.time()
    for b in range(0, len(todo), chunk):
        batch = todo[b:b + chunk]
        _seed(model_key, split, batch[0]["id"])
        style_texts = [P.format_prompt(tok, model_key, sys_p, P.task_user(it))
                       for it in batch for _, sys_p in P.STYLES]
        style_out = run_generate(tok, model, model_key, style_texts, 1)
        sc_texts = [P.format_prompt(tok, model_key, P.NEUTRAL_SYSTEM, P.task_user(it)) for it in batch]
        sc_out = run_generate(tok, model, model_key, sc_texts, config.K_SC)
        out = []
        for i, it in enumerate(batch):
            rec = {k: it[k] for k in ("id", "question", "answer", "type")}
            rec["style"] = []
            for j, (persona, _) in enumerate(P.STYLES):
                txt = style_out[i * config.K_STYLE + j]
                pred = P.extract(txt, it["type"])
                rec["style"].append({"persona": persona, "text": txt, "pred": pred,
                                     "correct": P.is_correct(pred, it["answer"], it["type"])})
            rec["sc"] = []
            for j in range(config.K_SC):
                txt = sc_out[i * config.K_SC + j]
                pred = P.extract(txt, it["type"])
                rec["sc"].append({"text": txt, "pred": pred,
                                  "correct": P.is_correct(pred, it["answer"], it["type"])})
            out.append(rec)
        append_jsonl(path, out)     # one checkpoint per batch, forced to disk
        n = b + len(batch)
        print(f"[gen] {model_key}/{split}: {n}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)


def _tail_tokens(tok, text, n):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= n else "..." + tok.decode(ids[-n:])


def judge_split(tok, model, model_key, split):
    gen_path, path = _paths(model_key, split)
    recs = read_jsonl(gen_path)
    repair_jsonl(path)
    done = _done_ids(path)
    todo = [r for r in recs if r["id"] not in done]
    if not todo:
        print(f"[judge] {model_key}/{split}: already complete")
        return
    n_tok = config.JUDGE_TRACE_TOKENS[model_key]
    bs = max(1, config.GEN_BATCH[model_key] // 4)
    for b in range(0, len(todo), bs):
        batch = todo[b:b + bs]
        texts, perms = [], []
        for r in batch:
            # Show the three traces in a per-item random order, so a judge with a
            # position bias (e.g. "always 1") cannot systematically favour one persona.
            perm = random.Random(zlib.crc32(f"{config.SEED}|judge|{model_key}|{split}|{r['id']}".encode())).sample(range(config.K_STYLE), config.K_STYLE)
            perms.append(perm)
            t = [_tail_tokens(tok, r["style"][a]["text"], n_tok) for a in perm]
            user = P.JUDGE_TEMPLATE.format(question=r["question"], t1=t[0], t2=t[1], t3=t[2])
            texts.append(P.format_prompt(tok, model_key, P.NEUTRAL_SYSTEM, user))
        outs = run_generate(tok, model, model_key, texts, 1, sample=False, max_new=8)
        out = []
        for r, o, perm in zip(batch, outs, perms):
            pos = P.parse_judge(o)   # displayed position the judge chose (0-based) or None
            out.append({"id": r["id"], "raw": o, "perm": perm, "pos": pos,
                                "pick": perm[pos] if pos is not None else None})
        append_jsonl(path, out)
    print(f"[judge] {model_key}/{split}: done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.MODELS))
    ap.add_argument("--splits", nargs="*", default=config.SPLITS)
    ap.add_argument("--stage", choices=["gen", "judge", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N items per split (smoke tests; use a separate SE_OUT)")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    a = ap.parse_args()
    data.build() if not os.path.exists(os.path.join(config.DATA_DIR, "ma2.jsonl")) else None
    todo = [s for s in a.splits if not split_complete(a.model, s, a.stage, a.limit)]
    if not todo:
        print(f"[skip] {a.model}: all requested splits already complete", flush=True)
        return
    tok, model = load_model(a.model, getattr(torch, a.dtype))
    for s in a.splits:
        if a.stage in ("gen", "all"):
            gen_split(tok, model, a.model, s, a.limit)
        if a.stage in ("judge", "all"):
            judge_split(tok, model, a.model, s)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
