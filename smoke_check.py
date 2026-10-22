"""Sanity-check a small generation run before launching the full one.

Prints, per split: answer parse rate, per-agent accuracy, oracle, and one
sample trace, so you can see whether the prompts, answer extraction and judge
parsing behave on the real models.   Usage: SE_OUT=<dir> python smoke_check.py
"""
import json
import os
import sys

import config
from io_utils import read_jsonl


def jl(p):
    return read_jsonl(p)


root = os.path.join(config.OUT_DIR, "gen")
if not os.path.isdir(root):
    sys.exit("nothing generated under " + root)
for m in sorted(os.listdir(root)):
    for s in config.SPLITS:
        G = jl(os.path.join(root, m, f"{s}.jsonl"))
        if not G:
            continue
        J = jl(os.path.join(root, m, f"{s}.judge.jsonl"))
        st = [t for r in G for t in r["style"]]
        sc = [t for r in G for t in r["sc"]]
        parse = lambda ts: 100 * sum(t["pred"] is not None for t in ts) / len(ts)
        fmt = lambda ts: 100 * sum("final answer" in t["text"].lower() for t in ts) / len(ts)
        acc = lambda ts: 100 * sum(t["correct"] for t in ts) / len(ts)
        orc = 100 * sum(any(t["correct"] for t in r["style"]) for r in G) / len(G)
        jp = 100 * sum(j["pick"] is not None for j in J) / len(J) if J else float("nan")
        print(f"\n=== {m}/{s}  n={len(G)} ===")
        print(f"style: has 'Final answer' line {fmt(st):.0f}%  parsed {parse(st):.0f}%  correct {acc(st):.0f}%  oracle {orc:.0f}%   "
              f"neutral: has line {fmt(sc):.0f}%  parsed {parse(sc):.0f}%  correct {acc(sc):.0f}%   judge parsed {jp:.0f}%")
        bad = next((t for t in st if t["pred"] is None), None)
        print("--- sample style trace (formal) ---\n" + G[0]["style"][0]["text"][-500:])
        if bad:
            print("--- an UNPARSED trace (tail) ---\n" + bad["text"][-300:])
        if J:
            pos = [j.get("pos") for j in J]
            print("--- judge: displayed position chosen (1/2/3/unparsed):",
                  [sum(p == i for p in pos) for i in range(3)], sum(p is None for p in pos),
                  "| raw:", [j["raw"][:12] for j in J[:5]])
