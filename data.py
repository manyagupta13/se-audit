"""Build the four evaluation splits deterministically and cache them as JSONL.

  wol4 : 50 items sampled from BIG-Bench-Hard web_of_lies (chain length 4)
  wol5 : 50 items generated in the identical BBH template at chain length 5
         (BBH ships only length 4), answers balanced Yes/No
  gsm3 : 50 GSM8K test problems whose reference solution has exactly 3 calculation steps
  gsm5 : 50 GSM8K test problems whose reference solution has 5 or more calculation steps
         (a step is one <<...>> calculator annotation in the reference solution)

Selection uses only the reference solutions, never any model output, and is
fixed before any model is run on these splits.

The original MA-1 / MA-2 (MultiArith) splits were dropped because they sit at
the accuracy ceiling for the Qwen models, which leaves selection efficiency
undefined or noise-dominated. Every MultiArith problem has exactly two
operations and the additive ones share one equation shape, so no fair
difficulty rule can create headroom there. The old MA outputs are in the earlier
raw_outputs.zip; they are no longer produced or analysed.

Run:  python data.py
"""
import json
import os
import random
import re
import urllib.request

import config
import wol
from io_utils import read_jsonl, write_jsonl_atomic

BBH_WOL_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/web_of_lies.json"
GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def _fetch(url, name):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, name)
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)   # never leave a half-downloaded file behind
    with open(path) as f:
        if name.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()]
        return json.load(f)


def _write(split, items):
    """Write a split, but never silently change one that already has data.

    Generations are matched to items by id, so rewriting an existing split with
    different content would let stale outputs be mistaken for current ones.
    """
    path = os.path.join(config.DATA_DIR, f"{split}.jsonl")
    if os.path.exists(path):
        if read_jsonl(path) == items:
            return
        raise SystemExit(f"{path} already exists with different content. Refusing to overwrite it, "
                         "because saved generations are keyed to it. Delete it (and "
                         f"gen/*/{split}*.jsonl, nli/*/*/{split}.jsonl, sim/*/{split}.jsonl) on purpose if you want a new split.")
    write_jsonl_atomic(path, items)
    print(f"{split}: {len(items)} items -> {path}")


def build():
    rng = random.Random(config.SEED)
    n = config.N_PER_SPLIT

    # --- Web of Lies -------------------------------------------------------
    bbh = _fetch(BBH_WOL_URL, "bbh_web_of_lies.json")["examples"]
    for e in bbh:  # sanity: the symbolic solver must agree with every BBH label
        assert wol.solve(e["input"])[1] == e["target"], e
    picked = rng.sample(range(len(bbh)), n)
    _write("wol4", [{"id": f"wol4-{i}", "question": bbh[i]["input"].removeprefix("Question: "),
                     "answer": bbh[i]["target"], "type": "bool", "source": f"bbh:{i}"} for i in picked])

    names = wol.names_from_bbh(bbh)
    items, seen, want = [], set(), {"Yes": n // 2, "No": n - n // 2}
    while len(items) < n:
        q, a = wol.make_item(5, names, rng)
        if q in seen or want[a] == 0:
            continue
        seen.add(q)
        want[a] -= 1
        items.append({"id": f"wol5-{len(items)}", "question": q.removeprefix("Question: "),
                      "answer": a, "type": "bool", "source": "generated:len5"})
    _write("wol5", items)

    # --- GSM8K by number of calculation steps -------------------------------
    gsm = _fetch(GSM8K_TEST_URL, "gsm8k_test.jsonl")
    pools = {"gsm3": [], "gsm5": []}
    for idx, e in enumerate(gsm):
        steps = len(re.findall(r"<<.*?>>", e["answer"]))
        gold = e["answer"].split("####")[-1].strip().replace(",", "")
        try:
            gold = float(gold)
        except ValueError:
            continue
        item = {"question": e["question"].strip(), "answer": gold, "type": "num",
                "source": f"gsm8k-test:{idx}", "ref_steps": steps}
        if steps == 3:
            pools["gsm3"].append((idx, item))
        elif steps >= 5:
            pools["gsm5"].append((idx, item))
    grng = random.Random(f"gsm-{config.SEED}")   # own stream: leaves the WoL items untouched
    for split, pool in pools.items():
        pick = grng.sample(pool, n)
        _write(split, [dict(it, id=f"{split}-{idx}") for idx, it in pick])


def load(split):
    path = os.path.join(config.DATA_DIR, f"{split}.jsonl")
    if not os.path.exists(path):
        build()
    return read_jsonl(path)


if __name__ == "__main__":
    build()
