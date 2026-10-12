"""Build the four evaluation splits deterministically and cache them as JSONL.

  wol4 : 50 items sampled from BIG-Bench-Hard web_of_lies (chain length 4)
  wol5 : 50 items generated in the identical BBH template at chain length 5
         (BBH ships only length 4), answers balanced Yes/No
  ma1  : 50 MultiArith problems whose equation uses only + and -
  ma2  : 50 MultiArith problems whose equation uses * or /
         (every MultiArith problem has exactly two operations, so operator
          type is the complexity axis)

Run:  python data.py
"""
import json
import os
import random
import urllib.request

import config
import wol

BBH_WOL_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/web_of_lies.json"
MULTIARITH_URL = "https://raw.githubusercontent.com/kojima-takeshi188/zero_shot_cot/main/dataset/MultiArith/MultiArith.json"


def _fetch(url, name):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        return json.load(f)


def _write(split, items):
    path = os.path.join(config.DATA_DIR, f"{split}.jsonl")
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
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

    # --- MultiArith --------------------------------------------------------
    ma = _fetch(MULTIARITH_URL, "multiarith.json")
    add = [e for e in ma if not any(o in e["lEquations"][0] for o in "*/")]
    mul = [e for e in ma if any(o in e["lEquations"][0] for o in "*/")]
    for split, pool in (("ma1", add), ("ma2", mul)):
        pick = rng.sample(pool, n)
        _write(split, [{"id": f"{split}-{e['iIndex']}", "question": e["sQuestion"].strip(),
                        "answer": float(e["lSolutions"][0]), "type": "num",
                        "source": f"multiarith:{e['iIndex']}", "equation": e["lEquations"][0]} for e in pick])


def load(split):
    path = os.path.join(config.DATA_DIR, f"{split}.jsonl")
    if not os.path.exists(path):
        build()
    with open(path) as f:
        return [json.loads(l) for l in f]


if __name__ == "__main__":
    build()
