"""Stage 3: compute every number in the paper from the cached outputs.

No model is loaded here; it runs on CPU in seconds. Every table is emitted
as LaTeX from the same per-item records, so the tables cannot disagree.

Outputs (OUT_DIR/tables/):
  tab_methods.tex       Table: mean SE by selector (all / Qwen)
  tab_methods_full.tex  Appendix: per-configuration SE, four selectors
  tab_toth.tex          Appendix: ToTh accuracy, random, oracle, SE
  tab_pool.tex          NEW: pool-vs-selector control (same selector, different pool; k-matched)
  tab_sim.tex           Appendix: trace similarity / agreement
  tab_decomp.tex        Appendix: gain decomposition
  tab_nli_ablation.tex  Appendix: RoBERTa vs BART NLI backbone, McNemar
  tab_mappings.tex      Appendix: ToTh accuracy under six trust mappings + inverted
  tab_verifier.tex      Appendix: symbolic verifier on Web of Lies
  tab_smol.tex          Appendix: SmolLM2 capability-floor case
  results.json          every per-config number
  claims.md             each quantitative claim in the text, recomputed

Definitions (state these in the paper):
  A_rand   expected accuracy of picking one style trace uniformly at random
           = mean per-item fraction of correct style traces (no random draw)
  A_oracle fraction of items with >=1 correct style trace
  SE       (A_sys - A_rand) / (A_oracle - A_rand); None if the gap is 0
  Shared-reference SE uses the style pool's A_rand and A_oracle for every
  selector. Own-pool SE (self-consistency only) uses its own samples.
  Ties (plurality votes, equal trust scores) are scored in expectation over
  the tied candidates, so no random tie-break enters any number.
"""
import itertools
import json
import math
import os
from collections import Counter

from scipy.stats import binomtest, combine_pvalues

import config
import wol
from io_utils import read_jsonl, write_text_atomic

T = os.path.join(config.OUT_DIR, "tables")
MAPPINGS = ["V0", "V1", "V2", "V3", "V4", "V5"]


# ---------------------------------------------------------------- loading ----
def _jl(path):
    return read_jsonl(path) if os.path.exists(path) else None


def load(model, split):
    g = _jl(os.path.join(config.OUT_DIR, "gen", model, f"{split}.jsonl"))
    if not g:
        return None
    j = _jl(os.path.join(config.OUT_DIR, "gen", model, f"{split}.judge.jsonl")) or []
    nli = {}
    for b in config.NLI_MODELS:
        x = _jl(os.path.join(config.OUT_DIR, "nli", b, model, f"{split}.jsonl"))
        if x:
            nli[b] = {r["id"]: r["edges"] for r in x}
    sim = _jl(os.path.join(config.OUT_DIR, "sim", model, f"{split}.jsonl")) or []
    return {"gen": g, "judge": {r["id"]: r["pick"] for r in j}, "judge_pos": [r.get("pos") for r in j], "nli": nli,
            "sim": {r["id"]: r["cosine"] for r in sim}}


# ------------------------------------------------------------ primitives ----
def _key(p):
    return None if p is None else (round(p, 6) if isinstance(p, float) else p)


def plurality(cands):
    """Expected accuracy of a plurality vote with uniform tie-breaking."""
    votes = Counter(_key(c["pred"]) for c in cands if c["pred"] is not None)
    if not votes:
        return sum(c["correct"] for c in cands) / len(cands)
    top = max(votes.values())
    winners = [a for a, v in votes.items() if v == top]
    right = {_key(c["pred"]) for c in cands if c["correct"]}
    return sum(a in right for a in winners) / len(winners)


def argmax_expected(scores, correct):
    best = max(scores)
    tied = [c for s, c in zip(scores, correct) if abs(s - best) < 1e-12]
    return sum(tied) / len(tied)


def argmax_first(scores):
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def trust(edges, mapping):
    if not edges:
        return 0.0
    e = [x[0] for x in edges]; n = [x[1] for x in edges]; c = [x[2] for x in edges]
    m = len(edges)
    if mapping == "V0":   # mean (entail - contradict)
        return sum(a - b for a, b in zip(e, c)) / m
    if mapping == "V1":   # mean entail
        return sum(e) / m
    if mapping == "V2":   # neutral as half credit
        return sum(a + 0.5 * b for a, b in zip(e, n)) / m
    if mapping == "V3":   # mean log-odds entail vs contradict
        return sum(math.log((a + 1e-6) / (b + 1e-6)) for a, b in zip(e, c)) / m
    if mapping == "V4":   # weakest link
        return min(a - b for a, b in zip(e, c))
    if mapping == "V5":   # hard labels: frac entail - frac contradict
        lab = [max(range(3), key=lambda k: x[k]) for x in edges]
        return (lab.count(0) - lab.count(2)) / m
    if mapping == "INV":  # inverted V0
        return -trust(edges, "V0")
    raise ValueError(mapping)


def se(acc, rand, oracle):
    gap = oracle - rand
    return None if abs(gap) < 1e-9 else (acc - rand) / gap


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(a, b):
    a, b = a.lower().split(), b.lower().split()
    l = lcs(a, b)
    if l == 0:
        return 0.0
    p, r = l / len(b), l / len(a)
    return 2 * p * r / (p + r)


# ------------------------------------------------------- per-config stats ----
def config_stats(model, split):
    d = load(model, split)
    if d is None:
        return None
    G = d["gen"]
    n = len(G)
    K = config.K_STYLE
    S = {"model": model, "split": split, "n": n}

    corr = [[t["correct"] for t in r["style"]] for r in G]
    S["agent_acc"] = [sum(c[k] for c in corr) / n for k in range(K)]
    S["rand"] = sum(sum(c) / K for c in corr) / n
    S["single_p"] = S["rand"]
    S["oracle"] = sum(any(c) for c in corr) / n
    prod = 1.0
    for p in S["agent_acc"]:
        prod *= (1 - p)
    S["indep_ceil"] = 1 - prod
    S["corr_loss"] = S["indep_ceil"] - S["oracle"]

    acc = {}
    acc["majority"] = sum(plurality(r["style"]) for r in G) / n
    acc["sc"] = sum(plurality(r["sc"]) for r in G) / n
    combos = list(itertools.combinations(range(config.K_SC), config.K_MATCHED))
    acc["sc_k3"] = sum(sum(plurality([r["sc"][i] for i in cb]) for cb in combos) / len(combos) for r in G) / n
    # judge: unparseable pick -> random choice in expectation
    jp = [d["judge"].get(r["id"]) for r in G]
    S["judge_coverage"] = sum(p is not None for p in jp) / n if d["judge"] else 0.0
    S["judge_pos_counts"] = [d["judge_pos"].count(i) for i in range(K)]   # by displayed position
    acc["judge"] = (sum((c[p] if p is not None else sum(c) / K) for c, p in zip(corr, jp)) / n
                    if d["judge"] else None)

    # ToTh under each backbone and mapping
    S["toth"] = {}
    S["entail_rate"] = {}
    S["toth_picks"] = {}
    for b, E in d["nli"].items():
        S["toth"][b] = {}
        for mp in MAPPINGS + ["INV"]:
            S["toth"][b][mp] = sum(argmax_expected([trust(e, mp) for e in E[r["id"]]], c)
                                   for r, c in zip(G, corr)) / n
        all_edges = [x for r in G for e in E[r["id"]] for x in e]
        S["entail_rate"][b] = (sum(max(range(3), key=lambda k: x[k]) == 0 for x in all_edges) / len(all_edges)
                               if all_edges else None)
        S["toth_picks"][b] = [argmax_first([trust(e, config.MAIN_MAPPING) for e in E[r["id"]]]) for r in G]
    if "roberta" in S["toth"]:
        acc["toth"] = S["toth"]["roberta"][config.MAIN_MAPPING]
    else:
        acc["toth"] = None
    S["acc"] = acc

    S["se"] = {k: (se(v, S["rand"], S["oracle"]) if v is not None else None) for k, v in acc.items()}
    # own-pool SE for self-consistency
    S["sc_rand"] = sum(sum(t["correct"] for t in r["sc"]) / config.K_SC for r in G) / n
    S["sc_oracle"] = sum(any(t["correct"] for t in r["sc"]) for r in G) / n
    S["sc_k3_oracle"] = sum(sum(any(r["sc"][i]["correct"] for i in cb) for cb in combos) / len(combos)
                            for r in G) / n
    S["se_own"] = {"sc": se(acc["sc"], S["sc_rand"], S["sc_oracle"]),
                   "sc_k3": se(acc["sc_k3"], S["sc_rand"], S["sc_k3_oracle"])}
    S["low_signal"] = (S["oracle"] - S["rand"]) < config.LOW_SIGNAL_GAP
    S["net_score"] = (acc["toth"] - S["single_p"]) if acc["toth"] is not None else None

    # similarity / agreement
    pair_idx = list(itertools.combinations(range(K), 2))
    S["cosine"] = mean([d["sim"].get(r["id"]) for r in G]) if d["sim"] else None
    S["rouge_l"] = sum(sum(rouge_l(r["style"][i]["text"], r["style"][j]["text"]) for i, j in pair_idx) / len(pair_idx)
                       for r in G) / n
    def agree(r, i, j):
        a, b = r["style"][i]["pred"], r["style"][j]["pred"]
        return a is not None and _key(a) == _key(b)
    S["pair_agree"] = sum(sum(agree(r, i, j) for i, j in pair_idx) / len(pair_idx) for r in G) / n
    S["all_agree"] = sum(all(agree(r, i, j) for i, j in pair_idx) for r in G) / n
    S["no_answer_rate"] = sum(t["pred"] is None for r in G for t in r["style"]) / (n * K)

    # McNemar RoBERTa vs BART (per-item, deterministic first-index argmax)
    if {"roberta", "bart"} <= set(S["toth_picks"]):
        pr, pb = S["toth_picks"]["roberta"], S["toth_picks"]["bart"]
        cr = [c[i] for c, i in zip(corr, pr)]
        cbt = [c[i] for c, i in zip(corr, pb)]
        b_ = sum(x and not y for x, y in zip(cr, cbt))
        c_ = sum(y and not x for x, y in zip(cr, cbt))
        S["mcnemar_p"] = binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
        S["same_picks"] = sum(x == y for x, y in zip(pr, pb)) / n

    # symbolic verifier (Web of Lies)
    if split in config.WOL_SPLITS:
        solver_ok, sel = 0, 0.0
        for r, c in zip(G, corr):
            _, ans = wol.solve(r["question"])
            solver_ok += ans == r["answer"]
            sc_ = [wol.verify_trace(r["question"], t["text"], t["pred"]) for t in r["style"]]
            score = [(1.0 if m else 0.0) + 0.1 * cons for m, cons in sc_]
            sel += argmax_expected(score, c)
        S["verifier_solver_acc"] = solver_ok / n
        S["verifier_select_acc"] = sel / n
    return S


# --------------------------------------------------------------- latex ----
def f3(x, bold=False):
    if x is None:
        return "undef"
    s = f"{x:.3f}"
    s = f"\\mathbf{{{s}}}" if bold else s
    return f"$\\negv{{{s}}}$" if x < 0 else f"${s}$"


def fpct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def fpp(x):
    return "--" if x is None else f"${100 * x:+.1f}$"


def fmean(x, bold=False):
    if x is None:
        return "--"
    s = f"{x:+.2f}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def lab(S):
    return config.MODEL_LABEL[S["model"]], config.SPLIT_LABEL[S["split"]]


def write(name, body):
    os.makedirs(T, exist_ok=True)
    write_text_atomic(os.path.join(T, name), body)


def table(spec, header, rows, caption, label, size="\\small", float_="[H]"):
    return ("\\begin{table}" + float_ + "\n\\centering" + size + "\n\\begin{tabular}{" + spec + "}\n\\toprule\n"
            + header + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            + "\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{table}\n")


SEL = [("toth", "ToTh"), ("majority", "Majority"), ("sc", "Self-Cons."), ("judge", "LLM-judge")]


def emit(main, floor):
    qwen = [S for S in main if S["model"] in config.QWEN_MODELS]
    means = {k: (mean([S["se"][k] for S in main]), mean([S["se"][k] for S in qwen])) for k, _ in SEL}

    # --- Table: mean SE ---
    order = [("sc", "Self-consistency"), ("majority", "Majority vote"), ("judge", "LLM-as-judge"),
             ("toth", "ToTh (NLI coherence)")]
    best_all = max((v[0] for v in means.values() if v[0] is not None), default=None)
    best_q = max((v[1] for v in means.values() if v[1] is not None), default=None)
    rows = [f"{nm} & {fmean(means[k][0], means[k][0] is not None and means[k][0] == best_all)} & {fmean(means[k][1], means[k][1] is not None and means[k][1] == best_q)} \\\\"
            for k, nm in order]
    write("tab_methods.tex", table("l r r", "Selector & Mean \\SE{} (all) & Mean \\SE{} (Qwen)", rows,
          f"\\textbf{{Mean Selection Efficiency by selector}} over all {len(main)} configurations and over the "
          f"{len(qwen)} Qwen configurations. All selectors share the style pool's random and oracle accuracy "
          "(shared reference). Per-configuration values in Appendix~\\ref{app:full}, Table~\\ref{tab:methods-full}.",
          "tab:methods", float_="[t]"))

    # --- Appendix: per-config SE ---
    rows = []
    for S in main:
        vals = [S["se"][k] for k, _ in SEL]
        best = max(v for v in vals if v is not None) if any(v is not None for v in vals) else None
        m, s = lab(S)
        cells = [f3(v, v is not None and best is not None and abs(v - best) < 1e-9) for v in vals]
        dag = "$^{\\dagger}$" if S["low_signal"] else ""
        rows.append(f"{m} & {s}{dag} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{2}{l}{\\textbf{Mean (all)}} & " + " & ".join(fmean(means[k][0]) for k, _ in SEL) + " \\\\")
    rows.append("\\multicolumn{2}{l}{\\textbf{Mean (Qwen)}} & " + " & ".join(fmean(means[k][1]) for k, _ in SEL) + " \\\\")
    write("tab_methods_full.tex", table("l l r r r r", "Model & Dataset & ToTh & Majority & Self-Cons. & LLM-judge", rows,
          "\\textbf{Selection Efficiency by method, all configurations} (shared reference: the style pool's random "
          "and oracle accuracy). Best per row in \\textbf{bold}. Self-consistency can exceed 1 when its own samples "
          "contain correct answers absent from the style pool. $^{\\dagger}$Low-signal case: oracle--random gap "
          f"below {100*config.LOW_SIGNAL_GAP:.0f}\\,pp.", "tab:methods-full"))

    # --- Appendix: ToTh detail ---
    rows = []
    for S in main:
        m, s = lab(S)
        dag = "$^{\\dagger}$" if S["low_signal"] else ""
        rows.append(f"{m} & {s}{dag} & {fpct(S['acc']['toth'])} & {fpct(S['rand'])} & {fpct(S['oracle'])} & {f3(S['se']['toth'])} \\\\")
    write("tab_toth.tex", table("l l r r r r", "Model & Dataset & ToTh\\,\\% & Rand\\,\\% & Oracle\\,\\% & $\\SE$", rows,
          "\\textbf{ToTh Selection Efficiency across all runs.} Rand is the expected accuracy of a uniformly random "
          "style trace. $^{\\dagger}$Low-signal case.", "tab:toth", size="\\footnotesize"))

    # --- NEW: pool control ---
    rows = []
    for S in main:
        m, s = lab(S)
        rows.append(f"{m} & {s} & {f3(S['se']['majority'])} & {f3(S['se']['sc_k3'])} & {f3(S['se']['sc'])} & "
                    f"{f3(S['se_own']['sc_k3'])} & {f3(S['se_own']['sc'])} \\\\")
    rows.append("\\midrule")
    mk = lambda f: fmean(mean([f(S) for S in main]))
    rows.append("\\multicolumn{2}{l}{\\textbf{Mean (all)}} & " + " & ".join([
        mk(lambda S: S["se"]["majority"]), mk(lambda S: S["se"]["sc_k3"]), mk(lambda S: S["se"]["sc"]),
        mk(lambda S: S["se_own"]["sc_k3"]), mk(lambda S: S["se_own"]["sc"])]) + " \\\\")
    write("tab_pool.tex", table("l l r r r r r",
          "& & \\multicolumn{3}{c}{Shared reference} & \\multicolumn{2}{c}{Own pool} \\\\\n"
          "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\nModel & Dataset & Style, $k{=}3$ & Neutral, $k{=}3$ & "
          "Neutral, $k{=}5$ & Neutral, $k{=}3$ & Neutral, $k{=}5$", rows,
          "\\textbf{Same selector, different pool.} Every column is a plurality vote over a pool of the stated source and size; only the pool changes. "
          "Neutral, $k{=}3$ averages the vote over all $\\binom{5}{3}$ subsets of the cached self-consistency "
          "samples, matching the style pool's $k$; Neutral, $k{=}5$ is self-consistency. Shared reference uses the "
          "style pool's random/oracle accuracy; own pool uses the neutral samples' own.", "tab:pool", size="\\footnotesize"))

    # --- similarity ---
    rows = []
    for S in main:
        m, s = lab(S)
        cos = "--" if S["cosine"] is None else f"{S['cosine']:.3f}"
        rows.append(f"{m} & {s} & {cos} & {S['rouge_l']:.3f} & {fpct(S['pair_agree'])} & {fpct(S['all_agree'])} \\\\")
    write("tab_sim.tex", table("l l r r r r", "Model & Dataset & Cosine & ROUGE-L & Pair Agr.\\,\\% & All-3 Agr.\\,\\%", rows,
          "Trace similarity across all runs (cosine: mean pairwise " + config.EMBED_MODEL.split("/")[-1] +
          " embedding similarity of the three style traces).", "tab:sim"))

    # --- decomposition ---
    rows = []
    for S in main:
        m, s = lab(S)
        rows.append(f"{m} & {s} & {fpct(S['single_p'])} & {fpct(S['indep_ceil'])} & {fpct(S['oracle'])} & "
                    f"{fpp(S['corr_loss'])} & {fpct(S['acc']['toth'])} & {fpp(S['net_score'])} \\\\")
    write("tab_decomp.tex", table("l l r r r r r r",
          "Model & Dataset & Single-$p$ & IndepCeil & Oracle & CorrDisc & System & NetScore", rows,
          "Gain decomposition in percentage points. IndepCeil $=1-\\prod_i(1-p_i)$ is the oracle if the three agents "
          "erred independently; CorrDisc $=$ IndepCeil $-$ Oracle is the coverage lost to correlated errors; System is "
          "ToTh accuracy; NetScore $=$ System $-$ Single-$p$.", "tab:decomp", size="\\footnotesize"))

    # --- NLI ablation ---
    rows, ps = [], []
    for S in main:
        if "mcnemar_p" not in S:
            continue
        m, s = lab(S)
        ps.append(S["mcnemar_p"])
        rows.append(f"{m} / {s} & {fpct(S['entail_rate']['roberta'])} & {fpct(S['entail_rate']['bart'])} & "
                    f"{fpct(S['toth']['roberta'][config.MAIN_MAPPING])} & {fpct(S['toth']['bart'][config.MAIN_MAPPING])} & "
                    f"{fpct(S['same_picks'])} & ${S['mcnemar_p']:.3f}$ \\\\")
    fisher = combine_pvalues(ps, method="fisher").pvalue if ps else None
    if rows:
        write("tab_nli_ablation.tex", table("l r r r r r r",
              "& \\multicolumn{2}{c}{Entail.\\,\\%} & \\multicolumn{2}{c}{ToTh\\,\\%} & & \\\\\n"
              "Config & RoB. & BART & RoB. & BART & Same pick\\,\\% & McNemar $p$", rows,
              f"\\textbf{{NLI backbone ablation.}} RoB.\\ = RoBERTa-large-MNLI, BART = BART-large-MNLI; ToTh uses the "
              f"{config.MAIN_MAPPING} mapping. Exact paired McNemar test; Fisher's combined $p={fisher:.3f}$.",
              "tab:nli-ablation", size="\\footnotesize"))

    # --- trust mappings ---
    rows = []
    for S in main:
        if "roberta" not in S["toth"]:
            continue
        m, s = lab(S)
        t = S["toth"]["roberta"]
        rows.append(f"{m} & {s} & " + " & ".join(fpct(t[k]) for k in MAPPINGS + ["INV"]) + f" & {fpct(S['rand'])} \\\\")
    if rows:
        write("tab_mappings.tex", table("l l " + "r " * 8, "Model & Dataset & " + " & ".join(MAPPINGS) + " & Inv. & Rand", rows,
              "ToTh accuracy (\\%) under six trust mappings and the inverted V0 mapping. V0: mean(e$-$c); V1: mean(e); "
              "V2: mean(e$+$n/2); V3: mean log(e/c); V4: min(e$-$c); V5: frac.\\ entail $-$ frac.\\ contradict "
              "(argmax labels).", "tab:mappings", size="\\footnotesize"))

    # --- verifier ---
    rows = []
    for S in main:
        if "verifier_solver_acc" not in S:
            continue
        m, s = lab(S)
        rows.append(f"{m} & {s} & {fpct(S['verifier_solver_acc'])} & {fpct(S['verifier_select_acc'])} & "
                    f"{fpct(S['oracle'])} & {fpct(S['acc']['toth'])} \\\\")
    if rows:
        write("tab_verifier.tex", table("l l r r r r", "Model & Dataset & Solver\\,\\% & Verifier pick\\,\\% & Oracle\\,\\% & ToTh\\,\\%", rows,
              "Symbolic verifier on Web of Lies. Solver: truth propagated from the anchor statement (no model call). "
              "Verifier pick: the style trace whose answer agrees with the propagation (ties broken by step-level "
              "consistency); it reaches the oracle whenever a correct trace exists.", "tab:verifier", size="\\footnotesize"))

    # --- SmolLM2 ---
    rows = []
    for S in floor:
        cos = "--" if S["cosine"] is None else f"{S['cosine']:.3f}"
        rows.append(f"{config.SPLIT_LABEL[S['split']]} & {cos} & {fpct(S['oracle'])} & {fpct(S['rand'])} & "
                    f"{f3(S['se']['toth']) if S['se']['toth'] is not None else 'undef'} & {fpct(S['no_answer_rate'])} \\\\")
    if rows:
        write("tab_smol.tex", table("l r r r r r", "Dataset & Cosine & Oracle\\,\\% & Rand\\,\\% & ToTh $\\SE$ & No answer\\,\\%", rows,
              "SmolLM2-1.7B capability-floor case. \\SE{} is undefined when oracle equals random.", "tab:smol"))
    return means, fisher


# ----------------------------------------------------------- sign tests ----
def sign_test(main, a, b, key="se"):
    w = l = t = 0
    for S in main:
        x, y = S[key].get(a), S[key].get(b)
        if x is None or y is None:
            continue
        if abs(x - y) < 1e-9:
            t += 1
        elif x > y:
            w += 1
        else:
            l += 1
    p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
    return {"wins": w, "losses": l, "ties": t, "p": p}


def main():
    main_s = [S for m in config.MAIN_MODELS for s in config.SPLITS if (S := config_stats(m, s))]
    floor_s = [S for m in config.FLOOR_MODELS for s in config.SPLITS if (S := config_stats(m, s))]
    if not main_s:
        raise SystemExit("No generations found under " + config.OUT_DIR)
    means, fisher = emit(main_s, floor_s)

    tests = {
        "majority>toth": sign_test(main_s, "majority", "toth"),
        "sc>toth": sign_test(main_s, "sc", "toth"),
        "majority>judge": sign_test(main_s, "majority", "judge"),
        "sc_k3>majority (same selector & k, different pool)": sign_test(main_s, "sc_k3", "majority"),
    }
    toth_se = [S["se"]["toth"] for S in main_s if S["se"]["toth"] is not None]
    qwen = [S for S in main_s if S["model"] in config.QWEN_MODELS]
    wol_s = [S for S in main_s if S["split"] in config.WOL_SPLITS]
    def rng(vals, fmt="{:.1f}", scale=100):
        vals = [v for v in vals if v is not None]
        return (fmt.format(scale * min(vals)) + " to " + fmt.format(scale * max(vals))) if vals else "n/a"

    def fm(k):
        a, q = means[k]
        return " / ".join("n/a" if x is None else f"{x:.2f}" for x in (a, q))
    ordered = all(means[k][0] is not None for k in ("sc", "majority", "judge", "toth")) and \
        means["sc"][0] > means["majority"][0] > max(means["judge"][0], means["toth"][0])
    claims = [
        ("Mean SE, self-consistency (all / Qwen)", fm("sc")),
        ("Mean SE, majority (all / Qwen)", fm("majority")),
        ("Mean SE, LLM-judge (all / Qwen)", fm("judge")),
        ("Mean SE, ToTh (all / Qwen)", fm("toth")),
        ("Ordering SC > majority > max(judge, ToTh) holds?", str(ordered)),
        ("Max ToTh SE (paper: <=0.18)", f"{max(toth_se):.3f}" if toth_se else "n/a"),
        ("ToTh SE == 0 count (paper: 2)", str(sum(abs(x) < 1e-9 for x in toth_se))),
        ("ToTh SE < 0 count (paper: 3)", str(sum(x < -1e-9 for x in toth_se))),
        ("Mean ToTh accuracy-to-oracle gap, pp (paper: 38)", f"{100*mean([S['oracle']-S['acc']['toth'] for S in main_s if S['acc']['toth'] is not None]):.1f}" if toth_se else "n/a"),
        ("SC SE > 1 configs", ", ".join(f"{lab(S)[0]}/{lab(S)[1]}" for S in main_s if (S['se']['sc'] or 0) > 1) or "none"),
        ("Qwen min cosine (paper: >0.85)", f"{min(S['cosine'] for S in qwen if S['cosine'] is not None):.3f}" if any(S['cosine'] is not None for S in qwen) else "n/a"),
        ("Phi-2 MultiArith cosine (paper: 0.60)", ", ".join(f"{S['cosine']:.3f}" for S in main_s if S['model']=='phi2' and S['split'].startswith('ma') and S['cosine'] is not None) or "n/a"),
        ("Verifier solver accuracy on WoL (paper: 100%)", rng([S.get('verifier_solver_acc') for S in wol_s])),
        ("Max ToTh accuracy on WoL (paper: 48%)", f"{100*max(S['acc']['toth'] for S in wol_s if S['acc']['toth'] is not None):.1f}" if any(S['acc']['toth'] is not None for S in wol_s) else "n/a"),
        ("Entailment rate range, RoBERTa (paper: 9-41%)", rng([S['entail_rate'].get('roberta') for S in main_s])),
        ("NetScore range, pp (paper: -12.0 to +2.7)", rng([S['net_score'] for S in main_s])),
        ("Inverted mapping >= V0 in how many configs", str(sum(S['toth']['roberta']['INV'] >= S['toth']['roberta']['V0'] - 1e-9 for S in main_s if 'roberta' in S['toth']))),
        ("Fisher combined McNemar p (paper: 0.791)", "n/a" if fisher is None else f"{fisher:.3f}"),
        ("Judge pick parse rate (range over configs)", rng([S['judge_coverage'] for S in main_s])),
        ("Judge picks by displayed position 1/2/3 (all configs; strong skew = position bias)",
         " / ".join(str(sum(S['judge_pos_counts'][i] for S in main_s)) for i in range(config.K_STYLE))),
        ("Largest single-position share of judge picks in any config", f"{100*max(max(S['judge_pos_counts'])/max(1,sum(S['judge_pos_counts'])) for S in main_s):.0f}%"),
        ("Unparseable-answer rate (max over configs)", f"{100*max(S['no_answer_rate'] for S in main_s):.1f}%"),
        ("Undefined-SE configs (oracle == random)", ", ".join(f"{lab(S)[0]}/{lab(S)[1]}" for S in main_s if S['se']['majority'] is None) or "none"),
    ]
    for k, v in tests.items():
        claims.append((f"Sign test {k}", f"{v['wins']} wins / {v['losses']} losses / {v['ties']} ties, p={v['p']:.3f}"))

    os.makedirs(T, exist_ok=True)
    write_text_atomic(os.path.join(T, "claims.md"),
                      "| Claim | Recomputed |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in claims))
    drop = {"toth_picks"}
    write_text_atomic(os.path.join(T, "results.json"), json.dumps(
        {"configs": [{k: v for k, v in S.items() if k not in drop} for S in main_s + floor_s],
         "means": means, "sign_tests": tests, "fisher_mcnemar": fisher}, indent=1))
    print(open(os.path.join(T, "claims.md")).read())
    print("Tables written to", T)


if __name__ == "__main__":
    main()
