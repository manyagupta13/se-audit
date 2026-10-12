"""Web of Lies: parser, symbolic solver, item generator, trace verifier.

Uses the exact BIG-Bench-Hard surface template:
  "Question: A tells the truth. B says A lies. C says B tells the truth. Does C tell the truth?"
Chain length = number of "says" statements (BBH ships only length 4).
"""
import random
import re

_ANCHOR = re.compile(r"^(?:Question:\s*)?([A-Z][a-z]+) (tells the truth|lies)\.")
_SAYS = re.compile(r"([A-Z][a-z]+) says ([A-Z][a-z]+) (tells the truth|lies)\.")
_ASK = re.compile(r"Does ([A-Z][a-z]+) tell the truth\?")


def parse(question):
    q = question.strip()
    m = _ANCHOR.match(q)
    if not m:
        raise ValueError(f"no anchor in: {q}")
    anchor, anchor_truth = m.group(1), m.group(2) == "tells the truth"
    chain = [(s, t, c == "tells the truth") for s, t, c in _SAYS.findall(q)]
    asked = _ASK.search(q).group(1)
    return {"anchor": anchor, "anchor_truth": anchor_truth, "chain": chain, "asked": asked}


def solve(question):
    """Propagate truth values from the anchor. Returns ({name: bool}, 'Yes'/'No')."""
    p = parse(question)
    truth = {p["anchor"]: p["anchor_truth"]}
    for speaker, target, claim in p["chain"]:
        if target not in truth:
            raise ValueError(f"chain out of order: {question}")
        # speaker is truthful iff their claim about target matches target's truth
        truth[speaker] = (claim == truth[target])
    return truth, ("Yes" if truth[p["asked"]] else "No")


def make_item(chain_len, names, rng):
    people = rng.sample(names, chain_len + 1)
    anchor_truth = rng.random() < 0.5
    parts = [f"Question: {people[0]} {'tells the truth' if anchor_truth else 'lies'}."]
    for i in range(1, chain_len + 1):
        claim = rng.random() < 0.5
        parts.append(f"{people[i]} says {people[i-1]} {'tells the truth' if claim else 'lies'}.")
    parts.append(f"Does {people[-1]} tell the truth?")
    q = " ".join(parts)
    return q, solve(q)[1]


def names_from_bbh(examples):
    names = set()
    for e in examples:
        p = parse(e["input"])
        names.add(p["anchor"])
        for s, t, _ in p["chain"]:
            names.update([s, t])
    return sorted(names)


# ---- trace verifier ---------------------------------------------------------
_TRUE_PAT = r"(?:tells the truth|is telling the truth|is truthful|is honest|tells the truth\.)"
_FALSE_PAT = r"(?:lies|is lying|is a liar|is not telling the truth|does not tell the truth|doesn't tell the truth|is dishonest)"


def trace_claims(trace, names):
    """Extract the trace's own conclusions about who is truthful.

    Sentences containing 'says' are skipped so restatements of the question are
    not counted as the model's conclusions.
    """
    claims = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", trace):
        if " says " in sent or " said " in sent:
            continue
        for n in names:
            if re.search(rf"\b{n}\b[^.]*?\b{_FALSE_PAT}", sent):
                claims.append((n, False))
            elif re.search(rf"\b{n}\b[^.]*?\b{_TRUE_PAT}", sent):
                claims.append((n, True))
    return claims


def verify_trace(question, trace, final_answer):
    """Score a trace against symbolic propagation.

    Returns (answer_matches_symbolic, step_consistency in [-1, 1]).
    """
    truth, ans = solve(question)
    claims = trace_claims(trace, list(truth))
    good = sum(truth[n] == v for n, v in claims)
    bad = len(claims) - good
    cons = (good - bad) / (len(claims) + 1)
    return final_answer == ans, cons
