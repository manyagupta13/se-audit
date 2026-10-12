"""Prompts, prompt formatting and answer extraction.

The three style personas define the style-prompted pool. Self-consistency
uses one neutral prompt. Report these verbatim in the appendix.
"""
import re

STYLES = [
    ("formal", "You are a meticulous logician. Reason formally and precisely, justifying every step from the facts given."),
    ("concise", "You are a practical engineer. Reason concisely, keeping only the steps that matter."),
    ("teacher", "You are a patient teacher. Reason in plain language as if explaining to a student, checking each step as you go."),
]
NEUTRAL_SYSTEM = "You are a helpful assistant."

TASK = {
    "num": ("Solve the following problem. Think step by step, writing one step per line. "
            "End with a line of the form 'Final answer: <number>'."),
    "bool": ("Answer the following question. Think step by step, writing one step per line. "
             "End with a line of the form 'Final answer: Yes' or 'Final answer: No'."),
}

JUDGE_TEMPLATE = (
    "Below is a question followed by three candidate solutions.\n\n"
    "Question: {question}\n\n"
    "Solution 1:\n{t1}\n\nSolution 2:\n{t2}\n\nSolution 3:\n{t3}\n\n"
    "Which solution reaches its final answer through the best-supported reasoning? "
    "Reply with only the number 1, 2, or 3."
)

PHI2_STOPS = ["\nInstruct:", "<|endoftext|>", "\nExercise", "\nUser:", "\n\n\n\n"]


def format_prompt(tokenizer, model_key, system, user):
    if model_key == "phi2":  # base-ish model: use its documented QA format
        return f"Instruct: {system}\n{user}\nOutput:"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def task_user(item):
    return f"{TASK[item['type']]}\n\n{item['question']}"


def clean(text, model_key):
    if model_key == "phi2":
        for s in PHI2_STOPS:
            i = text.find(s)
            if i != -1:
                text = text[:i]
    return text.strip()


_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def _to_float(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract(text, kind):
    """Return the final answer as float (num), 'Yes'/'No' (bool), or None."""
    m = list(re.finditer(r"final answer\s*(?:is)?\s*[:=]?\s*\**\s*(.+)", text, flags=re.I))
    tail = m[-1].group(1) if m else None
    if kind == "num":
        if tail:
            nums = re.findall(_NUM, tail)
            if nums:
                return _to_float(nums[0])
        boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxed:
            nums = re.findall(_NUM, boxed[-1])
            if nums:
                return _to_float(nums[0])
        nums = re.findall(_NUM, text)
        return _to_float(nums[-1]) if nums else None
    # bool
    src = tail if tail else text[-300:]
    hits = re.findall(r"\b(yes|no)\b", src, flags=re.I)
    if hits:
        return hits[0 if tail else -1].capitalize()
    if re.search(r"tells? the truth|is truthful", src, re.I) and not re.search(r"\b(not|lies|liar)\b", src, re.I):
        return "Yes"
    if re.search(r"\blies\b|\bliar\b|not tell", src, re.I):
        return "No"
    return None


def is_correct(pred, gold, kind):
    if pred is None:
        return False
    if kind == "num":
        return abs(pred - float(gold)) < 1e-6
    return pred == gold


def split_steps(text, max_steps=40):
    """Split a trace into reasoning steps: one per line, long lines split into sentences."""
    steps = []
    for line in text.splitlines():
        line = line.strip(" -*\t")
        if not line:
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            s = s.strip()
            if len(s) > 3:
                steps.append(s)
    return steps[:max_steps]


def parse_judge(text):
    m = re.search(r"[123]", text)
    return int(m.group(0)) - 1 if m else None
