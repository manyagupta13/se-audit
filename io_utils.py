"""Crash-safe JSONL helpers.

A Colab/Kaggle session can die in the middle of a write, and Google Drive can
lag behind the local disk. These helpers make every output file safe to resume:

  read_jsonl    skips a truncated or corrupt line and drops duplicate ids
  repair_jsonl  rewrites a file to its valid, de-duplicated lines (atomic)
  append_jsonl  appends one batch and forces it to disk (flush + fsync)
  write_*_atomic  write to a temp file, then rename over the target, so a
                  reader never sees a half-written file
"""
import json
import os


def _parse(path):
    """Return (records, n_bad, n_dup) for a JSONL file; missing file -> ([], 0, 0)."""
    if not os.path.exists(path):
        return [], 0, 0
    with open(path, "rb") as f:
        raw = f.read()
    recs, seen, bad, dup = [], set(), 0, 0
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            bad += 1
            continue
        rid = r.get("id") if isinstance(r, dict) else None
        if rid is not None:
            if rid in seen:
                dup += 1
                continue
            seen.add(rid)
        recs.append(r)
    return recs, bad, dup


def read_jsonl(path):
    return _parse(path)[0]


def repair_jsonl(path):
    """Drop corrupt/duplicate lines in place. Returns the number of lines dropped."""
    recs, bad, dup = _parse(path)
    if bad or dup:
        write_jsonl_atomic(path, recs)
        print(f"[repair] {path}: dropped {bad} corrupt and {dup} duplicate line(s)", flush=True)
    return bad + dup


def append_jsonl(path, recs):
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_text_atomic(path, text):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl_atomic(path, recs):
    write_text_atomic(path, "".join(json.dumps(r) + "\n" for r in recs))
