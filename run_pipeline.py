"""Run the whole experiment on Colab, Kaggle or any GPU box. Safe to interrupt.

  python run_pipeline.py --work DIR --smoke    # ~10 min check on a few items
  python run_pipeline.py --work DIR --full     # whole experiment
  python run_pipeline.py --work DIR --status   # what is done / still to do

Point --work at Google Drive on Colab (e.g. /content/drive/MyDrive/se_audit_run).
Every batch of generations is appended and forced to disk as soon as it is
produced, so when a session dies you lose at most the batch in flight: rerun
the same command and it continues from the last checkpoint. Splits that are
already complete are skipped without even loading the model.

Full run: generation (one model per GPU if two are visible) -> completeness
check -> NLI/cosine scoring -> all tables -> tables.zip and raw_outputs.zip in
--work. If generation is incomplete it says which files are short and stops
before scoring.
"""
import argparse
import atexit
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from io_utils import read_jsonl, write_text_atomic   # noqa: E402

GPU_PLAN = [["qwen1.5b", "phi2"], ["qwen3b", "smollm2"]]   # one list per GPU
ALL_MODELS = [m for chain in GPU_PLAN for m in chain]
SPLITS = ["wol4", "wol5", "ma1", "ma2"]
N = 50
LOCK_STALE_S = 150


def default_work():
    if os.environ.get("SE_WORK"):
        return os.environ["SE_WORK"]
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(HERE, "work")


class Paths:
    def __init__(self, work):
        self.work = work
        self.out = os.path.join(work, "se_audit_out")
        self.smoke = os.path.join(work, "smoke")
        self.input = os.environ.get("SE_INPUT", "/kaggle/input")


def sh(cmd, se_out, **kw):
    env = dict(os.environ, SE_OUT=se_out)
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=HERE, env=env, **kw)


def n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, len([l for l in out.splitlines() if l.startswith("GPU")]))
    except FileNotFoundError:
        return 1


# --------------------------------------------------------------- locking ----
def acquire_lock(P, force):
    """Refuse to start if another live session is writing to the same folder."""
    path = os.path.join(P.work, ".run.lock")
    if os.path.exists(path) and not force:
        try:
            info = json.load(open(path))
            age = time.time() - info.get("heartbeat", 0)
        except Exception:
            info, age = {}, LOCK_STALE_S + 1
        if age < LOCK_STALE_S:
            sys.exit(f"Another session ({info.get('host')}, pid {info.get('pid')}) wrote to {P.work} "
                     f"{age:.0f}s ago and may still be running. Two sessions writing the same files would "
                     f"duplicate work. Wait {LOCK_STALE_S - age:.0f}s and rerun, or pass --force if you are "
                     "sure the other session is gone.")
    me = {"host": socket.gethostname(), "pid": os.getpid(), "started": time.time()}

    def beat():
        while True:
            write_text_atomic(path, json.dumps(dict(me, heartbeat=time.time())))
            time.sleep(30)
    beat_once = dict(me, heartbeat=time.time())
    write_text_atomic(path, json.dumps(beat_once))
    threading.Thread(target=beat, daemon=True).start()

    def release():
        try:
            if json.load(open(path)).get("pid") == me["pid"]:
                os.remove(path)
        except Exception:
            pass
    atexit.register(release)


# ---------------------------------------------------------------- status ----
def counts(P, model, split):
    d = os.path.join(P.out, "gen", model)
    g = len(read_jsonl(os.path.join(d, f"{split}.jsonl")))
    j = len(read_jsonl(os.path.join(d, f"{split}.judge.jsonl")))
    return g, j


def missing_files(P):
    miss = []
    for m in ALL_MODELS:
        for s in SPLITS:
            g, j = counts(P, m, s)
            if g < N:
                miss.append(f"{m}/{s}.jsonl ({g}/{N})")
            if j < N:
                miss.append(f"{m}/{s}.judge.jsonl ({j}/{N})")
    return miss


def status(P):
    print(f"Working folder: {P.work}")
    total = done = 0
    for m in ALL_MODELS:
        cells = []
        for s in SPLITS:
            g, j = counts(P, m, s)
            total += 2 * N
            done += min(g, N) + min(j, N)
            cells.append(f"{s} gen {g:>2}/{N} judge {j:>2}/{N}")
        print(f"  {m:9s} " + " | ".join(cells))
    print(f"Generation: {100 * done / total:.0f}% done")
    for d in ("nli/roberta", "nli/bart", "sim"):
        n = len(glob.glob(os.path.join(P.out, d, "*", "*.jsonl")))
        print(f"  scored {d:12s}: {n}/{len(ALL_MODELS) * len(SPLITS)} files")
    t = os.path.join(P.out, "tables", "claims.md")
    print("Tables:", "built" if os.path.exists(t) else "not built yet")


def tail(path, k=3):
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        lines = [l.rstrip() for l in f if l.strip()]
    return "\n    ".join(lines[-k:])


# ------------------------------------------------------------------ runs ----
def restore_previous(P):
    """Kaggle only: copy an earlier run's output (added as a notebook input) back."""
    if os.path.isdir(os.path.join(P.out, "gen")):
        return
    prev = [p for p in glob.glob(os.path.join(P.input, "**", "se_audit_out"), recursive=True) if os.path.isdir(p)]
    if prev:
        shutil.copytree(prev[0], P.out, dirs_exist_ok=True)
        print("RESUMING from earlier output:", prev[0], flush=True)


def smoke(P):
    sh("python data.py", P.smoke)
    sh("python generate.py --model qwen1.5b --splits wol4 ma1 --limit 10", P.smoke)
    sh("python generate.py --model phi2 --splits wol4 --limit 6", P.smoke)
    sh("python smoke_check.py", P.smoke)


def full(P):
    restore_previous(P)
    sh("python data.py", P.out, check=True)
    miss = missing_files(P)
    if miss:
        gpus = n_gpus()
        plan = GPU_PLAN if gpus >= 2 else [ALL_MODELS]
        print(f"{gpus} GPU(s) visible; plan: {plan}", flush=True)
        status(P)
        procs, logs = [], []
        for g, chain in enumerate(plan):
            log = os.path.join(P.work, f"log_gpu{g}.txt")
            # ';' not '&&': one model failing must not stop the next model on that GPU
            cmd = " ; ".join(f"python generate.py --model {m}" for m in chain)
            env = dict(os.environ, SE_OUT=P.out, CUDA_VISIBLE_DEVICES=str(g))
            procs.append(subprocess.Popen(cmd, shell=True, cwd=HERE, env=env,
                                          stdout=open(log, "a"), stderr=subprocess.STDOUT))
            logs.append(log)
        t0, last = time.time(), 0
        while any(p.poll() is None for p in procs):
            time.sleep(20)
            if time.time() - last > 300:   # progress every 5 min
                last = time.time()
                print(f"\n[{(time.time() - t0) / 60:.0f} min]", flush=True)
                for l in logs:
                    print(f"  {os.path.basename(l)}:\n    {tail(l)}", flush=True)
        print(f"generation finished after {(time.time() - t0) / 60:.0f} min", flush=True)
        for l in logs:
            print(f"--- {os.path.basename(l)} (last lines)\n    {tail(l, 6)}")
        miss = missing_files(P)

    if miss:
        print("\nGENERATION INCOMPLETE, so no tables were built:", *miss, sep="\n  ")
        print("Check log_gpu*.txt, fix the cause if there is an error, then run the same command again; "
              "it continues from the last checkpoint.")
        save_zips(P, tables=False)
        sys.exit(1)

    sh("python score.py", P.out, check=False)
    sh("python analyze.py", P.out, check=False)
    ok = os.path.exists(os.path.join(P.out, "tables", "claims.md"))
    save_zips(P, tables=ok)
    if ok:
        print("\n" + open(os.path.join(P.out, "tables", "claims.md")).read())
        print(f"DONE: tables.zip and raw_outputs.zip are in {P.work}")
    else:
        sys.exit("Generation complete but analysis did not produce tables; see the errors above.")


def save_zips(P, tables):
    def zipdirs(name, dirs):
        tmp = os.path.join(P.work, name + ".part")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for d in dirs:
                for root, _, files in os.walk(os.path.join(P.out, d)):
                    for f in files:
                        if ".tmp" in f or f.endswith(".part"):
                            continue
                        p = os.path.join(root, f)
                        z.write(p, os.path.relpath(p, P.out))
        os.replace(tmp, os.path.join(P.work, name))
        print(f"saved {name}", flush=True)
    zipdirs("raw_outputs.zip", ["gen", "nli", "sim", "data"])
    if tables:
        zipdirs("tables.zip", ["tables"])


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--work", default=default_work(), help="folder for all outputs (put it on Drive in Colab)")
    ap.add_argument("--force", action="store_true", help="ignore another session's lock")
    a = ap.parse_args()
    P = Paths(os.path.abspath(a.work))
    os.makedirs(P.work, exist_ok=True)
    if a.status:
        return status(P)
    acquire_lock(P, a.force)
    smoke(P) if a.smoke else full(P)


if __name__ == "__main__":
    main()
