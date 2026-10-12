"""Everything the Kaggle notebook does, in one script, so fixes arrive via `git pull`.

  python kaggle_run.py --smoke   # ~10-15 min check on a few items (separate folder)
  python kaggle_run.py --full    # whole experiment; resumable

Full run: generation (one model per GPU when two are visible) -> completeness
check -> NLI/cosine scoring -> all tables -> zips in the working directory.
If generation is incomplete it skips scoring, says which files are short, and
still saves raw_outputs.zip so the next run can resume from it.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("SE_WORK", "/kaggle/working")
INPUT = os.environ.get("SE_INPUT", "/kaggle/input")
OUT = os.path.join(WORK, "se_audit_out")
SMOKE = os.path.join(WORK, "smoke")
GPU_PLAN = [["qwen1.5b", "phi2"], ["qwen3b", "smollm2"]]   # one list per GPU
ALL_MODELS = [m for chain in GPU_PLAN for m in chain]
SPLITS = ["wol4", "wol5", "ma1", "ma2"]
N = 50


def sh(cmd, env_out, **kw):
    env = dict(os.environ, SE_OUT=env_out)
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=HERE, env=env, **kw)


def n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
        return max(1, len([l for l in out.splitlines() if l.startswith("GPU")]))
    except FileNotFoundError:
        return 1


def restore_previous():
    """Copy an earlier run's output back (added as a notebook input) so generation resumes."""
    if os.path.isdir(os.path.join(OUT, "gen")):
        return
    prev = [p for p in glob.glob(os.path.join(INPUT, "**", "se_audit_out"), recursive=True) if os.path.isdir(p)]
    if prev:
        shutil.copytree(prev[0], OUT, dirs_exist_ok=True)
        print("RESUMING from earlier output:", prev[0], flush=True)


def missing_files():
    miss = []
    for m in ALL_MODELS:
        for s in SPLITS:
            for suffix in (".jsonl", ".judge.jsonl"):
                p = os.path.join(OUT, "gen", m, s + suffix)
                n = sum(1 for _ in open(p)) if os.path.exists(p) else 0
                if n < N:
                    miss.append(f"{m}/{s}{suffix} ({n}/{N})")
    return miss


def tail(path, k=3):
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        lines = [l.rstrip() for l in f if l.strip()]
    return "\n    ".join(lines[-k:])


def smoke():
    sh("python data.py", SMOKE)
    sh("python generate.py --model qwen1.5b --splits wol4 ma1 --limit 10", SMOKE)
    sh("python generate.py --model phi2 --splits wol4 --limit 6", SMOKE)
    sh("python smoke_check.py", SMOKE)


def full():
    restore_previous()
    sh("python data.py", OUT)
    gpus = n_gpus()
    plan = GPU_PLAN if gpus >= 2 else [ALL_MODELS]
    print(f"{gpus} GPU(s) visible; plan: {plan}", flush=True)
    procs, logs = [], []
    for g, chain in enumerate(plan):
        log = os.path.join(WORK, f"log_gpu{g}.txt")
        # ';' not '&&': one model failing must not stop the next model on that GPU
        cmd = " ; ".join(f"python generate.py --model {m}" for m in chain)
        env = dict(os.environ, SE_OUT=OUT, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen(cmd, shell=True, cwd=HERE, env=env,
                                      stdout=open(log, "a"), stderr=subprocess.STDOUT))
        logs.append(log)
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        if int(time.time() - t0) % 600 < 30:   # progress every ~10 min in the commit log
            print(f"[{(time.time()-t0)/60:.0f} min]", *[f"\n  {os.path.basename(l)}:\n    {tail(l)}" for l in logs], flush=True)
    print(f"generation finished after {(time.time()-t0)/60:.0f} min", flush=True)
    for l in logs:
        print(f"--- {os.path.basename(l)} (last lines)\n    {tail(l, 6)}")

    miss = missing_files()
    complete = not miss
    if complete:
        sh("python score.py", OUT, check=False)
        sh("python analyze.py", OUT, check=False)
    else:
        print("\nGENERATION INCOMPLETE, so no tables were built:", *miss, sep="\n  ")
        print("Check log_gpu*.txt. To resume: add this notebook's output as an input and run again.")

    with zipfile.ZipFile(os.path.join(WORK, "raw_outputs.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for d in ("gen", "nli", "sim", "data"):
            for root, _, files in os.walk(os.path.join(OUT, d)):
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, OUT))
    print("saved raw_outputs.zip", flush=True)
    tables = os.path.join(OUT, "tables")
    if complete and os.path.exists(os.path.join(tables, "claims.md")):
        shutil.make_archive(os.path.join(WORK, "tables"), "zip", OUT, "tables")
        print("\n" + open(os.path.join(tables, "claims.md")).read())
        print("DONE: download tables.zip and raw_outputs.zip from the Output tab.")
    elif complete:
        print("Generation complete but analysis did not produce tables; see the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    smoke() if a.smoke else full()
