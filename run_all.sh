#!/usr/bin/env bash
# Full rerun on one GPU. Every step is resumable: rerun this script after an
# interruption and finished work is skipped.
set -euo pipefail
cd "$(dirname "$0")"
python data.py
for m in qwen1.5b qwen3b phi2 smollm2; do
  python generate.py --model "$m"
done
python score.py
python analyze.py
