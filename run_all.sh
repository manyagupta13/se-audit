#!/usr/bin/env bash
# Full rerun on one machine; resumable (see README "Interruptions and checkpoints").
set -euo pipefail
cd "$(dirname "$0")"
python run_pipeline.py --full "$@"
