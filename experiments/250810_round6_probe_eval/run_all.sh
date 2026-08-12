#!/usr/bin/env bash
# Run all round-6 probe eval lineages (skips any with existing outputs).
set -euo pipefail
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
python -u experiments/2508_round6_probe_eval/run_round6.py "$@" >> /workspace/experiments/2508_round6_probe_eval/run.log 2>&1
