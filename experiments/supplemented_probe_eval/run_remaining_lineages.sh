#!/usr/bin/env bash
set -euo pipefail

source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

LOG=/workspace/experiments/supplemented_probe_eval/run_remaining.log
exec > >(tee -a "$LOG") 2>&1

echo "=== supplemented probe eval: remaining 8 lineages ==="
echo "started: $(date -Is)"

run_one() {
  local trial=$1
  local variant=$2
  local label="trial${trial}"
  if [[ -n "$variant" ]]; then
    label="${label}_${variant}"
  fi
  echo ""
  echo ">>> ${label} $(date -Is)"
  if [[ -n "$variant" ]]; then
    python /workspace/experiments/supplemented_probe_eval/run_eval.py --trial "$trial" --variant "$variant"
    python /workspace/experiments/supplemented_probe_eval/plot_shifts.py --trial "$trial" --variant "$variant"
  else
    python /workspace/experiments/supplemented_probe_eval/run_eval.py --trial "$trial" --variant ""
    python /workspace/experiments/supplemented_probe_eval/plot_shifts.py --trial "$trial" --variant ""
  fi
  echo "<<< done ${label} $(date -Is)"
}

run_one 4 v2
run_one 4 frozen
run_one 4 early_retraining_v2
run_one 6 ""
run_one 6 frozen
run_one 6 early_retraining
run_one 8 frozen
run_one 8 early_retraining

echo ""
echo "=== all remaining lineages complete ==="
echo "finished: $(date -Is)"
