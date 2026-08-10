#!/bin/bash
set -euo pipefail
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
LOG=/workspace/experiments/supplemented_probe_eval/run_all.log
exec > >(tee -a "$LOG") 2>&1

run_one() {
  local trial=$1 variant=$2
  local label="trial${trial}"
  if [[ -n "$variant" ]]; then label="${label}_${variant}"; fi
  echo "===== START ${label} $(date -Is) ====="
  python /workspace/experiments/supplemented_probe_eval/run_eval.py --trial "$trial" --variant "$variant"
  echo "===== DONE ${label} $(date -Is) ====="
}

run_one 4 v2
run_one 4 frozen
run_one 4 early_retraining_v2
run_one 6 ""
run_one 6 frozen
run_one 6 early_retraining
run_one 8 frozen
run_one 8 early_retraining

python /workspace/experiments/supplemented_probe_eval/plot_shifts.py
echo "ALL DONE $(date -Is)"
