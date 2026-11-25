#!/bin/bash
# Smoke check (f): the changed eval's probes and labels stages on 261117's saved base eval (seed 42), then the
# comparison (check_eval_rerun.py). Log: logs/check_f_<host>.log, ends "EXIT <code>".
set -u
source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
cd /workspace/qwen_experiment/261120_qwen_port_checks
LOG=logs/check_f_a2_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
SAVED=/workspace/sycophancy_experiments/261117_no_penalty_vs_frozen_400/no_penalty/seed_42/evals/base_rescored
OUT=check_f_a2/base_rescored
PROBE_RUN_DIR=/workspace/results/run_results/261117_no_penalty_vs_frozen_400/np400_s42_44s8k5wr
{
  $PY check_eval_rerun.py rescore --rerun-dir $OUT \
    && $PY $SETUP_DIR/eval_checkpoint.py probes --out-dir $OUT --probe-run-dir $PROBE_RUN_DIR --model meta-llama/Meta-Llama-3-8B-Instruct \
    && $PY $SETUP_DIR/eval_checkpoint.py labels --out-dir $OUT \
    && $PY check_eval_rerun.py compare --rerun-dir $OUT
  echo "EXIT $?"
} > "$LOG" 2>&1
