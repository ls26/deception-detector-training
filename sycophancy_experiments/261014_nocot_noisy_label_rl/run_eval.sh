#!/bin/bash
# Usage (inside its own tmux session): bash run_eval.sh base50|adapter50|base|final <attempt 1-3>
#
# Runs eval_checkpoint.py's three stages (generate, selfreport, labels) for one checkpoint and writes
# logs/eval_<name>_a<attempt>_<host>.log, ending with "EXIT <code>" (the first failing stage's code, or 0).
#   base50     dry run: the base model on the first 50 eval questions        -> dry/evals/base50/
#   adapter50  dry run: 261013's final adapter on the same 50 questions        -> dry/evals/adapter50/
#   base       the base model on the 547 eval questions                        -> evals/base/
#   final      the RL run's final adapter (run id from run_metrics_nocot_noisy_label_s42.json) -> evals/final/
# Refuses if that attempt's log exists; eval_checkpoint.py refuses to overwrite its outputs, so before a
# rerun after a fix the failed attempt's output folder must be moved to failed_attempts/.
set -u
NAME=$1
ATTEMPT=$2
[[ $NAME =~ ^(base50|adapter50|base|final)$ ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: run_eval.sh base50|adapter50|base|final <attempt 1-3>" >&2; exit 1; }
source /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/run_env.sh
LOG=$RUN_DIR/logs/eval_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"
OUT_ROOT=$RUN_DIR/evals
LIMIT_ARGS=()
ADAPTER_ARGS=()
case $NAME in
  base50) OUT_ROOT=$RUN_DIR/dry/evals; LIMIT_ARGS=(--limit 50) ;;
  adapter50) OUT_ROOT=$RUN_DIR/dry/evals; LIMIT_ARGS=(--limit 50)
    ADAPTER_ARGS=(--adapter /workspace/results/run_results/261013_noisy_label_rl/noisy_label_s42_6vcs39wj/checkpoints/final) ;;
  final)
    RUN_ID=$($PY -c "import json; print(json.load(open('$RUN_DIR/run_metrics_nocot_noisy_label_s42.json'))['run_id'])")
    ADAPTER_ARGS=(--adapter "/workspace/results/run_results/261014_nocot_noisy_label_rl/nocot_noisy_label_s42_$RUN_ID/checkpoints/final") ;;
esac
{
  cd "$RUN_DIR"
  $PY eval_checkpoint.py generate --name "$NAME" --out-root "$OUT_ROOT" "${LIMIT_ARGS[@]}" "${ADAPTER_ARGS[@]}" \
    && $PY eval_checkpoint.py selfreport --name "$NAME" --out-root "$OUT_ROOT" "${ADAPTER_ARGS[@]}" \
    && $PY eval_checkpoint.py labels --name "$NAME" --out-root "$OUT_ROOT"
  echo "EXIT $?"
} > "$LOG" 2>&1
