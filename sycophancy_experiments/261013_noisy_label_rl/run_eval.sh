#!/bin/bash
# Usage (inside its own tmux session): bash run_eval.sh base|final
#
# Runs eval_checkpoint.py's three stages (generate, selfreport, labels) for one checkpoint into
# evals/<name>/ and writes logs/eval_<name>_<host>.log, ending with "EXIT <code>" (the first
# failing stage's code, or 0). base = the base model; final = the RL run's final adapter, found
# from the run id in run_metrics_noisy_label_s42.json. Refuses if the log exists.
set -u
NAME=$1
[[ $NAME == base || $NAME == final ]] || { echo "usage: run_eval.sh base|final" >&2; exit 1; }
source /workspace/sycophancy_experiments/261013_noisy_label_rl/run_env.sh
LOG=$RUN_DIR/logs/eval_${NAME}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: an eval step is never re-run" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"
ADAPTER_ARGS=()
if [[ $NAME == final ]]; then
  RUN_ID=$($PY -c "import json; print(json.load(open('$RUN_DIR/run_metrics_noisy_label_s42.json'))['run_id'])")
  ADAPTER_ARGS=(--adapter "/workspace/results/run_results/261013_noisy_label_rl/noisy_label_s42_$RUN_ID/checkpoints/final")
fi
{
  cd "$RUN_DIR"
  $PY eval_checkpoint.py generate --name "$NAME" "${ADAPTER_ARGS[@]}" \
    && $PY eval_checkpoint.py selfreport --name "$NAME" "${ADAPTER_ARGS[@]}" \
    && $PY eval_checkpoint.py labels --name "$NAME"
  echo "EXIT $?"
} > "$LOG" 2>&1
