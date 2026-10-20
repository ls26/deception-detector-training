#!/bin/bash
# Usage (inside its own tmux session): bash run_eval.sh <name> <attempt 1-3>
#
# Runs the setup's own eval_checkpoint.py (261013's for CoT, 261014's for no-CoT, unchanged, by absolute
# path) through its three stages (generate, selfreport, labels) for one checkpoint and writes
# logs/eval_<name>_a<attempt>_<host>.log, ending with "EXIT <code>" (the first failing stage's code, or 0).
#   cot_s43 cot_s44 nocot_s43 nocot_s44   the run's final adapter (run id from run_metrics_<run_key>.json),
#                                         all eval questions                         -> evals/<name>/
#   dry50_cot dry50_nocot                 the seed-42 run's final adapter, first 50 eval questions
#                                                                                    -> dry/evals/<name>/
# Refuses if that attempt's log exists; eval_checkpoint.py refuses to overwrite its outputs, so before a
# rerun the failed attempt's output folder must be moved to failed_attempts/.
set -u
NAME=$1
ATTEMPT=$2
[[ $NAME =~ ^(cot_s43|cot_s44|nocot_s43|nocot_s44|dry50_cot|dry50_nocot)$ ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: run_eval.sh cot_s43|cot_s44|nocot_s43|nocot_s44|dry50_cot|dry50_nocot <attempt 1-3>" >&2; exit 1; }
source /workspace/sycophancy_experiments/261018_noisy_label_rl_seeds/run_env.sh
LOG=$RUN_DIR/logs/eval_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"
case $NAME in
  cot_*|dry50_cot) SETUP_DIR=/workspace/sycophancy_experiments/261013_noisy_label_rl ;;
  *) SETUP_DIR=/workspace/sycophancy_experiments/261014_nocot_noisy_label_rl ;;
esac
LIMIT_ARGS=()
case $NAME in
  dry50_cot) OUT_ROOT=$RUN_DIR/dry/evals; LIMIT_ARGS=(--limit 50)
    ADAPTER=/workspace/results/run_results/261013_noisy_label_rl/noisy_label_s42_6vcs39wj/checkpoints/final ;;
  dry50_nocot) OUT_ROOT=$RUN_DIR/dry/evals; LIMIT_ARGS=(--limit 50)
    ADAPTER=/workspace/results/run_results/261014_nocot_noisy_label_rl/nocot_noisy_label_s42_5b0x5xp6/checkpoints/final ;;
  *) OUT_ROOT=$RUN_DIR/evals
    RUN_KEY=$($PY -c "import yaml; print(yaml.safe_load(open('$RUN_DIR/$NAME.yaml'))['run_key'])")
    RUN_ID=$($PY -c "import json; print(json.load(open('$RUN_DIR/run_metrics_$RUN_KEY.json'))['run_id'])")
    ADAPTER=/workspace/results/run_results/261018_noisy_label_rl_seeds/${RUN_KEY}_$RUN_ID/checkpoints/final ;;
esac
{
  echo "SETUP_DIR $SETUP_DIR"
  echo "ADAPTER $ADAPTER"
  cd "$RUN_DIR"
  $PY "$SETUP_DIR/eval_checkpoint.py" generate --name "$NAME" --out-root "$OUT_ROOT" "${LIMIT_ARGS[@]}" --adapter "$ADAPTER" \
    && $PY "$SETUP_DIR/eval_checkpoint.py" selfreport --name "$NAME" --out-root "$OUT_ROOT" --adapter "$ADAPTER" \
    && $PY "$SETUP_DIR/eval_checkpoint.py" labels --name "$NAME" --out-root "$OUT_ROOT"
  echo "EXIT $?"
} > "$LOG" 2>&1
