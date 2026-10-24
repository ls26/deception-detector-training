#!/bin/bash
# Usage (inside its own tmux session): bash run_eval.sh <name> <attempt 1-3>
#
# Runs the setup's own eval_checkpoint.py (261013's for CoT, 261014's for no-CoT, unchanged, by absolute
# path) through its three stages (generate, selfreport, labels) for one checkpoint and writes
# <format>/logs/eval_<name>_a<attempt>_<host>.log, ending with "EXIT <code>" (the first failing stage's code,
# or 0). <format> is cot or nocot, the name's first word.
#   <run>               an RL run with a config <format>/<run>.yaml: its final adapter (run id from
#                       <format>/run_metrics_<run>.json), all eval questions            -> <format>/evals/<run>/
#   <format>_smoke_s42  the smoke run's final adapter, the first 50 eval questions        -> <format>/dry/evals/<name>/
#   nocot_base50        the base model, the first 50 eval questions (no-CoT only)        -> nocot/dry/evals/nocot_base50/
# Refuses if that attempt's log exists; eval_checkpoint.py refuses to overwrite its outputs, so before a
# rerun the failed attempt's output folder must be moved to <format>/failed_attempts/.
set -u
NAME=$1
ATTEMPT=$2
FORMAT=${NAME%%_*}
source /workspace/sycophancy_experiments/261020_reference_rl/run_env.sh
[[ $FORMAT =~ ^(cot|nocot)$ ]] && { [[ -e $RUN_DIR/$FORMAT/$NAME.yaml ]] || [[ $NAME == nocot_base50 ]]; } && [[ $ATTEMPT =~ ^[1-3]$ ]] \
  || { echo "usage: run_eval.sh <run with a config <format>/<run>.yaml> | nocot_base50 <attempt 1-3>" >&2; exit 1; }
LOG=$RUN_DIR/$FORMAT/logs/eval_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
mkdir -p "$RUN_DIR/$FORMAT/logs"
case $FORMAT in
  cot) SETUP_DIR=/workspace/sycophancy_experiments/261013_noisy_label_rl ;;
  nocot) SETUP_DIR=/workspace/sycophancy_experiments/261014_nocot_noisy_label_rl ;;
esac
LIMIT_ARGS=()
ADAPTER_ARGS=()
case $NAME in
  nocot_base50) OUT_ROOT=$RUN_DIR/nocot/dry/evals; LIMIT_ARGS=(--limit 50) ;;
  *)
    RUN_KEY=$($PY -c "import yaml; print(yaml.safe_load(open('$RUN_DIR/$FORMAT/$NAME.yaml'))['run_key'])")
    GROUP=$($PY -c "import yaml; print(yaml.safe_load(open('$RUN_DIR/$FORMAT/$NAME.yaml'))['group'])")
    RUN_ID=$($PY -c "import json; print(json.load(open('$RUN_DIR/$FORMAT/run_metrics_$RUN_KEY.json'))['run_id'])")
    ADAPTER_ARGS=(--adapter "/workspace/results/run_results/$GROUP/${RUN_KEY}_$RUN_ID/checkpoints/final")
    case $NAME in
      *_smoke_s42) OUT_ROOT=$RUN_DIR/$FORMAT/dry/evals; LIMIT_ARGS=(--limit 50) ;;
      *) OUT_ROOT=$RUN_DIR/$FORMAT/evals ;;
    esac ;;
esac
{
  echo "SETUP_DIR $SETUP_DIR"
  echo "ADAPTER ${ADAPTER_ARGS[*]:-base model}"
  cd "$RUN_DIR/$FORMAT"
  $PY "$SETUP_DIR/eval_checkpoint.py" generate --name "$NAME" --out-root "$OUT_ROOT" "${LIMIT_ARGS[@]}" "${ADAPTER_ARGS[@]}" \
    && $PY "$SETUP_DIR/eval_checkpoint.py" selfreport --name "$NAME" --out-root "$OUT_ROOT" "${ADAPTER_ARGS[@]}" \
    && $PY "$SETUP_DIR/eval_checkpoint.py" labels --name "$NAME" --out-root "$OUT_ROOT"
  echo "EXIT $?"
} > "$LOG" 2>&1
