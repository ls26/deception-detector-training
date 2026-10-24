#!/bin/bash
# Usage (inside its own tmux session): bash run_eval.sh <experiment folder> <name> <attempt 1-3> [<probe run>]
#
# Runs this package's eval_checkpoint.py through its three stages (generate, selfreport, labels) for one
# checkpoint and writes <folder>/logs/eval_<name>_a<attempt>_<host>.log, ending with "EXIT <code>" (the first
# failing stage's code, or 0).
#   <run>         an RL run with a config <folder>/<run>.yaml: its final adapter, scored with the run's own
#                 start-of-run and fresh probes, all eval questions                      -> <folder>/evals/<run>/
#                 (a run whose name contains "smoke": the first 50 eval questions        -> <folder>/dry/evals/<run>/)
#   base          the base model, scored with the start-of-run probe of <probe run>      -> <folder>/evals/base/
#                 (so it runs after that run has finished: its run id is in its run_metrics json)
# A run's folder is /workspace/results/run_results/<group>/<run_key>_<run id> (run id from
# <output_dir>/run_metrics_<run_key>.json). Refuses if that attempt's log exists; eval_checkpoint.py refuses to
# overwrite its outputs, so before a rerun the failed attempt's output folder must be moved away.
set -u
FOLDER=$1
NAME=$2
ATTEMPT=$3
source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
run_dir() {  # the run folder of the RL run with the config $FOLDER/$1.yaml
  $PY -c "
import json, yaml
config = yaml.safe_load(open('$FOLDER/$1.yaml'))
run_id = json.load(open(config['output_dir'] + '/run_metrics_' + config['run_key'] + '.json'))['run_id']
print('/workspace/results/run_results/' + config['group'] + '/' + config['run_key'] + '_' + run_id)"
}
[[ $FOLDER == /* ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] && { [[ -e $FOLDER/$NAME.yaml ]] || { [[ $NAME == base ]] && [[ -e $FOLDER/${4:-}.yaml ]]; }; } \
  || { echo "usage: run_eval.sh <experiment folder (absolute)> <run with a config <folder>/<run>.yaml> <attempt 1-3> | <folder> base <attempt> <probe run>" >&2; exit 1; }
LOG=$FOLDER/logs/eval_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
mkdir -p "$FOLDER/logs"
LIMIT_ARGS=()
ADAPTER_ARGS=()
case $NAME in
  base) PROBE_RUN_DIR=$(run_dir "$4") || exit 1; OUT_DIR=$FOLDER/evals/base ;;
  *)
    RUN_DIR=$(run_dir "$NAME") || exit 1
    PROBE_RUN_DIR=$RUN_DIR
    ADAPTER_ARGS=(--adapter "$RUN_DIR/checkpoints/final")
    case $NAME in
      *smoke*) OUT_DIR=$FOLDER/dry/evals/$NAME; LIMIT_ARGS=(--limit 50) ;;
      *) OUT_DIR=$FOLDER/evals/$NAME ;;
    esac ;;
esac
{
  echo "ADAPTER ${ADAPTER_ARGS[*]:-base model}"
  echo "PROBE_RUN_DIR $PROBE_RUN_DIR"
  cd "$FOLDER"
  $PY "$SETUP_DIR/eval_checkpoint.py" generate --out-dir "$OUT_DIR" "${LIMIT_ARGS[@]}" "${ADAPTER_ARGS[@]}" \
    && $PY "$SETUP_DIR/eval_checkpoint.py" selfreport --out-dir "$OUT_DIR" --probe-run-dir "$PROBE_RUN_DIR" "${ADAPTER_ARGS[@]}" \
    && $PY "$SETUP_DIR/eval_checkpoint.py" labels --out-dir "$OUT_DIR"
  echo "EXIT $?"
} > "$LOG" 2>&1
