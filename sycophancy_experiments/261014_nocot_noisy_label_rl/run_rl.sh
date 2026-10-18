#!/bin/bash
# Usage (inside its own tmux session): bash run_rl.sh smoke|run <attempt 1-3>
#
# Runs the copy's launcher on <name>.yaml in this folder's environment and writes
# logs/<name>_a<attempt>_<host>.log. The log starts with the import paths a Python started like the
# trainer sees (same interpreter, same environment, the trainer's working directory) and ends with
# "EXIT <launcher exit code>". Refuses if that attempt's log exists. Before a rerun after a fix, a
# completed earlier attempt's run_metrics_<run_key>.json must be moved to failed_attempts/ (the
# launcher skips a run whose metrics file exists).
set -u
NAME=$1
ATTEMPT=$2
[[ $NAME == smoke || $NAME == run ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: run_rl.sh smoke|run <attempt 1-3>" >&2; exit 1; }
source /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/run_env.sh
LOG=$RUN_DIR/logs/${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"
{
  cd /workspace/sycophancy-rl/obfuscation-atlas  # the trainer's working directory (grpo_launcher.OBFUSCATION_ATLAS_DIR)
  $PY -c 'import obfuscation_atlas, afterburner; print("IMPORT_CHECK", obfuscation_atlas.__file__); print("IMPORT_CHECK", afterburner.__file__)'
  cd /workspace/sycophancy-rl
  $PY -m training.run_grpo "$RUN_DIR/$NAME.yaml"
  echo "EXIT $?"
} > "$LOG" 2>&1
