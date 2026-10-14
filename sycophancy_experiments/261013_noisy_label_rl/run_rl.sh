#!/bin/bash
# Usage (inside its own tmux session): bash run_rl.sh smoke|run
#
# Runs the copy's launcher on <name>.yaml in this folder's environment and writes
# logs/<name>_<host>.log. The log starts with the import paths a Python started like the trainer
# sees (same interpreter, same environment, the trainer's working directory) and ends with
# "EXIT <launcher exit code>". An RL step is never re-run: if its log exists, this refuses.
set -u
NAME=$1
[[ $NAME == smoke || $NAME == run ]] || { echo "usage: run_rl.sh smoke|run" >&2; exit 1; }
source /workspace/sycophancy_experiments/261013_noisy_label_rl/run_env.sh
LOG=$RUN_DIR/logs/${NAME}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: an RL step is never re-run" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"
{
  cd /workspace/sycophancy-rl/obfuscation-atlas  # the trainer's working directory (grpo_launcher.OBFUSCATION_ATLAS_DIR)
  $PY -c 'import obfuscation_atlas, afterburner; print("IMPORT_CHECK", obfuscation_atlas.__file__); print("IMPORT_CHECK", afterburner.__file__)'
  cd /workspace/sycophancy-rl
  $PY -m training.run_grpo "$RUN_DIR/$NAME.yaml"
  echo "EXIT $?"
} > "$LOG" 2>&1
