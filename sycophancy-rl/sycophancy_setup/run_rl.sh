#!/bin/bash
# Usage (inside its own tmux session): bash run_rl.sh <experiment folder> <run> <attempt 1-3>
#
# Runs the copy's launcher on the config <folder>/<run>.yaml in this package's environment and writes
# <folder>/logs/rl_<run>_a<attempt>_<host>.log. The log starts with the import paths a Python started like the
# trainer sees (same interpreter, same environment, the trainer's working directory) and ends with
# "EXIT <launcher exit code>". Refuses if that attempt's log exists. The launcher skips a run whose
# run_metrics_<run_key>.json exists in the config's output_dir.
set -u
FOLDER=$1
NAME=$2
ATTEMPT=$3
source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
[[ $FOLDER == /* ]] && [[ -e $FOLDER/$NAME.yaml ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: run_rl.sh <experiment folder (absolute)> <run with a config <folder>/<run>.yaml> <attempt 1-3>" >&2; exit 1; }
grep -q FILL_IN "$FOLDER/$NAME.yaml" && { echo "$FOLDER/$NAME.yaml still has a FILL_IN field" >&2; exit 1; }
LOG=$FOLDER/logs/rl_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
mkdir -p "$FOLDER/logs"
{
  cd /workspace/sycophancy-rl/obfuscation-atlas  # the trainer's working directory (grpo_launcher.OBFUSCATION_ATLAS_DIR)
  $PY -c 'import obfuscation_atlas, afterburner; print("IMPORT_CHECK", obfuscation_atlas.__file__); print("IMPORT_CHECK", afterburner.__file__)'
  cd /workspace/sycophancy-rl
  $PY -m training.run_grpo "$FOLDER/$NAME.yaml"
  echo "EXIT $?"
} > "$LOG" 2>&1
