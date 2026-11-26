#!/bin/bash
# Usage (inside its own tmux session): bash run_rl.sh <experiment folder> <run> <attempt 1-3>
#
# Runs the launcher on the config <folder>/<run>.yaml and writes <folder>/logs/rl_<run>_a<attempt>_<host>.log. The
# log starts with the import paths a Python started like the trainer sees (same interpreter, same environment, the
# trainer's working directory) and ends with "EXIT <launcher exit code>"; the script exits with that code. Refuses if
# that attempt's log exists. The launcher skips a run whose run_metrics_<run_key>.json exists in the config's
# output_dir, and stops on a run folder without one (launcher/grpo_launcher.py).
set -u
[[ $# -eq 3 ]] || { echo "usage: run_rl.sh <experiment folder (absolute)> <run with a config <folder>/<run>.yaml> <attempt 1-3>" >&2; exit 1; }
FOLDER=$1
NAME=$2
ATTEMPT=$3
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
[[ $FOLDER == /* ]] && [[ -e $FOLDER/$NAME.yaml ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: run_rl.sh <experiment folder (absolute)> <run with a config <folder>/<run>.yaml> <attempt 1-3>" >&2; exit 1; }
grep -q FILL_IN "$FOLDER/$NAME.yaml" && { echo "$FOLDER/$NAME.yaml still has a FILL_IN field" >&2; exit 1; }
LOG=$FOLDER/logs/rl_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
mkdir -p "$FOLDER/logs"
cd "$COPY/obfuscation-atlas" || exit 1  # the trainer's working directory (launcher.common.OBFUSCATION_ATLAS_DIR)
{
  $PY -c 'import obfuscation_atlas, afterburner; print("IMPORT_CHECK", obfuscation_atlas.__file__); print("IMPORT_CHECK", afterburner.__file__)'
  cd "$COPY" && $PY -m launcher.run_grpo "$FOLDER/$NAME.yaml"
  CODE=$?
  echo "EXIT $CODE"
} > "$LOG" 2>&1
exit $CODE
