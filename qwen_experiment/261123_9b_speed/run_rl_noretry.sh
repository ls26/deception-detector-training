#!/bin/bash
# Usage: bash run_rl_noretry.sh <experiment folder> <run> <attempt 1-3>
#
# sycophancy_setup/run_rl.sh with the launcher's crash retries set to 0 (grpo_launcher.MAX_GRPO_RETRIES, read when a
# launch starts), for the 261124 speed tests: a crashed test run ends instead of having its folder deleted and being
# started again. Writes <folder>/logs/rl_<run>_a<attempt>_<host>.log, ending with "EXIT <launcher exit code>".
set -u
FOLDER=$1
NAME=$2
ATTEMPT=$3
source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
[[ $FOLDER == /* ]] && [[ -e $FOLDER/$NAME.yaml ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: run_rl_noretry.sh <experiment folder (absolute)> <run with a config <folder>/<run>.yaml> <attempt 1-3>" >&2; exit 1; }
grep -q FILL_IN "$FOLDER/$NAME.yaml" && { echo "$FOLDER/$NAME.yaml still has a FILL_IN field" >&2; exit 1; }
LOG=$FOLDER/logs/rl_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
mkdir -p "$FOLDER/logs"
{
  cd /workspace/sycophancy-rl/obfuscation-atlas  # the trainer's working directory (grpo_launcher.OBFUSCATION_ATLAS_DIR)
  $PY -c 'import obfuscation_atlas, afterburner; print("IMPORT_CHECK", obfuscation_atlas.__file__); print("IMPORT_CHECK", afterburner.__file__)'
  cd /workspace/sycophancy-rl
  $PY -c "import runpy, sys; from training import grpo_launcher; grpo_launcher.MAX_GRPO_RETRIES = 0; sys.argv = ['training.run_grpo', '$FOLDER/$NAME.yaml']; runpy.run_module('training.run_grpo', run_name='__main__', alter_sys=True)"
  echo "EXIT $?"
} > "$LOG" 2>&1
