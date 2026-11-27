#!/bin/bash
# Usage (inside its own tmux session): bash run_rl_worktree.sh <run> <attempt 1-3>
#
# probe-rl/launcher/run_rl.sh run from the /workspace/eff-wip checkout (folder_env.sh's import paths, its launcher and
# trainer) with the live copy's venv. Config: this folder's <run>.yaml. Writes logs/rl_<run>_a<attempt>_<host>.log,
# which starts with the import paths the trainer sees and ends with "EXIT <launcher exit code>".
# Phase 2 (before main's reorganization) ran 261123_9b_speed/run_rl_noretry.sh's launch line (training.run_grpo with
# the launcher's crash retries set to 0); main's launcher no longer retries a crash, so phase 3 runs it as it is.
set -u
NAME=$1
ATTEMPT=$2
source /workspace/eff-wip/qwen_experiment/261126_low_memory_passes/folder_env.sh
[[ -e $F/$NAME.yaml ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: run_rl_worktree.sh <run with a config $F/<run>.yaml> <attempt 1-3>" >&2; exit 1; }
LOG=$F/logs/rl_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
{
  cd $CODE/obfuscation-atlas  # the trainer's working directory (launcher.common.OBFUSCATION_ATLAS_DIR)
  $PY -c 'import obfuscation_atlas, afterburner, launcher; print("IMPORT_CHECK", obfuscation_atlas.__file__); print("IMPORT_CHECK", afterburner.__file__); print("IMPORT_CHECK", launcher.__file__)'
  cd $CODE
  $PY -m launcher.run_grpo "$F/$NAME.yaml"
  echo "EXIT $?"
} > "$LOG" 2>&1
