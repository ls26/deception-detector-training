#!/bin/bash
# Usage (inside its own tmux session): bash run_rl.sh cot_s43|cot_s44|nocot_s43|nocot_s44 <attempt 1-3>
#
# Runs the copy's launcher on <run>.yaml in this folder's environment and writes
# logs/rl_<run>_a<attempt>_<host>.log. The log starts with the import paths a Python started like the
# trainer sees (same interpreter, same environment, the trainer's working directory) and ends with
# "EXIT <launcher exit code>". Refuses if that attempt's log exists. The launcher skips a run whose
# run_metrics_<run_key>.json exists in this folder.
set -u
NAME=$1
ATTEMPT=$2
[[ $NAME =~ ^(cot_s43|cot_s44|nocot_s43|nocot_s44)$ ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: run_rl.sh cot_s43|cot_s44|nocot_s43|nocot_s44 <attempt 1-3>" >&2; exit 1; }
source /workspace/sycophancy_experiments/261018_noisy_label_rl_seeds/run_env.sh
LOG=$RUN_DIR/logs/rl_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists: this attempt already ran" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"
{
  cd /workspace/sycophancy-rl/obfuscation-atlas  # the trainer's working directory (grpo_launcher.OBFUSCATION_ATLAS_DIR)
  $PY -c 'import obfuscation_atlas, afterburner; print("IMPORT_CHECK", obfuscation_atlas.__file__); print("IMPORT_CHECK", afterburner.__file__)'
  cd /workspace/sycophancy-rl
  $PY -m training.run_grpo "$RUN_DIR/$NAME.yaml"
  echo "EXIT $?"
} > "$LOG" 2>&1
