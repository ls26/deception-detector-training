#!/bin/bash
# Usage (inside its own tmux session): bash start_vllm.sh <experiment folder> <run> <attempt 1-3>
#
# Starts the vLLM server from the copy's vendored GRPO framework on the single GPU (0.40 of its memory;
# training shares the GPU) and writes <folder>/logs/vllm_<run>_a<attempt>_<host>.log, which starts with the
# afterburner import path and ends with "EXIT <code>" when the server stops (not when its tmux session is
# killed). <run> is the RL run the server is for: the name of a config <folder>/<run>.yaml. Refuses if that
# log exists.
set -u
FOLDER=$1
NAME=$2
ATTEMPT=$3
source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
[[ $FOLDER == /* ]] && [[ -e $FOLDER/$NAME.yaml ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: start_vllm.sh <experiment folder (absolute)> <run with a config <folder>/<run>.yaml> <attempt 1-3>" >&2; exit 1; }
LOG=$FOLDER/logs/vllm_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
mkdir -p "$FOLDER/logs"
{
  cd /workspace/sycophancy-rl/obfuscation-atlas
  $PY -c 'import afterburner; print("IMPORT_CHECK", afterburner.__file__)'
  $PY -m afterburner.generation.vllm.server --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1
  echo "EXIT $?"
} > "$LOG" 2>&1
