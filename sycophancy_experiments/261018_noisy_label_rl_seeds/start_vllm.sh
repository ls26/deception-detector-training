#!/bin/bash
# Usage (inside its own tmux session): bash start_vllm.sh <tag> <attempt 1-3>
#
# Starts the vLLM server from the copy's vendored GRPO framework on the single GPU (0.40 of its
# memory; training shares the GPU) and writes logs/vllm_<tag>_a<attempt>_<host>.log, which starts with
# the afterburner import path and ends with "EXIT <code>" when the server stops (not when its tmux
# session is killed). <tag> is the RL run the server is for (cot_s43, cot_s44, nocot_s43, nocot_s44) or
# dry (the pre-launch start/stop check). Refuses if that log exists.
set -u
TAG=$1
ATTEMPT=$2
[[ $TAG =~ ^(cot_s43|cot_s44|nocot_s43|nocot_s44|dry)$ ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: start_vllm.sh cot_s43|cot_s44|nocot_s43|nocot_s44|dry <attempt 1-3>" >&2; exit 1; }
source /workspace/sycophancy_experiments/261018_noisy_label_rl_seeds/run_env.sh
LOG=$RUN_DIR/logs/vllm_${TAG}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"
{
  cd /workspace/sycophancy-rl/obfuscation-atlas
  $PY -c 'import afterburner; print("IMPORT_CHECK", afterburner.__file__)'
  $PY -m afterburner.generation.vllm.server --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1
  echo "EXIT $?"
} > "$LOG" 2>&1
