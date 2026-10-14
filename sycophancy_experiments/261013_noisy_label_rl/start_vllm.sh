#!/bin/bash
# Usage (inside its own tmux session): bash start_vllm.sh <tag>
#
# Starts the vLLM server from the copy's vendored GRPO framework on the single GPU (0.40 of its
# memory; training shares the GPU) and writes logs/vllm_<tag>_<host>.log, which starts with the
# afterburner import path and ends with "EXIT <code>" when the server stops. <tag> names the RL
# step the server is for (smoke or run), so each start gets its own log.
set -u
TAG=$1
[[ $TAG == smoke || $TAG == run ]] || { echo "usage: start_vllm.sh smoke|run" >&2; exit 1; }
source /workspace/sycophancy_experiments/261013_noisy_label_rl/run_env.sh
LOG=$RUN_DIR/logs/vllm_${TAG}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
mkdir -p "$RUN_DIR/logs"
{
  cd /workspace/sycophancy-rl/obfuscation-atlas
  $PY -c 'import afterburner; print("IMPORT_CHECK", afterburner.__file__)'
  $PY -m afterburner.generation.vllm.server --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1
  echo "EXIT $?"
} > "$LOG" 2>&1
