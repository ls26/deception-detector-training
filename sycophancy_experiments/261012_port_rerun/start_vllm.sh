#!/bin/bash
# Usage (inside its own tmux session): bash start_vllm.sh
#
# Starts the vLLM server from the copy's vendored GRPO framework on the single GPU (0.40 of its
# memory; training shares the GPU) and writes logs/vllm_<host>.log, which starts with the
# afterburner import path and ends with "EXIT <code>" when the server stops.
set -u
source /workspace/sycophancy_experiments/261012_port_rerun/run_env.sh
LOG=$RUN_DIR/logs/vllm_$HOST.log
{
  cd /workspace/sycophancy-rl/obfuscation-atlas
  $PY -c 'import afterburner; print("IMPORT_CHECK", afterburner.__file__)'
  $PY -m afterburner.generation.vllm.server --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1
  echo "EXIT $?"
} > "$LOG" 2>&1
