#!/bin/bash
# Usage (inside its own tmux session): bash start_vllm.sh <run> <attempt 1-3>
#
# Starts the vLLM server from the copy's vendored GRPO framework on the single GPU (0.40 of its
# memory; training shares the GPU) and writes <format>/logs/vllm_<run>_a<attempt>_<host>.log, which starts
# with the afterburner import path and ends with "EXIT <code>" when the server stops (not when its tmux
# session is killed). <run> is the RL run the server is for: the name of a config <format>/<run>.yaml
# (<format> = cot or nocot, the run name's first word). Refuses if that log exists.
set -u
NAME=$1
ATTEMPT=$2
FORMAT=${NAME%%_*}
source /workspace/sycophancy_experiments/261020_reference_rl/run_env.sh
[[ $FORMAT =~ ^(cot|nocot)$ ]] && [[ -e $RUN_DIR/$FORMAT/$NAME.yaml ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] || { echo "usage: start_vllm.sh <run with a config <format>/<run>.yaml> <attempt 1-3>" >&2; exit 1; }
LOG=$RUN_DIR/$FORMAT/logs/vllm_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
mkdir -p "$RUN_DIR/$FORMAT/logs"
{
  cd /workspace/sycophancy-rl/obfuscation-atlas
  $PY -c 'import afterburner; print("IMPORT_CHECK", afterburner.__file__)'
  $PY -m afterburner.generation.vllm.server --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1
  echo "EXIT $?"
} > "$LOG" 2>&1
