#!/bin/bash
# Usage (inside its own tmux session): bash start_vllm_gpu.sh <experiment folder> <run> <attempt 1-3> <gpu> <port>
#
# Copied from 261123_9b_speed (the vLLM server code is the same in both checkouts, so it runs from the live copy).
# sycophancy_setup/start_vllm.sh with the GPU and the port given (the runs use port 8000: the launcher's start check
# is fixed at 8000, so a pod runs one server; on the 2-GPU pod f6bf0473cbc9 the MBPP run's server and trainer use
# GPU 1 and GPU 0 does the offline-vLLM work). Otherwise unchanged:
# 0.40 of the GPU's memory (training shares the GPU), the model the run trains, --max-model-len 4096,
# --max-num-seqs 512. Writes <folder>/logs/vllm_<run>_a<attempt>_<host>.log, ending with "EXIT <code>".
set -u
FOLDER=$1
NAME=$2
ATTEMPT=$3
GPU=$4
PORT=$5
source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
[[ $FOLDER == /* ]] && [[ -e $FOLDER/$NAME.yaml ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] && [[ $GPU =~ ^[0-9]$ ]] && [[ $PORT =~ ^[0-9]+$ ]] \
  || { echo "usage: start_vllm_gpu.sh <experiment folder (absolute)> <run> <attempt 1-3> <gpu> <port>" >&2; exit 1; }
LOG=$FOLDER/logs/vllm_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
mkdir -p "$FOLDER/logs"
MODEL=$(cd "$SETUP_DIR" && $PY -c "from pathlib import Path; from spec import model_of_run; print(model_of_run(Path('$FOLDER/$NAME.yaml')))") || exit 1
{
  cd /workspace/sycophancy-rl/obfuscation-atlas
  $PY -c 'import afterburner; print("IMPORT_CHECK", afterburner.__file__)'
  echo "MODEL $MODEL GPU $GPU PORT $PORT"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m afterburner.generation.vllm.server --model "$MODEL" --max-model-len 4096 --max-num-seqs 512 \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1 --port "$PORT"
  echo "EXIT $?"
} > "$LOG" 2>&1
