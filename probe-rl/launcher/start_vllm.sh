#!/bin/bash
# Usage (inside its own tmux session): bash start_vllm.sh <experiment folder> <run> <attempt 1-3> <gpu>
#
# Starts the vLLM server from the copy's vendored GRPO framework on GPU <gpu>, with 0.40 of its memory (training
# shares the GPU), for the model the run trains (its config's model preset, launcher.common.model_of_run; the server
# loads the model class the trainer trains, text-only for Qwen3.5). It listens on port 8000, the only port the
# launcher's start check uses, so a pod runs one server at a time. <run> is the RL run the server is for: the name of
# a config <folder>/<run>.yaml. Writes <folder>/logs/vllm_<run>_a<attempt>_<host>.log, which starts with the
# afterburner import path and ends with "EXIT <code>" when the server stops (not when its tmux session is killed).
# Refuses if that log exists.
set -u
[[ $# -eq 4 ]] || { echo "usage: start_vllm.sh <experiment folder (absolute)> <run with a config <folder>/<run>.yaml> <attempt 1-3> <gpu>" >&2; exit 1; }
FOLDER=$1
NAME=$2
ATTEMPT=$3
GPU=$4
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
[[ $FOLDER == /* ]] && [[ -e $FOLDER/$NAME.yaml ]] && [[ $ATTEMPT =~ ^[1-3]$ ]] && [[ $GPU =~ ^[0-9]$ ]] \
  || { echo "usage: start_vllm.sh <experiment folder (absolute)> <run with a config <folder>/<run>.yaml> <attempt 1-3> <gpu>" >&2; exit 1; }
LOG=$FOLDER/logs/vllm_${NAME}_a${ATTEMPT}_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
mkdir -p "$FOLDER/logs"
MODEL=$(cd "$COPY" && $PY -c "from pathlib import Path; from launcher.common import model_of_run; print(model_of_run(Path('$FOLDER/$NAME.yaml')))") || exit 1
cd "$COPY/obfuscation-atlas" || exit 1
{
  $PY -c 'import afterburner; print("IMPORT_CHECK", afterburner.__file__)'
  echo "MODEL $MODEL GPU $GPU"
  # --max-model-len 4096: the Qwen3.5 presets' 2048-token prompt and answer caps (the server's default, 2048, holds
  # Llama's 1024 + 1024). --max-num-seqs 512: a training step's answers (64 prompts x 8); at 0.40 of the GPU,
  # Qwen3.5-4B's linear-attention state has room for 978 sequences, under vLLM's default of 1024.
  CUDA_VISIBLE_DEVICES=$GPU $PY -m afterburner.generation.vllm.server --model "$MODEL" --max-model-len 4096 \
    --max-num-seqs 512 --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1
  CODE=$?
  echo "EXIT $CODE"
} > "$LOG" 2>&1
exit $CODE
