#!/bin/bash
# Usage (inside its own tmux session): bash start_vllm.sh <experiment folder> <run> <attempt 1-3>
#
# Starts the vLLM server from the copy's vendored GRPO framework on the single GPU (0.40 of its memory;
# training shares the GPU) for the model the run trains (its config's model preset, spec.model_of_run; the server
# loads the model class the trainer trains, text-only for Qwen3.5) and writes
# <folder>/logs/vllm_<run>_a<attempt>_<host>.log, which starts with the afterburner import path and ends with
# "EXIT <code>" when the server stops (not when its tmux session is killed). <run> is the RL run the server is for: the name of a config <folder>/<run>.yaml. Refuses if that
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
MODEL=$(cd "$SETUP_DIR" && $PY -c "from pathlib import Path; from spec import model_of_run; print(model_of_run(Path('$FOLDER/$NAME.yaml')))") || exit 1
{
  cd /workspace/sycophancy-rl/obfuscation-atlas
  $PY -c 'import afterburner; print("IMPORT_CHECK", afterburner.__file__)'
  echo "MODEL $MODEL"
  # --max-model-len 4096: the Qwen3.5 presets' 2048-token prompt and answer caps (the server's default, 2048, holds
  # Llama's 1024 + 1024). --max-num-seqs 512: a training step's answers (64 prompts x 8); at 0.40 of the GPU,
  # Qwen3.5-4B's linear-attention state has room for 978 sequences, under vLLM's default of 1024.
  $PY -m afterburner.generation.vllm.server --model "$MODEL" --max-model-len 4096 --max-num-seqs 512 \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1
  echo "EXIT $?"
} > "$LOG" 2>&1
