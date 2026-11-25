#!/bin/bash
# 261124 sorting test on pod fdc2d68d6537 (1 H100, GPU 0): sycophancy, 3 steps each, cuDNN attention off and the cheap
# adapter switch on: srt4_h100 (4 answers per pass, sorted by length), then uns4_h100 (4 per pass, unsorted).
# Waits for the GPU to be free (the profile runs first), starts each run's vLLM at once, and starts each trainer only
# once $F/CODE_READY exists. Each run stops 90 s after step 3 (run_to_step3 in queue_common.sh).
# Launch: tmux new-session -d -s 261124-sort "bash /workspace/qwen_experiment/261123_9b_speed/queue_h100_sort.sh"
# Writes logs/queue_h100_sort_<host>.log, logs/gpu_mem_<run>_<host>.log, sycophancy/logs/{vllm,rl}_<run>_a1_<host>.log.
set -uo pipefail
QUEUE=h100_sort
source /workspace/qwen_experiment/261123_9b_speed/queue_common.sh
exec > >(tee -a "$F/logs/queue_${QUEUE}_$HOST.log") 2>&1
[[ $HOST == fdc2d68d6537 ]] || { echo "wrong pod: $HOST"; exit 1; }
export CUDA_VISIBLE_DEVICES=0
export MIN_FREE_GB=40   # the launcher's free-disk floor (default 50); each test writes under 1 GB
echo "start $(date -u +%H:%M:%S) | $(uptime) | $(lscpu | grep 'Model name' | sed 's/  */ /g') | $(nproc) cores"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
wait_gpu_free 0
run_to_step3 sycophancy srt4_h100 0
run_to_step3 sycophancy uns4_h100 0
echo "end $(date -u +%H:%M:%S) | $(uptime)"
echo QUEUE_DONE
