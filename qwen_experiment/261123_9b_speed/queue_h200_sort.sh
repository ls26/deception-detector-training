#!/bin/bash
# 261124 sorting test on the H200 pod (GPU 0 must be an H200): sycophancy, 3 steps each, cuDNN attention off and the
# cheap adapter switch on: srt8_h200 (8 answers per pass, sorted by length), then uns8_h200 (8 per pass, unsorted).
# Starts each run's vLLM at once and each trainer only once $F/CODE_READY exists. Each run stops 90 s after step 3
# (run_to_step3 in queue_common.sh).
# Launch (in the H200 pod's terminal):
#   tmux new-session -d -s 261124-sort "bash /workspace/qwen_experiment/261123_9b_speed/queue_h200_sort.sh"
# Writes logs/queue_h200_sort_<host>.log, logs/gpu_mem_<run>_<host>.log, sycophancy/logs/{vllm,rl}_<run>_a1_<host>.log.
set -uo pipefail
QUEUE=h200_sort
source /workspace/qwen_experiment/261123_9b_speed/queue_common.sh
exec > >(tee -a "$F/logs/queue_${QUEUE}_$HOST.log") 2>&1
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | grep -q H200 || { echo "GPU 0 is not an H200 on $HOST"; exit 1; }
export CUDA_VISIBLE_DEVICES=0
export MIN_FREE_GB=40   # the launcher's free-disk floor (default 50); each test writes under 1 GB
echo "start $(date -u +%H:%M:%S) | host $HOST | $(uptime) | $(lscpu | grep 'Model name' | sed 's/  */ /g') | $(nproc) cores"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
wait_gpu_free 0
run_to_step3 sycophancy srt8_h200 0
run_to_step3 sycophancy uns8_h200 0
echo "end $(date -u +%H:%M:%S) | $(uptime)"
echo QUEUE_DONE
