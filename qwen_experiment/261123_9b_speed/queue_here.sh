#!/bin/bash
# 261123 speed test on pod fdc2d68d6537 (1 GPU): sycophancy at 4 answers per forward/backward pass, 3 steps.
# Launch: tmux new-session -d -s 261123-speed "bash /workspace/qwen_experiment/261123_9b_speed/queue_here.sh"
# Writes logs/queue_here_<host>.log (this script), logs/gpu_mem_spd4_syc9b_<host>.log (GPU memory every 5 s),
# sycophancy/logs/{vllm,rl}_spd4_syc9b_a1_<host>.log.
set -uo pipefail
QUEUE=here
source /workspace/qwen_experiment/261123_9b_speed/queue_common.sh
exec > >(tee -a "$F/logs/queue_${QUEUE}_$HOST.log") 2>&1
[[ $HOST == fdc2d68d6537 ]] || { echo "wrong pod: $HOST"; exit 1; }
export CUDA_VISIBLE_DEVICES=0
export MIN_FREE_GB=40   # the launcher's free-disk floor (default 50); /workspace had 51 GB free, each test writes ~3 GB
echo "start $(date -u +%H:%M:%S) | $(uptime)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
wait_gpu_free 0
nvidia-smi -i 0 --query-gpu=timestamp,memory.used --format=csv,noheader -l 5 > "$F/logs/gpu_mem_spd4_syc9b_$HOST.log" &
MEM_PID=$!
run_with_server sycophancy spd4_syc9b 0 1
kill $MEM_PID
echo "end $(date -u +%H:%M:%S) | $(uptime)"
echo QUEUE_DONE
