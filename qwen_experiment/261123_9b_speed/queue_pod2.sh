#!/bin/bash
# 261123 speed test on the 2-GPU pod f6bf0473cbc9, GPU 1 (as its MBPP calibration run): MBPP at 4 answers per
# forward/backward pass, 3 steps.
# Launch: tmux new-session -d -s 261123-speed "bash /workspace/qwen_experiment/261123_9b_speed/queue_pod2.sh"
# Writes logs/queue_pod2_<host>.log (this script), logs/gpu_mem_spd4_mbpp9b_<host>.log (GPU memory every 5 s),
# mbpp/logs/{vllm,rl}_spd4_mbpp9b_a1_<host>.log.
set -uo pipefail
QUEUE=pod2
source /workspace/qwen_experiment/261123_9b_speed/queue_common.sh
exec > >(tee -a "$F/logs/queue_${QUEUE}_$HOST.log") 2>&1
[[ $HOST == f6bf0473cbc9 ]] || { echo "wrong pod: $HOST"; exit 1; }
export CUDA_VISIBLE_DEVICES=1
export MIN_FREE_GB=40   # the launcher's free-disk floor (default 50); /workspace had 51 GB free, each test writes ~3 GB
echo "start $(date -u +%H:%M:%S) | $(uptime)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
wait_gpu_free 1
nvidia-smi -i 1 --query-gpu=timestamp,memory.used --format=csv,noheader -l 5 > "$F/logs/gpu_mem_spd4_mbpp9b_$HOST.log" &
MEM_PID=$!
run_with_server mbpp spd4_mbpp9b 1 1
kill $MEM_PID
echo "end $(date -u +%H:%M:%S) | $(uptime)"
echo QUEUE_DONE
