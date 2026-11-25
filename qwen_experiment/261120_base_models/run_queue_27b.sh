#!/bin/bash
# Rerun of run_queue.sh's 27B steps after its first attempt failed at vLLM startup (max_num_seqs; see generate.py
# MAX_NUM_SEQS). The MBPP prompts from the first attempt are reused (that step succeeded); then the tables.
set -uo pipefail
cd /workspace/qwen_experiment/261120_base_models
source folder_env.sh
step() { local name=$1; shift; echo "=== $name $(date -u +%H:%M:%S)"; "$@" > "logs/${name}_a2_${HOST}.log" 2>&1; echo "EXIT $? $name $(date -u +%H:%M:%S)"; }
step syc_qwen3_5_27b $PY generate.py --model Qwen/Qwen3.5-27B --out-dir runs/qwen3_5_27b/sycophancy
step mbpp_generate_qwen3_5_27b $PY mbpp_generate.py generate --model Qwen/Qwen3.5-27B --out-dir runs/qwen3_5_27b/mbpp
step mbpp_score_qwen3_5_27b $PY mbpp_generate.py score --out-dir runs/qwen3_5_27b/mbpp
step summarize $PY summarize.py --runs-dir runs
echo QUEUE_DONE
