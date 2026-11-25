#!/bin/bash
# Smoke checks (a) and (b): bash run_check_ab.sh (inside tmux). Log: logs/check_ab_<host>.log, ends with "EXIT <code>".
set -u
source /workspace/sycophancy-rl/env.sh
export HF_HUB_OFFLINE=1
cd /workspace/qwen_experiment/261120_qwen_port_checks
LOG=logs/check_ab_a2_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
{
  $PY check_vllm_lora.py adapters --out-dir check_ab_a2 \
    && $PY check_vllm_lora.py vllm --out-dir check_ab_a2 \
    && $PY check_vllm_lora.py hf --out-dir check_ab_a2
  echo "EXIT $?"
} > "$LOG" 2>&1
