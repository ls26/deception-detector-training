#!/bin/bash
# Text-only copies of Qwen3.5-4B and -9B (attempt 2: 9B only; 4B done in attempt 1) (make_text_only_checkpoint.py). Log: logs/text_only_<host>.log, ends "EXIT <code>".
set -u
source /workspace/sycophancy-rl/env.sh
export HF_HUB_OFFLINE=1
cd /workspace/qwen_experiment/261120_qwen_port_checks
LOG=logs/text_only_a2_$HOST.log
[[ -e $LOG ]] && { echo "$LOG exists" >&2; exit 1; }
{
  true \
    && $PY make_text_only_checkpoint.py Qwen/Qwen3.5-9B /workspace/hf_cache/text_only/Qwen3.5-9B
  echo "EXIT $?"
} > "$LOG" 2>&1
