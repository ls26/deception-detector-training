#!/bin/bash
# Dry run: 50 questions (500 answers) on Qwen 4B and on Llama, and the full MBPP set on Qwen 4B, into dry/.
set -uo pipefail
cd /workspace/qwen_experiment/261120_base_models
source folder_env.sh
step() { local name=$1; shift; echo "=== $name $(date -u +%H:%M:%S)"; "$@" > "logs/dry_${name}_${HOST}.log" 2>&1; echo "EXIT $? $name $(date -u +%H:%M:%S)"; }
step syc_qwen3_5_4b $PY generate.py --model Qwen/Qwen3.5-4B --out-dir dry/qwen3_5_4b/sycophancy --limit 50
step syc_llama3_8b $PY generate.py --model meta-llama/Meta-Llama-3-8B-Instruct --out-dir dry/llama3_8b/sycophancy --limit 50
step mbpp_prompts_qwen3_5_4b $PY mbpp_generate.py prompts --model Qwen/Qwen3.5-4B --out-dir dry/qwen3_5_4b/mbpp
step mbpp_generate_qwen3_5_4b $PY mbpp_generate.py generate --model Qwen/Qwen3.5-4B --out-dir dry/qwen3_5_4b/mbpp
step mbpp_score_qwen3_5_4b $PY mbpp_generate.py score --out-dir dry/qwen3_5_4b/mbpp
echo DRY_DONE
