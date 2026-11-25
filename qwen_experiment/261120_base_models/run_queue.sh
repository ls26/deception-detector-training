#!/bin/bash
# The measurement queue: per model, sycophancy (all 2290 questions, 8 + 1 + 1 answers, 4096-token cap) and MBPP (400
# problems, 1024-token cap), thinking off; then Qwen 9B with thinking on (300 eval questions; 400 MBPP problems; 8192);
# then the tables. A failed step is logged and the queue goes on to the next one.
set -uo pipefail
cd /workspace/qwen_experiment/261120_base_models
source folder_env.sh
step() { local name=$1; shift; echo "=== $name $(date -u +%H:%M:%S)"; "$@" > "logs/${name}_${HOST}.log" 2>&1; echo "EXIT $? $name $(date -u +%H:%M:%S)"; }
measure() {  # key, model, extra flags
  local key=$1 model=$2; shift 2
  step "syc_${key}" $PY generate.py --model "$model" --out-dir "runs/${key}/sycophancy" "$@"
  step "mbpp_prompts_${key}" $PY mbpp_generate.py prompts --model "$model" --out-dir "runs/${key}/mbpp" "$@"
  step "mbpp_generate_${key}" $PY mbpp_generate.py generate --model "$model" --out-dir "runs/${key}/mbpp" "$@"
  step "mbpp_score_${key}" $PY mbpp_generate.py score --out-dir "runs/${key}/mbpp"
}
measure llama3_8b meta-llama/Meta-Llama-3-8B-Instruct
# The pipeline check against the release's Llama artifact, made at the RL's 256-token cap (not a measurement column)
step mbpp_prompts_llama3_8b_check_256 $PY mbpp_generate.py prompts --model meta-llama/Meta-Llama-3-8B-Instruct --out-dir runs/llama3_8b/mbpp_check_256
step mbpp_generate_llama3_8b_check_256 $PY mbpp_generate.py generate --model meta-llama/Meta-Llama-3-8B-Instruct --out-dir runs/llama3_8b/mbpp_check_256 --max-tokens 256
step mbpp_score_llama3_8b_check_256 $PY mbpp_generate.py score --out-dir runs/llama3_8b/mbpp_check_256
measure qwen3_5_4b Qwen/Qwen3.5-4B
measure qwen3_5_9b Qwen/Qwen3.5-9B
measure qwen3_5_9b_thinking Qwen/Qwen3.5-9B --thinking
step summarize_before_27b $PY summarize.py --runs-dir runs
measure qwen3_5_27b Qwen/Qwen3.5-27B
step summarize $PY summarize.py --runs-dir runs
echo QUEUE_DONE
