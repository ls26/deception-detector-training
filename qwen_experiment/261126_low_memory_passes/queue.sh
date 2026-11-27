#!/bin/bash
# The 261126 GPU tests on fdc2d68d6537 (one H100), run in tmux `eff-queue`; its log is logs/queue_<host>.log, one
# "=== step" and one "EXIT <code> step" line per step, each step's output in logs/<step>_<host>.log.
#   0. Wait until 261125's eval queue has ended (no `eval_queue.py runs` process) and the GPU has no process.
#   1. memory_test.py next to a vLLM server at 0.40 of the GPU (the runs' server, started with lmp_on4's model).
#   2. lmp_on4, lmp_off4: 3-step runs, settings on / off, 4 answers per pass.
#   3. lmp_on8: 3 steps, settings on, 8 per pass, only if step 1 left at least 4 GiB free at 8 x (1024 + 1024).
# Every run: its own vLLM server (tmux vllm-<run>), GPU memory by nvidia-smi every 5 s (logs/gpu_mem_<run>_<host>.log).
# `bash queue.sh merged [run]` (phase 3, after this branch moved onto main's reorganized code): one 3-step run with
# both settings on, once the GPU has no process: lmp_on4_main (the default; tmux eff-merged, its log
# logs/queue_merged_<host>.log), then lmp_mbpp_on4 (MBPP; tmux eff-mbpp, logs/queue_mbpp_<host>.log).
# The vLLM server comes from the live copy's probe-rl/launcher/start_vllm.sh (port 8000, GPU 0); its server code is the
# same as this checkout's. Phase 2 started it with start_vllm_gpu.sh, whose paths predate the reorganization.
source /workspace/eff-wip/qwen_experiment/261126_low_memory_passes/folder_env.sh
cd $F
step() { local name=$1; shift; echo "=== $name $(date -u +%H:%M:%S)"; "$@" > "$F/logs/${name}_${HOST}.log" 2>&1; local code=$?; echo "EXIT $code $name $(date -u +%H:%M:%S)"; return $code; }
gpu_busy() { [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; }
wait_gpu_free() { for i in $(seq 120); do gpu_busy || return 0; sleep 5; done; echo "GPU still busy after 10 min"; return 1; }
start_vllm() {  # <run> <label>: a server at 0.40 of GPU 0, port 8000, in tmux vllm-<label>
  tmux new-session -d -s "vllm-$2" "bash /workspace/probe-rl/launcher/start_vllm.sh $F $1 1 0"
  echo "... waiting for the vLLM server vllm-$2 $(date -u +%H:%M:%S)"
  for i in $(seq 360); do
    curl -sf http://127.0.0.1:8000/v1/models > /dev/null && { echo "... vLLM up $(date -u +%H:%M:%S)"; return 0; }
    tmux has-session -t "vllm-$2" 2>/dev/null || { echo "vLLM session vllm-$2 ended"; return 1; }
    sleep 10
  done
  echo "vLLM not up after 60 min"; tmux kill-session -t "vllm-$2"; return 1
}
stop_vllm() { tmux kill-session -t "vllm-$1"; wait_gpu_free; }
run_with_server() {  # <run>
  start_vllm "$1" "$1" || return 1
  nvidia-smi -i 0 --query-gpu=timestamp,memory.used --format=csv,noheader -l 5 > "$F/logs/gpu_mem_$1_$HOST.log" &
  local mem_pid=$!
  step "$1" bash $F/run_rl_worktree.sh "$1" 1
  grep -h "^EXIT" "$F/logs/rl_$1_a1_$HOST.log"
  kill $mem_pid
  stop_vllm "$1"
}

if [[ ${1:-} == merged ]]; then
  wait_gpu_free || exit 1
  run_with_server "${2:-lmp_on4_main}"
  echo "QUEUE_DONE $(date -u +%H:%M:%S)"
  exit 0
fi

echo "... waiting for 261125's eval queue to end and the GPU to be free $(date -u +%H:%M:%S)"
until ! pgrep -f "eval_queue.py runs" > /dev/null && ! gpu_busy; do sleep 60; done
echo "... 261125's eval queue has ended $(date -u +%H:%M:%S)"

# vLLM's log for the memory test: logs/vllm_lmp_on4_a1 is the runs' name, so the memory test's server is started from
# a copy of lmp_on4.yaml named memtest_server.yaml (same model)
cp $F/lmp_on4.yaml $F/memtest_server.yaml
start_vllm memtest_server memtest && step memory_test $PY $F/memory_test.py $F/memory
stop_vllm memtest

run_with_server lmp_on4
run_with_server lmp_off4
if $PY -c "import json, sys; r = json.load(open('$F/memory/memory.json'))['low_memory_passes on, 8 x (1024 + 1024)']; sys.exit(0 if isinstance(r, dict) and r['free_after_pass_GiB'] >= 4 else 1)"; then
  run_with_server lmp_on8
else
  echo "lmp_on8 not run: the memory test left under 4 GiB free at 8 x (1024 + 1024), or ran out of memory"
fi
echo "QUEUE_DONE $(date -u +%H:%M:%S)"
