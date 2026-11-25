# Sourced by the queue scripts: the folder's environment, a step runner, and helpers. Each step's output goes to
# logs/<queue>_<step>_<host>.log; the queue log gets one "=== step" and one "EXIT <code> step" line per step.
F=/workspace/qwen_experiment/261123_9b_speed
SETUP=/workspace/sycophancy-rl/sycophancy_setup
cd $F
source $F/folder_env.sh
step() { local name=$1; shift; echo "=== $name $(date -u +%H:%M:%S)"; "$@" > "$F/logs/${QUEUE}_${name}_${HOST}.log" 2>&1; local code=$?; echo "EXIT $code $name $(date -u +%H:%M:%S)"; return $code; }
wait_gpu_free() { for i in $(seq 120); do [[ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader)" ]] && return 0; sleep 5; done; echo "GPU $1 still busy after 10 min"; }
wait_for_file() { echo "... waiting for $1 $(date -u +%H:%M:%S)"; until [[ -e $1 ]]; do sleep 60; done; }
wait_for_filled() { echo "... waiting for $1 to have no FILL_IN $(date -u +%H:%M:%S)"; while grep -q FILL_IN "$1"; do sleep 60; done; }
# A run: its vLLM server (port 8000, the port the launcher's start check and the trainer's default server_url use; one
# server per pod) in tmux session vllm-<run> on the GPU; once the server answers, run_rl.sh (its log ends with EXIT <code>)
run_with_server() {  # <subfolder> <run> <gpu> <attempt>
  tmux new-session -d -s "vllm-$2" "bash $F/start_vllm_gpu.sh $F/$1 $2 $4 $3 8000"
  echo "... waiting for the vLLM server of $2 $(date -u +%H:%M:%S)"
  for i in $(seq 360); do
    curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break
    tmux has-session -t "vllm-$2" 2>/dev/null || { echo "vLLM session of $2 ended (see $1/logs/vllm_$2_a$4_$HOST.log)"; return 1; }
    sleep 10
  done
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { echo "vLLM of $2 not up after 60 min"; tmux kill-session -t "vllm-$2"; return 1; }
  echo "... vLLM up $(date -u +%H:%M:%S)"
  step "$2" env CUDA_VISIBLE_DEVICES=$3 bash $SETUP/run_rl.sh $F/$1 $2 $4
  grep -h "^EXIT" "$F/$1/logs/rl_$2_a$4_$HOST.log"
  tmux kill-session -t "vllm-$2"
  wait_gpu_free $3
}
# 261124 speed tests: a run stopped after step 3. Starts the GPU-memory log and vLLM (tmux vllm-<run>), waits for the
# server and for $F/CODE_READY, then runs run_rl_noretry.sh (no launcher retries). A watcher stops the run 90 s after
# "Step 3 timings" appears in its log (so W&B has saved step 3): the launcher first, so it doesn't delete the run
# folder, then the trainer. Then vLLM is stopped and the GPU waited for.
run_to_step3() {  # <subfolder> <run> <gpu>
  local log=$F/$1/logs/rl_$2_a1_$HOST.log
  nvidia-smi -i $3 --query-gpu=timestamp,memory.used --format=csv,noheader -l 5 > "$F/logs/gpu_mem_$2_$HOST.log" &
  local mem_pid=$!
  tmux new-session -d -s "vllm-$2" "bash $F/start_vllm_gpu.sh $F/$1 $2 1 $3 8000"
  echo "... waiting for the vLLM server of $2 $(date -u +%H:%M:%S)"
  for i in $(seq 360); do
    curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break
    tmux has-session -t "vllm-$2" 2>/dev/null || { echo "vLLM session of $2 ended (see $1/logs/vllm_$2_a1_$HOST.log)"; kill $mem_pid; return 1; }
    sleep 10
  done
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { echo "vLLM of $2 not up after 60 min"; tmux kill-session -t "vllm-$2"; kill $mem_pid; return 1; }
  echo "... vLLM up $(date -u +%H:%M:%S)"
  wait_for_file "$F/CODE_READY"
  ( until grep -q "Step 3 timings" "$log" 2>/dev/null; do sleep 10; done
    echo "step 3 logged $(date -u +%H:%M:%S); stopping $2 in 90 s"; sleep 90
    pkill -f "MAX_GRPO_RETRIES = 0.*/$2.yaml"; sleep 5
    pkill -f "train_grpo_model_against_probe.*run_name_prefix=$2 "; echo "stopped $2 $(date -u +%H:%M:%S)" ) &
  local watcher=$!
  step "$2" env CUDA_VISIBLE_DEVICES=$3 bash $F/run_rl_noretry.sh $F/$1 $2 1
  grep -h "^EXIT" "$log"
  kill $watcher 2>/dev/null
  tmux kill-session -t "vllm-$2"
  kill $mem_pid
  wait_gpu_free $3
}
