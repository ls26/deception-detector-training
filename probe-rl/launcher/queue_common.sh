# Helpers for a queue script (one experiment's ordered steps on one pod). Set F (the experiment folder, absolute) and
# QUEUE (the queue's name, used in log names), source this file (it sources ../env.sh), then source any environment of
# the folder's own, so that its settings (such as HF_HUB_OFFLINE=1) win over env.sh's. Each step's output goes to
# $F/logs/<queue>_<step>_<host>.log; the queue's own log gets one "=== <step>" and one "EXIT <code> <step>" line per
# step. Every helper returns non-zero when it fails; run_with_server also when the GPU is still busy 10 minutes after
# the run, since the next step would share it.
[[ -n ${F:-} && -n ${QUEUE:-} ]] || { echo "queue_common.sh: set F and QUEUE before sourcing it" >&2; return 1; }
LAUNCHER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$LAUNCHER/../env.sh" || return 1
cd "$F" || return 1
mkdir -p "$F/logs"

step() {  # <name> <command...>: runs the command with its output in the step's log; returns its exit code
  local name=$1; shift
  echo "=== $name $(date -u +%H:%M:%S)"
  "$@" > "$F/logs/${QUEUE}_${name}_${HOST}.log" 2>&1
  local code=$?
  echo "EXIT $code $name $(date -u +%H:%M:%S)"
  return $code
}
wait_gpu_free() {  # <gpu>: waits up to 10 min for no process on the GPU
  for i in $(seq 120); do
    [[ -z "$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader)" ]] && return 0
    sleep 5
  done
  echo "GPU $1 still busy after 10 min"
  return 1
}
wait_for_file() { echo "... waiting for $1 $(date -u +%H:%M:%S)"; until [[ -e $1 ]]; do sleep 60; done; }
wait_for_filled() { echo "... waiting for $1 to have no FILL_IN $(date -u +%H:%M:%S)"; while grep -q FILL_IN "$1"; do sleep 60; done; }
# A run: its vLLM server (port 8000, the port the launcher's start check and the trainer's default server_url use; one
# server per pod) in tmux session vllm-<run> on the GPU; once the server answers, run_rl.sh on the same GPU. Returns the
# run's exit code (its log ends with EXIT <code>), or 1 if the server did not come up or the GPU did not free up after.
run_with_server() {  # <subfolder of $F> <run> <gpu> <attempt>
  tmux new-session -d -s "vllm-$2" "bash $LAUNCHER/start_vllm.sh $F/$1 $2 $4 $3" || { echo "could not start tmux session vllm-$2"; return 1; }
  echo "... waiting for the vLLM server of $2 $(date -u +%H:%M:%S)"
  for i in $(seq 360); do
    curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break
    tmux has-session -t "vllm-$2" 2>/dev/null || { echo "vLLM session of $2 ended (see $1/logs/vllm_$2_a$4_$HOST.log)"; return 1; }
    sleep 10
  done
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { echo "vLLM of $2 not up after 60 min"; tmux kill-session -t "vllm-$2"; return 1; }
  echo "... vLLM up $(date -u +%H:%M:%S)"
  step "$2" env CUDA_VISIBLE_DEVICES="$3" bash "$LAUNCHER/run_rl.sh" "$F/$1" "$2" "$4"
  local code=$?
  grep -h "^EXIT" "$F/$1/logs/rl_$2_a$4_$HOST.log"
  tmux kill-session -t "vllm-$2"
  wait_gpu_free "$3" || return 1
  return $code
}
