#!/usr/bin/env bash
# Sequential mixed-reward GRPO sweep: kl=1e-3, det=3, 128 steps,
# mixed_reward_pass_all_prob = 0.35 (s41, s42) → 0.40 (s41, s42).
#
# All runs use the nofu mixed-probe manifest (300/class Phase-0 probe),
# no prompt mixing (shown_prompt_ratio left at default 1.0).
#
# Prerequisites:
#   - vLLM on http://127.0.0.1:8000 (GPU 1), keep it running across all runs
#   - tmux recommended so the sweep survives disconnects
#   - source /workspace/atlas_setup.sh already done
#   - >= 50 GB free on /workspace before first run
#
# Usage:
#   bash /workspace/scratch/grpo_mixed_reward_sweep.sh
#
# Completion checks use filesystem artifacts (final adapter + response JSONL
# count), NOT log grepping — avoids the false-negative bug in
# grpo_det1_nofu_sweep.sh where grep missed "Total number of steps" due to
# log flush timing.
#
# Safe cleanup: only removes checkpoints/lora/ and checkpoints/step_*/ after
# verifying final/adapter exists. Never touches runs from other sweeps.

set -euo pipefail

MAX_STEPS=128
NUM_EPOCHS=16   # 16 × ceil(473/64) = 128 with default grad_accum_steps=8
MIN_FREE_GB=50

PROBS=(0.35 0.35 0.40 0.40)
PROB_INTS=(35 35 40 40)
SEEDS=(41 42 41 42)

LOG=/workspace/scratch/grpo_mixed_reward_sweep_p35_p40.log
RUNS_BASE=/workspace/results/run_results/full_grpo_runs

source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/obfuscation-atlas

run_dir_for_id() {
  local run_id="$1"
  echo "${RUNS_BASE}/trigger_none_model_${run_id}_layers_32"
}

assert_safe_run_dir() {
  local run_dir="$1"
  case "$run_dir" in
    "${RUNS_BASE}/trigger_none_model_"[a-z0-9]*"_layers_32") return 0 ;;
    *)
      echo "REFUSE: run_dir outside expected pattern: $run_dir" >&2
      return 1
      ;;
  esac
}

final_adapter_exists() {
  local run_dir="$1"
  [[ -f "$run_dir/checkpoints/final/adapter_model.bin" ]] \
    || [[ -f "$run_dir/checkpoints/final/lora/adapter_model.bin" ]]
}

count_response_steps() {
  local run_dir="$1"
  find "$run_dir/checkpoints/responses" -maxdepth 1 -name 'step_*.jsonl' 2>/dev/null | wc -l
}

cleanup_intermediates() {
  local run_dir="$1"
  local label="$2"
  local min_resp="${3:-$((MAX_STEPS - 20))}"

  assert_safe_run_dir "$run_dir"

  if [[ ! -d "$run_dir" ]]; then
    echo "SKIP cleanup ($label): directory missing — $run_dir"
    return 0
  fi

  if ! final_adapter_exists "$run_dir"; then
    echo "REFUSE cleanup ($label): no final adapter under $run_dir/checkpoints/final/" >&2
    return 1
  fi

  local n_resp
  n_resp=$(count_response_steps "$run_dir")
  if [[ "$n_resp" -lt "$min_resp" ]]; then
    echo "REFUSE cleanup ($label): only $n_resp response JSONLs (expected >= $min_resp)" >&2
    return 1
  fi

  echo "===== CLEANUP $label $(date -Is) =====" | tee -a "$LOG"
  echo "  run_dir=$run_dir" | tee -a "$LOG"
  echo "  responses=$n_resp final=OK" | tee -a "$LOG"

  if [[ -d "$run_dir/checkpoints/lora" ]]; then
    echo "  removing checkpoints/lora/" | tee -a "$LOG"
    rm -rf "$run_dir/checkpoints/lora"
  fi

  shopt -s nullglob
  for d in "$run_dir/checkpoints"/step_*; do
    [[ -d "$d" ]] || continue
    echo "  removing $(basename "$d")/" | tee -a "$LOG"
    rm -rf "$d"
  done
  shopt -u nullglob

  echo "  kept: checkpoints/final/, checkpoints/responses/, eval artifacts" | tee -a "$LOG"
  du -sh "$run_dir" 2>/dev/null | tee -a "$LOG" || true
  df -h /workspace | tail -1 | tee -a "$LOG"
}

extract_run_id() {
  local prefix="$1"
  local run_id
  run_id=$(
    sed -n "/===== START ${prefix}/,/===== END ${prefix}/p" "$LOG" \
      | grep -oE 'setting up run [a-z0-9]{8}' \
      | tail -1 \
      | awk '{print $NF}'
  )
  if [[ -z "${run_id:-}" ]]; then
    run_id=$(
      sed -n "/===== START ${prefix}/,/===== END ${prefix}/p" "$LOG" \
        | grep -oE "trigger_none_model_[a-z0-9]+_layers_32" \
        | tail -1 \
        | sed 's/trigger_none_model_\(.*\)_layers_32/\1/'
    )
  fi
  if [[ -z "${run_id:-}" ]]; then
    echo "FAILED to extract RUN_ID for $prefix from $LOG" >&2
    return 1
  fi
  echo "$run_id"
}

check_disk() {
  local free_gb
  free_gb=$(df -BG /workspace | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
  echo "Disk free on /workspace: ${free_gb}G (need >= ${MIN_FREE_GB}G)" | tee -a "$LOG"
  if [[ "$free_gb" -lt "$MIN_FREE_GB" ]]; then
    echo "ERROR: insufficient disk space" >&2
    return 1
  fi
}

check_vllm() {
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null || {
    echo "ERROR: vLLM not reachable on :8000 — start server before sweep" >&2
    return 1
  }
  echo "vLLM OK on :8000" | tee -a "$LOG"
}

run_grpo() {
  local prob="$1"
  local prefix="$2"
  local seed="$3"

  PYTHONUNBUFFERED=1 python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
      +model_preset=llama3_8b \
      loss.kl=0.001 \
      loss.detector=3 \
      +data/probe_dataset=mbpp_probe \
      training.seed="$seed" \
      data.seed=53 \
      training.num_epochs="$NUM_EPOCHS" \
      training.max_steps="$MAX_STEPS" \
      grpo.num_epochs="$NUM_EPOCHS" \
      grpo.optimizer.max_steps="$MAX_STEPS" \
      task_args.mixed_reward_pass_all_prob="$prob" \
      data.probe_dataset.congruent_follow_up_prompts=[] \
      data.probe_dataset.incongruent_follow_up_prompts=[] \
      data.task_dataset.congruent_follow_up_prompts=[] \
      data.task_dataset.incongruent_follow_up_prompts=[] \
      detector.directory_save_prefix=nofu_mixed \
      detector.retrain=false \
      ++detector.mixed_probe_manifest_path=/workspace/experiments/archive/mixed_round1_nofu/probe_train_dataset.json \
      grpo.model.attn_implementation=sdpa \
      training.save_path_base=/workspace/results/run_results \
      detector.model_cache_path=/workspace/results/activations \
      corrected_detector_gradient=false \
      misalignment_eval.enabled=false \
      wandb.project=deception_detection \
      wandb.group=full_grpo_runs \
      wandb.run_name_prefix="${prefix}" \
      2>&1 | tee -a "$LOG"
}

verify_run_complete() {
  local run_dir="$1"
  local run_id="$2"
  local min_resp=$((MAX_STEPS - 20))

  assert_safe_run_dir "$run_dir"

  if [[ ! -d "$run_dir" ]]; then
    echo "ERROR: run $run_id directory missing: $run_dir" | tee -a "$LOG" >&2
    return 1
  fi

  if ! final_adapter_exists "$run_dir"; then
    echo "ERROR: run $run_id missing final adapter under checkpoints/final/" | tee -a "$LOG" >&2
    return 1
  fi

  local n_resp
  n_resp=$(count_response_steps "$run_dir")
  if [[ "$n_resp" -lt "$min_resp" ]]; then
    echo "ERROR: run $run_id only has $n_resp response JSONLs (expected >= $min_resp)" | tee -a "$LOG" >&2
    return 1
  fi

  echo "Verified $run_id complete: final=OK responses=$n_resp (>= $min_resp)" | tee -a "$LOG"
  return 0
}

finish_run() {
  local prefix="$1"
  local prob="$2"
  local rc="$3"

  echo "===== END ${prefix} exit=${rc} $(date -Is) =====" | tee -a "$LOG"

  if [[ "$rc" -ne 0 ]]; then
    echo "Run failed (prob=$prob); stopping sweep. Re-run script to resume if step_* exist." | tee -a "$LOG"
    exit "$rc"
  fi

  local run_id run_dir
  run_id=$(extract_run_id "$prefix")
  run_dir=$(run_dir_for_id "$run_id")
  echo "Completed $prefix (prob=$prob) → RUN_ID=$run_id" | tee -a "$LOG"

  if ! verify_run_complete "$run_dir" "$run_id"; then
    echo "ERROR: run $run_id failed completion checks — stopping sweep" | tee -a "$LOG" >&2
    exit 1
  fi

  cleanup_intermediates "$run_dir" "$prefix"
  FINISH_ENTRY="${prob}:${run_id}"
}

COMPLETED=()
FINISH_ENTRY=""

echo "===== SWEEP START $(date -Is) =====" | tee "$LOG"
echo "  mixed_reward_pass_all_prob sweep: ${PROBS[*]}" | tee -a "$LOG"
echo "  seeds: ${SEEDS[*]}" | tee -a "$LOG"
check_vllm
check_disk

for i in "${!PROBS[@]}"; do
  PROB="${PROBS[$i]}"
  PROB_INT="${PROB_INTS[$i]}"
  SEED="${SEEDS[$i]}"
  PREFIX="mixed-reward-p${PROB_INT}-kl1e-3-det-3-s${SEED}"

  echo "" | tee -a "$LOG"
  echo "===== START ${PREFIX} (prob=${PROB}, seed=${SEED}) $(date -Is) =====" | tee -a "$LOG"
  check_disk
  check_vllm
  run_grpo "$PROB" "$PREFIX" "$SEED"
  RC=${PIPESTATUS[0]}
  finish_run "$PREFIX" "$PROB" "$RC"
  COMPLETED+=("$FINISH_ENTRY")
done

echo "" | tee -a "$LOG"
echo "===== SWEEP DONE $(date -Is) =====" | tee -a "$LOG"
echo "Completed runs:" | tee -a "$LOG"
for entry in "${COMPLETED[@]}"; do
  echo "  prob ${entry%%:*} → ${entry##*:}" | tee -a "$LOG"
done
df -h /workspace | tail -1 | tee -a "$LOG"
