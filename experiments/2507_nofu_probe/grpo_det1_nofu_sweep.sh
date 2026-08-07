#!/usr/bin/env bash
# Sequential nofu GRPO sweep: kl=1e-3, det=1, seeds 1–3, 200 train steps.
#
# Prerequisites:
#   - vLLM on http://127.0.0.1:8000 (GPU 1), keep it running across all seeds
#   - tmux recommended so the sweep survives disconnects
#
# Usage:
#   bash /workspace/scratch/grpo_det1_nofu_sweep.sh
#
# Safe cleanup: only removes checkpoints/lora/ and checkpoints/step_*/ (FSDP resume
# dirs) after verifying final/adapter exists. Never touches final/, responses/,
# activations/, or other run dirs.

set -euo pipefail

MAX_STEPS=200
NUM_EPOCHS=25   # 25 × ceil(473/64) = 200 with default grad_accum_steps=8
SEEDS=(1 2 3)
PRIOR_RUN_ID=i4o8jkck   # failed 32-step seed-1 run — clean intermediates at startup

LOG=/workspace/scratch/grpo_det1_nofu_sweep.log
RUNS_BASE=/workspace/results/run_results/full_grpo_runs
MIN_FREE_GB=50

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

echo "===== SWEEP START $(date -Is) =====" | tee "$LOG"
check_vllm
check_disk

PRIOR_RUN_DIR=$(run_dir_for_id "$PRIOR_RUN_ID")
if [[ -d "$PRIOR_RUN_DIR/checkpoints/lora" ]] || compgen -G "$PRIOR_RUN_DIR/checkpoints/step_*" >/dev/null; then
  echo "Cleaning prior run $PRIOR_RUN_ID intermediates..." | tee -a "$LOG"
  cleanup_intermediates "$PRIOR_RUN_DIR" "prior $PRIOR_RUN_ID" 25
else
  echo "Prior run $PRIOR_RUN_ID: no intermediates to clean (or already cleaned)" | tee -a "$LOG"
fi

COMPLETED=()

for SEED in "${SEEDS[@]}"; do
  PREFIX="kl1e-3-det1-s${SEED}-mbpp-nofu"
  echo "" | tee -a "$LOG"
  echo "===== START ${PREFIX} $(date -Is) =====" | tee -a "$LOG"
  check_disk

  PYTHONUNBUFFERED=1 python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
      +model_preset=llama3_8b \
      loss.kl=0.001 \
      loss.detector=1 \
      +data/probe_dataset=mbpp_probe \
      training.seed="$SEED" \
      data.seed=53 \
      training.num_epochs="$NUM_EPOCHS" \
      training.max_steps="$MAX_STEPS" \
      grpo.num_epochs="$NUM_EPOCHS" \
      grpo.optimizer.max_steps="$MAX_STEPS" \
      data.probe_dataset.congruent_follow_up_prompts=[] \
      data.probe_dataset.incongruent_follow_up_prompts=[] \
      data.task_dataset.congruent_follow_up_prompts=[] \
      data.task_dataset.incongruent_follow_up_prompts=[] \
      detector.directory_save_prefix=trial3_nofu_mbpp \
      detector.retrain=false \
      grpo.model.attn_implementation=sdpa \
      training.save_path_base=/workspace/results/run_results \
      detector.model_cache_path=/workspace/results/activations \
      corrected_detector_gradient=false \
      misalignment_eval.enabled=false \
      wandb.project=deception_detection \
      wandb.group=full_grpo_runs \
      wandb.run_name_prefix="${PREFIX}" \
      2>&1 | tee -a "$LOG"

  RC=${PIPESTATUS[0]}
  echo "===== END ${PREFIX} exit=${RC} $(date -Is) =====" | tee -a "$LOG"

  if [[ "$RC" -ne 0 ]]; then
    echo "Run failed (seed=$SEED); stopping sweep. Fix and re-run — wandb resume may work if step_* still exist." | tee -a "$LOG"
    exit "$RC"
  fi

  RUN_ID=$(extract_run_id "$PREFIX")
  RUN_DIR=$(run_dir_for_id "$RUN_ID")
  echo "Completed $PREFIX → RUN_ID=$RUN_ID" | tee -a "$LOG"

  if ! sed -n "/===== START ${PREFIX}/,/===== END ${PREFIX}/p" "$LOG" \
      | grep -q "Total number of steps: ${MAX_STEPS}"; then
    echo "ERROR: run $RUN_ID did not train for ${MAX_STEPS} steps — check log before continuing sweep" | tee -a "$LOG"
    exit 1
  fi

  cleanup_intermediates "$RUN_DIR" "$PREFIX"
  COMPLETED+=("${SEED}:${RUN_ID}")
done

echo "" | tee -a "$LOG"
echo "===== SWEEP DONE $(date -Is) =====" | tee -a "$LOG"
echo "Completed runs:" | tee -a "$LOG"
for entry in "${COMPLETED[@]}"; do
  echo "  seed ${entry%%:*} → ${entry##*:}" | tee -a "$LOG"
done
df -h /workspace | tail -1 | tee -a "$LOG"
