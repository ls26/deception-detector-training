#!/usr/bin/env bash
# Sequential det0 seed sweep: baseline (mixed_reward_pass_all_prob=0.3) and
# ceiling (mixed_reward_pass_all_prob=1.0), each rerun at training.seed=43
# and training.seed=44. Everything else is identical to the original
# training.seed=42 runs (experiments/run_det0_baseline.py,
# experiments/run_det0_ceiling.py): kl=1e-3, det=0, data.seed=53, nofu,
# pass_only_shown, nofu_mixed Phase-0 probe manifest (monitoring only).
#
# Prerequisites:
#   - vLLM on http://127.0.0.1:8000 (GPU 1), keep it running across all runs
#   - tmux recommended so the sweep survives disconnects
#   - source /workspace/atlas_setup.sh already done
#   - >= 50 GB free on /workspace before each run
#
# Usage:
#   bash /workspace/scratch/grpo_det0_seed_sweep.sh
#
# Completion checks use filesystem artifacts (final adapter + response JSONL
# count), NOT log grepping — avoids the false-negative bug in
# grpo_det1_nofu_sweep.sh where grep missed "Total number of steps" due to
# log flush timing.
#
# Resilience: unlike grpo_mixed_reward_sweep.sh (which exits the whole sweep
# on any single-run failure), this sweep logs failures and continues to the
# next run — a bad run or a failed W&B export should not block the rest of
# the sweep. Cleanup is always attempted, but is refused (not forced) if the
# run doesn't look complete, so partial/failed runs are left intact for
# manual inspection.
#
# Safe cleanup: only removes checkpoints/lora/ and checkpoints/step_*/ after
# verifying final/adapter exists. Never touches runs from other sweeps.

set -uo pipefail

MAX_STEPS=128
MIN_FREE_GB=50

PROBE_MANIFEST=/workspace/experiments/archive/mixed_round1_nofu/probe_train_dataset.json

# type:prob:seed. Defaults to all 4; pass specific "type:prob:seed" entries as
# args to rerun a subset (e.g. skipped runs), leaving the default untouched.
if [[ $# -gt 0 ]]; then
  RUNS=("$@")
else
  RUNS=(
    "baseline:0.3:43"
    "baseline:0.3:44"
    "ceiling:1.0:43"
    "ceiling:1.0:44"
  )
fi

LOG=/workspace/scratch/grpo_det0_seed_sweep.log
RUNS_BASE=/workspace/results/run_results/full_grpo_runs
OUTPUT_BASE=/workspace/experiments/trials/trial3/rounds

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

  assert_safe_run_dir "$run_dir" || return 1

  if [[ ! -d "$run_dir" ]]; then
    echo "SKIP cleanup ($label): directory missing — $run_dir" | tee -a "$LOG"
    return 0
  fi

  if ! final_adapter_exists "$run_dir"; then
    echo "REFUSE cleanup ($label): no final adapter under $run_dir/checkpoints/final/" | tee -a "$LOG" >&2
    return 1
  fi

  local n_resp
  n_resp=$(count_response_steps "$run_dir")
  if [[ "$n_resp" -lt "$min_resp" ]]; then
    echo "REFUSE cleanup ($label): only $n_resp response JSONLs (expected >= $min_resp)" | tee -a "$LOG" >&2
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
    echo "ERROR: insufficient disk space" | tee -a "$LOG" >&2
    return 1
  fi
}

check_vllm() {
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null || {
    echo "ERROR: vLLM not reachable on :8000 — start server before sweep" | tee -a "$LOG" >&2
    return 1
  }
  echo "vLLM OK on :8000" | tee -a "$LOG"
}

run_grpo() {
  local prob="$1"
  local prefix="$2"
  local seed="$3"

  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
      +model_preset=llama3_8b \
      loss.kl=0.001 \
      loss.detector=0 \
      +data/probe_dataset=mbpp_probe \
      training.seed="$seed" \
      data.seed=53 \
      data.probe_dataset.congruent_follow_up_prompts=[] \
      data.probe_dataset.incongruent_follow_up_prompts=[] \
      data.task_dataset.congruent_follow_up_prompts=[] \
      data.task_dataset.incongruent_follow_up_prompts=[] \
      grpo.model.attn_implementation=sdpa \
      training.save_path_base=/workspace/results/run_results \
      detector.model_cache_path=/workspace/results/activations \
      detector.directory_save_prefix=nofu_mixed \
      detector.retrain=false \
      ++detector.mixed_probe_manifest_path="$PROBE_MANIFEST" \
      corrected_detector_gradient=false \
      misalignment_eval.enabled=false \
      wandb.project=deception_detection \
      wandb.group=full_grpo_runs \
      wandb.run_name_prefix="${prefix}" \
      task_args.deceptive_definition=pass_only_shown \
      task_args.mixed_reward_pass_all_prob="$prob" \
      2>&1 | tee -a "$LOG"
}

verify_run_complete() {
  local run_dir="$1"
  local run_id="$2"
  local min_resp=$((MAX_STEPS - 20))

  assert_safe_run_dir "$run_dir" || return 1

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

export_metrics_by_id() {
  local run_id="$1"
  local output_path="$2"

  python -m obfuscation_atlas.scripts.export_wandb_run_metrics \
      --entity lilylixin6-the-harker-school \
      --project deception_detection \
      --run-id="$run_id" \
      --output="$output_path" \
      2>&1 | tee -a "$LOG"
}

process_run() {
  local run_type="$1"
  local prob="$2"
  local seed="$3"
  local prefix="trial3_det0_${run_type}_s${seed}"
  local output_dir="${OUTPUT_BASE}/det0_${run_type}"

  echo "" | tee -a "$LOG"
  echo "===== START ${prefix} (type=${run_type}, prob=${prob}, seed=${seed}) $(date -Is) =====" | tee -a "$LOG"

  if ! check_disk; then
    echo "SKIP ${prefix}: disk check failed" | tee -a "$LOG" >&2
    echo "===== END ${prefix} exit=skip-disk $(date -Is) =====" | tee -a "$LOG"
    return 1
  fi
  if ! check_vllm; then
    echo "SKIP ${prefix}: vLLM check failed" | tee -a "$LOG" >&2
    echo "===== END ${prefix} exit=skip-vllm $(date -Is) =====" | tee -a "$LOG"
    return 1
  fi

  mkdir -p "$output_dir"

  run_grpo "$prob" "$prefix" "$seed"
  local rc=${PIPESTATUS[0]}
  echo "===== END ${prefix} exit=${rc} $(date -Is) =====" | tee -a "$LOG"

  if [[ "$rc" -ne 0 ]]; then
    echo "ERROR: run ${prefix} failed (exit=${rc}); logging and continuing to next run." | tee -a "$LOG" >&2
    return 1
  fi

  local run_id run_dir
  if ! run_id=$(extract_run_id "$prefix"); then
    echo "ERROR: could not determine RUN_ID for ${prefix} — skipping export/cleanup" | tee -a "$LOG" >&2
    return 1
  fi
  run_dir=$(run_dir_for_id "$run_id")
  echo "Completed ${prefix} → RUN_ID=$run_id" | tee -a "$LOG"

  if ! verify_run_complete "$run_dir" "$run_id"; then
    echo "ERROR: run $run_id failed completion checks — skipping export/cleanup" | tee -a "$LOG" >&2
    return 1
  fi

  local run_metrics_path="${output_dir}/run_metrics_${prefix}.json"
  if ! export_metrics_by_id "$run_id" "$run_metrics_path"; then
    echo "ERROR: metrics export failed for $run_id ($prefix) — will still attempt cleanup" | tee -a "$LOG" >&2
  else
    echo "Exported metrics to $run_metrics_path" | tee -a "$LOG"
  fi

  if ! cleanup_intermediates "$run_dir" "$prefix"; then
    echo "WARNING: cleanup skipped/failed for $run_id ($prefix) — check manually" | tee -a "$LOG" >&2
  fi

  PROCESS_RESULT="${prefix}:${run_id}"
  return 0
}

COMPLETED=()
FAILED=()
PROCESS_RESULT=""

echo "===== SWEEP START $(date -Is) =====" | tee "$LOG"
echo "  runs: ${RUNS[*]}" | tee -a "$LOG"
if ! check_vllm; then
  echo "ABORT: vLLM not reachable at sweep start — fix and rerun." | tee -a "$LOG" >&2
  exit 1
fi
if ! check_disk; then
  echo "ABORT: insufficient disk at sweep start — fix and rerun." | tee -a "$LOG" >&2
  exit 1
fi

for entry in "${RUNS[@]}"; do
  IFS=':' read -r RUN_TYPE PROB SEED <<< "$entry"
  PROCESS_RESULT=""
  if process_run "$RUN_TYPE" "$PROB" "$SEED"; then
    COMPLETED+=("$PROCESS_RESULT")
  else
    FAILED+=("trial3_det0_${RUN_TYPE}_s${SEED}")
  fi
done

echo "" | tee -a "$LOG"
echo "===== SWEEP DONE $(date -Is) =====" | tee -a "$LOG"
echo "Completed runs:" | tee -a "$LOG"
for entry in "${COMPLETED[@]}"; do
  echo "  ${entry%%:*} → ${entry##*:}" | tee -a "$LOG"
done
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed/skipped runs (check log for details):" | tee -a "$LOG"
  for entry in "${FAILED[@]}"; do
    echo "  $entry" | tee -a "$LOG"
  done
fi
df -h /workspace | tail -1 | tee -a "$LOG"
