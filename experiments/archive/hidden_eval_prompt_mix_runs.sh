#!/usr/bin/env bash
set -euo pipefail

source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/obfuscation-atlas

LOG=/workspace/scratch/hidden_eval_prompt_mix.log
echo "===== HIDDEN EVAL SWEEP START $(date -Is) =====" | tee "$LOG"

run_hidden_eval() {
  local source_run_id="$1"
  local prefix="$2"
  echo "" | tee -a "$LOG"
  echo "===== START ${prefix} (source=${source_run_id}) $(date -Is) =====" | tee -a "$LOG"

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
      +model_preset=llama3_8b \
      loss.kl=0.001 \
      loss.detector=3 \
      +data/probe_dataset=mbpp_probe \
      training.seed=43 \
      data.seed=53 \
      task_args.eval_show_expected_output=false \
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
      ++training.only_eval_from_other_run=True \
      ++training.resume_from_checkpoint=False \
      ++training.model_wandb_id="${source_run_id}" \
      wandb.project=deception_detection \
      wandb.group=full_grpo_runs \
      wandb.run_name_prefix="${prefix}" \
      2>&1 | tee -a "$LOG"

  local rc=${PIPESTATUS[0]}
  echo "===== END ${prefix} exit=${rc} $(date -Is) =====" | tee -a "$LOG"
  if [[ "$rc" -ne 0 ]]; then
    echo "Hidden eval failed for ${source_run_id}; stopping." | tee -a "$LOG"
    exit "$rc"
  fi
}

# 50% prompt-mix run (a8cmusu2)
run_hidden_eval a8cmusu2 hidden-eval-prompt-mix-shown50-kl1e-3-det-3-s43

# 60% prompt-mix run (5grn9xef)
run_hidden_eval 5grn9xef hidden-eval-prompt-mix-kl1e-3-det-3-s43

echo "" | tee -a "$LOG"
echo "===== HIDDEN EVAL SWEEP DONE $(date -Is) =====" | tee -a "$LOG"
