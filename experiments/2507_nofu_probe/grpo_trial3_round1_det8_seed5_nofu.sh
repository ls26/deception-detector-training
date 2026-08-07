#!/usr/bin/env bash
# No-follow-up probe GRPO run (last-token-generation on completion EOT).
# See docs/FULL_GRPO_RUN.md § No-follow-up probe extraction.
set -euo pipefail

source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/obfuscation-atlas

curl -sf http://127.0.0.1:8000/v1/models >/dev/null && echo vLLM_OK || {
  echo "vLLM not reachable on :8000 — start server before training"
  exit 1
}

LOG=/workspace/scratch/grpo_trial3_round1_det8_seed5_nofu.log
PREFIX=trial3_round1_det8_seed5_nofu

echo "===== START ${PREFIX} $(date -Is) =====" | tee "$LOG"

PYTHONUNBUFFERED=1 python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    +model_preset=llama3_8b \
    loss.kl=0.001 \
    loss.detector=8 \
    +data/probe_dataset=mbpp_probe \
    training.seed=5 \
    data.seed=53 \
    data.probe_dataset.congruent_follow_up_prompts=[] \
    data.probe_dataset.incongruent_follow_up_prompts=[] \
    data.task_dataset.congruent_follow_up_prompts=[] \
    data.task_dataset.incongruent_follow_up_prompts=[] \
    detector.directory_save_prefix=trial3_nofu_mbpp \
    detector.retrain=true \
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
exit "$RC"
