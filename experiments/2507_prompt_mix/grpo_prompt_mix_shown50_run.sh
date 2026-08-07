#!/usr/bin/env bash
set -euo pipefail

source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/obfuscation-atlas

LOG=/workspace/scratch/grpo_prompt-mix-shown50-kl1e-3-det-3-s43.log
echo "===== START prompt-mix-shown50 $(date -Is) =====" | tee "$LOG"

PYTHONUNBUFFERED=1 python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    +model_preset=llama3_8b \
    loss.kl=0.001 \
    loss.detector=3 \
    +data/probe_dataset=mbpp_probe \
    training.seed=43 \
    data.seed=53 \
    task_args.shown_prompt_ratio=0.5 \
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
    wandb.run_name_prefix=prompt-mix-shown50-kl1e-3-det-3-s43 \
    2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
echo "===== END prompt-mix-shown50 exit=${rc} $(date -Is) =====" | tee -a "$LOG"
exit "$rc"
