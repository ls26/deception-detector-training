# %%
import gc
import subprocess

import torch
from transformers import AutoTokenizer

from generation import (
    bootstrap_hf_env,
    print_pipeline_df_stats,
    run_dry_generation,
    run_on_policy_pipeline,
)
from build_dataset import build_train_dataset_split_sources
from probe_training import (
    eval_probe_on_sources,
    load_final_policy_model,
    run_probe_train_eval_pipeline,
)
from extract_run_results import print_all_results
from round_config import RoundConfig

cfg = RoundConfig(trial=3, round=4, previous_run_id="wu6k5j61")
# Set cfg.grpo_run_id manually to re-export; auto-resolved from W&B after GRPO otherwise.

# If True, GRPO uses the calibrated probe penalty det * clip((z - z_h) / (z_d - z_h), 0, 1)
# (z_h/z_d = median honest/deceptive on-policy logits from cfg.probe_metrics, computed by
# run_probe_train_eval_pipeline) instead of the default det * sigmoid(z).
USE_CALIBRATED_PENALTY = False

BASE_HONEST_JSONL = cfg.workspace / "experiments/artifacts/base/base_train_generations.jsonl"
DET0_DECEPTIVE_JSONL = cfg.workspace / "experiments/artifacts/det0/det0_train_generations.jsonl"

# %%
bootstrap_hf_env()
tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint_final)

# %%
run_dry_generation(
    model_key=cfg.producer_key,
    dataset="train",
    tokenizer=tokenizer,
    lora_path=cfg.adapter_path,
    n_generations=1,
    deceptive_definition=cfg.deceptive_definition,
)

# %%
train_df = run_on_policy_pipeline(
    model_key=cfg.producer_key,
    dataset="train",
    lora_path=cfg.adapter_path,
    tokenizer=tokenizer,
    require_adapter=True,
    n_generations=5,
    output_dir=cfg.buffer,
    run_id_config=cfg.previous_run_id,
    deceptive_definition=cfg.deceptive_definition,
)
print_pipeline_df_stats(train_df)

# %%
eval_df = run_on_policy_pipeline(
    model_key=cfg.producer_key,
    dataset="eval",
    lora_path=cfg.adapter_path,
    tokenizer=tokenizer,
    require_adapter=True,
    n_generations=1,
    output_dir=cfg.buffer,
    run_id_config=cfg.previous_run_id,
    deceptive_definition=cfg.deceptive_definition,
)
print_pipeline_df_stats(eval_df)

# %%
model, _ = load_final_policy_model(
    checkpoint_final=cfg.checkpoint_final,
    adapter_path=cfg.adapter_path,
)
train_pos, train_neg, manifest = build_train_dataset_split_sources(
    tokenizer,
    trial=cfg.trial,
    round=cfg.round,
    manifest_path=cfg.probe_train_dataset,
    buffer_dir=cfg.buffer,
    base_honest_jsonl=BASE_HONEST_JSONL,
    det0_deceptive_jsonl=DET0_DECEPTIVE_JSONL,
    on_policy_per_source=100,
    n_fixed_per_class=200,
)

# %%
run_probe_train_eval_pipeline(
    model,
    tokenizer,
    train_pos=train_pos,
    train_neg=train_neg,
    manifest=manifest,
    weights_dir=cfg.probe_weights,
    activations_dir=cfg.probe_activations,
    metrics_path=cfg.probe_metrics,
    manifest_path=cfg.probe_train_dataset,
    val_jsonl=cfg.eval_jsonl,
    run_id=cfg.previous_run_id,
    deceptive_definition=cfg.deceptive_definition,
)

# %%
per_source_results = eval_probe_on_sources(
    model,
    tokenizer,
    source_jsonls={
        "trial3_round1": cfg.workspace / "experiments/trials/trial3/buffer/trial3_round1_eval_generations.jsonl",
        "trial3_round2": cfg.workspace / "experiments/trials/trial3/buffer/trial3_round2_eval_generations.jsonl",
        "trial3_round3": cfg.workspace / "experiments/trials/trial3/buffer/trial3_round3_eval_generations.jsonl",
        "det0": cfg.workspace / "experiments/artifacts/det0/det0_eval_generations.jsonl",
        "base": cfg.workspace / "experiments/artifacts/base/base_eval_generations.jsonl",
    },
    weights_dir=cfg.probe_weights,
    activations_dir=cfg.probe_activations,
    deceptive_definition=cfg.deceptive_definition,
)

1# %%
del model
gc.collect()
torch.cuda.empty_cache()

# %%
grpo_args = [
    "python",
    "-m",
    "obfuscation_atlas.scripts.train_grpo_model_against_probe",
    "+model_preset=llama3_8b",
    "loss.kl=0.001",
    "loss.detector=3",
    "+data/probe_dataset=mbpp_probe",
    "training.seed=42",
    "data.seed=53",
    "data.probe_dataset.congruent_follow_up_prompts=[]",
    "data.probe_dataset.incongruent_follow_up_prompts=[]",
    "data.task_dataset.congruent_follow_up_prompts=[]",
    "data.task_dataset.incongruent_follow_up_prompts=[]",
    "grpo.model.attn_implementation=sdpa",
    f"training.save_path_base={cfg.workspace / 'results/run_results'}",
    "detector.model_cache_path=/workspace/results/activations",
    f"detector.directory_save_prefix={cfg.run_key}",
    f"++detector.mixed_probe_manifest_path={cfg.probe_train_dataset}",
    f"++grpo.model.reference_adapter_path={cfg.checkpoint_final}",
    "++training.resume_from_checkpoint=False",
    "corrected_detector_gradient=false",
    "misalignment_eval.enabled=false",
    "wandb.project=deception_detection",
    "wandb.group=full_grpo_runs",
    f"wandb.run_name_prefix={cfg.run_key}",
    f"task_args.deceptive_definition={cfg.deceptive_definition}",
    "task_args.mixed_reward_pass_all_prob=0.3",
]
if USE_CALIBRATED_PENALTY:
    grpo_args.append(f"++detector.calibration_logits_path={cfg.probe_metrics}")

subprocess.run(
    grpo_args,
    cwd=cfg.workspace / "obfuscation-atlas",
    check=True,
)

# Terminal equivalent (from obfuscation-atlas/, after source atlas_setup.sh && activate venv):
# python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
#     +model_preset=llama3_8b \
#     loss.kl=0.001 \
#     loss.detector=3 \
#     +data/probe_dataset=mbpp_probe \
#     training.seed=43 \
#     data.seed=53 \
#     grpo.model.attn_implementation=sdpa \
#     training.save_path_base=/workspace/results/run_results \
#     detector.model_cache_path=/workspace/results/activations \
#     detector.directory_save_prefix=trial1_round6 \
#     ++detector.mixed_probe_manifest_path=/workspace/experiments/trials/trial1/rounds/round6/probe_train_dataset.json \
#     ++grpo.model.reference_adapter_path=/workspace/results/run_results/full_grpo_runs/trigger_none_model_xgnqzx4k_layers_32/checkpoints/final \
#     ++training.resume_from_checkpoint=False \
#     corrected_detector_gradient=false \
#     misalignment_eval.enabled=false \
#     wandb.project=deception_detection \
#     wandb.group=full_grpo_runs \
#     wandb.run_name_prefix=trial1_round6

# %%
if cfg.grpo_run_id is None:
    import wandb

    api = wandb.Api()
    project = "lilylixin6-the-harker-school/deception_detection"
    matching = [
        run
        for run in api.runs(project, order="-created_at", per_page=50)
        if run.name.startswith(cfg.run_key)
    ]
    if not matching:
        raise RuntimeError(
            f"No W&B run with name prefix {cfg.run_key!r}; set cfg.grpo_run_id manually."
        )
    cfg.grpo_run_id = matching[0].id

subprocess.run(
    [
        "python",
        "-m",
        "obfuscation_atlas.scripts.export_wandb_run_metrics",
        "--entity",
        "lilylixin6-the-harker-school",
        "--project",
        "deception_detection",
        f"--run-id={cfg.grpo_run_id}",
        f"--output={cfg.run_metrics}",
    ],
    cwd=cfg.workspace / "obfuscation-atlas",
    check=True,
)

# Terminal equivalent (from obfuscation-atlas/, after source atlas_setup.sh && activate venv):
# python -m obfuscation_atlas.scripts.export_wandb_run_metrics \
#     --entity lilylixin6-the-harker-school \
#     --project deception_detection \
#     --run-id=<grpo_run_id> \
#     --output=/workspace/experiments/trials/trial1/rounds/round6/run_metrics_trial1_round6.json

# %%
print_all_results(cfg.run_metrics)

# %%
