#!/usr/bin/env python3
"""Round 2 with calibrated probe penalty: GRPO using clip((z - z_h) / (z_d - z_h), 0, 1)
instead of sigmoid(z) for the per-layer probe penalty.

Everything else matches trial 3's round 2 (`trial3_round2_v2`): same round-2 probe
manifest, same round-1 checkpoint as the policy starting point, same hyperparams.
The only difference is ++detector.calibration_logits_path, which loads per-layer
median logits from the starting policy's on-policy eval to replace sigmoid with a
linearly calibrated clip penalty.

Results saved to trial3/rounds/round2_calibrated_penalty/.
"""

import gc
import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]  # archive/ is one level deeper than experiments/
TRIAL_DIR = WORKSPACE / "experiments/trials/trial3"
ROUND2_DIR = TRIAL_DIR / "rounds/round2"
OUTPUT_DIR = TRIAL_DIR / "rounds/round2_calibrated_penalty"

CHECKPOINT_FINAL = (
    WORKSPACE
    / "results/run_results/full_grpo_runs"
    / "trigger_none_model_62es59py_layers_32/checkpoints/final"
)

RUN_KEY = "trial3_round2_calibrated_penalty"


def ensure_calibration_logits(probe_metrics_path: Path) -> dict:
    """Compute calibration_logits (scalar z_h/z_d from layer-mean-aggregated logits) and
    update probe_metrics.json if missing or stale (e.g. from the old per-layer format)."""
    probe_metrics = json.loads(probe_metrics_path.read_text())
    calib = probe_metrics.get("calibration_logits")
    if calib is not None:
        if isinstance(calib["z_h"], dict):
            print("Found stale per-layer calibration_logits; recomputing as aggregate scalars.")
            del probe_metrics["calibration_logits"]
        else:
            print(f"OK: calibration_logits already present (z_h={calib['z_h']:.4f}, z_d={calib['z_d']:.4f})")
            return probe_metrics

    print("Computing calibration_logits (reusing cached activations)...")

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer

    sys.path.insert(0, str(WORKSPACE / "experiments"))
    from build_dataset import load_jsonl
    from probe_training import (
        EVAL_SEQUENCE_AGGREGATOR,
        FOLLOW_UP,
        MAX_SEQUENCE_LENGTH,
        MODEL_BATCH_SIZE,
        compute_calibration_logits,
        load_final_policy_model,
    )
    from obfuscation_atlas.detectors.train import load_model
    from obfuscation_atlas.tasks.code_generation import DEFAULT_DECEPTIVE_DEFINITION

    adapter_path = CHECKPOINT_FINAL / "lora"
    if not adapter_path.is_dir():
        adapter_path = CHECKPOINT_FINAL

    model, tokenizer = load_final_policy_model(
        checkpoint_final=CHECKPOINT_FINAL,
        adapter_path=adapter_path,
    )
    probe_dict = load_model(str(ROUND2_DIR / "probe_weights"))
    filtered_layers = probe_metrics["filtered_layers"]

    val_jsonl = TRIAL_DIR / "buffer/trial3_round2_eval_generations.jsonl"
    val_rows = load_jsonl(val_jsonl)
    val_gens = Dataset.from_dict(
        {"prompt": [r["prompt"] for r in val_rows], "completion": [r["completion"] for r in val_rows]}
    )
    # val_honest uses on-dataset honest code (same as eval_probe_on_policy does)
    from probe_training import load_eval_task_dataset
    eval_task_dataset = load_eval_task_dataset(tokenizer)
    _, val_neg = eval_task_dataset
    val_honest = Dataset.from_dict({"prompt": list(val_neg["prompt"]), "completion": list(val_neg["code"])})

    calibration_logits = compute_calibration_logits(
        model,
        tokenizer,
        probe_dict=probe_dict,
        filtered_layers=filtered_layers,
        val_gens=val_gens,
        val_honest=val_honest,
        val_rows=val_rows,
        activations_dir=ROUND2_DIR / "probe_activations",
        follow_up=FOLLOW_UP,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        model_batch_size=MODEL_BATCH_SIZE,
        eval_sequence_aggregator=EVAL_SEQUENCE_AGGREGATOR,
        obfuscate_over="last-token-generation",
        deceptive_definition=DEFAULT_DECEPTIVE_DEFINITION,
    )

    # Update probe_metrics.json in place
    probe_metrics["calibration_logits"] = calibration_logits
    probe_metrics_path.write_text(json.dumps(probe_metrics, indent=2))
    print(f"OK: saved calibration_logits to {probe_metrics_path}")

    # Print summary for verification
    z_h_val = calibration_logits["z_h"]
    z_d_val = calibration_logits["z_d"]
    print(f"  z_h (honest median aggregated logit):    {z_h_val:.4f}")
    print(f"  z_d (deceptive median aggregated logit): {z_d_val:.4f}")
    print(f"  z_d - z_h (separation):                  {z_d_val - z_h_val:.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return probe_metrics


def main() -> int:
    probe_manifest = ROUND2_DIR / "probe_train_dataset.json"
    probe_metrics_path = ROUND2_DIR / "probe_metrics.json"

    for p in [probe_manifest, probe_metrics_path, CHECKPOINT_FINAL]:
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Ensure calibration logits exist in probe_metrics.json
    probe_metrics = ensure_calibration_logits(probe_metrics_path)

    print(f"\nRound 2 probe on_dataset_auc: {probe_metrics['on_dataset_auc']:.4f}")
    print(f"Manifest: {probe_manifest}")
    print(f"Checkpoint (policy init): {CHECKPOINT_FINAL}")
    print(f"Penalty: calibrated clip (calibration_logits_path={probe_metrics_path})")
    print(f"Output dir: {OUTPUT_DIR}")

    # Step 2: GRPO with calibrated penalty
    subprocess.run(
        [
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
            f"training.save_path_base={WORKSPACE / 'results/run_results'}",
            "detector.model_cache_path=/workspace/results/activations",
            f"detector.directory_save_prefix={RUN_KEY}",
            f"++detector.mixed_probe_manifest_path={probe_manifest}",
            f"++grpo.model.reference_adapter_path={CHECKPOINT_FINAL}",
            f"++detector.calibration_logits_path={probe_metrics_path}",
            "++training.resume_from_checkpoint=False",
            "corrected_detector_gradient=false",
            "misalignment_eval.enabled=false",
            "wandb.project=deception_detection",
            "wandb.group=full_grpo_runs",
            f"wandb.run_name_prefix={RUN_KEY}",
            "task_args.deceptive_definition=pass_only_shown",
            "task_args.mixed_reward_pass_all_prob=0.3",
        ],
        cwd=WORKSPACE / "obfuscation-atlas",
        check=True,
    )

    # Step 3: Export W&B metrics
    import wandb

    api = wandb.Api()
    project = "lilylixin6-the-harker-school/deception_detection"
    matching = [
        run
        for run in api.runs(project, order="-created_at", per_page=50)
        if run.name.startswith(RUN_KEY)
    ]
    if not matching:
        print(f"WARNING: no W&B run with prefix {RUN_KEY!r}; export manually.",
              file=sys.stderr)
        return 0

    grpo_run_id = matching[0].id
    run_metrics_path = OUTPUT_DIR / f"run_metrics_{RUN_KEY}.json"

    subprocess.run(
        [
            "python",
            "-m",
            "obfuscation_atlas.scripts.export_wandb_run_metrics",
            "--entity", "lilylixin6-the-harker-school",
            "--project", "deception_detection",
            f"--run-id={grpo_run_id}",
            f"--output={run_metrics_path}",
        ],
        cwd=WORKSPACE / "obfuscation-atlas",
        check=True,
    )
    print(f"Exported metrics to {run_metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
