#!/usr/bin/env python3
"""KL-from-base ablation of round 2: GRPO with the KL penalty measured against the
base model instead of the round 1 checkpoint.

Everything else matches trial 3's round 2 (`trial3_round2_v2`): same round-2 probe
manifest, same round-1 checkpoint as the policy's starting point, same hyperparams.
The only difference is `++grpo.model.kl_reference_from_base=true`, which zeroes the
reference adapter's LoRA weights after loading so the KL penalty is computed relative
to the pristine base model while the policy still starts from round 1's weights.
Results saved to trial3/rounds/round2_kl_from_base/.
"""

import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]  # archive/ is one level deeper than experiments/
TRIAL_DIR = WORKSPACE / "experiments/trials/trial3"
ROUND2_DIR = TRIAL_DIR / "rounds/round2"
OUTPUT_DIR = TRIAL_DIR / "rounds/round2_kl_from_base"

CHECKPOINT_FINAL = (
    WORKSPACE
    / "results/run_results/trial3/original"
    / "trial3_round1_62es59py/checkpoints/final"
)

RUN_KEY = "trial3_round2_kl_from_base"


def main() -> int:
    probe_manifest = ROUND2_DIR / "probe_train_dataset.json"
    probe_metrics_path = ROUND2_DIR / "probe_metrics.json"

    for p in [probe_manifest, probe_metrics_path, CHECKPOINT_FINAL]:
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    probe_metrics = json.loads(probe_metrics_path.read_text())
    print(f"Round 2 probe on_dataset_auc: {probe_metrics['on_dataset_auc']:.4f}")
    print(f"Manifest: {probe_manifest}")
    print(f"Checkpoint (policy init): {CHECKPOINT_FINAL}")
    print("KL reference: base model (kl_reference_from_base=true)")
    print(f"Output dir: {OUTPUT_DIR}")

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
            "++grpo.model.kl_reference_from_base=true",
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

    # --- Export W&B metrics ---
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
