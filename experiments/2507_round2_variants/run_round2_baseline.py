#!/usr/bin/env python3
"""Baseline round 2: GRPO against round 1's probe (no probe retraining).

Uses the same probe manifest as round 1 (base+det0 mixed, 300/class) instead
of building a new dataset from round 1 on-policy generations. All other
hyperparams match trial 3. Results saved to trial3/rounds/round2_baseline/.
"""

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]  # archive/ is one level deeper than experiments/
TRIAL_DIR = WORKSPACE / "experiments/trials/trial3"
ROUND1_DIR = TRIAL_DIR / "rounds/round1"
OUTPUT_DIR = TRIAL_DIR / "rounds/round2_baseline"

CHECKPOINT_FINAL = (
    WORKSPACE
    / "results/run_results/full_grpo_runs"
    / "trigger_none_model_62es59py_layers_32/checkpoints/final"
)

RUN_KEY = "trial3_round2_baseline"


def main() -> int:
    probe_manifest = ROUND1_DIR / "probe_train_dataset.json"
    probe_metrics_path = ROUND1_DIR / "probe_metrics.json"

    for p in [probe_manifest, probe_metrics_path, CHECKPOINT_FINAL]:
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {probe_manifest}")
    print(f"Checkpoint: {CHECKPOINT_FINAL}")
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
