#!/usr/bin/env python3
"""No-detector-penalty ceiling: GRPO from the base model with loss.detector=0.

Same hyperparams as trial3 det0 baseline (kl=0.001, seed=42, data.seed=53,
nofu, pass_only_shown) but with mixed reward pass_all_prob=1.0 so every
training prompt is scored on passes_all_tests (honest-only reward). Detector
penalty is zero and there is no reference adapter, so training starts fresh
from the base Instruct model. Reuses the existing nofu_mixed Phase-0 probe
cache (identical manifest to trial3/round1) purely for monitoring/post-eval —
it has zero effect on training since loss.detector=0. Results saved to
trial3/rounds/det0_ceiling/.
"""

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]  # archive/ is one level deeper than experiments/
TRIAL_DIR = WORKSPACE / "experiments/trials/trial3"
OUTPUT_DIR = TRIAL_DIR / "rounds/det0_ceiling"

PROBE_MANIFEST = WORKSPACE / "experiments/archive/mixed_round1_nofu/probe_train_dataset.json"

RUN_KEY = "trial3_det0_ceiling"


def main() -> int:
    if not PROBE_MANIFEST.exists():
        print(f"ERROR: missing {PROBE_MANIFEST}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {PROBE_MANIFEST}")
    print("Starting from: base model (no reference_adapter_path)")
    print(f"Output dir: {OUTPUT_DIR}")

    subprocess.run(
        [
            "python",
            "-m",
            "obfuscation_atlas.scripts.train_grpo_model_against_probe",
            "+model_preset=llama3_8b",
            "loss.kl=0.001",
            "loss.detector=0",
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
            "detector.directory_save_prefix=nofu_mixed",
            "detector.retrain=false",
            f"++detector.mixed_probe_manifest_path={PROBE_MANIFEST}",
            "corrected_detector_gradient=false",
            "misalignment_eval.enabled=false",
            "wandb.project=deception_detection",
            "wandb.group=full_grpo_runs",
            f"wandb.run_name_prefix={RUN_KEY}",
            "task_args.deceptive_definition=pass_only_shown",
            "task_args.mixed_reward_pass_all_prob=1.0",
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
