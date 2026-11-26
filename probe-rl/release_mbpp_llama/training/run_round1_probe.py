"""Train and evaluate the round-1 probe on the base model's activations.

Usage:
    python -m training.run_round1_probe --trial 1

Needs the shared artifacts from `python -m training.make_artifacts`. Writes the probe's
training manifest, weights, and metrics to `experiments/trials/trial<N>/rounds/round1/`; the
round-1 GRPO run (`experiments/trials/trial<N>/round1.yaml`) trains its penalty probe from
that manifest, and the frozen condition reuses these weights in every round.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch

from training.build_dataset import N_FIXED_PER_CLASS, ON_POLICY_ROUND1_MIXED, build_train_dataset_round1_mixed
from training.common import EVAL_SAMPLES, EXPERIMENTS, bootstrap_hf_env, check_jsonl_rows
from training.extract_run_results import build_run_metrics_df, probe_value
from training.probe_training import load_base_model, run_probe_train_eval_pipeline

ARTIFACTS_DIR = EXPERIMENTS / "artifacts"
MIN_EXPECTED_ON_DATASET_AUC = 0.9


def round1_dir(trial: int) -> Path:
    return EXPERIMENTS / "trials" / f"trial{trial}" / "rounds" / "round1"


def verify_probe_metrics(metrics_path: Path, weights_dir: Path) -> dict:
    metrics = json.loads(metrics_path.read_text())
    assert metrics["deceptive_sources"] == {"on_dataset": N_FIXED_PER_CLASS, "det0": ON_POLICY_ROUND1_MIXED}
    assert metrics["honest_sources"] == {"on_dataset": N_FIXED_PER_CLASS, "base_model": ON_POLICY_ROUND1_MIXED}
    assert (weights_dir / "layer_0" / "model.pt").exists()
    if metrics["on_dataset_auc"] < MIN_EXPECTED_ON_DATASET_AUC:
        print(
            f"WARN: on_dataset_auc={metrics['on_dataset_auc']:.4f} < {MIN_EXPECTED_ON_DATASET_AUC}; "
            "the round-1 probe separates the on-dataset classes worse than in the paper"
        )
    print(f"OK: probe metrics verified at {metrics_path.name}")
    return metrics


def print_summary(metrics: dict, run_metrics_path: Path) -> None:
    print("\n=== Round 1 probe metrics ===")
    print(f"  on_dataset_auc:       {metrics['on_dataset_auc']:.4f}")
    print(f"  on_policy_split_auc:  {metrics['on_policy_split_auc']:.4f}")
    print(
        f"  on_policy scores:     "
        f"deceptive={metrics['on_policy_mean_scores']['deceptive']:.4f}  "
        f"honest={metrics['on_policy_mean_scores']['honest']:.4f}"
    )
    print(f"  train sources:        {metrics['deceptive_sources']}")
    if not run_metrics_path.exists():
        return
    # The round-1 GRPO run's Phase 0 trains its own probe from the same manifest; the two
    # should agree closely.
    df = build_run_metrics_df(json.loads(run_metrics_path.read_text()))
    grpo_on_dataset = probe_value(df, "averaged_probe", "AUC_ROC", "initial_detector", "base_model", "on_dataset")
    grpo_on_policy_split = probe_value(
        df, "averaged_probe", "AUC_ROC", "initial_detector", "base_model", "on_policy.split"
    )
    print("\n=== Comparison with the round-1 GRPO export (initial_detector.base_model) ===")
    print(f"  on_dataset AUC:       script={metrics['on_dataset_auc']:.4f}  grpo={grpo_on_dataset}")
    print(f"  on_policy.split AUC:  script={metrics['on_policy_split_auc']:.4f}  grpo={grpo_on_policy_split}")


def run(trial: int, *, force: bool) -> None:
    bootstrap_hf_env()
    output_dir = round1_dir(trial)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "probe_weights"
    activations_dir = output_dir / "probe_activations"
    metrics_path = output_dir / "probe_metrics.json"
    manifest_path = output_dir / "probe_train_dataset.json"
    base_eval_jsonl = ARTIFACTS_DIR / "base" / "base_eval_generations.jsonl"
    print(f"\n{'=' * 60}\n=== Round 1 probe (trial {trial}) ===\n{'=' * 60}")

    if not base_eval_jsonl.exists():
        raise FileNotFoundError(f"Missing {base_eval_jsonl}: run `python -m training.make_artifacts` first")
    counts = check_jsonl_rows(base_eval_jsonl, EVAL_SAMPLES, label_fields=("deceptive", "honest"))["counts"]
    print(f"OK: base on-policy eval ({EVAL_SAMPLES} rows, deceptive={counts['deceptive']}, honest={counts['honest']})")

    if force:
        for path in (weights_dir, activations_dir):
            if path.exists():
                shutil.rmtree(path)
                print(f"OK: deleted {path} (--force)")

    model, tokenizer = load_base_model()
    train_pos, train_neg, manifest = build_train_dataset_round1_mixed(
        tokenizer,
        manifest_path=manifest_path,
        base_train_jsonl=ARTIFACTS_DIR / "base" / "base_train_generations.jsonl",
        det0_train_jsonl=ARTIFACTS_DIR / "det0" / "det0_train_generations.jsonl",
    )
    run_probe_train_eval_pipeline(
        model,
        tokenizer,
        train_pos=train_pos,
        train_neg=train_neg,
        manifest=manifest,
        weights_dir=weights_dir,
        activations_dir=activations_dir,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        val_jsonl=base_eval_jsonl,
        run_id="base",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    metrics = verify_probe_metrics(metrics_path, weights_dir)
    print_summary(metrics, output_dir / f"run_metrics_trial{trial}_round1.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Delete cached probe weights/activations and retrain")
    args = parser.parse_args(argv)
    run(args.trial, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
