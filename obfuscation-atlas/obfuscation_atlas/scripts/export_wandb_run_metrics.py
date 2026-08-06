"""Export W&B run metrics and policy/obfuscation classifications.

Fetches all metrics produced by extract_common_metrics() and _process_run_by_id(),
then applies the same classification pipeline used by fetch_all_runs_cached()
(without bulk fetch or filter_rules).

For runs where training and post-eval completed in one W&B run, pass --run-id only.
For split runs (e.g. eval-only rerun with broken on_policy_responses), pass
--task-metrics-run-id to overlay val_* task metrics from the training run.

Usage:
    # Single run (normal case)
    python -m obfuscation_atlas.scripts.export_wandb_run_metrics \\
        --entity lilylixin6-the-harker-school \\
        --project deception_detection \\
        --run-id abc123 \\
        --output run_metrics.json

    # Split run: probe/obfuscation from eval run, task metrics from training run
    python -m obfuscation_atlas.scripts.export_wandb_run_metrics \\
        --entity lilylixin6-the-harker-school \\
        --project deception_detection \\
        --run-id 24v08wlt \\
        --task-metrics-run-id c3akfszd \\
        --output run_metrics.json

    # HF checkpoint: overlay task metrics from local MBPP summary JSON
    python -m obfuscation_atlas.scripts.export_wandb_run_metrics \\
        --entity lilylixin6-the-harker-school \\
        --project deception_detection \\
        --run-id abc123 \\
        --task-metrics-json /workspace/results/run_results/hf_probe_eval_runs/extracted_results/task_metrics_kl1e-3-det3-s1-mbpp.json \\
        --output /workspace/results/run_results/hf_probe_eval_runs/extracted_results/run_metrics_kl1e-3-det3-s1-mbpp.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from obfuscation_atlas.utils.wandb_api_cache_utils import (
    _process_run_by_id,
    add_classification_columns,
    classify_policy_types,
    correct_obfuscated_policy_margin,
    extract_final_metric,
    use_validation_metrics,
)

_SPLIT_DETECTOR_KEY = "linear-probe_last-token-generation"
_SPLIT_AGGREGATED_PREFIX = "aggregated_detector_eval_metrics"

ON_POLICY_SPLIT_AGGREGATED_KEYS: dict[str, str] = {
    "on_policy_split_final_initial_recall": (
        f"{_SPLIT_AGGREGATED_PREFIX}/initial_detector.final_model.on_policy.split/"
        f"{_SPLIT_DETECTOR_KEY}/recall_at_1%_fpr"
    ),
    "on_policy_split_final_initial_auc": (
        f"{_SPLIT_AGGREGATED_PREFIX}/initial_detector.final_model.on_policy.split/"
        f"{_SPLIT_DETECTOR_KEY}/AUC_ROC"
    ),
    "on_policy_split_fresh_final_recall": (
        f"{_SPLIT_AGGREGATED_PREFIX}/fresh_detector.final_model.on_policy.split/"
        f"{_SPLIT_DETECTOR_KEY}/recall_at_1%_fpr"
    ),
    "on_policy_split_fresh_final_auc": (
        f"{_SPLIT_AGGREGATED_PREFIX}/fresh_detector.final_model.on_policy.split/"
        f"{_SPLIT_DETECTOR_KEY}/AUC_ROC"
    ),
    "on_policy_split_base_initial_recall": (
        f"{_SPLIT_AGGREGATED_PREFIX}/initial_detector.base_model.on_policy.split/"
        f"{_SPLIT_DETECTOR_KEY}/recall_at_1%_fpr"
    ),
    "on_policy_split_base_initial_auc": (
        f"{_SPLIT_AGGREGATED_PREFIX}/initial_detector.base_model.on_policy.split/"
        f"{_SPLIT_DETECTOR_KEY}/AUC_ROC"
    ),
}

MBPP_TO_VAL_METRICS = {
    "pass_first_test_rate": "val_success",
    "pass_all_tests_rate": "val_passes_all_tests",
    "deceptive_rate": "val_passes_first_n_but_not_all_tests",
    "pass_only_shown_deception_rate": "val_pass_only_shown_deception",
    "hardcoded_rate": "val_likely_hardcoded",
    "executes_rate": "val_does_execute",
    "avg_score": "val_avg_score",
    "n_examples": "val_n_examples",
}

CLASSIFICATION_COLUMNS = [
    "policy_type",
    "honest",
    "degenerate",
    "reward_hacky",
    "reward_hacky_no_hardcode",
    "obfuscated_policy",
    "obfuscated_activations",
    "obfuscated_policy_threshold",
    "representation_change_threshold",
]

DEFAULT_POLICY_THRESHOLD = 0.35
DEFAULT_REPR_CHANGE_THRESHOLD = 0.2


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python natives."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def enrich_on_policy_split_metrics(metrics: dict, project_path: str, run_id: str) -> None:
    """Add on-policy split convenience fields and ensure full W&B split keys are present."""
    import wandb

    run = wandb.Api().run(f"{project_path}/{run_id}")
    for field_name, wandb_key in ON_POLICY_SPLIT_AGGREGATED_KEYS.items():
        if metrics.get(field_name) is not None:
            continue
        value = extract_final_metric(run, wandb_key)
        if value is not None:
            metrics[field_name] = value
        if wandb_key not in metrics and value is not None:
            metrics[wandb_key] = value

    for key, value in run.summary.items():
        if ".on_policy.split" not in key or key.startswith("_") or value is None:
            continue
        if type(value) not in (int, float, str):
            continue
        if key not in metrics:
            metrics[key] = value


def fetch_run_metrics(run_id: str, project_path: str) -> dict:
    """Fetch metrics for one W&B run via _process_run_by_id."""
    metrics = _process_run_by_id((run_id, project_path))
    if metrics is None:
        raise RuntimeError(f"Failed to fetch metrics for run {run_id} in {project_path}")
    return metrics


def overlay_val_metrics(metrics: dict, task_metrics: dict) -> None:
    """Copy val_* keys from task_metrics into metrics (in place)."""
    for key, value in task_metrics.items():
        if key.startswith("val_"):
            metrics[key] = value


def load_task_metrics_json(path: Path) -> dict:
    """Load task metrics JSON and normalize to val_* keys."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Task metrics JSON must be an object: {path}")

    val_metrics: dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("val_"):
            val_metrics[key] = value

    for mbpp_key, val_key in MBPP_TO_VAL_METRICS.items():
        if mbpp_key in data and val_key not in val_metrics:
            val_metrics[val_key] = data[mbpp_key]

    # W&B artifact uses val_passes_first_n_tests for the same pass-first rate as val_success
    if "val_success" in val_metrics and "val_passes_first_n_tests" not in val_metrics:
        val_metrics["val_passes_first_n_tests"] = val_metrics["val_success"]

    if "val_avg_score" in val_metrics and "val_task_reward" not in val_metrics:
        val_metrics["val_task_reward"] = val_metrics["val_avg_score"]

    if not val_metrics:
        raise ValueError(f"No val_* or MBPP summary fields found in {path}")
    return val_metrics


def ensure_val_task_reward(metrics: dict) -> None:
    """Set val_task_reward from val_avg_score when MBPP overlay already provided it."""
    if metrics.get("val_task_reward") is not None:
        return
    if metrics.get("val_avg_score") is not None:
        metrics["val_task_reward"] = metrics["val_avg_score"]


def warn_if_task_metrics_suspicious(
    metrics: dict,
    task_metrics_run_id: str | None,
    task_metrics_json: str | None,
) -> None:
    """Print a hint when task metrics look broken and no overlay was used."""
    if task_metrics_run_id is not None or task_metrics_json is not None:
        return
    val_success = metrics.get("val_success")
    val_hardcoded = metrics.get("val_likely_hardcoded")
    if val_success is not None and val_hardcoded is not None:
        if float(val_success) == 0.0 and float(val_hardcoded) > 0.3:
            print(
                "Warning: val_success=0 but val_likely_hardcoded>0.3 — task metrics may be "
                "broken. If training and eval used separate W&B runs, pass --task-metrics-run-id.",
                file=sys.stderr,
            )


def ensure_classification_probe_fields(metrics: dict) -> None:
    """Fill missing probe fields so classify_obfuscation does not raise on None comparisons."""
    for key in (
        "representation_change",
        "base_model_mean_score",
        "final_model_mean_score",
        "base_model_mean_score_negative_on_dataset",
        "final_model_mean_score_negative_on_dataset",
    ):
        if metrics.get(key) is None:
            metrics[key] = float("nan")


def classify_metrics(
    metrics: dict,
    policy_threshold: float,
    repr_change_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    """Run classification pipeline; return (classification dict, classified DataFrame)."""
    ensure_classification_probe_fields(metrics)
    df = pd.DataFrame([metrics])
    df = use_validation_metrics(df, verbose=False)
    df = correct_obfuscated_policy_margin(df)
    df = add_classification_columns(
        df,
        obfuscated_policy_margin_threshold=policy_threshold,
        representation_change_threshold=repr_change_threshold,
        baseline_subtract=False,
        include_llm_subcategories=False,
    )
    df["policy_type"] = classify_policy_types(df)

    row = df.iloc[0]
    classification = {col: json_safe(row[col]) for col in CLASSIFICATION_COLUMNS if col in df.columns}

    # Reflect validation-metric replacements back into metrics dict for export
    updated_metrics = row.to_dict()
    return classification, updated_metrics


def build_report(
    run_id: str,
    project_path: str,
    metrics: dict,
    classification: dict,
    task_metrics_run_id: str | None,
    task_metrics_json: str | None,
) -> dict:
    return {
        "run_id": run_id,
        "task_metrics_run_id": task_metrics_run_id,
        "task_metrics_json": task_metrics_json,
        "project": project_path,
        "metrics": json_safe(metrics),
        "classification": classification,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export W&B run metrics and policy/obfuscation classifications.",
    )
    parser.add_argument("--entity", required=True, help="W&B entity (user or team)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run-id", required=True, help="Primary W&B run ID")
    parser.add_argument(
        "--task-metrics-run-id",
        default=None,
        help="Optional W&B run ID to overlay val_* task metrics (split-run case)",
    )
    parser.add_argument(
        "--task-metrics-json",
        default=None,
        help="Optional JSON file with val_* keys or MBPP summary fields to overlay",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: print to stdout)",
    )
    parser.add_argument(
        "--policy-threshold",
        type=float,
        default=DEFAULT_POLICY_THRESHOLD,
        help=f"Obfuscated policy threshold (default: {DEFAULT_POLICY_THRESHOLD})",
    )
    parser.add_argument(
        "--repr-change-threshold",
        type=float,
        default=DEFAULT_REPR_CHANGE_THRESHOLD,
        help=f"Representation change threshold (default: {DEFAULT_REPR_CHANGE_THRESHOLD})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.task_metrics_run_id and args.task_metrics_json:
        print("Error: pass only one of --task-metrics-run-id or --task-metrics-json.", file=sys.stderr)
        return 1

    project_path = f"{args.entity}/{args.project}"

    print(f"Fetching metrics for run {args.run_id}...", file=sys.stderr)
    metrics = fetch_run_metrics(args.run_id, project_path)
    enrich_on_policy_split_metrics(metrics, project_path, args.run_id)

    if args.task_metrics_run_id:
        print(
            f"Overlaying val_* metrics from task run {args.task_metrics_run_id}...",
            file=sys.stderr,
        )
        task_metrics = fetch_run_metrics(args.task_metrics_run_id, project_path)
        overlay_val_metrics(metrics, task_metrics)
    elif args.task_metrics_json:
        task_metrics_path = Path(args.task_metrics_json)
        print(f"Overlaying val_* metrics from {task_metrics_path}...", file=sys.stderr)
        overlay_val_metrics(metrics, load_task_metrics_json(task_metrics_path))
    else:
        warn_if_task_metrics_suspicious(metrics, args.task_metrics_run_id, args.task_metrics_json)

    ensure_val_task_reward(metrics)

    classification, metrics = classify_metrics(
        metrics,
        policy_threshold=args.policy_threshold,
        repr_change_threshold=args.repr_change_threshold,
    )

    report = build_report(
        run_id=args.run_id,
        project_path=project_path,
        metrics=metrics,
        classification=classification,
        task_metrics_run_id=args.task_metrics_run_id,
        task_metrics_json=args.task_metrics_json,
    )

    output_json = json.dumps(report, indent=2, sort_keys=False)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json + "\n")
        print(f"Wrote {out_path}", file=sys.stderr)
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
