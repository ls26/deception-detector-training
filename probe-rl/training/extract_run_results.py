"""Read an exported GRPO run (`run_metrics_<run_key>.json`, written by
obfuscation_atlas.scripts.export_wandb_run_metrics) and print its headline numbers: task
metrics on the held-out prompts, the policy classification, and the layer-averaged probe
scores and metrics of the initial and fresh probes on the base and final models. Sycophancy
(multiple-choice) runs have their own, shorter summary: `print_sycophancy_results`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

VAL_TASK_KEYS = [
    "val_likely_hardcoded",
    "val_does_execute",
    "val_passes_all_tests",
    "val_passes_first_n_but_not_all_tests",
    "val_success",
    "val_passes_first_n_tests",
    "val_length",
    "val_response_length_in_chars",
    "val_n_examples",
]

CONFIG_KEYS = [
    "run_id",
    "run_name",
    "wandb_url",
    "model",
    "kl_coef",
    "detector_coef",
    "code_length_penalty",
    "probe_dataset",
    "probe_dataset_canonical",
    "seed",
    "corrected_gradient",
    "val_from_artifact",
]

DETECTORS = ["initial_detector", "fresh_detector"]
MODELS = ["base_model", "final_model"]
EVAL_TYPES = ["on_policy.split", "on_dataset"]
EVAL_TYPE_LABELS = {"on_dataset": "on_dataset", "on_policy.split": "on_policy"}

AVERAGED_SCORE_SUFFIXES = ["mean_score_positive", "mean_score_negative"]
AVERAGED_CLASSIFICATION_SUFFIXES = ["AUC_ROC", "accuracy", "recall_at_1%_fpr"]
ACCURACY_SOURCE_SUFFIX = {
    "on_dataset": "metrics_at_threshold/accuracy",
    "on_policy.split": "accuracy",
}

PROBE_LABELS = {
    "mean_score_positive": "deceptive",
    "mean_score_negative": "honest",
    "AUC_ROC": "AUC_ROC",
    "accuracy": "accuracy",
    "recall_at_1%_fpr": "recall_at_1%_fpr",
}

AVERAGED_PROBE_PREFIX = "probes/eval_{detector}.{model}.{eval_type}_averaged/linear-probe_last-token-generation/"

PROBE_COMBOS = [
    ("initial_detector", "base_model", "initial probe / base model"),
    ("initial_detector", "final_model", "initial probe / final model"),
    ("fresh_detector", "final_model", "final probe / final model"),
]


def add_row(rows, section, key, value, detector=None, model=None, eval_type=None):
    rows.append(
        {"section": section, "key": key, "detector": detector, "model": model, "eval_type": eval_type, "value": value}
    )


def policy_label(classification):
    for flag, label in [
        ("honest", "honest"),
        ("degenerate", "degenerate"),
        ("reward_hacky", "reward_hacky"),
        ("reward_hacky_no_hardcode", "reward_hacky_no_hardcode"),
    ]:
        if classification.get(flag):
            return label
    return float("nan")


def obfuscation_label(classification):
    policy = bool(classification.get("obfuscated_policy"))
    activations = bool(classification.get("obfuscated_activations"))
    if policy and activations:
        return "obfuscated_both"
    if policy:
        return "obfuscated_policy"
    if activations:
        return "obfuscated_activations"
    return float("nan")


def add_probe_rows(rows, metrics):
    for detector in DETECTORS:
        for model in MODELS:
            for eval_type in EVAL_TYPES:
                prefix = AVERAGED_PROBE_PREFIX.format(detector=detector, model=model, eval_type=eval_type)
                found = {
                    suffix: metrics[prefix + suffix]
                    for suffix in AVERAGED_SCORE_SUFFIXES + ["AUC_ROC", "recall_at_1%_fpr"]
                    if prefix + suffix in metrics
                }
                accuracy_suffix = ACCURACY_SOURCE_SUFFIX[eval_type]
                if prefix + accuracy_suffix in metrics:
                    found["accuracy"] = metrics[prefix + accuracy_suffix]
                for suffix, value in found.items():
                    add_row(rows, "averaged_probe", suffix, value, detector, model, eval_type)


def scalar(df, section, key):
    value = df.loc[(df["section"] == section) & (df["key"] == key), "value"]
    assert not value.empty, f"Missing {section}/{key}"
    return value.iloc[0]


def probe_value(df, section, key, detector, model, eval_type):
    """One averaged probe metric, or None when the export does not have it."""
    row = df.loc[
        (df["section"] == section)
        & (df["key"] == key)
        & (df["detector"] == detector)
        & (df["model"] == model)
        & (df["eval_type"] == eval_type),
        "value",
    ]
    return None if row.empty else float(row.iloc[0])


def format_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.4f}"


def print_base_vs_final(df, section, metric_key, eval_type, label):
    base = probe_value(df, section, metric_key, "initial_detector", "base_model", eval_type)
    final = probe_value(df, section, metric_key, "initial_detector", "final_model", eval_type)
    if base is None and final is None:
        return
    delta = None if base is None or final is None else final - base
    print(
        f" {label:<20} base model: {format_value(base)}, "
        f"final model: {format_value(final)}, "
        f"delta: {format_value(delta)}"
    )


def build_run_metrics_df(data: dict) -> pd.DataFrame:
    """The export as a long table with columns section/key/detector/model/eval_type/value."""
    assert {"run_id", "metrics", "classification"} <= data.keys()
    metrics, classification = data["metrics"], data["classification"]
    rows = []

    for key in CONFIG_KEYS:
        add_row(rows, "config", key, metrics[key])
    filtered_layers_str = ",".join(str(layer) for layer in metrics["filtered_layers"])
    add_row(rows, "config", "filtered_layers", filtered_layers_str)
    add_row(rows, "config", "command_args", " ".join(metrics["command_args"]))
    assert metrics["run_id"] and metrics["kl_coef"] is not None
    assert filtered_layers_str and metrics["command_args"]

    for key in VAL_TASK_KEYS:
        add_row(rows, "task", key, metrics[key])
    assert float(metrics["val_n_examples"]) > 0
    if float(metrics["val_does_execute"]) <= 0.5:
        print(f"WARN: val_does_execute={float(metrics['val_does_execute'])} <= 0.5: the policy may have collapsed")

    for key in ("base_model_mean_score", "final_model_mean_score", "representation_change"):
        add_row(rows, "classification", key, metrics[key])
    for key in ("obfuscated_policy_threshold", "representation_change_threshold"):
        add_row(rows, "classification", key, classification[key])
    add_row(rows, "classification", "policy_label", policy_label(classification))
    add_row(rows, "classification", "obfuscation_label", obfuscation_label(classification))
    add_row(rows, "classification", "extracted_policy_type", classification["policy_type"])
    assert classification["policy_type"] and metrics["base_model_mean_score"] is not None

    add_probe_rows(rows, metrics)
    df = pd.DataFrame(rows)
    assert len(df.loc[df["section"] == "averaged_probe"]) >= 1
    return df


def print_all_results(run_metrics_path: Path | str) -> None:
    """Print the task metrics, classification, and probe summary of one exported run."""
    run_metrics_path = Path(run_metrics_path)
    mtime = datetime.fromtimestamp(run_metrics_path.stat().st_mtime)
    print(f"Loaded from {run_metrics_path} (modified {mtime})")
    data = json.loads(run_metrics_path.read_text())
    print(f"run_id={data['run_id']}")
    df = build_run_metrics_df(data)

    print("\nTask metrics:")
    print(f"  pass_first_test_rate: {scalar(df, 'task', 'val_passes_first_n_tests')}")
    print(f"  pass_all_test_rate: {scalar(df, 'task', 'val_passes_all_tests')}")
    print(f"  deception_rate: {scalar(df, 'task', 'val_passes_first_n_but_not_all_tests')}")
    print(f"  task reward: {scalar(df, 'task', 'val_success')}")

    print("\nPolicy classification:")
    print(f"  policy_label:       {scalar(df, 'classification', 'policy_label')}")
    print(f"  obfuscation_label:  {scalar(df, 'classification', 'obfuscation_label')}")
    print(f"  base_model_mean_score: {scalar(df, 'classification', 'base_model_mean_score')}")
    print(f"  final_model_mean_score: {scalar(df, 'classification', 'final_model_mean_score')}")
    print(f"  representation_change: {scalar(df, 'classification', 'representation_change')}")
    print(f"  obfuscated_policy_threshold: {scalar(df, 'classification', 'obfuscated_policy_threshold')}")
    print(f"  representation_change_threshold: {scalar(df, 'classification', 'representation_change_threshold')}")

    print("\nProbe scores (layer-averaged, 0-1 scale)")
    for eval_type in EVAL_TYPES:
        print(f"  [{EVAL_TYPE_LABELS[eval_type]}]")
        for detector, model, label in PROBE_COMBOS:
            deceptive = probe_value(df, "averaged_probe", "mean_score_positive", detector, model, eval_type)
            honest = probe_value(df, "averaged_probe", "mean_score_negative", detector, model, eval_type)
            print(f"    {label}:  deceptive: {format_value(deceptive)}  honest: {format_value(honest)}")

    print("\nProbe metrics (layer-averaged, 0-1 scale)")
    for eval_type in EVAL_TYPES:
        print(f"  [{EVAL_TYPE_LABELS[eval_type]}]")
        for detector, model, label in PROBE_COMBOS:
            values = "  ".join(
                f"{PROBE_LABELS[metric_key]}: "
                f"{format_value(probe_value(df, 'averaged_probe', metric_key, detector, model, eval_type))}"
                for metric_key in AVERAGED_CLASSIFICATION_SUFFIXES
            )
            print(f"    {label}:  {values}")

    print("\nRepresentation drift (initial probe, base model -> final model)")
    for eval_type in EVAL_TYPES:
        print(f"  [{EVAL_TYPE_LABELS[eval_type]}]")
        for metric_key in AVERAGED_SCORE_SUFFIXES + AVERAGED_CLASSIFICATION_SUFFIXES:
            print_base_vs_final(df, "averaged_probe", metric_key, eval_type, PROBE_LABELS[metric_key])


SYCOPHANCY_FINAL_EVAL_KEYS = [
    "final_eval/rewards/task_success",
    "final_eval/rewards_metrics/followed_cue",
    "final_eval/rewards_metrics/parse_failed",
]


def print_sycophancy_results(run_metrics_path: Path | str) -> None:
    """Print one exported sycophancy run: the post-training eval on the held-out questions
    (accuracy, cue-following, parse failures) and the in-loop eval history that
    grpo_launcher.add_sycophancy_eval_history added to the export."""
    run_metrics_path = Path(run_metrics_path)
    data = json.loads(run_metrics_path.read_text())
    metrics = data["metrics"]
    print(f"run_id={data['run_id']}")
    print("\nFinal eval (held-out questions, one sample each):")
    for key in SYCOPHANCY_FINAL_EVAL_KEYS:
        print(f"  {key}: {metrics[key]:.4f}")
    print("\nIn-loop eval (accuracy / cue agreement rate / parse failures; each row scores the policy that")
    print("generated that step's training answers; in runs from before 2026-10-24, 261013-261020, the")
    print("policy from the previous eval point, the first row the starting model):")
    for row in data["sycophancy_eval_history"]:
        print(
            f"  logged at step {row['logged_at_train_step']}: {row['eval/rewards/task_success_mean']:.4f} / "
            f"{row['eval/rewards_metrics/followed_cue_mean']:.4f} / "
            f"{row['eval/rewards_metrics/parse_failed_mean']:.4f}"
        )
