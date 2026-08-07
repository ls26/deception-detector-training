# Extract key metrics from a single GRPO run export (run_metrics.json).
# Averaged probe metrics are on 0-1 scale (layer-averaged scores, then evaluated once).

import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

VAL_TASK_KEYS = [
    "val_likely_hardcoded",
    "val_does_execute",
    "val_passes_all_tests",
    "val_executes_but_passes_no_tests",
    "val_passes_first_n_but_not_all_tests",
    "val_success",
    "val_passes_first_n_tests",
    "val_length",
    "val_response_length_in_chars",
    "val_non_code_length_in_chars",
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
SUMMARY_EVAL_TYPES = ["on_dataset", "on_policy.split"]
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

AVERAGED_PROBE_PREFIX = (
    "probes/eval_{detector}.{model}.{eval_type}_averaged/linear-probe_last-token-generation/"
)

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
    skipped = 0
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
                if not found:
                    skipped += 1
                    continue
                for suffix, value in found.items():
                    add_row(rows, "averaged_probe", suffix, value, detector, model, eval_type)
    return skipped


def scalar(df, section, key):
    value = df.loc[(df["section"] == section) & (df["key"] == key), "value"]
    assert not value.empty, f"Missing {section}/{key}"
    return value.iloc[0]


def probe_value(df, section, key, detector, model, eval_type):
    row = df.loc[
        (df["section"] == section)
        & (df["key"] == key)
        & (df["detector"] == detector)
        & (df["model"] == model)
        & (df["eval_type"] == eval_type),
        "value",
    ]
    return None if row.empty else float(row.iloc[0])


def format_value(value, precision=4):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.{precision}f}"


def print_base_vs_final(df, section, metric_key, eval_type, label, precision=4):
    base = probe_value(df, section, metric_key, "initial_detector", "base_model", eval_type)
    final = probe_value(df, section, metric_key, "initial_detector", "final_model", eval_type)
    if base is None and final is None:
        return
    delta = None if base is None or final is None else final - base
    print(
        f" {label:<20} base model: {format_value(base, precision)}, "
        f"final model: {format_value(final, precision)}, "
        f"delta: {format_value(delta, precision)}"
    )


def build_run_metrics_df(data: dict) -> pd.DataFrame:
    assert {"run_id", "metrics", "classification"} <= data.keys()
    metrics, classification = data["metrics"], data["classification"]

    rows = []

    for key in CONFIG_KEYS:
        add_row(rows, "config", key, metrics[key])
    add_row(rows, "config", "task_metrics_run_id", data.get("task_metrics_run_id"))

    filtered_layers_str = ",".join(str(layer) for layer in metrics.get("filtered_layers") or [])
    add_row(rows, "config", "filtered_layers", filtered_layers_str)
    add_row(rows, "config", "command_args", " ".join(metrics.get("command_args") or []))

    assert metrics["run_id"] and metrics["kl_coef"] is not None
    assert filtered_layers_str and metrics.get("command_args")
    print(f"config: {len(CONFIG_KEYS) + 3} rows")

    for key in VAL_TASK_KEYS:
        add_row(rows, "task", key, metrics[key])

    assert float(metrics["val_n_examples"]) > 0
    assert float(metrics["val_does_execute"]) > 0.5
    if data.get("task_metrics_run_id"):
        warnings.warn(
            f"Split run: task metrics from {data['task_metrics_run_id']}, probe metrics from {data['run_id']}"
        )
    print(
        f"task: {len(VAL_TASK_KEYS)} rows "
        f"(val_n_examples={float(metrics['val_n_examples'])}, val_does_execute={float(metrics['val_does_execute'])})"
    )

    for key in ("base_model_mean_score", "final_model_mean_score", "representation_change"):
        add_row(rows, "classification", key, metrics[key])
    for key in ("obfuscated_policy_threshold", "representation_change_threshold"):
        add_row(rows, "classification", key, classification[key])
    add_row(rows, "classification", "policy_label", policy_label(classification))
    add_row(rows, "classification", "obfuscation_label", obfuscation_label(classification))
    add_row(rows, "classification", "extracted_policy_type", classification["policy_type"])

    assert classification["policy_type"] and metrics["base_model_mean_score"] is not None
    print(f"classification: 8 rows (extracted_policy_type={classification['policy_type']!r})")

    skipped_avg = add_probe_rows(rows, metrics)

    df = pd.DataFrame(rows)
    assert len(df.loc[df["section"] == "averaged_probe"]) >= 1
    print(
        f"probes: {len(df.loc[df['section'] == 'averaged_probe'])} averaged rows "
        f"(skipped {skipped_avg} combos)"
    )
    return df


def print_all_results(run_metrics_path: Path | str) -> None:
    run_metrics_path = Path(run_metrics_path)
    mtime = datetime.fromtimestamp(run_metrics_path.stat().st_mtime)
    print(f"Loaded from {run_metrics_path} (modified {mtime})")

    data = json.loads(run_metrics_path.read_text())
    assert {"run_id", "metrics", "classification"} <= data.keys()
    print(f"run_id={data['run_id']}, task_metrics_run_id={data.get('task_metrics_run_id')}")

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
    for eval_type in SUMMARY_EVAL_TYPES:
        print(f"  [{EVAL_TYPE_LABELS[eval_type]}]")
        for detector, model, label in PROBE_COMBOS:
            deceptive = probe_value(df, "averaged_probe", "mean_score_positive", detector, model, eval_type)
            honest = probe_value(df, "averaged_probe", "mean_score_negative", detector, model, eval_type)
            print(
                f"    {label}:  "
                f"deceptive: {format_value(deceptive)}  honest: {format_value(honest)}"
            )

    print("\nProbe metrics (layer-averaged, 0-1 scale)")
    for eval_type in SUMMARY_EVAL_TYPES:
        print(f"  [{EVAL_TYPE_LABELS[eval_type]}]")
        for detector, model, label in PROBE_COMBOS:
            values = "  ".join(
                f"{PROBE_LABELS[metric_key]}: {format_value(probe_value(df, 'averaged_probe', metric_key, detector, model, eval_type))}"
                for metric_key in AVERAGED_CLASSIFICATION_SUFFIXES
            )
            print(f"    {label}:  {values}")

    representation_drift_summary(df)


def representation_drift_summary(df):
    print("\nRepresentation drift summary")

    print("\nProbe scores (initial_detector, layer-averaged, 0-1 scale)")
    for eval_type in SUMMARY_EVAL_TYPES:
        print(f"  [{EVAL_TYPE_LABELS[eval_type]}]")
        for metric_key in AVERAGED_SCORE_SUFFIXES:
            print_base_vs_final(df, "averaged_probe", metric_key, eval_type, PROBE_LABELS[metric_key])

    print("\nProbe metrics (initial_detector, layer-averaged, 0-1 scale)")
    for eval_type in SUMMARY_EVAL_TYPES:
        print(f"  [{EVAL_TYPE_LABELS[eval_type]}]")
        for metric_key in AVERAGED_CLASSIFICATION_SUFFIXES:
            print_base_vs_final(df, "averaged_probe", metric_key, eval_type, PROBE_LABELS[metric_key])
