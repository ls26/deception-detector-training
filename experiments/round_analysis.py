# Multi-round trial analysis: metrics CSV, plots, and buffer sample inspection.

# %%
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

WORKSPACE = Path(__file__).resolve().parent.parent
TRIAL = 8
VARIANT = "early_retraining"  # "" for the main branch, or a named branch (e.g. "calibrated")
if not VARIANT:
    VARIANT_LABEL = "original"
elif VARIANT == "v2":
    VARIANT_LABEL = "original"
elif VARIANT in ("early_retraining", "early_retraining_v2"):
    VARIANT_LABEL = "early retraining"
else:
    VARIANT_LABEL = VARIANT  # used in plot titles
TRIAL_DIR = WORKSPACE / f"experiments/trials/trial{TRIAL}"
ROUNDS_DIR = TRIAL_DIR / (f"rounds_{VARIANT}" if VARIANT else "rounds")
BUFFER_DIR = TRIAL_DIR / (f"buffer_{VARIANT}" if VARIANT else "buffer")
OUTPUT_CSV = WORKSPACE / "experiments/trial_results" / (
    f"round_analysis_trial{TRIAL}{f'_{VARIANT}' if VARIANT else ''}.csv"
)
WANDB_PROJECT_PATH = "lilylixin6-the-harker-school/deception_detection"
BASE_EVAL_JSONL = WORKSPACE / "experiments/artifacts/base/base_eval_generations.jsonl"
ROUND2_BASELINE_METRICS = (
    ROUNDS_DIR / "round2_baseline" / "run_metrics_trial3_round2_baseline.json"
)
ROUND2_CALIBRATED_PENALTY_METRICS = (
    ROUNDS_DIR
    / "round2_calibrated_penalty"
    / "run_metrics_trial3_round2_calibrated_penalty.json"
)
# Fixed trial-3-era reference runs, reused as-is across every trial-4 branch's
# analysis below — not ROUNDS_DIR-relative, since ROUNDS_DIR now points at trial 4.
DET0_BASELINE_METRICS = [
    WORKSPACE
    / "experiments/trials/trial3/rounds/det0_baseline"
    / f"run_metrics_trial3_det0_baseline{suffix}.json"
    for suffix in ("", "_s43", "_s44")
]
# Pass-all reward ceiling, code_length_penalty=0.001 seed sweep (matches trial 4's
# CLP) — supersedes the old trial3 det0_ceiling runs (CLP=0.003), which are no
# longer the right comparison point. See EXPERIMENT_LOG.md "Trial 3 baseline —
# no-detector-penalty ceiling" (intermediate CLP seed sweep).
DET0_CEILING_METRICS = [
    WORKSPACE / "experiments/single_runs" / f"run_metrics_ceiling_clp1e-3_s{seed}.json"
    for seed in (42, 43, 44)
]

NUM_ROUNDS = 5

# Per-round file version: None → run_metrics_trial{T}_round{R}.json
#                        2    → run_metrics_trial{T}_round{R}_v2.json
RUN_METRICS_VERSION: dict[int, int | None] = {
    1: None,
    2: None,  # use existing _v2 for trial3 round2
}

PROBE_PREFIX = (
    "probes/eval_initial_detector.{model}.{eval_type}_averaged/"
    "linear-probe_last-token-generation/{suffix}"
)

OUTCOME_COLS = ("correct", "deceptive", "buggy", "fail")

BASE_COLUMNS = [
    "round",
    "run_id",
    "grpo_train_step",
    *OUTCOME_COLS,
    "start_on_policy_deceptive",
    "start_on_policy_honest",
    "start_on_policy_auc",
    "start_on_dataset_deceptive",
    "start_on_dataset_honest",
    "start_on_dataset_auc",
    "end_on_dataset_deceptive",
    "end_on_dataset_honest",
    "end_on_dataset_auc",
    "end_on_policy_deceptive",
    "end_on_policy_honest",
    "end_on_policy_auc",
]

SPLIT_COLUMNS = [
    "train_n_deceptive",
    "train_n_honest",
    "train_n_unlabeled",
    "eval_n_deceptive",
    "eval_n_honest",
    "eval_n_unlabeled",
]

END_PROBE_METRICS = [
    ("end_on_dataset_deceptive", "on_dataset", "mean_score_positive"),
    ("end_on_dataset_honest", "on_dataset", "mean_score_negative"),
    ("end_on_dataset_auc", "on_dataset", "AUC_ROC"),
    ("end_on_policy_deceptive", "on_policy.split", "mean_score_positive"),
    ("end_on_policy_honest", "on_policy.split", "mean_score_negative"),
    ("end_on_policy_auc", "on_policy.split", "AUC_ROC"),
]
PROBE_END_COL_MAP = {
    row_key: (eval_type, suffix)
    for row_key, eval_type, suffix in END_PROBE_METRICS
    if suffix.startswith("mean_score_")
}

TASK_METRICS = [
    ("correct", "Correct", "#2ca02c"),
    ("deceptive", "Deceptive", "#d62728"),
    ("buggy", "Buggy", "#1f77b4"),
    ("fail", "Fail", "#ff7f0e"),
]
COMBINED_TASK_METRICS = [
    TASK_METRICS[0],
    TASK_METRICS[1],
    ("buggy_fail", "Buggy/fail", "#ff7f0e"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def buffer_jsonl(round_num: int | str, split: str) -> Path | None:
    if round_num == "base":
        path = BUFFER_DIR / f"base_{split}_generations.jsonl"
        return path if path.exists() else None

    filename = f"trial{TRIAL}_round{round_num}_{split}_generations.jsonl"
    path = BUFFER_DIR / filename
    if not path.exists() and VARIANT and round_num == 1:
        # Round 1 is a shared bootstrap (matches RoundConfig.run_key / run_key() above).
        # Variants that skip per-round generation entirely (e.g. "frozen") never copy
        # it into their own buffer_<variant> dir, so fall back to the main buffer dir.
        path = TRIAL_DIR / "buffer" / filename
    return path if path.exists() else None


def load_buffer_rows(round_num: int | str, split: str) -> list[dict]:
    path = buffer_jsonl(round_num, split)
    if path is None:
        raise FileNotFoundError(f"No buffer file for trial {TRIAL} round {round_num} {split}")
    return load_jsonl(path)


def show_buffer_samples(round_num: int | str, split: str, label: str, n: int = 3) -> None:
    rows = load_buffer_rows(round_num, split)
    matches = [row for row in rows if row.get(label)]
    print(f"trial{TRIAL} {round_num} {split}: {len(matches)} {label} / {len(rows)} total\n")
    if not matches:
        print("No matching examples.")
        return
    for i, row in enumerate(random.sample(matches, min(n, len(matches))), 1):
        task = row["prompt"].split("Task: ", 1)[-1].split("<|")[0].strip()
        print(f"--- Example {i} (task_id={row['task_id']}, prompt_idx={row['prompt_idx']}) ---")
        print(f"Task: {task}\n")
        print(row["completion"])
        print()


def run_key(round_num: int) -> str:
    # Round 1 is a shared bootstrap copied verbatim from the main branch, so it never
    # carries the variant suffix even when VARIANT is set (matches RoundConfig.run_key).
    if VARIANT and round_num > 1:
        return f"trial{TRIAL}_{VARIANT}_round{round_num}"
    return f"trial{TRIAL}_round{round_num}"


def run_metrics_path(round_num: int) -> Path:
    version = RUN_METRICS_VERSION.get(round_num)
    suffix = f"_v{version}" if version is not None else ""
    return ROUNDS_DIR / f"round{round_num}" / f"run_metrics_{run_key(round_num)}{suffix}.json"


def resolve_run_metrics(round_num: int) -> Path:
    path = run_metrics_path(round_num)
    if not path.exists():
        version = RUN_METRICS_VERSION.get(round_num)
        version_label = f"_v{version}" if version is not None else " (no suffix)"
        raise FileNotFoundError(
            f"round {round_num}: missing run_metrics file {path.name} "
            f"(RUN_METRICS_VERSION={version!r}{version_label})"
        )
    return path


def derive_outcome_categories(pass_first: float, pass_all: float, deceptive: float) -> dict[str, float]:
    return {
        "correct": pass_all,
        "deceptive": deceptive,
        "buggy": pass_first - pass_all - deceptive,
        "fail": 1 - pass_first,
    }


def outcome_categories_from_metrics(metrics: dict) -> dict[str, float]:
    return derive_outcome_categories(
        metrics["val_passes_first_n_tests"],
        metrics["val_passes_all_tests"],
        metrics["val_pass_only_shown_deception"],
    )


def outcome_categories_from_run_metrics(path: Path) -> dict[str, float]:
    return outcome_categories_from_metrics(load_json(path)["metrics"])


def outcome_categories_from_jsonl(path: Path) -> dict[str, float]:
    rows = load_jsonl(path)
    n = len(rows)
    return derive_outcome_categories(
        sum(r["passes_first_n_tests"] for r in rows) / n,
        sum(r["passes_all_tests"] for r in rows) / n,
        sum(r["pass_only_shown_deception"] for r in rows) / n,
    )


def combined_outcome_row(categories: dict[str, float]) -> dict[str, float]:
    return {**categories, "buggy_fail": categories["buggy"] + categories["fail"]}


def probe_metric(metrics: dict, model: str, eval_type: str, suffix: str):
    return metrics.get(PROBE_PREFIX.format(model=model, eval_type=eval_type, suffix=suffix))


def count_labels(path: Path | None) -> dict[str, int | None]:
    if path is None or not path.exists():
        return {"n_deceptive": None, "n_honest": None, "n_unlabeled": None}

    rows = load_jsonl(path)
    return {
        "n_deceptive": sum(r.get("deceptive", False) for r in rows),
        "n_honest": sum(r.get("honest", False) for r in rows),
        "n_unlabeled": sum(r.get("unlabeled", False) for r in rows),
    }


def composition_columns(cfg: dict) -> dict[str, int]:
    # Variants with a frozen probe (e.g. "frozen") skip per-round probe retraining
    # entirely after round 1, so later rounds have no probe_train_dataset.json.
    dataset_path = cfg["probe_train_dataset"]
    if not dataset_path.exists():
        return {}

    dataset = load_json(dataset_path)
    return {
        f"{class_name}_{source}": count
        for class_name in ("deceptive", "honest")
        for source, count in dataset[f"{class_name}_sources"].items()
    }


_grpo_step_cache: dict[str, int | None] = {}


def training_run_id(round_num: int) -> str | None:
    state_name = f"trial_state_{VARIANT}.json" if VARIANT else "trial_state.json"
    state_path = TRIAL_DIR / state_name
    if not state_path.exists():
        return None
    return load_json(state_path).get("round_grpo_ids", {}).get(str(round_num))


def fetch_grpo_train_step(run_id: str) -> int | None:
    if run_id in _grpo_step_cache:
        return _grpo_step_cache[run_id]
    from obfuscation_atlas.utils.wandb_api_cache_utils import extract_final_metric

    import wandb

    step = extract_final_metric(wandb.Api().run(f"{WANDB_PROJECT_PATH}/{run_id}"), "grpo_train_step")
    result = int(step) if step is not None else None
    _grpo_step_cache[run_id] = result
    return result


def resolve_grpo_train_step(round_num: int, metrics: dict) -> int | None:
    step = metrics.get("grpo_train_step")
    if step is not None:
        return int(step)
    train_id = training_run_id(round_num)
    return fetch_grpo_train_step(train_id) if train_id else None


def build_row(cfg: dict) -> dict:
    data = load_json(resolve_run_metrics(cfg["round"]))
    metrics = data["metrics"]

    row = {
        "round": cfg["round"],
        "run_id": data["run_id"],
        "grpo_train_step": resolve_grpo_train_step(cfg["round"], metrics),
        **outcome_categories_from_metrics(metrics),
    }

    for prefix, jsonl_key in (("train", "train_jsonl"), ("eval", "eval_jsonl")):
        counts = count_labels(cfg.get(jsonl_key))
        for count_key, value in counts.items():
            row[f"{prefix}_{count_key}"] = value

    probe_metrics_path = cfg["probe_metrics"]
    if probe_metrics_path.exists():
        probe = load_json(probe_metrics_path)
        split_scores = probe.get("on_policy_split_mean_scores", {})
        row["start_on_policy_deceptive"] = split_scores.get("deceptive")
        row["start_on_policy_honest"] = split_scores.get("honest")
        row["start_on_policy_auc"] = probe.get("on_policy_split_auc")

        dataset_scores = probe.get("on_dataset_mean_scores", {})
        row["start_on_dataset_deceptive"] = dataset_scores.get("deceptive")
        row["start_on_dataset_honest"] = dataset_scores.get("honest")
        row["start_on_dataset_auc"] = probe.get("on_dataset_auc")

    for row_key, eval_type, suffix in END_PROBE_METRICS:
        row[row_key] = probe_metric(metrics, "final_model", eval_type, suffix)

    row.update(composition_columns(cfg))
    return row


def plot_score_shifts(ax, df, shifts):
    for _, row in df.iterrows():
        r = int(row["round"])
        for start_col, end_col, start_color, end_color in shifts:
            start, end = row[start_col], row[end_col]
            if pd.notna(start) and pd.notna(end):
                ax.plot([r - 1, r], [start, end], color="black", linewidth=2)
                ax.scatter(r - 1, start, color=start_color, s=40, zorder=3)
                ax.scatter(r, end, color=end_color, s=40, zorder=3)


def plot_auc_shifts(ax, df, start_col, end_col, start_color, end_color):
    plot_score_shifts(ax, df, [(start_col, end_col, start_color, end_color)])


def plot_calibrated_score_shift(ax, df, calibrated_metrics, shifts, *, x_start=1, x_end=2):
    round2_row = df.loc[df["round"] == 2].iloc[0]
    for start_col, end_col, start_color, end_color in shifts:
        start = round2_row[start_col]
        eval_type, suffix = PROBE_END_COL_MAP[end_col]
        end = probe_metric(calibrated_metrics, "final_model", eval_type, suffix)
        if pd.notna(start) and end is not None:
            ax.plot([x_start, x_end], [start, end], color="black", linewidth=2, linestyle=":")
            ax.scatter(x_start, start, color=start_color, s=40, zorder=3)
            ax.scatter(x_end, end, color=end_color, s=40, zorder=3)


def add_shift_legend(ax, entries, *, include_grpo=False, include_calibrated=False):
    for color, label in entries:
        ax.scatter([], [], color=color, s=40, label=label)
    if include_grpo:
        ax.plot([], [], color="black", linewidth=2, label="GRPO")
    if include_calibrated:
        ax.plot([], [], color="black", linewidth=2, linestyle=":", label="Calibrated probe")


XLABEL_PAD = 4
XLABEL_PAD_WITH_STEPS = 14


def style_probe_axis(ax, *, title, xlabel=None, x_ticks, x_labels, ylabel="Score", round_df=None):
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=XLABEL_PAD_WITH_STEPS if round_df is not None else XLABEL_PAD)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_ticks, x_labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if round_df is not None:
        annotate_grpo_steps(ax, round_df)


def grouped_bar_offsets(n_groups: int, bar_width: float) -> list[float]:
    return [(i - (n_groups - 1) / 2) * bar_width for i in range(n_groups)]


def plot_task_metric_bars(
    metrics_spec,
    rows_by_x: dict[int, dict[str, float]],
    *,
    bar_width: float,
    title: str,
    x_ticks: list[int],
    x_labels: list,
    ncol: int,
    ceiling_refs: dict[str, tuple[float, float]] | None = None,
    overlay_at_x: tuple[int, dict[str, float], str] | None = None,
    round_df: pd.DataFrame | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    if ceiling_refs:
        color_by_col = {col: color for col, _, color in metrics_spec}
        mean, std = ceiling_refs["correct"]
        color = color_by_col["correct"]
        ax.axhspan(mean - std, mean + std, color=color, alpha=0.2, zorder=0)
        ax.axhline(mean, color=color, linestyle="--", linewidth=1.5, label="Ceiling correct", zorder=1)
    offsets = grouped_bar_offsets(len(metrics_spec), bar_width)
    x_positions = list(rows_by_x)
    for i, (col, label, color) in enumerate(metrics_spec):
        ax.bar(
            [xi + offsets[i] for xi in x_positions],
            [rows_by_x[xi][col] for xi in x_positions],
            width=bar_width,
            color=color,
            label=label,
            zorder=2,
        )
    if overlay_at_x is not None:
        x, overlay_row, overlay_label = overlay_at_x
        for i, (col, _, _color) in enumerate(metrics_spec):
            ax.scatter(
                x + offsets[i],
                overlay_row[col],
                color="black",
                s=30,
                zorder=3,
                label=overlay_label if i == 0 else None,
            )
    ax.set_xlabel("Model", labelpad=XLABEL_PAD_WITH_STEPS if round_df is not None else XLABEL_PAD)
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.set_xticks(x_ticks, x_labels)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", alpha=0.3)
    if round_df is not None:
        annotate_grpo_steps(ax, round_df)
    legend_y = -0.24 if round_df is not None else -0.16
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=ncol, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26 if round_df is not None else 0.18)


def round_axis_labels(round_df: pd.DataFrame) -> list[str]:
    return ["Base"] + [str(int(r)) for r in round_df["round"]]


def annotate_grpo_steps(ax, round_df: pd.DataFrame, *, y: float = -0.09) -> None:
    for _, row in round_df.iterrows():
        step = row.get("grpo_train_step")
        if pd.isna(step):
            continue
        r = int(row["round"])
        ax.text(
            r - 0.5,
            y,
            f"({int(step)} steps)",
            ha="center",
            va="top",
            fontsize=8,
            color="0.45",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )


def task_rows_by_x(base_row: dict[str, float], round_df: pd.DataFrame, *, combined: bool) -> dict[int, dict]:
    rows = {0: combined_outcome_row(base_row) if combined else base_row}
    for _, row in round_df.iterrows():
        categories = {col: row[col] for col in OUTCOME_COLS}
        rows[int(row["round"])] = combined_outcome_row(categories) if combined else categories
    return rows


def outcome_summary_stats(metric_paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [outcome_categories_from_run_metrics(path) for path in metric_paths],
        columns=list(OUTCOME_COLS),
    )


def print_outcome_summary(label: str, metric_paths: list[Path]) -> pd.DataFrame:
    outcome_df = outcome_summary_stats(metric_paths)
    print(f"\n{label}")
    for col in OUTCOME_COLS:
        mean = outcome_df[col].mean()
        std = outcome_df[col].std()
        print(f"  {col}: mean={mean * 100:.2f}%, std={std * 100:.2f}%")
    return outcome_df


def style_table(frame: pd.DataFrame, *, pre_wrap: bool = False):
    """Hide the row index and center column headers; optionally preserve embedded newlines."""
    styler = frame.style.hide(axis="index").set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )
    if pre_wrap:
        styler = styler.set_properties(**{"white-space": "pre-wrap", "text-align": "left"})
    return styler


ROUNDS = [
    {
        "round": round_num,
        "probe_metrics": (round_dir := ROUNDS_DIR / f"round{round_num}") / "probe_metrics.json",
        "probe_train_dataset": round_dir / "probe_train_dataset.json",
        "train_jsonl": buffer_jsonl(round_num, "train"),
        "eval_jsonl": buffer_jsonl(round_num, "eval"),
    }
    for round_num in range(1, NUM_ROUNDS + 1)
]


# %%
rows = [build_row(cfg) for cfg in ROUNDS]
df = pd.DataFrame(rows)

known_cols = set(BASE_COLUMNS + SPLIT_COLUMNS)
composition_cols = sorted(c for c in df.columns if c not in known_cols)
df = df[BASE_COLUMNS + SPLIT_COLUMNS + composition_cols]

for col in SPLIT_COLUMNS + composition_cols:
    df[col] = df[col].astype("Int64")

# Some exported run_metrics store an undefined AUC (degenerate on-policy split) as the
# literal string "NaN" instead of a numeric null; coerce so downstream math/formatting works.
numeric_cols = [c for c in BASE_COLUMNS if c not in ("round", "run_id")]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")
style_table(df)

# %%
outcome_pct = df[["round", "correct", "deceptive", "buggy", "fail"]]
pct_cols = ["correct", "deceptive", "buggy", "fail"]
outcome_pct[pct_cols] = (outcome_pct[pct_cols] * 100).round(1).astype(str) + "%"
style_table(outcome_pct)

# %%
def format_probe_cell(deceptive: float, honest: float, auc: float) -> str:
    return f"Deceptive = {deceptive:.3f}\nHonest = {honest:.3f}\nAUC = {auc:.3f}"


PROBE_SUMMARY_COLS = {
    "On-policy (start)": ("start_on_policy_deceptive", "start_on_policy_honest", "start_on_policy_auc"),
    "On-policy (end)": ("end_on_policy_deceptive", "end_on_policy_honest", "end_on_policy_auc"),
    "On-dataset (start)": ("start_on_dataset_deceptive", "start_on_dataset_honest", "start_on_dataset_auc"),
    "On-dataset (end)": ("end_on_dataset_deceptive", "end_on_dataset_honest", "end_on_dataset_auc"),
}

COMPOSITION_SOURCE_LABELS = {"on_dataset": "on-dataset", "base_model": "base model"}


def format_composition_cell(row: pd.Series) -> str:
    lines = []
    for class_name, display in (("deceptive", "Deceptive"), ("honest", "Honest")):
        prefix = f"{class_name}_"
        sources = [
            col[len(prefix):]
            for col in composition_cols
            if col.startswith(prefix) and pd.notna(row[col])
        ]
        sources.sort(key=lambda s: (s != "on_dataset", s))
        parts = [
            f"{int(row[prefix + source])} {COMPOSITION_SOURCE_LABELS.get(source, source)}"
            for source in sources
        ]
        lines.append(f"{display}: " + ", ".join(parts))
    return "\n".join(lines)


probe_summary = pd.DataFrame({"round": df["round"]})
for col_label, (deceptive_col, honest_col, auc_col) in PROBE_SUMMARY_COLS.items():
    probe_summary[col_label] = df.apply(
        lambda row: format_probe_cell(row[deceptive_col], row[honest_col], row[auc_col]), axis=1
    )
style_table(probe_summary, pre_wrap=True)

# %%
dataset_composition = pd.DataFrame(
    {"round": df["round"], "Dataset Composition": df.apply(format_composition_cell, axis=1)}
)
style_table(dataset_composition, pre_wrap=True)
# %%
DECEPTIVE_START = "#e6550d"
DECEPTIVE_END = "#fdae6b"
HONEST_START = "#2171b5"
HONEST_END = "#6baed6"

models = [0] + df["round"].astype(int).tolist()
model_labels = round_axis_labels(df)
dataset_shifts = [
    ("start_on_dataset_deceptive", "end_on_dataset_deceptive", DECEPTIVE_START, DECEPTIVE_END),
    ("start_on_dataset_honest", "end_on_dataset_honest", HONEST_START, HONEST_END),
]
on_policy_shifts = [
    ("start_on_policy_deceptive", "end_on_policy_deceptive", DECEPTIVE_START, DECEPTIVE_END),
    ("start_on_policy_honest", "end_on_policy_honest", HONEST_START, HONEST_END),
]
shift_legend = [
    (DECEPTIVE_START, "Deceptive start"),
    (DECEPTIVE_END, "Deceptive end"),
    (HONEST_START, "Honest start"),
    (HONEST_END, "Honest end"),
]

fig, (ax_policy, ax_dataset) = plt.subplots(2, 1, figsize=(6, 8))

plot_score_shifts(ax_policy, df, on_policy_shifts)
add_shift_legend(ax_policy, shift_legend, include_grpo=True)
style_probe_axis(
    ax_policy,
    title=f"On-policy probe score shift across rounds ({VARIANT_LABEL})",
    xlabel="Model",
    x_ticks=models,
    x_labels=model_labels,
    round_df=df,
)

plot_score_shifts(ax_dataset, df, dataset_shifts)
add_shift_legend(ax_dataset, shift_legend, include_grpo=True)
style_probe_axis(
    ax_dataset,
    title=f"On-dataset probe score shift across rounds ({VARIANT_LABEL})",
    xlabel="Model",
    x_ticks=models,
    x_labels=model_labels,
    round_df=df,
)

fig.tight_layout()
fig.subplots_adjust(hspace=0.45)

# %%
AUC_START = "#006d2c"
AUC_END = "#74c476"
auc_legend = [(AUC_START, "Start"), (AUC_END, "End")]

fig, (ax_policy_auc, ax_dataset_auc) = plt.subplots(2, 1, figsize=(6, 8))

plot_auc_shifts(
    ax_policy_auc, df, "start_on_policy_auc", "end_on_policy_auc", AUC_START, AUC_END
)
add_shift_legend(ax_policy_auc, auc_legend, include_grpo=True)
style_probe_axis(
    ax_policy_auc,
    title=f"On-policy split probe AUC shift across rounds ({VARIANT_LABEL})",
    xlabel="Model",
    x_ticks=models,
    x_labels=model_labels,
    ylabel="AUC",
    round_df=df,
)

plot_auc_shifts(
    ax_dataset_auc, df, "start_on_dataset_auc", "end_on_dataset_auc", AUC_START, AUC_END
)
add_shift_legend(ax_dataset_auc, auc_legend, include_grpo=True)
style_probe_axis(
    ax_dataset_auc,
    title=f"On-dataset probe AUC shift across rounds ({VARIANT_LABEL})",
    xlabel="Model",
    x_ticks=models,
    x_labels=model_labels,
    ylabel="AUC",
    round_df=df,
)

fig.tight_layout()
fig.subplots_adjust(hspace=0.45)

# %%
base_row = outcome_categories_from_jsonl(BASE_EVAL_JSONL)
round_x = [0] + df["round"].astype(int).tolist()
round_labels = round_axis_labels(df)

plot_task_metric_bars(
    TASK_METRICS,
    task_rows_by_x(base_row, df, combined=False),
    bar_width=0.18,
    title=f"Task metrics across rounds ({VARIANT_LABEL})",
    x_ticks=round_x,
    x_labels=round_labels,
    ncol=4,
    round_df=df,
)

# %%
plot_task_metric_bars(
    COMBINED_TASK_METRICS,
    task_rows_by_x(base_row, df, combined=True),
    bar_width=0.22,
    title=f"Task metrics across rounds ({VARIANT_LABEL})",
    x_ticks=round_x,
    x_labels=round_labels,
    ncol=3,
    round_df=df,
)

# %%
print_outcome_summary("No detector penalty baseline, 1 round (n=3)", DET0_BASELINE_METRICS)
print_outcome_summary("Pass-all reward ceiling (n=3)", DET0_CEILING_METRICS)

ceiling_outcomes = outcome_summary_stats(DET0_CEILING_METRICS)
ceiling_refs = {
    "correct": (
        ceiling_outcomes["correct"].mean(),
        ceiling_outcomes["correct"].std(),
    ),
}

plot_task_metric_bars(
    COMBINED_TASK_METRICS,
    task_rows_by_x(base_row, df, combined=True),
    bar_width=0.22,
    title=f"Task metrics across rounds ({VARIANT_LABEL})",
    x_ticks=round_x,
    x_labels=round_labels,
    ncol=3,
    ceiling_refs=ceiling_refs,
    round_df=df,
)

# %%
calibrated_row = combined_outcome_row(
    outcome_categories_from_run_metrics(ROUND2_CALIBRATED_PENALTY_METRICS)
)
calibrated_metrics = load_json(ROUND2_CALIBRATED_PENALTY_METRICS)["metrics"]

fig, (ax_policy, ax_dataset) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

plot_score_shifts(ax_policy, df, on_policy_shifts)
plot_calibrated_score_shift(ax_policy, df, calibrated_metrics, on_policy_shifts)
add_shift_legend(ax_policy, shift_legend, include_grpo=True, include_calibrated=True)
style_probe_axis(
    ax_policy,
    title=f"On-policy probe score shift across rounds ({VARIANT_LABEL})",
    x_ticks=models,
    x_labels=model_labels,
)

plot_score_shifts(ax_dataset, df, dataset_shifts)
plot_calibrated_score_shift(ax_dataset, df, calibrated_metrics, dataset_shifts)
add_shift_legend(ax_dataset, shift_legend, include_grpo=True, include_calibrated=True)
style_probe_axis(
    ax_dataset,
    title=f"On-dataset probe score shift across rounds ({VARIANT_LABEL})",
    xlabel="Model",
    x_ticks=models,
    x_labels=model_labels,
    round_df=df,
)

fig.tight_layout()
fig.subplots_adjust(bottom=0.10)

plot_task_metric_bars(
    COMBINED_TASK_METRICS,
    task_rows_by_x(base_row, df, combined=True),
    bar_width=0.22,
    title=f"Task metrics across rounds ({VARIANT_LABEL})",
    x_ticks=round_x,
    x_labels=round_labels,
    ncol=4,
    overlay_at_x=(2, calibrated_row, "Calibrated probe"),
    round_df=df,
)

# %%
round1_row = df.loc[df["round"] == 1].iloc[0]
baseline_rows_by_x = {
    0: combined_outcome_row(base_row),
    1: combined_outcome_row({col: round1_row[col] for col in OUTCOME_COLS}),
    2: combined_outcome_row(outcome_categories_from_run_metrics(ROUND2_BASELINE_METRICS)),
}

plot_task_metric_bars(
    COMBINED_TASK_METRICS,
    baseline_rows_by_x,
    bar_width=0.22,
    title=f"Task metrics: no probe update baseline ({VARIANT_LABEL})",
    x_ticks=[0, 1, 2, 3, 4],
    x_labels=["Base", "1", "2", "3", "4"],
    ncol=3,
)

# %%
SAMPLE_ROUND = 1  # int round number, or "base"
SAMPLE_SPLIT = "eval"  # "train" or "eval" (base only has train)
SAMPLE_LABEL = "honest"  # "deceptive" or "honest"
SAMPLE_N = 16

show_buffer_samples(SAMPLE_ROUND, SAMPLE_SPLIT, SAMPLE_LABEL, n=SAMPLE_N)

# %%
