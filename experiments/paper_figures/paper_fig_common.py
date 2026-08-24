"""Shared helpers for fig2a/fig2b: on-policy probe score/AUC shift panels for
trials 4 / 6 / 8 (columns Seed 1 / 2 / 3), comparing two probe-retraining conditions
(rows): early retraining (probe score) vs frozen probe.

Reads only the existing `supplemented_probe_eval_{variant}.json` outputs
(see docs/SUPPLEMENTED_PROBE_EVAL.md) -- nothing is regenerated here.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("/workspace")))

from experiments.supplemented_probe_eval.common import results_path

# Column order in the figure: Seed 1 = trial 4, Seed 2 = trial 6, Seed 3 = trial 8,
# matching make_fig1.py's TRIAL_PANELS. Each dict maps condition key -> the variant
# suffix used in supplemented_probe_eval_{suffix}.json for that trial.
TRIAL_PANELS = [
    ("Seed 1", 4, {"probe_early": "early_retraining_v2", "frozen": "frozen"}),
    ("Seed 2", 6, {"probe_early": "early_retraining", "frozen": "frozen"}),
    ("Seed 3", 8, {"probe_early": "early_retraining", "frozen": "frozen"}),
]

# Row order: early retraining (probe score) first, then frozen probe.
ROW_CONDITIONS = [
    ("probe_early", "Early retraining\n(probe score)"),
    ("frozen", "Frozen probe"),
]

# Full 4-condition variant, matching make_fig1.py's TRIAL_PANELS exactly (adds
# "fixed" and "auc_early" alongside "probe_early"/"frozen").
FULL_TRIAL_PANELS = [
    (
        "Seed 1",
        4,
        {
            "probe_early": "early_retraining_v2",
            "auc_early": "early_retraining_auc_v2",
            "fixed": "v2",
            "frozen": "frozen",
        },
    ),
    (
        "Seed 2",
        6,
        {
            "probe_early": "early_retraining",
            "auc_early": "early_retraining_auc_v3",
            "fixed": "",
            "frozen": "frozen",
        },
    ),
    (
        "Seed 3",
        8,
        {
            "probe_early": "early_retraining",
            "auc_early": "early_retraining_auc_v3",
            "fixed": "",
            "frozen": "frozen",
        },
    ),
]

FULL_ROW_CONDITIONS = [
    ("probe_early", "Early retraining\n(probe score)"),
    ("auc_early", "Early retraining\n(AUC)"),
    ("fixed", "Fixed-schedule\nretraining"),
    ("frozen", "Frozen probe"),
]


def load_condition_df(trial: int, variant: str) -> pd.DataFrame:
    """One (trial, variant) lineage's supplemented on-policy start/end metrics, one row per round."""
    data = json.loads(results_path(trial, variant).read_text())
    rows = []
    for round_str, metrics in sorted(data["rounds"].items(), key=lambda x: int(x[0])):
        start = metrics.get("start") or {}
        end = metrics.get("end") or {}
        rows.append(
            {
                "round": int(round_str),
                "start_deceptive": start.get("deceptive"),
                "start_honest": start.get("honest"),
                "start_auc": start.get("auc"),
                "end_deceptive": end.get("deceptive"),
                "end_honest": end.get("honest"),
                "end_auc": end.get("auc"),
            }
        )
    return pd.DataFrame(rows)


def csv_path(trial: int, variant: str) -> Path:
    """Per-round round_analysis CSV for one lineage (same helper as make_fig1.py)."""
    stem = f"round_analysis_trial{trial}" + (f"_{variant}" if variant else "")
    return Path("/workspace") / f"experiments/trials/trial{trial}" / "results" / f"{stem}.csv"


# On-dataset column pairs -> the schema load_condition_df already emits, so the
# on-dataset figures reuse plot_shift_cell and the output CSV layout unchanged.
ON_DATASET_COLUMNS = {
    "start_on_dataset_deceptive": "start_deceptive",
    "start_on_dataset_honest": "start_honest",
    "start_on_dataset_auc": "start_auc",
    "end_on_dataset_deceptive": "end_deceptive",
    "end_on_dataset_honest": "end_honest",
    "end_on_dataset_auc": "end_auc",
}


def load_condition_df_on_dataset(trial: int, variant: str) -> pd.DataFrame:
    """One lineage's start/end probe metrics on the fixed on-dataset reference set.

    Start comes from round{K}/probe_metrics.json (probe K on model_{K-1}), end from that
    round's run_metrics `final_model.on_dataset_averaged` (probe K on model_K); both are
    already joined per round by pipeline/round_analysis.py. See the figure scripts for the
    cross-fit / AUC-definition / frozen-backfill caveats this inherits.
    """
    df = pd.read_csv(csv_path(trial, variant))
    out = df[["round", *ON_DATASET_COLUMNS]].rename(columns=ON_DATASET_COLUMNS)
    return out.sort_values("round").reset_index(drop=True)


def build_long_table(
    trial_panels=TRIAL_PANELS,
    row_conditions=ROW_CONDITIONS,
    loader=load_condition_df,
) -> pd.DataFrame:
    records = []
    for panel_label, trial, suffixes in trial_panels:
        for cond_key, cond_label in row_conditions:
            cond_df = loader(trial, suffixes[cond_key])
            for _, row in cond_df.iterrows():
                records.append(
                    {
                        "panel": panel_label,
                        "trial": trial,
                        "condition": cond_key,
                        "condition_label": cond_label,
                        **row.to_dict(),
                    }
                )
    return pd.DataFrame.from_records(records)


X_TICKS = [0, 1, 2, 3, 4, 5]
X_LABELS = ["Base", "1", "2", "3", "4", "5"]

POSTER_RCPARAMS = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
}
POSTER_FIG1_FIGSIZE = (11, 6.5)
POSTER_PAD_INCHES = 0.35
POSTER_MARKER_SIZE = 6
POSTER_LEGEND_MARKER_SIZE = 8
POSTER_SCATTER_SIZE = 50
POSTER_ROW_LABEL_FONTSIZE = 12


def apply_poster_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(POSTER_RCPARAMS)


def style_grid_axis(
    ax,
    *,
    row_idx: int,
    col_idx: int,
    n_rows: int,
    cond_label: str,
    panel_label: str,
    ylabel: str,
    row_label_fontsize: int = 10,
):
    ax.set_xticks(X_TICKS, X_LABELS)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if row_idx == 0:
        ax.set_title(panel_label)
    if col_idx == 0:
        ax.set_ylabel(ylabel)
        ax.text(
            -0.42,
            0.5,
            cond_label,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=row_label_fontsize,
            fontweight="bold",
        )
    else:
        ax.tick_params(labelleft=True)
    if row_idx == n_rows - 1:
        ax.set_xlabel("Round")


def plot_shift_cell(ax, df: pd.DataFrame, shifts, *, scatter_size: int = 40) -> None:
    """shifts: list of (start_col, end_col, start_color, end_color)."""
    for _, row in df.iterrows():
        r = int(row["round"])
        for start_col, end_col, start_color, end_color in shifts:
            start, end = row[start_col], row[end_col]
            if pd.notna(start) and pd.notna(end):
                ax.plot([r - 1, r], [start, end], color="black", linewidth=2, zorder=2)
                ax.scatter(r - 1, start, color=start_color, s=scatter_size, zorder=3)
                ax.scatter(r, end, color=end_color, s=scatter_size, zorder=3)
