"""Shared helpers for fig2a/fig2b: on-policy probe score/AUC shift panels for
trials 4 / 6 / 8 (columns A / B / C), comparing two probe-retraining conditions
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

# Column order in the figure: Trial A = trial 4, Trial B = trial 6, Trial C = trial 8,
# matching make_fig1.py's TRIAL_PANELS. Each dict maps condition key -> the variant
# suffix used in supplemented_probe_eval_{suffix}.json for that trial.
TRIAL_PANELS = [
    ("Trial A", 4, {"probe_early": "early_retraining_v2", "frozen": "frozen"}),
    ("Trial B", 6, {"probe_early": "early_retraining", "frozen": "frozen"}),
    ("Trial C", 8, {"probe_early": "early_retraining", "frozen": "frozen"}),
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
        "Trial A",
        4,
        {
            "probe_early": "early_retraining_v2",
            "auc_early": "early_retraining_auc_v2",
            "fixed": "v2",
            "frozen": "frozen",
        },
    ),
    (
        "Trial B",
        6,
        {
            "probe_early": "early_retraining",
            "auc_early": "early_retraining_auc_v3",
            "fixed": "",
            "frozen": "frozen",
        },
    ),
    (
        "Trial C",
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


def build_long_table(trial_panels=TRIAL_PANELS, row_conditions=ROW_CONDITIONS) -> pd.DataFrame:
    records = []
    for panel_label, trial, suffixes in trial_panels:
        for cond_key, cond_label in row_conditions:
            cond_df = load_condition_df(trial, suffixes[cond_key])
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


def style_grid_axis(ax, *, row_idx: int, col_idx: int, n_rows: int, cond_label: str, panel_label: str, ylabel: str):
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
            fontsize=10,
            fontweight="bold",
        )
    else:
        ax.tick_params(labelleft=True)
    if row_idx == n_rows - 1:
        ax.set_xlabel("Round")


def plot_shift_cell(ax, df: pd.DataFrame, shifts) -> None:
    """shifts: list of (start_col, end_col, start_color, end_color)."""
    for _, row in df.iterrows():
        r = int(row["round"])
        for start_col, end_col, start_color, end_color in shifts:
            start, end = row[start_col], row[end_col]
            if pd.notna(start) and pd.notna(end):
                ax.plot([r - 1, r], [start, end], color="black", linewidth=2, zorder=2)
                ax.scatter(r - 1, start, color=start_color, s=40, zorder=3)
                ax.scatter(r, end, color=end_color, s=40, zorder=3)
