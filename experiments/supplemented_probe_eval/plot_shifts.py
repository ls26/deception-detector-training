#!/usr/bin/env python3
"""Plot supplemented on-policy probe score/AUC shift graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path("/workspace")))

from experiments.supplemented_probe_eval.common import (
    resolve_targets,
    results_path,
    trial_dir,
    variant_label,
)

DECEPTIVE_START = "#e6550d"
DECEPTIVE_END = "#fdae6b"
HONEST_START = "#2171b5"
HONEST_END = "#6baed6"
AUC_START = "#006d2c"
AUC_END = "#74c476"


def load_results(trial: int, variant: str) -> dict:
    path = results_path(trial, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}")
    return json.loads(path.read_text())


def to_dataframe(data: dict) -> pd.DataFrame:
    rows = []
    for round_str, metrics in sorted(data["rounds"].items(), key=lambda x: int(x[0])):
        row = {"round": int(round_str)}
        start = metrics.get("start") or {}
        end = metrics.get("end") or {}
        row["start_on_policy_deceptive"] = start.get("deceptive")
        row["start_on_policy_honest"] = start.get("honest")
        row["start_on_policy_auc"] = start.get("auc")
        row["end_on_policy_deceptive"] = end.get("deceptive")
        row["end_on_policy_honest"] = end.get("honest")
        row["end_on_policy_auc"] = end.get("auc")
        rows.append(row)
    return pd.DataFrame(rows)


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


def style_axis(ax, *, title, xlabel, x_ticks, x_labels, ylabel="Score", round_df=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x_ticks, x_labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)


FIGSIZE = (6, 4)  # matches one panel of round_analysis.py's (6, 8) probe-shift figures

def plot_lineage(trial: int, variant: str, *, show: bool = False) -> tuple[Path, Path]:
    data = load_results(trial, variant)
    df = to_dataframe(data)
    label = variant_label(variant)
    suffix = f"_{variant}" if variant else ""
    out_dir = trial_dir(trial) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [0] + df["round"].astype(int).tolist()
    model_labels = ["Base"] + [str(int(r)) for r in df["round"]]
    on_policy_shifts = [
        ("start_on_policy_deceptive", "end_on_policy_deceptive", DECEPTIVE_START, DECEPTIVE_END),
        ("start_on_policy_honest", "end_on_policy_honest", HONEST_START, HONEST_END),
    ]
    shift_legend = [
        (DECEPTIVE_START, "Deceptive start"),
        (DECEPTIVE_END, "Deceptive end"),
        (HONEST_START, "Honest start"),
        (HONEST_END, "Honest end"),
        ("black", "GRPO"),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_score_shifts(ax, df, on_policy_shifts)
    for color, lbl in shift_legend:
        if color == "black":
            ax.plot([], [], color="black", linewidth=2, label=lbl)
        else:
            ax.scatter([], [], color=color, s=40, label=lbl)
    style_axis(
        ax,
        title=f"On-policy probe score shift ({label})",
        xlabel="Model",
        x_ticks=models,
        x_labels=model_labels,
        round_df=df,
    )
    score_path = out_dir / f"supplemented_on_policy_score_shift_trial{trial}{suffix}.png"
    fig.tight_layout()
    fig.savefig(score_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_auc_shifts(ax, df, "start_on_policy_auc", "end_on_policy_auc", AUC_START, AUC_END)
    ax.scatter([], [], color=AUC_START, s=40, label="Start")
    ax.scatter([], [], color=AUC_END, s=40, label="End")
    ax.plot([], [], color="black", linewidth=2, label="GRPO")
    style_axis(
        ax,
        title=f"On-policy probe AUC shift ({label})",
        xlabel="Model",
        x_ticks=models,
        x_labels=model_labels,
        ylabel="AUC",
        round_df=df,
    )
    auc_path = out_dir / f"supplemented_on_policy_auc_shift_trial{trial}{suffix}.png"
    fig.tight_layout()
    fig.savefig(auc_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    print(f"Wrote {score_path}")
    print(f"Wrote {auc_path}")
    return score_path, auc_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, action="append", dest="trials")
    parser.add_argument("--variant", type=str, action="append", dest="variants")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    targets = resolve_targets(trials=args.trials, variants=args.variants)

    for trial, variant in targets:
        plot_lineage(trial, variant, show=args.show)


if __name__ == "__main__":
    main()
