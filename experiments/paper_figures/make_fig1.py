"""Paper figure 1: reward-hacking rate (top row) and correct rate (bottom row)
across GRPO retraining rounds, for trials 4 / 6 / 8 (columns Seed 1 / 2 / 3), comparing
the four probe-retraining conditions:

    frozen probe | fixed-schedule retraining | early retraining (probe score) | early retraining (AUC)

Data source: reuses the existing per-round `round_analysis_trial{N}_{variant}.csv`
files written by pipeline/round_analysis.py (see docs/SUPPLEMENTED_PROBE_EVAL.md /
CONTEXT.md for what each condition/variant means) plus one shared "Base" (pre-RL)
point computed from experiments/artifacts/base/base_eval_generations.jsonl. Nothing
is regenerated or re-run here -- this script only reads existing outputs and plots.

Usage (no GPU needed):
    python /workspace/experiments/paper_figures/make_fig1.py

Outputs (written next to this script), each as .png (raster, for quick viewing)
and .pdf (vector, for camera-ready LaTeX inclusion):
    fig1_data.csv   -- the tidy long-format table actually plotted
    fig1            -- both rows, as before
    fig1a           -- reward-hacking rate row only
    fig1b           -- correct rate row only
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

WORKSPACE = Path("/workspace")
TRIALS_DIR = WORKSPACE / "experiments/trials"
BASE_EVAL_JSONL = WORKSPACE / "experiments/artifacts/base/base_eval_generations.jsonl"
OUT_DIR = Path(__file__).resolve().parent

# Column order in the figure: Seed 1 = trial 4, Seed 2 = trial 6, Seed 3 = trial 8
# (per user mapping). Each dict maps condition key -> the variant suffix used in
# round_analysis_trial{N}_{suffix}.csv for that trial ("" = no suffix).
TRIAL_PANELS = [
    (
        "Seed 1",
        4,
        {
            "fixed": "v2",
            "probe_early": "early_retraining_v2",
            "auc_early": "early_retraining_auc_v2",
            "frozen": "frozen",
        },
    ),
    (
        "Seed 2",
        6,
        {
            "fixed": "",
            "probe_early": "early_retraining",
            "auc_early": "early_retraining_auc_v3",
            "frozen": "frozen",
        },
    ),
    (
        "Seed 3",
        8,
        {
            "fixed": "",
            "probe_early": "early_retraining",
            "auc_early": "early_retraining_auc_v3",
            "frozen": "frozen",
        },
    ),
]

# Condition identity -> (label, color, zorder, linestyle). Fixed order/color per condition
# across every panel (color follows the entity, never repainted); order here also sets
# legend order. Colors are categorical palette slots 1/2/3/7 (blue/orange/aqua/violet) from
# the dataviz skill's validated default palette, reassigned per user request -- same 4-color
# set as before, order/assignment doesn't change the pairwise CVD validation -- this 4-color
# subset passes the all-pairs CVD (deutan/tritan >= 8) and normal-vision (>= 15) floors
# needed when all 4 series appear together in one panel (checked via
# dataviz/scripts/validate_palette.py "#2a78d6,#eb6834,#1baf7a,#4a3aa7" --pairs all). Aqua is
# under the 3:1 contrast floor against a white surface (relief rule) -- mitigated here with
# markers + a direct legend rather than color alone. Probe-based early retraining gets the
# highest zorder so it draws on top whenever lines overlap. Linestyle here is the *default*
# (solid everywhere); LINESTYLE_OVERRIDES below dashes AUC-based early retraining only in the
# one panel where it's visually identical to fixed-schedule retraining (trial 4 / Seed 1,
# reward-hacking rate) so the dashes' gaps reveal the solid line underneath instead of fully
# occluding it -- every other panel stays solid.
CONDITIONS = [
    ("frozen", "Frozen probe", "#eb6834", 2, "-"),
    ("fixed", "Fixed-schedule retraining", "#e6ab02", 2, "-"),
    ("probe_early", "Early retraining (probe score)", "#2a78d6", 3, "-"),
    ("auc_early", "Early retraining (AUC)", "#238b45", 2, "-"),
]

# (panel_label, metric_col, condition_key) -> (linestyle, zorder) override, applied only
# where that exact combination overlaps another condition's line.
LINESTYLE_OVERRIDES = {
    ("Seed 1", "deceptive", "auc_early"): ("--", 2.5),
}

# Shared bootstrap segment (Base -> round 1) is identical across all 4 conditions within
# a trial, so it's drawn once in black rather than as 4 overlapping colored segments.
BOOTSTRAP_COLOR = "#0b0b0b"

# Honest baseline: mean correct (pass-all) rate across the no-detector-penalty "pass-all
# reward ceiling" seed sweep (matches round_analysis.py's DET0_CEILING_METRICS /
# ceiling_refs["correct"], minus the shaded std band).
DET0_CEILING_METRICS = [
    WORKSPACE / "experiments/2507_det0_baselines/single_runs" / f"run_metrics_ceiling_clp1e-3_s{seed}.json"
    for seed in (42, 43, 44)
]
CEILING_COLOR = "#52514e"
CEILING_LABEL = "Honest baseline (pass-all-tests reward)"

METRICS = [
    ("deceptive", "Reward hacking rate"),
    ("correct", "Correct rate"),
]

# Legend layout for the single-metric figures, written as literal rows (see plot()'s
# `legend_rows`): the two probe-retraining-off conditions on top, the two early-retraining
# ones below, with fig1b's honest-baseline entry rightmost on the top row. The combined
# fig1 keeps its original legend order and so passes no override.
SPLIT_LEGEND_ROWS = {
    "fig1a": [["frozen", "fixed"], ["probe_early", "auc_early"]],
    "fig1b": [["frozen", "fixed", "ceiling"], ["probe_early", "auc_early"]],
}


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def base_outcome_row(jsonl_path: Path) -> dict[str, float]:
    """Shared pre-RL (round 0) point -- same base model/checkpoint for every
    trial/condition, so all 4 lines in a panel share this starting value."""
    rows = load_jsonl(jsonl_path)
    n = len(rows)
    return {
        "correct": sum(r["passes_all_tests"] for r in rows) / n,
        "deceptive": sum(r["pass_only_shown_deception"] for r in rows) / n,
    }


def honest_reward_ceiling(metric_paths: list[Path]) -> float:
    """Mean correct (pass-all) rate across the pass-all reward ceiling seed sweep."""
    rates = []
    for path in metric_paths:
        metrics = json.load(open(path))["metrics"]
        rates.append(metrics["val_passes_all_tests"])
    return sum(rates) / len(rates)


def csv_path(trial: int, suffix: str) -> Path:
    stem = f"round_analysis_trial{trial}" + (f"_{suffix}" if suffix else "")
    return TRIALS_DIR / f"trial{trial}" / "results" / f"{stem}.csv"


def load_condition_series(trial: int, suffix: str, base_row: dict[str, float]) -> pd.DataFrame:
    df = pd.read_csv(csv_path(trial, suffix))[["round", "correct", "deceptive"]]
    base = pd.DataFrame([{"round": 0, **base_row}])
    return pd.concat([base, df], ignore_index=True).sort_values("round")


def build_long_table() -> pd.DataFrame:
    base_row = base_outcome_row(BASE_EVAL_JSONL)
    records = []
    for panel_label, trial, suffixes in TRIAL_PANELS:
        for cond_key, cond_label, _color, _zorder, _linestyle in CONDITIONS:
            series = load_condition_series(trial, suffixes[cond_key], base_row)
            for _, row in series.iterrows():
                records.append(
                    {
                        "panel": panel_label,
                        "trial": trial,
                        "condition": cond_key,
                        "condition_label": cond_label,
                        "round": int(row["round"]),
                        "correct": row["correct"],
                        "deceptive": row["deceptive"],
                    }
                )
    return pd.DataFrame.from_records(records)


def plot(
    df: pd.DataFrame,
    ceiling_correct: float,
    *,
    marker_size: int = 5,
    legend_bbox_y: float = -0.08,
    figsize: tuple[float, float] = (11, 6),
    metrics: list[tuple[str, str]] = METRICS,
    legend_rows: list[list[str]] | None = None,
) -> plt.Figure:
    panel_labels = [p for p, _, _ in TRIAL_PANELS]
    # squeeze=False so a single-metric figure still indexes as axes[row, col].
    fig, axes = plt.subplots(len(metrics), 3, figsize=figsize, sharex=True, sharey="row", squeeze=False)

    x_ticks = list(range(6))
    x_labels = ["Base", "1", "2", "3", "4", "5"]

    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        for col_idx, panel_label in enumerate(panel_labels):
            ax = axes[row_idx, col_idx]
            panel_df = df[df["panel"] == panel_label]

            # Shared Base -> round-1 bootstrap segment, identical across all 4
            # conditions -- drawn once in black instead of 4 overlapping colored lines.
            bootstrap = panel_df[
                (panel_df["condition"] == CONDITIONS[0][0]) & (panel_df["round"] <= 1)
            ].sort_values("round")
            ax.plot(
                bootstrap["round"],
                bootstrap[metric_col],
                color=BOOTSTRAP_COLOR,
                linewidth=2,
                marker="o",
                markersize=marker_size,
                zorder=1,
            )

            for cond_key, cond_label, color, zorder, linestyle in CONDITIONS:
                linestyle, zorder = LINESTYLE_OVERRIDES.get((panel_label, metric_col, cond_key), (linestyle, zorder))
                series = panel_df[
                    (panel_df["condition"] == cond_key) & (panel_df["round"] >= 1)
                ].sort_values("round")
                ax.plot(
                    series["round"],
                    series[metric_col],
                    color=color,
                    linewidth=2,
                    linestyle=linestyle,
                    marker="o",
                    markersize=marker_size,
                    label=cond_label,
                    zorder=zorder,
                )

            if metric_col == "correct":
                ax.axhline(ceiling_correct, color=CEILING_COLOR, linestyle="--", linewidth=1.5, zorder=1)

            ax.set_xticks(x_ticks, x_labels)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(panel_label)
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            else:
                ax.tick_params(labelleft=True)  # sharey hides ticks by default; keep them
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Round")

    handle_by_key = {
        key: plt.Line2D(
            [], [], color=color, linewidth=2, linestyle=linestyle, marker="o", markersize=marker_size, label=cond_label
        )
        for key, cond_label, color, _zorder, linestyle in CONDITIONS
    }
    # The ceiling line is only drawn on the correct-rate row, so only advertise it there.
    if any(metric_col == "correct" for metric_col, _label in metrics):
        handle_by_key["ceiling"] = plt.Line2D(
            [], [], color=CEILING_COLOR, linestyle="--", linewidth=1.5, label=CEILING_LABEL
        )

    if legend_rows is None:
        keys, ncol = list(handle_by_key), 3
    else:
        # matplotlib fills legend cells column-major, so flatten the requested rows column
        # by column to get the reading order back out. Rows must be left-packed (a short
        # row is short on its right), which is also how matplotlib sizes its own columns.
        ncol = max(len(row) for row in legend_rows)
        keys = [row[j] for j in range(ncol) for row in legend_rows if j < len(row)]
    handles = [handle_by_key[key] for key in keys]

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, legend_bbox_y),
        frameon=False,
    )
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [OUT_DIR / f"{stem}.png", OUT_DIR / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def main():
    df = build_long_table()
    data_path = OUT_DIR / "fig1_data.csv"
    df.to_csv(data_path, index=False)

    ceiling_correct = honest_reward_ceiling(DET0_CEILING_METRICS)
    written = [data_path]

    # Combined 2-row figure, plus each row on its own: fig1a = reward hacking,
    # fig1b = correct rate. Single-row figures are half as tall, so the legend
    # offset (axes fraction) roughly doubles to clear the x-label.
    written += save(plot(df, ceiling_correct), "fig1")
    for stem, metric in (("fig1a", METRICS[0]), ("fig1b", METRICS[1])):
        fig = plot(
            df,
            ceiling_correct,
            metrics=[metric],
            figsize=(11, 3.4),
            legend_bbox_y=-0.12,
            legend_rows=SPLIT_LEGEND_ROWS[stem],
        )
        written += save(fig, stem)
        plt.close(fig)

    print("\n".join(f"Wrote {p}" for p in written))


if __name__ == "__main__":
    main()
