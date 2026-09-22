"""Paper figure 1 (simplified): seed-averaged reward-hacking rate across GRPO rounds,
comparing frozen probe vs adaptive (probe-score) retraining only.

Reuses the same data pipeline as make_fig1.py — reads existing round_analysis CSVs,
does not re-run experiments.

Usage (no GPU needed):
    python /workspace/experiments/paper_figures/make_fig1_new.py

Outputs (written next to this script):
    fig1_new_data.csv  -- seed-averaged tidy table
    fig1_new.png       -- raster preview
    fig1_new.pdf       -- vector for LaTeX
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from make_fig1 import BOOTSTRAP_COLOR, build_long_table
from paper_fig_common import apply_figure_title

OUT_DIR = Path(__file__).resolve().parent
METRIC_COL = "deceptive"
METRIC_LABEL = "Reward hacking rate"
FIG_TITLE = "Adaptive Probe Retraining Reverses Reward Hacking"

# Subset of make_fig1.CONDITIONS: frozen probe + probe-score early retraining only.
# probe_early label renamed per paper wording.
CONDITIONS = [
    ("frozen", "Frozen probe", "#eb6834", 2),
    ("probe_early", "Adaptive retraining (probe score)", "#2a78d6", 3),
]

X_TICKS = list(range(6))
X_LABELS = ["Base", "1", "2", "3", "4", "5"]


def build_averaged_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean reward-hacking rate across trials 4/6/8 at each (condition, round)."""
    keys = ["condition", "round"]
    cond_labels = dict((k, label) for k, label, *_ in CONDITIONS)
    avg = (
        df[df["condition"].isin(cond_labels)]
        .groupby(keys, as_index=False)[METRIC_COL]
        .mean()
        .rename(columns={METRIC_COL: "deceptive_mean"})
    )
    avg["condition_label"] = avg["condition"].map(cond_labels)
    return avg.sort_values(["condition", "round"]).reset_index(drop=True)


def plot(df: pd.DataFrame, *, marker_size: int = 5) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))

    # Shared Base -> round 1 bootstrap: mean across seeds (round 0 is identical everywhere).
    bootstrap = (
        df[df["condition"] == CONDITIONS[0][0]]
        .sort_values("round")
        .loc[lambda d: d["round"] <= 1]
    )
    ax.plot(
        bootstrap["round"],
        bootstrap["deceptive_mean"],
        color=BOOTSTRAP_COLOR,
        linewidth=2,
        marker="o",
        markersize=marker_size,
        zorder=1,
    )

    for cond_key, cond_label, color, zorder in CONDITIONS:
        series = df[(df["condition"] == cond_key) & (df["round"] >= 1)].sort_values("round")
        ax.plot(
            series["round"],
            series["deceptive_mean"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=marker_size,
            label=cond_label,
            zorder=zorder,
        )

    ax.set_xticks(X_TICKS, X_LABELS)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel(METRIC_LABEL)
    ax.set_xlabel("Round")
    ax.grid(True, alpha=0.3)
    ax.set_title(FIG_TITLE, fontsize=15, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [OUT_DIR / f"{stem}.png", OUT_DIR / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def main():
    raw = build_long_table()
    df = build_averaged_table(raw)

    data_path = OUT_DIR / "fig1_new_data.csv"
    df.to_csv(data_path, index=False)

    fig = plot(df)
    written = [data_path] + save(fig, "fig1_new")
    plt.close(fig)

    print("\n".join(f"Wrote {p}" for p in written))


if __name__ == "__main__":
    main()
