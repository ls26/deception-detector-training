"""Paper figure 2a (full): on-policy probe score shift across GRPO retraining
rounds, for trials 4 / 6 / 8 (columns A / B / C), comparing all four
probe-retraining conditions (rows): early retraining (probe score) | early
retraining (AUC) | fixed-schedule retraining | frozen probe.

Data source: experiments/trials/trial{N}/results/supplemented_probe_eval_{variant}.json
(see docs/SUPPLEMENTED_PROBE_EVAL.md). Nothing is regenerated or re-run here -- this
script only reads existing outputs and plots.

Usage (no GPU needed):
    python /workspace/experiments/paper_figures/make_fig2a_full.py

Outputs (written next to this script):
    fig2a_full_data.csv  -- the tidy long-format table actually plotted
    fig2a_full.png       -- raster, for quick viewing
    fig2a_full.pdf       -- vector, for camera-ready LaTeX inclusion
"""

from pathlib import Path

import matplotlib.pyplot as plt

from paper_fig_common import FULL_ROW_CONDITIONS, FULL_TRIAL_PANELS, build_long_table, plot_shift_cell, style_grid_axis

OUT_DIR = Path(__file__).resolve().parent

DECEPTIVE_START = "#e6550d"
DECEPTIVE_END = "#fdae6b"
HONEST_START = "#2171b5"
HONEST_END = "#6baed6"

SHIFTS = [
    ("start_deceptive", "end_deceptive", DECEPTIVE_START, DECEPTIVE_END),
    ("start_honest", "end_honest", HONEST_START, HONEST_END),
]


def plot(df) -> plt.Figure:
    panel_labels = [p for p, _, _ in FULL_TRIAL_PANELS]
    n_rows, n_cols = len(FULL_ROW_CONDITIONS), len(panel_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 12.5), sharex=True, sharey="row")

    for row_idx, (cond_key, cond_label) in enumerate(FULL_ROW_CONDITIONS):
        for col_idx, panel_label in enumerate(panel_labels):
            ax = axes[row_idx, col_idx]
            cell_df = df[(df["panel"] == panel_label) & (df["condition"] == cond_key)].sort_values("round")
            plot_shift_cell(ax, cell_df, SHIFTS)
            style_grid_axis(
                ax,
                row_idx=row_idx,
                col_idx=col_idx,
                n_rows=n_rows,
                cond_label=cond_label,
                panel_label=panel_label,
                ylabel="Probe score",
            )

    legend_handles = [
        plt.Line2D([], [], color="black", linewidth=2, label="GRPO"),
        plt.Line2D([], [], marker="o", linestyle="", color=DECEPTIVE_START, markersize=7, label="Deceptive start"),
        plt.Line2D([], [], marker="o", linestyle="", color=DECEPTIVE_END, markersize=7, label="Deceptive end"),
        plt.Line2D([], [], marker="o", linestyle="", color=HONEST_START, markersize=7, label="Honest start"),
        plt.Line2D([], [], marker="o", linestyle="", color=HONEST_END, markersize=7, label="Honest end"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )
    fig.tight_layout()
    return fig


def main():
    df = build_long_table(FULL_TRIAL_PANELS, FULL_ROW_CONDITIONS)
    data_path = OUT_DIR / "fig2a_full_data.csv"
    df.to_csv(data_path, index=False)

    fig = plot(df)
    png_path = OUT_DIR / "fig2a_full.png"
    pdf_path = OUT_DIR / "fig2a_full.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {data_path}\nWrote {png_path}\nWrote {pdf_path}")


if __name__ == "__main__":
    main()
