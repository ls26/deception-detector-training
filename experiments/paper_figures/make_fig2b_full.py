"""Paper figure 2b (full): on-policy probe AUC shift across GRPO retraining
rounds, for trials 4 / 6 / 8 (columns Seed 1 / 2 / 3), comparing all four
probe-retraining conditions (rows): early retraining (probe score) | early
retraining (AUC) | fixed-schedule retraining | frozen probe.

Data source: experiments/trials/trial{N}/results/supplemented_probe_eval_{variant}.json
(see docs/SUPPLEMENTED_PROBE_EVAL.md). Nothing is regenerated or re-run here -- this
script only reads existing outputs and plots.

Usage (no GPU needed):
    python /workspace/experiments/paper_figures/make_fig2b_full.py

Outputs (written next to this script):
    fig2b_full_data.csv  -- the tidy long-format table actually plotted
    fig2b_full.png       -- raster, for quick viewing
    fig2b_full.pdf       -- vector, for camera-ready LaTeX inclusion
"""

from pathlib import Path

import matplotlib.pyplot as plt

from paper_fig_common import FULL_ROW_CONDITIONS, FULL_TRIAL_PANELS, build_long_table, plot_shift_cell, style_grid_axis

OUT_DIR = Path(__file__).resolve().parent

AUC_START = "#006d2c"
AUC_END = "#74c476"

SHIFTS = [("start_auc", "end_auc", AUC_START, AUC_END)]


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
                ylabel="AUC",
            )

    legend_handles = [
        plt.Line2D([], [], color="black", linewidth=2, label="Within-round change"),
        plt.Line2D([], [], marker="o", linestyle="", color=AUC_START, markersize=7, label="Start"),
        plt.Line2D([], [], marker="o", linestyle="", color=AUC_END, markersize=7, label="End"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )
    fig.tight_layout()
    return fig


def main():
    df = build_long_table(FULL_TRIAL_PANELS, FULL_ROW_CONDITIONS)
    data_path = OUT_DIR / "fig2b_full_data.csv"
    df.to_csv(data_path, index=False)

    fig = plot(df)
    png_path = OUT_DIR / "fig2b_full.png"
    pdf_path = OUT_DIR / "fig2b_full.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {data_path}\nWrote {png_path}\nWrote {pdf_path}")


if __name__ == "__main__":
    main()
