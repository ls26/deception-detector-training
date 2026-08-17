"""Paper figure 6: ON-DATASET probe score shift across GRPO retraining rounds, for
trials 4 / 6 / 8 (columns A / B / C), comparing all four probe-retraining conditions
(rows): early retraining (probe score) | early retraining (AUC) | fixed-schedule
retraining | frozen probe.

The on-dataset counterpart of figure 4 (make_fig2a_full.py). Same panel grid, same
colors, same start/end shift encoding -- but scores are measured on the fixed MBPP
reference set (400 programmatically hardcoded + 400 human-written correct completions,
identical every round and every lineage) instead of the policy's own completions.

Together with figure 7 this is the evidence for the Appendix A.8 / Discussion claim that
evaded probes still separate the reference set while their on-dataset scores compress
within each round -- i.e. evasion is primarily behavioral, with some representation shift.

Data source: experiments/trials/trial{N}/results/round_analysis_trial{N}[_{variant}].csv,
columns start_on_dataset_* / end_on_dataset_*, written by pipeline/round_analysis.py.
Nothing is regenerated or re-run here -- this script only reads existing outputs and plots.

Provenance and caveats (see also the docstring of make_fig2b_ondataset.py):

  start  probe K on model_{K-1}, from rounds*/round{K}/probe_metrics.json
         (on_dataset_mean_scores = mean_scores_layer_averaged)
  end    probe K on model_K, from that round's run_metrics
         (initial_detector.final_model.on_dataset_averaged/mean_score_{positive,negative})

1. Cross-fit: the start uses the pipeline's reporting probe, the end uses the penalty
   probe fit inside the GRPO run -- separate fits of the same recipe on identical data,
   as disclosed in Appendix A.2. Both mean scores are layer-averaged, so unlike the AUC
   panels of figure 7 this figure is aggregation-consistent across the segment.
2. Frozen backfill: frozen lineages train no probe after probe 1, so they have no
   probe_metrics.json for rounds 2-5; round_analysis.backfill_frozen_on_dataset_starts()
   fills round K's start from round K-1's end. Legitimate for a frozen probe (same probe,
   same checkpoint) but it joins the frozen row's segments by construction.
3. Do NOT substitute run_metrics `base_model.on_dataset_averaged` for the start: it is
   re-logged from the shared Phase-0 detector cache computed on plain Llama-3-8B, not on
   the round's starting policy. In the frozen condition, where start-of-round-K is by
   definition end-of-round-(K-1), the two disagree in all 12 cases (~0.97 vs 0.48-0.68).

Round 1 is a shared bootstrap that all four conditions branch from, so its start->end
segment is identical across the four rows within a trial. Figures 4 and 5 behave the same.

Usage (no GPU needed):
    python /workspace/experiments/paper_figures/make_fig2a_ondataset.py

Outputs (written next to this script):
    fig2a_ondataset_data.csv  -- the tidy long-format table actually plotted
    fig2a_ondataset.png       -- raster, for quick viewing
    fig2a_ondataset.pdf       -- vector, for camera-ready LaTeX inclusion
"""

from pathlib import Path

import matplotlib.pyplot as plt

from paper_fig_common import (
    FULL_ROW_CONDITIONS,
    FULL_TRIAL_PANELS,
    build_long_table,
    load_condition_df_on_dataset,
    plot_shift_cell,
    style_grid_axis,
)

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
    df = build_long_table(FULL_TRIAL_PANELS, FULL_ROW_CONDITIONS, load_condition_df_on_dataset)
    data_path = OUT_DIR / "fig2a_ondataset_data.csv"
    df.to_csv(data_path, index=False)

    fig = plot(df)
    png_path = OUT_DIR / "fig2a_ondataset.png"
    pdf_path = OUT_DIR / "fig2a_ondataset.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {data_path}\nWrote {png_path}\nWrote {pdf_path}")


if __name__ == "__main__":
    main()
