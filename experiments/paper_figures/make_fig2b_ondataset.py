"""Paper figure 7: ON-DATASET probe AUC shift across GRPO retraining rounds, for trials
4 / 6 / 8 (columns Seed 1 / 2 / 3), comparing all four probe-retraining conditions (rows):
early retraining (probe score) | early retraining (AUC) | fixed-schedule retraining |
frozen probe.

The on-dataset counterpart of figure 5 (make_fig2b_full.py), measured on the fixed MBPP
reference set (400 programmatically hardcoded + 400 human-written correct completions,
identical every round and every lineage) instead of the policy's own completions.

Expect flat lines at the top of the axis. Every one of the 120 start/end values lies in
[0.998, 1.000] -- on-dataset AUC never degrades, in any round of any condition, including
the frozen probe that is thoroughly evaded on-policy. That saturation IS the finding: the
same probes that fail on current on-policy completions still separate the reference set,
so evasion is the policy moving outside the probe's training coverage rather than the
reward-hacking signal disappearing. The y-axis is deliberately left at 0-1, matching
figure 5, so the contrast with the on-policy AUC collapse is read directly off the pair;
a zoomed axis would magnify a 0.002 spread into apparent structure.

Data source: experiments/trials/trial{N}/results/round_analysis_trial{N}[_{variant}].csv,
columns start_on_dataset_auc / end_on_dataset_auc, written by pipeline/round_analysis.py.
Nothing is regenerated or re-run here -- this script only reads existing outputs and plots.

Provenance and caveats:

  start  probe K on model_{K-1}, from rounds*/round{K}/probe_metrics.json
         (on_dataset_auc = auc_filtered = MEAN OF PER-LAYER AUCs; range 0.9977-1.0000)
  end    probe K on model_K, from that round's run_metrics
         (initial_detector.final_model.on_dataset_averaged/AUC_ROC
          = AUC OF THE LAYER-AVERAGED SCORE; range 0.9993-1.0000)

1. AUC definition differs across the segment: the start aggregates per-layer AUCs, the end
   takes the AUC of the layer-averaged score. A start->end shift here is therefore partly
   definitional. It cannot mislead at this scale -- both definitions are >= 0.998 in every
   round of every lineage, which is the only claim the figure supports -- but do not read
   the within-round direction of an individual segment as a measurement. The mean-score
   panels of figure 6 do not have this problem: both sides there are layer-averaged.
2. Cross-fit: the start uses the pipeline's reporting probe, the end uses the penalty probe
   fit inside the GRPO run -- separate fits of the same recipe on identical data, as
   disclosed in Appendix A.2.
3. Frozen backfill: frozen lineages train no probe after probe 1 and so have no
   probe_metrics.json for rounds 2-5; round_analysis.backfill_frozen_on_dataset_starts()
   fills round K's start from round K-1's end.
4. Do NOT substitute run_metrics `base_model.on_dataset_averaged` for the start: it is
   re-logged from the shared Phase-0 detector cache computed on plain Llama-3-8B, not on
   the round's starting policy. In the frozen condition, where start-of-round-K is by
   definition end-of-round-(K-1), the two disagree in all 12 cases.

Round 1 is a shared bootstrap that all four conditions branch from, so its start->end
segment is identical across the four rows within a trial. Figures 4 and 5 behave the same.

Usage (no GPU needed):
    python /workspace/experiments/paper_figures/make_fig2b_ondataset.py

Outputs (written next to this script):
    fig2b_ondataset_data.csv  -- the tidy long-format table actually plotted
    fig2b_ondataset.png       -- raster, for quick viewing
    fig2b_ondataset.pdf       -- vector, for camera-ready LaTeX inclusion
"""

from pathlib import Path

import matplotlib.pyplot as plt

from paper_fig_common import (
    FULL_ROW_CONDITIONS,
    FULL_TRIAL_PANELS,
    apply_figure_title,
    build_long_table,
    load_condition_df_on_dataset,
    plot_shift_cell,
    style_grid_axis,
)

OUT_DIR = Path(__file__).resolve().parent
FIG_TITLE = "On-Dataset Probe AUC (All Conditions)"

AUC_START = "#006d2c"
AUC_END = "#74c476"

SHIFTS = [("start_auc", "end_auc", AUC_START, AUC_END)]


def plot(df) -> plt.Figure:
    panel_labels = [p for p, _, _ in FULL_TRIAL_PANELS]
    n_rows, n_cols = len(FULL_ROW_CONDITIONS), len(panel_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 10), sharex=True, sharey="row")

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
    apply_figure_title(fig, FIG_TITLE)
    return fig


def main():
    df = build_long_table(FULL_TRIAL_PANELS, FULL_ROW_CONDITIONS, load_condition_df_on_dataset)
    data_path = OUT_DIR / "fig2b_ondataset_data.csv"
    df.to_csv(data_path, index=False)

    fig = plot(df)
    png_path = OUT_DIR / "fig2b_ondataset.png"
    pdf_path = OUT_DIR / "fig2b_ondataset.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {data_path}\nWrote {png_path}\nWrote {pdf_path}")


if __name__ == "__main__":
    main()
