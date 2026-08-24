"""Poster PDFs for figures 1 and 2a: slightly larger fonts and extra padding.

Reuses the same data/plot logic as make_fig1.py and make_fig2a.py; only styling
and output names differ. No GPU needed.

Usage:
    python /workspace/experiments/paper_figures/make_poster_figures.py

Outputs (written next to this script):
    fig1_poster.pdf
    fig2a_poster.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt

from make_fig1 import DET0_CEILING_METRICS, build_long_table, honest_reward_ceiling, plot as plot_fig1
from make_fig2a import plot as plot_fig2a
from paper_fig_common import (
    POSTER_LEGEND_MARKER_SIZE,
    POSTER_MARKER_SIZE,
    POSTER_FIG1_FIGSIZE,
    POSTER_PAD_INCHES,
    POSTER_ROW_LABEL_FONTSIZE,
    POSTER_SCATTER_SIZE,
    apply_poster_style,
    build_long_table as build_fig2a_table,
)

OUT_DIR = Path(__file__).resolve().parent


def main():
    apply_poster_style()

    df1 = build_long_table()
    fig1 = plot_fig1(
        df1,
        honest_reward_ceiling(DET0_CEILING_METRICS),
        marker_size=POSTER_MARKER_SIZE,
        legend_bbox_y=-0.10,
        figsize=POSTER_FIG1_FIGSIZE,
    )
    fig1_path = OUT_DIR / "fig1_poster.pdf"
    fig1.savefig(fig1_path, bbox_inches="tight", pad_inches=POSTER_PAD_INCHES)
    plt.close(fig1)

    df2a = build_fig2a_table()
    fig2a = plot_fig2a(
        df2a,
        scatter_size=POSTER_SCATTER_SIZE,
        legend_marker_size=POSTER_LEGEND_MARKER_SIZE,
        row_label_fontsize=POSTER_ROW_LABEL_FONTSIZE,
        legend_bbox_y=-0.12,
    )
    fig2a_path = OUT_DIR / "fig2a_poster.pdf"
    fig2a.savefig(fig2a_path, bbox_inches="tight", pad_inches=POSTER_PAD_INCHES)
    plt.close(fig2a)

    print(f"Wrote {fig1_path}\nWrote {fig2a_path}")


if __name__ == "__main__":
    main()
