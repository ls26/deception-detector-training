"""Paper figure 3: initial vs. final probe AUC on the shared evaluation set, for
trials 4 / 6 / 8 (Seed 1 / 2 / 3), across the three probe-retraining conditions that
have a round-6 probe (frozen is excluded -- it has no round-6 probe weights):

    early retraining (probe score) | early retraining (AUC) | fixed-schedule retraining

"Final" = probe A (final_round6): trained on the round-6 manifest, scored on
round-5 (final policy) activations.
"Initial" = probe C (initial_round6_data): trained on the SAME round-6 manifest
as A, scored on base-model activations -- the initial-model probe with the
shared training set (probe B, a different/smaller training set, is not used).

Data source: experiments/260814_shared_eval_probe_compare/summary.json (see that
experiment's README.md). Nothing is regenerated or re-run here -- this script only
reads the existing committed summary and plots.

Usage (no GPU needed):
    python /workspace/experiments/paper_figures/make_fig3.py

Outputs (written next to this script):
    fig3_data.csv  -- the tidy long-format table actually plotted
    fig3.png       -- raster, for quick viewing
    fig3.pdf       -- vector, for camera-ready LaTeX inclusion
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

WORKSPACE = Path("/workspace")
OUT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = WORKSPACE / "experiments/260814_shared_eval_probe_compare/summary.json"

PROBE_FINAL = "final_round6"
PROBE_INITIAL = "initial_round6_data"

# Group order/labels match make_fig1.py's CONDITIONS labels exactly (frozen excluded --
# no round-6 probe for that variant). Trial -> variant mapping matches
# paper_fig_common.py / make_fig1.py / 260814_shared_eval_probe_compare/results_table.py.
GROUPS = [
    ("probe_early", "Early retraining (probe score)", {4: "early_retraining_v2", 6: "early_retraining", 8: "early_retraining"}),
    ("auc_early", "Early retraining (AUC)", {4: "early_retraining_auc_v2", 6: "early_retraining_auc_v3", 8: "early_retraining_auc_v3"}),
    ("fixed", "Fixed-schedule retraining", {4: "v2", 6: "", 8: ""}),
]
TRIALS = [("Seed 1", 4), ("Seed 2", 6), ("Seed 3", 8)]

FINAL_COLOR = "#74c476"
INITIAL_COLOR = "#006d2c"

BAR_WIDTH = 0.32
INTRA_PAIR_GAP = 0.04
TRIAL_GAP = 0.3
GROUP_GAP = 0.9


def load_summary() -> dict[tuple[int, str], dict]:
    entries = json.loads(SUMMARY_JSON.read_text())
    return {(e["trial"], e["variant"]): e for e in entries}


def build_long_table() -> pd.DataFrame:
    by_lineage = load_summary()
    records = []
    for group_key, group_label, variant_by_trial in GROUPS:
        for trial_label, trial in TRIALS:
            variant = variant_by_trial[trial]
            entry = by_lineage[(trial, variant or "original")]
            probes = entry["probes"]
            records.append(
                {
                    "group": group_key,
                    "group_label": group_label,
                    "trial_label": trial_label,
                    "trial": trial,
                    "variant": variant or "original",
                    "final_auc": probes[PROBE_FINAL]["auc"],
                    "initial_auc": probes[PROBE_INITIAL]["auc"],
                }
            )
    return pd.DataFrame.from_records(records)


def bar_positions(df: pd.DataFrame):
    """Running-cursor layout: bar_w-wide Final/Initial pairs, grouped by trial,
    grouped again by condition. Returns per-row x positions plus group label spans."""
    rows = []
    group_bounds = []
    x = 0.0
    for group_key, group_label, _ in GROUPS:
        group_start = x
        for trial_label, trial in TRIALS:
            row = df[(df["group"] == group_key) & (df["trial"] == trial)].iloc[0]
            initial_x = x + BAR_WIDTH / 2
            final_x = initial_x + BAR_WIDTH + INTRA_PAIR_GAP
            rows.append(
                {
                    "trial_label": trial_label,
                    "tick_x": (final_x + initial_x) / 2,
                    "final_x": final_x,
                    "initial_x": initial_x,
                    "final_auc": row["final_auc"],
                    "initial_auc": row["initial_auc"],
                }
            )
            x = final_x + BAR_WIDTH / 2 + TRIAL_GAP
        group_end = x - TRIAL_GAP
        group_bounds.append((group_label, group_start, group_end))
        x = group_end + GROUP_GAP
    return rows, group_bounds


def plot(df: pd.DataFrame) -> plt.Figure:
    rows, group_bounds = bar_positions(df)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for row in rows:
        ax.bar(row["initial_x"], row["initial_auc"], width=BAR_WIDTH, color=INITIAL_COLOR, zorder=2)
        ax.bar(row["final_x"], row["final_auc"], width=BAR_WIDTH, color=FINAL_COLOR, zorder=2)

    ax.set_xticks([r["tick_x"] for r in rows], [r["trial_label"] for r in rows])
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    # Group labels below the trial ticks, with a light separator between groups.
    for i, (label, start, end) in enumerate(group_bounds):
        ax.text(
            (start + end) / 2,
            -0.1,
            label,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        if i > 0:
            sep_x = (group_bounds[i - 1][2] + start) / 2
            ax.axvline(sep_x, color="0.85", linewidth=1, zorder=1)

    ax.set_xlim(-BAR_WIDTH, group_bounds[-1][2] + BAR_WIDTH)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=INITIAL_COLOR, label="Initial probe (base activations)"),
        plt.Rectangle((0, 0), 1, 1, color=FINAL_COLOR, label="Final probe (round-5 activations)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout()
    return fig


def main():
    df = build_long_table()
    data_path = OUT_DIR / "fig3_data.csv"
    df.to_csv(data_path, index=False)

    fig = plot(df)
    png_path = OUT_DIR / "fig3.png"
    pdf_path = OUT_DIR / "fig3.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {data_path}\nWrote {png_path}\nWrote {pdf_path}")


if __name__ == "__main__":
    main()
