"""Figures of the main results, in the style of /workspace/plot_style_guide.ipynb.

    python plot_results.py     # -> figures/*.png and figures/*.pdf

Reads seed_table.json ("mean_sd", written by seed_table.py). Rates are shown as percentages. Every panel has
two groups, CoT and no-CoT, each with the base model (hatched) and the three seeds after RL (solid). On bias and
the self-report rows the base bar is the mean of the seeds' base values (they differ by at most 0.007, because
each seed counts its own questions). Color = what the bar measures.

  fig1_answers_and_reward   2 x 3 panels. Top: cue agreement rate, bias and reward on the training cue mix.
                            Bottom: true accuracy with the cue at the correct answer, at the wrong answer, and on
                            the training cue mix
  fig2_self_report          1 x 3 panels, training cue mix: average P(Yes) on biased answers, on unbiased answers,
                            and deception
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = EXP_DIR / "figures"

# Design tokens from the style guide
COLORS = {
    "teal": "#5fa89d", "orange": "#d9844f", "blue": "#6b7fa3", "purple": "#a67fb8", "green": "#7fb88a",
    "rose": "#c97b84", "gray": "#888888", "slate": "#708090", "gold": "#d9a066", "ink": "#333333",
}
PANEL_BACKGROUND = "#f5f5f5"
GRID_COLOR = "#b5b5b5"
GRID_ALPHA = 0.60
GRID_LINEWIDTH = 0.9
FONT = {"title": 15, "panel_title": 13, "label": 13, "tick": 11, "annotation": 9}
EXPORT_DPI = 300

SETUPS = {"cot": "CoT", "nocot": "No-CoT"}
SEEDS = ("42", "43", "44")
BASE_HATCH = "///"  # hatching = base model, solid = after RL
METRIC_COLORS = {  # color = what the bar measures
    "followed_cue": COLORS["rose"], "bias": COLORS["orange"], "reward": COLORS["blue"], "accuracy": COLORS["green"],
    "said_yes_biased_soft": COLORS["purple"], "false_admission_soft": COLORS["gold"], "deception_soft": COLORS["slate"],
}
GROUP_GAP = 1.0  # empty bar slot between the CoT and no-CoT groups

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT["tick"],
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "hatch.color": COLORS["ink"],
    "hatch.linewidth": 0.8,
    "savefig.dpi": EXPORT_DPI,
})


def style_axis(ax, *, title=None, ylabel=None) -> None:
    """The style guide's axis style, with rates as percentages from 0 to 100."""
    ax.set_facecolor(PANEL_BACKGROUND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["ink"])
        ax.spines[side].set_linewidth(1.4)
    ax.grid(True, axis="y", linestyle="--", color=GRID_COLOR, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=FONT["tick"], length=3, pad=3)
    if title:
        ax.set_title(title, fontsize=FONT["panel_title"], fontweight="bold", pad=10)
    ax.set_ylabel(ylabel or "", fontsize=FONT["label"], fontweight="bold")
    ax.set_ylim(0, 108)  # room for the value labels above bars near 100%
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def annotate_bar(ax, bar, value: float) -> None:
    ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=FONT["annotation"], alpha=0.85,
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})  # keeps gridlines off the label


def seed_panel(ax, stats: dict, metric: str, condition: str, title: str, ylabel: str | None = None) -> None:
    """Base model and the three seeds after RL, touching bars per setup, the two setups side by side."""
    color = METRIC_COLORS[metric]
    tick_positions, tick_labels = [], []
    for group_index, setup in enumerate(SETUPS):
        row = stats[setup][metric][condition]
        start = group_index * (1 + len(SEEDS) + GROUP_GAP)
        values = [row["base_mean"], *row["after_rl"]]
        for offset, value in enumerate(values):
            is_base = offset == 0
            bar = ax.bar(start + offset, 100 * value, 1.0, color=color, edgecolor=COLORS["ink"], linewidth=0.7,
                         hatch=BASE_HATCH if is_base else None, alpha=0.72 if is_base else 1.0)[0]
            annotate_bar(ax, bar, 100 * value)
        tick_positions += [start + offset for offset in range(len(values))]
        tick_labels += ["Base", *SEEDS]
        ax.text(start + len(SEEDS) / 2, -0.14, SETUPS[setup], transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=FONT["label"], fontweight="bold")
    ax.set_xticks(tick_positions, tick_labels)
    style_axis(ax, title=title, ylabel=ylabel)


def legend(fig, y: float) -> None:
    handles = [
        Patch(facecolor="white", edgecolor=COLORS["ink"], hatch=BASE_HATCH, label="Base model"),
        Patch(facecolor="white", edgecolor=COLORS["ink"], label="After RL (seed 42 / 43 / 44)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote figures/{name}.png, .pdf")


def fig1(stats: dict) -> None:
    panels = (
        ("followed_cue", "train_mix", "Cue Agreement Rate\nTraining Cue Mix"),
        ("bias", "train_mix", "Bias\nTraining Cue Mix"),
        ("reward", "train_mix", "Reward\nTraining Cue Mix"),
        ("accuracy", "cue_gold", "True Accuracy\nCue at Correct Answer"),
        ("accuracy", "cue_dist", "True Accuracy\nCue at Wrong Answer"),
        ("accuracy", "train_mix", "True Accuracy\nTraining Cue Mix"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    for index, (ax, (metric, condition, title)) in enumerate(zip(axes.flat, panels)):
        seed_panel(ax, stats, metric, condition, title, ylabel="Rate" if index % 3 == 0 else None)
    fig.suptitle("Answers and Reward: Base Model and Each Seed After RL", fontsize=FONT["title"], fontweight="bold")
    legend(fig, 0.04)  # right under the bottom row's group labels, which tight_layout keeps above 0.04
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.0)
    save(fig, "fig1_answers_and_reward")


def fig2(stats: dict) -> None:
    panels = (
        ("said_yes_biased_soft", "Average P(Yes)\nBiased Answers"),
        ("false_admission_soft", "Average P(Yes)\nUnbiased Answers"),
        ("deception_soft", "Deception\n(P(No) on Biased, P(Yes) on Unbiased)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    for index, (ax, (metric, title)) in enumerate(zip(axes, panels)):
        seed_panel(ax, stats, metric, "train_mix", title, ylabel="Average probability" if index == 0 else None)
    fig.suptitle("Self-Report and Deception on the Training Cue Mix", fontsize=FONT["title"], fontweight="bold")
    legend(fig, 0.07)  # right under the group labels, which tight_layout keeps above 0.07
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig2_self_report")


def main() -> None:
    stats = json.loads((EXP_DIR / "seed_table.json").read_text())["mean_sd"]
    fig1(stats)
    fig2(stats)


if __name__ == "__main__":
    main()
