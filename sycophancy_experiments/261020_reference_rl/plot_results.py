"""Figures of the main results, in the style of /workspace/plot_style_guide.ipynb (design tokens copied from 261018's
plot_results.py).

    python plot_results.py     # -> figures/*.png and figures/*.pdf

Reads the condition tables ("mean_sd", written by condition_table.py). Rates are shown as percentages. Each bar is a
training condition's true accuracy after RL, the mean over its seeds, with a vertical line for the SD over the seeds
(n - 1); the dotted line across the panel is the base model, with its value written above it.

  fig1a_accuracy_no_cue  1 x 2 panels (CoT, no-CoT): true accuracy on eval prompts without a cue
  fig1b_accuracy_cued    2 x 3 panels (rows CoT, no-CoT): true accuracy with the cue at the correct answer, at the
                         wrong answer, and on the training cue mix. Conditions 1 and 2, trained without a cue, are scored
                         on the mix of the cued run with the same labels (1 on the baseline's, 2 on condition 3's), so
                         that panel has one base value per mix
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = EXP_DIR / "figures"
TABLE_FILES = {"cot": EXP_DIR / "cot" / "condition_table.json", "nocot": EXP_DIR / "nocot" / "condition_table.json"}

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
METRIC_COLORS = {"accuracy": COLORS["green"]}  # color = what the bar measures
CONDITIONS = {
    "baseline": "0. Noisy, cue\n(previous\nexperiment)",
    "noisy_nocue": "1. Noisy,\nno cue",
    "correct_nocue": "2. Correct,\nno cue",
    "correct_cue": "3. Correct,\nwith cue",
}
CUED_PANELS = {
    "cue_gold": "Cue at the Correct Answer",
    "cue_dist": "Cue at the Wrong Answer",
    "train_mix": "Training Cue Mix\n0, 1: cue at the noisy label 75%\n2, 3: cue at the correct answer 75%",
}
BAR_WIDTH = 0.6

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT["tick"],
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
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


def annotate_value(ax, x: float, top: float, value: float) -> None:
    ax.annotate(f"{value:.1f}", (x, top), xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=FONT["annotation"], alpha=0.85, zorder=6,  # above the base model's dotted line
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})  # keeps lines off the label


def base_lines(ax, base_values: list[float]) -> None:
    """The base model as a dotted line across the whole panel. On the training cue mix the base value changes between
    neighboring bars with different mixes: the line steps there, halfway between the bars. The last value is written
    above its line at the panel's right edge; the others at the step, on the side away from the next line."""
    x_min, x_max = ax.get_xlim()
    groups = []  # (first bar, last bar, base value) of neighboring bars that share a base value
    for x, value in enumerate(base_values):
        if groups and round(groups[-1][2], 1) == round(value, 1):
            groups[-1] = (groups[-1][0], x, groups[-1][2])
        else:
            groups.append((x, x, value))
    for index, (first, last, value) in enumerate(groups):
        is_last = index == len(groups) - 1
        left = x_min if index == 0 else first - 0.5
        right = x_max if is_last else last + 0.5
        ax.plot([left, right], [value, value], linestyle=":", color=COLORS["ink"], linewidth=2.2, zorder=5)
        below = not is_last and groups[index + 1][2] > value  # the next line is higher: write this value under its line
        ax.annotate(f"{value:.1f}", (right - 0.02 if is_last else right, value), xytext=(0, -3 if below else 3),
                    textcoords="offset points", ha="right" if is_last else "center", va="top" if below else "bottom",
                    fontsize=FONT["annotation"], fontstyle="italic", color=COLORS["ink"], zorder=6,
                    bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})


def accuracy_panel(ax, stats: dict, eval_condition: str, title: str | None, ylabel: str | None) -> None:
    """Each training condition's mean over its seeds with its SD, and the base model as a dotted line."""
    for x, condition in enumerate(CONDITIONS):
        row = stats[condition]["accuracy"][eval_condition]
        mean, sd = 100 * row["after_rl_mean"], 100 * row["after_rl_sd"]
        ax.bar(x, mean, BAR_WIDTH, color=METRIC_COLORS["accuracy"], edgecolor=COLORS["ink"], linewidth=0.7)
        ax.errorbar(x, mean, yerr=sd, color=COLORS["ink"], linewidth=1.3, capsize=5, capthick=1.3, zorder=4)
        annotate_value(ax, x, mean + sd, mean)
    ax.set_xticks(range(len(CONDITIONS)), list(CONDITIONS.values()))
    ax.set_xlim(-0.55, len(CONDITIONS) - 0.05)  # room right of the last bar for the base value
    base_lines(ax, [100 * stats[condition]["accuracy"][eval_condition]["base_mean"] for condition in CONDITIONS])
    style_axis(ax, title=title, ylabel=ylabel)


def legend(fig, y: float) -> None:
    # an empty error bar, so the legend shows the same capped line as the bars
    sd_handle = fig.axes[0].errorbar([float("nan")], [float("nan")], yerr=[1.0], fmt="none", ecolor=COLORS["ink"],
                                     elinewidth=1.3, capsize=5, capthick=1.3, label="SD over the 3 seeds")
    handles = [
        Patch(facecolor=METRIC_COLORS["accuracy"], edgecolor=COLORS["ink"], label="After RL, mean of 3 seeds"),
        sd_handle,
        Line2D([], [], color=COLORS["ink"], linestyle=":", linewidth=2.2, label="Base model"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote figures/{name}.png, .pdf")


def fig1a(stats: dict) -> None:
    fig, axes = plt.subplots(1, len(SETUPS), figsize=(13, 5.8), sharey=True)
    for column, setup in enumerate(SETUPS):
        accuracy_panel(axes[column], stats[setup], "none", SETUPS[setup], "True accuracy" if column == 0 else None)
        axes[column].set_xlabel("Training condition", fontsize=FONT["label"], fontweight="bold")
    fig.suptitle("True Accuracy on Eval Prompts Without a Cue", fontsize=FONT["title"], fontweight="bold")
    legend(fig, 0.05)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig1a_accuracy_no_cue")


def fig1b(stats: dict) -> None:
    fig, axes = plt.subplots(len(SETUPS), len(CUED_PANELS), figsize=(19, 11), sharey=True)
    for row, setup in enumerate(SETUPS):
        for column, (eval_condition, title) in enumerate(CUED_PANELS.items()):
            ax = axes[row, column]
            accuracy_panel(ax, stats[setup], eval_condition, title if row == 0 else None,
                           f"{SETUPS[setup]}\nTrue accuracy" if column == 0 else None)
            if row == len(SETUPS) - 1:
                ax.set_xlabel("Training condition", fontsize=FONT["label"], fontweight="bold")
    fig.suptitle("True Accuracy on Eval Prompts With a Cue", fontsize=FONT["title"], fontweight="bold")
    legend(fig, 0.035)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=2.0)
    save(fig, "fig1b_accuracy_cued")


def main() -> None:
    stats = {setup: json.loads(path.read_text())["mean_sd"] for setup, path in TABLE_FILES.items()}
    fig1a(stats)
    fig1b(stats)


if __name__ == "__main__":
    main()
