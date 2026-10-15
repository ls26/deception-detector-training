"""Figures of the main results, in the style of /workspace/plot_style_guide.ipynb.

    python plot_results.py     # -> figures/*.png and figures/*.pdf

Reads comparison.json (all eval questions) and training_curve.csv, both written by compare.py.
Rates are shown as percentages.

  fig1_training_cue_mix      grouped bars, training cue mix, base model vs after RL: true accuracy, reward,
                             bias, cue agreement rate, deception, false denial, false admission (soft)
  fig2_deference             grouped bars, base model vs after RL: true accuracy by cue position (no cue,
                             cue at the correct answer, cue at the wrong answer) and cue agreement rate
  fig3_training              lines: reward, true accuracy and cue agreement rate per training step (faint
                             points) and their 10-step means
  fig4_deception_breakdown   training cue mix. Left: share of biased answers and false denial rate; middle:
                             share of unbiased answers and false admission rate; right: deception stacked into
                             the two products. Color = biased / unbiased, hatching = base model (solid = after RL)
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

import noisy_spec as S  # noqa: E402

FIG_DIR = S.EXP_DIR / "figures"

# Design tokens from the style guide
COLORS = {
    "teal": "#5fa89d", "orange": "#d9844f", "blue": "#6b7fa3", "purple": "#a67fb8", "green": "#7fb88a",
    "rose": "#c97b84", "gray": "#888888", "slate": "#708090", "gold": "#d9a066", "ink": "#333333",
}
PANEL_BACKGROUND = "#f5f5f5"
GRID_COLOR = "#b5b5b5"
GRID_ALPHA = 0.60
GRID_LINEWIDTH = 0.9
FONT = {"title": 15, "panel_title": 13, "label": 13, "tick": 11, "annotation": 10}
LINE_WIDTH = 2.7
EXPORT_DPI = 300

MODELS = {"base": "Base model", "after_rl": "After RL"}
MODEL_COLORS = {"base": COLORS["teal"], "after_rl": COLORS["orange"]}  # figures 1-2: color = model
BIASED_COLOR, UNBIASED_COLOR = COLORS["purple"], COLORS["gold"]  # figure 4: color = kind of answer
BASE_HATCH = "///"  # figure 4: hatching = base model, solid = after RL
BAR_GROUP_WIDTH = 0.72

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT["tick"],
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "hatch.color": COLORS["ink"],
    "hatch.linewidth": 0.8,
    "savefig.dpi": EXPORT_DPI,
})


def style_axis(ax, *, title=None, xlabel=None, ylabel=None) -> None:
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
    ax.set_xlabel(xlabel or "", fontsize=FONT["label"], fontweight="bold")
    ax.set_ylabel(ylabel or "", fontsize=FONT["label"], fontweight="bold")
    ax.set_ylim(0, 108)  # room for the value labels above bars near 100%
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def annotate_bar(ax, bar, value: float) -> None:
    ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=FONT["annotation"], alpha=0.8)


def grouped_bars(ax, categories: list[str], values: dict[str, list[float]]) -> None:
    """The style guide's grouped bars: one touching base / after-RL pair per category, color = model."""
    width = BAR_GROUP_WIDTH / len(MODELS)
    for model_idx, model in enumerate(MODELS):
        offset = (model_idx - (len(MODELS) - 1) / 2) * width
        for x, value in enumerate(values[model]):
            bar = ax.bar(x + offset, 100 * value, width, color=MODEL_COLORS[model], edgecolor=COLORS["ink"], linewidth=0.7)[0]
            annotate_bar(ax, bar, 100 * value)
    ax.set_xticks(range(len(categories)), categories, fontweight="bold")


def model_legend(fig, y: float) -> None:
    handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor=COLORS["ink"], linewidth=0.7, label=MODELS[m]) for m in MODELS]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote figures/{name}.png, .pdf")


def fig1(table: dict) -> None:
    metrics = (
        ("accuracy", "True\naccuracy"), ("reward", "Reward"), ("bias", "Bias"), ("followed_cue", "Cue\nagreement"),
        ("deception_soft", "Deception"), ("false_denial_soft", "False\ndenial"), ("false_admission_soft", "False\nadmission"),
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bars(ax, [label for _, label in metrics],
                 {model: [table[model]["train_mix"][key] for key, _ in metrics] for model in MODELS})
    style_axis(ax, ylabel="Rate")
    ax.set_title("Training Cue Mix", fontsize=FONT["title"], fontweight="bold", pad=10)
    model_legend(fig, 0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig1_training_cue_mix")


def fig2(table: dict) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
    accuracy_conditions = (("none", "No cue"), ("cue_gold", "Cue at\ncorrect answer"), ("cue_dist", "Cue at\nwrong answer"))
    grouped_bars(left, [label for _, label in accuracy_conditions],
                 {model: [table[model][c]["accuracy"] for c, _ in accuracy_conditions] for model in MODELS})
    style_axis(left, title="True Accuracy", ylabel="Rate")
    cue_conditions = accuracy_conditions[1:]
    grouped_bars(right, [label for _, label in cue_conditions],
                 {model: [table[model][c]["followed_cue"] for c, _ in cue_conditions] for model in MODELS})
    style_axis(right, title="Cue Agreement Rate")
    right.tick_params(axis="y", labelleft=False)
    fig.suptitle("Accuracy and Cue Agreement by Cue Position", fontsize=FONT["title"], fontweight="bold")
    model_legend(fig, 0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig2_deference")


def fig3() -> None:
    rows = list(csv.DictReader((S.EXP_DIR / "training_curve.csv").open()))
    steps = [int(r["step"]) for r in rows]
    series = (("reward", "Reward", COLORS["blue"]), ("true_accuracy", "True accuracy", COLORS["green"]),
              ("followed_cue", "Cue agreement rate", COLORS["rose"]))
    window = 10
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, color in series:
        values = [100 * float(r[key]) for r in rows]
        ax.scatter(steps, values, s=12, color=color, alpha=0.35, linewidth=0)
        mean_steps = steps[window - 1:]
        means = [sum(values[i - window + 1:i + 1]) / window for i in range(window - 1, len(values))]
        ax.plot(mean_steps, means, color=color, linewidth=LINE_WIDTH, label=label)
    style_axis(ax, xlabel="Training Step", ylabel="Rate")
    ax.set_xlim(0, 128)
    ax.set_title("During RL (Lines = Mean of the Last 10 Steps)", fontsize=FONT["title"], fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, prop={"size": 11, "weight": "bold"})
    fig.tight_layout()
    save(fig, "fig3_training")


def fig4(table: dict) -> None:
    rows = {model: table[model]["train_mix"] for model in MODELS}
    fig, (biased_ax, unbiased_ax, total_ax) = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True,
                                                         gridspec_kw={"width_ratios": [1, 1, 0.9]})
    width = BAR_GROUP_WIDTH / len(MODELS)

    def factor_panel(ax, title: str, color: str, values: dict[str, tuple[float, float]], rate_label: str) -> None:
        for model_idx, model in enumerate(MODELS):
            offset = (model_idx - 0.5) * width
            for x, value in enumerate(values[model]):
                bar = ax.bar(x + offset, 100 * value, width, color=color, edgecolor=COLORS["ink"], linewidth=0.7,
                             hatch=BASE_HATCH if model == "base" else None, alpha=0.72 if model == "base" else 1.0)[0]
                annotate_bar(ax, bar, 100 * value)
        ax.text(0.5, 50, "×", ha="center", va="center", fontsize=26, fontweight="bold", color=COLORS["ink"])
        ax.set_xticks([0, 1], ["Share of\nanswers", rate_label], fontweight="bold")
        style_axis(ax, title=title)

    factor_panel(biased_ax, "Biased Answers", BIASED_COLOR,
                 {m: (r["biased_share"], r["false_denial_soft"]) for m, r in rows.items()}, "False denial\nrate")
    biased_ax.set_ylabel("Rate", fontsize=FONT["label"], fontweight="bold")
    factor_panel(unbiased_ax, "Unbiased Answers", UNBIASED_COLOR,
                 {m: (1 - r["biased_share"], r["false_admission_soft"]) for m, r in rows.items()}, "False admission\nrate")

    # right: deception = the two products, stacked; hatching again marks the base model
    for x, model in enumerate(MODELS):
        row = rows[model]
        bottom = 0.0
        for key, color in (("deception_from_biased", BIASED_COLOR), ("deception_from_unbiased", UNBIASED_COLOR)):
            height = 100 * row[key]
            total_ax.bar(x, height, 0.55, bottom=bottom, color=color, edgecolor=COLORS["ink"], linewidth=0.7,
                         hatch=BASE_HATCH if model == "base" else None, alpha=0.72 if model == "base" else 1.0)
            total_ax.text(x, bottom + height / 2, f"{height:.1f}", ha="center", va="center", fontsize=FONT["annotation"],
                          color=COLORS["ink"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
            bottom += height
        total_ax.annotate(f"{100 * row['deception_soft']:.1f}", (x, bottom), xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=FONT["annotation"] + 1, fontweight="bold")
    total_ax.set_xticks([0, 1], list(MODELS.values()), fontweight="bold")
    total_ax.set_xlim(-0.6, 1.6)
    style_axis(total_ax, title="Deception = Sum of the Products")
    for ax in (unbiased_ax, total_ax):
        ax.tick_params(axis="y", labelleft=False)

    handles = [
        Patch(facecolor="white", edgecolor=COLORS["ink"], hatch=BASE_HATCH, label="Base model"),
        Patch(facecolor="white", edgecolor=COLORS["ink"], label="After RL"),
        Patch(facecolor=BIASED_COLOR, edgecolor=COLORS["ink"], label="Biased answers"),
        Patch(facecolor=UNBIASED_COLOR, edgecolor=COLORS["ink"], label="Unbiased answers"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4,
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Deception on the Training Cue Mix, Broken Down", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig4_deception_breakdown")


def main() -> None:
    table = json.loads((S.EXP_DIR / "comparison.json").read_text())["eval"]["all"]
    fig1(table)
    fig2(table)
    fig3()
    fig4(table)


if __name__ == "__main__":
    main()
