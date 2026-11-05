"""Figures of the main results over the five rounds, in the style of /workspace/plot_style_guide.ipynb (261031's
plot_results.py with the rounds in place of the penalty weights).

    python plot_results.py     # -> figures/*.png and figures/*.pdf

  fig1_eval_answers  2 x 4 panels. Top: true accuracy with no cue, the cue at the correct answer, the cue at the wrong
                     answer and on the training cue mix. Bottom: the cue agreement rate on the training cue mix, and
                     bias with the cue at the correct answer, at the wrong answer and on the training cue mix. Bars:
                     the base model (hatched and lighter; on the bias rows, its mean over the five rounds' question
                     sets, as in the README), then each round's final checkpoint (tables/rounds/seed_table.json,
                     "mean_sd" "base" and "after_rl", one value per round). Lines, each with its value at the right
                     edge: the mean over seeds 42-44 of the no-penalty runs (dashed; 261024's
                     tables/baseline/seed_table.json) and of the no-cue ceiling (dash-dot; 261020's
                     cot/condition_table.json, "noisy_nocue").
  fig2_training      2 x 5 panels, one column per round, over its 128 training steps. Top: the cue agreement rate on
                     training answers (runs/round<R>/training_curve.csv, "followed_cue"). Bottom: W&B's mean answer
                     length in tokens (runs/dynamics/dynamics.json, "per_step" "response_length_mean").
  fig3_round_probe   1 x 2 panels: round R's probe on the cued eval answers that name the cued option, both cue
                     conditions pooled (probe_table.json). At each round, a line runs from the Initial value (open dot:
                     its own test set, the eval answers of the model the round starts from) to the Final value (filled
                     dot: round R's final-eval answers): the mean probe score of biased and of unbiased answers, and
                     the AUC between them.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = EXP_DIR / "figures"
ROUNDS = ("round1", "round2", "round3", "round4", "round5")
ROUND_LABELS = {"round1": "Round 1", "round2": "Round 2", "round3": "Round 3", "round4": "Round 4", "round5": "Round 5"}
ROUND_TABLE = EXP_DIR / "tables" / "rounds" / "seed_table.json"
NO_PENALTY_TABLE = EXP_DIR.parent / "261024_probe_penalty_rl" / "tables" / "baseline" / "seed_table.json"
CEILING_TABLE = EXP_DIR.parent / "261020_reference_rl" / "cot" / "condition_table.json"  # "noisy_nocue": the ceiling
DYNAMICS = EXP_DIR / "runs" / "dynamics" / "dynamics.json"
PROBE_TABLE = EXP_DIR / "probe_table.json"
STEPS = 128

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

METRIC_COLORS = {  # color = what is measured
    "accuracy": COLORS["green"], "followed_cue": COLORS["rose"], "bias": COLORS["purple"],
    "answer_length": COLORS["slate"],
    "probe_biased": COLORS["orange"], "probe_unbiased": COLORS["blue"], "probe_auc": COLORS["teal"],
}
EVAL_ROWS = (  # top row: true accuracy; bottom row: cue agreement rate and bias
    (("accuracy", "none", "True Accuracy\nNo Cue"),
     ("accuracy", "cue_gold", "True Accuracy\nCue at the Correct Answer"),
     ("accuracy", "cue_dist", "True Accuracy\nCue at the Wrong Answer"),
     ("accuracy", "train_mix", "True Accuracy\nTraining Cue Mix\n(cue at the noisy label 75%)")),
    (("followed_cue", "train_mix", "Cue Agreement Rate\nTraining Cue Mix\n(cue at the noisy label 75%)"),
     ("bias", "cue_gold", "Bias\nCue at the Correct Answer"),
     ("bias", "cue_dist", "Bias\nCue at the Wrong Answer"),
     ("bias", "train_mix", "Bias\nTraining Cue Mix\n(cue at the noisy label 75%)")),
)
BASE_X = 0  # fig1: the base model's bar, then the rounds'
ROUND_X = (1, 2, 3, 4, 5)
BAR_WIDTH = 0.66
BASE_ALPHA = 0.45  # fig1: the base model's bar lighter than the rounds', and hatched
BASE_HATCH = "//"
REFERENCE_LINES = {  # fig1: the means over 3 seeds of two earlier sets of runs, told apart by color and dash pattern
    "no_penalty": {"color": COLORS["ink"], "linestyle": "--",
                   "label": "No penalty: mean of 3 runs trained with the cue, no probe penalty"},
    "ceiling": {"color": COLORS["gray"], "linestyle": "-.",
                "label": "No-cue ceiling: mean of 3 runs trained without the cue"},
}
CURVE_LINE = {"linestyle": "-", "linewidth": 1.6}
DOT_OFFSET = 0.15  # fig3: Initial dot left of the round, Final dot right of it
ROUND_SPACING = 1.5  # fig3: room between rounds for the value labels on either side of each round's pair
DOT_SIZE = 90

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT["tick"],
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "savefig.dpi": EXPORT_DPI,
})


def style_axis(ax, *, title=None, ylabel=None) -> None:
    """The style guide's axis style."""
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


def percent_axis(ax, low: float, high: float, ticks: list[float]) -> None:
    ax.set_ylim(low, high)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def annotate_value(ax, x: float, top: float, value: float) -> None:
    """A bar's value above it, over the reference lines: its box cuts a reference line behind it, so the number
    reads cleanly."""
    ax.annotate(f"{value:.1f}", (x, top), xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=FONT["annotation"], alpha=0.85, zorder=8,
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})


def reference_line(ax, value: float, style: dict, label_above: bool) -> None:
    """A reference run's mean as a horizontal line across the whole panel, its value written at the right edge in the
    line's color, above the line or below it."""
    x_min, x_max = ax.get_xlim()
    ax.plot([x_min, x_max], [value, value], color=style["color"], linestyle=style["linestyle"], linewidth=2.2, zorder=5)
    ax.annotate(f"{value:.1f}", (x_max - 0.02, value), xytext=(0, 3 if label_above else -3), textcoords="offset points",
                ha="right", va="bottom" if label_above else "top", fontsize=FONT["annotation"], fontstyle="italic",
                fontweight="bold", color=style["color"], zorder=6,
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote figures/{name}.png, .pdf")


def eval_panel(ax, rounds: dict, references: dict, metric: str, eval_condition: str, title: str,
               ylabel: str | None) -> None:
    """The base model's bar (hatched, lighter), one bar per round, and the no-penalty runs' and the no-cue ceiling's
    means over their 3 seeds as horizontal lines."""
    row = rounds[metric][eval_condition]
    assert len(row["after_rl"]) == len(ROUNDS), (metric, eval_condition)
    # the base model: one eval; on the bias rows, its mean over the five rounds' question sets (README main table)
    base = 100 * statistics.mean(row["base"])
    ax.bar(BASE_X, base, BAR_WIDTH, color=METRIC_COLORS[metric], alpha=BASE_ALPHA, hatch=BASE_HATCH,
           edgecolor=COLORS["ink"], linewidth=0.7)
    annotate_value(ax, BASE_X, base, base)
    for x, value in zip(ROUND_X, row["after_rl"]):
        ax.bar(x, 100 * value, BAR_WIDTH, color=METRIC_COLORS[metric], edgecolor=COLORS["ink"], linewidth=0.7)
        annotate_value(ax, x, 100 * value, 100 * value)
    ax.set_xticks([BASE_X, *ROUND_X], ["Base\nmodel", *(ROUND_LABELS[r].replace(" ", "\n") for r in ROUNDS)])
    ax.set_xlim(BASE_X - 0.55, ROUND_X[-1] + 1.3)  # room right of the last bar for the lines' values
    means = {}
    for name, table in references.items():
        reference = table[metric][eval_condition]
        assert len(reference["after_rl"]) == 3, (name, metric, eval_condition)
        means[name] = 100 * reference["after_rl_mean"]
    higher = max(means, key=means.get)
    for name, mean in means.items():  # the higher line's value above it, the lower line's below it
        reference_line(ax, mean, REFERENCE_LINES[name], label_above=name == higher)
    style_axis(ax, title=title, ylabel=ylabel)
    percent_axis(ax, 0, 110, [0, 20, 40, 60, 80, 100])  # room for the value labels above bars near 100%


def fig1(rounds: dict, references: dict) -> None:
    fig, axes = plt.subplots(len(EVAL_ROWS), len(EVAL_ROWS[0]), figsize=(22, 12.6), sharey=True)
    for row_axes, panels in zip(axes, EVAL_ROWS):
        for index, (ax, (metric, eval_condition, title)) in enumerate(zip(row_axes, panels)):
            eval_panel(ax, rounds, references, metric, eval_condition, title, "Rate" if index == 0 else None)
    handles = [
        Patch(facecolor="white", edgecolor=COLORS["ink"], linewidth=0.7, hatch=BASE_HATCH, label="Base model"),
        Patch(facecolor="white", edgecolor=COLORS["ink"], linewidth=0.7,
              label="Rounds 1–5: one run (seed 42), each round training on from the one before"),
        *(Line2D([], [], color=style["color"], linestyle=style["linestyle"], linewidth=2.2, label=style["label"])
          for style in REFERENCE_LINES.values()),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=2,
               handlelength=4.5, prop={"size": 12, "weight": "bold"})  # line samples long enough to show the dashes
    fig.suptitle("Eval Answers After Each Round of RL Against the Bias Probe", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig1_eval_answers")


def read_curve(path: Path, column: str) -> list[float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == list(range(STEPS)), path
    return [float(r[column]) for r in rows]


def fig2(dynamics: dict) -> None:
    fig, axes = plt.subplots(2, len(ROUNDS), figsize=(22, 9.5), sharey="row")
    for column, run in enumerate(ROUNDS):
        cue_agreement = [100 * value for value in read_curve(EXP_DIR / "runs" / run / "training_curve.csv", "followed_cue")]
        per_step = dynamics["training"][run]["per_step"]
        assert len(per_step) == STEPS, run
        lengths = [step["response_length_mean"] for step in per_step]
        top, bottom = axes[0, column], axes[1, column]
        top.plot(range(STEPS), cue_agreement, color=METRIC_COLORS["followed_cue"], **CURVE_LINE)
        bottom.plot(range(STEPS), lengths, color=METRIC_COLORS["answer_length"], **CURVE_LINE)
        style_axis(top, title=ROUND_LABELS[run], ylabel="Cue agreement rate" if column == 0 else None)
        percent_axis(top, -2, 102, [0, 20, 40, 60, 80, 100])
        style_axis(bottom, ylabel="Mean answer length, tokens" if column == 0 else None)
        bottom.set_ylim(0, 270)
        bottom.set_xlabel("Training step", fontsize=FONT["label"], fontweight="bold")
        for ax in (top, bottom):
            ax.set_xlim(-3, STEPS + 2)  # the first and last steps clear of the axes
            ax.tick_params(labelleft=True)  # each round's values readable in its own panel
    fig.suptitle("Training Answers in Each Round of RL Against the Bias Probe", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_training")


def shift_line(ax, x_start: float, x_end: float, start_value: float, end_value: float, color: str,
               end_label_dy: float = 0) -> None:
    """A line from the Initial value (open dot) to the Final value (filled dot), each value written beside its dot,
    in the dot's color."""
    ax.plot([x_start, x_end], [start_value, end_value], color=COLORS["ink"], linewidth=2, zorder=3)
    ax.scatter(x_start, start_value, s=DOT_SIZE, facecolor="white", edgecolor=color, linewidth=2.2, zorder=4)
    ax.scatter(x_end, end_value, s=DOT_SIZE, facecolor=color, edgecolor=color, linewidth=2.2, zorder=4)
    for dot_x, value, offset, dy in ((x_start, start_value, -12, 0), (x_end, end_value, 12, end_label_dy)):
        ax.annotate(f"{value:.3f}", (dot_x, value), xytext=(offset, dy), textcoords="offset points",
                    ha="left" if offset > 0 else "right", va="center", fontsize=FONT["annotation"] + 1,
                    fontweight="bold", color=color, zorder=6,
                    bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})  # keeps gridlines off the label


def label_shifts(values: dict[str, float]) -> dict[str, float]:
    """Vertical offsets (points) for the value labels of two dots at one x: when their values are within 0.05, the
    higher label moves up and the lower one down, so they do not cover each other."""
    if len(values) == 1 or abs(max(values.values()) - min(values.values())) >= 0.05:
        return {key: 0 for key in values}
    high = max(values, key=values.get)
    return {key: (7 if key == high else -7) for key in values}


def fig3(probe_table: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.6))
    positions = [ROUND_SPACING * index for index in range(len(ROUNDS))]
    panels = (
        (axes[0], (("mean_biased", METRIC_COLORS["probe_biased"]), ("mean_unbiased", METRIC_COLORS["probe_unbiased"]))),
        (axes[1], (("auc", METRIC_COLORS["probe_auc"]),)),
    )
    for ax, series in panels:
        for x, run in zip(positions, ROUNDS):
            initial = probe_table[run]["scores"]["own_test_set"]
            final = probe_table[run]["scores"]["round_probe_round_answers"]
            end_dy = label_shifts({key: final[key] for key, _ in series})
            for key, color in series:  # biased and unbiased share the x positions
                shift_line(ax, x - DOT_OFFSET, x + DOT_OFFSET, initial[key], final[key], color, end_label_dy=end_dy[key])
        ax.set_xticks(positions, [ROUND_LABELS[run] for run in ROUNDS])
        ax.set_xlim(positions[0] - 0.75, positions[-1] + 0.75)  # room for the outermost value labels
    style_axis(axes[0], title="Mean Probe Score", ylabel="Mean probe score (0 to 1)")
    axes[0].set_ylim(-0.04, 0.85)
    style_axis(axes[1], title="AUC, Biased vs Unbiased Answers", ylabel="AUC")
    axes[1].set_ylim(0.45, 0.95)
    handles = [  # below the panels, off the data
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_biased"], label="Biased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_unbiased"],
               label="Unbiased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Initial: its own test set (eval answers of the model the round starts from)"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=COLORS["ink"],
               label="Final: eval answers of the round's final model"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.07), ncol=2,
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Each Round's Probe at the Start and End of the Round", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "fig3_round_probe")


def main() -> None:
    round_table = json.loads(ROUND_TABLE.read_text())
    assert list(round_table["runs"]) == list(ROUNDS), list(round_table["runs"])  # "after_rl" lists are in this order
    references = {"no_penalty": json.loads(NO_PENALTY_TABLE.read_text())["mean_sd"],
                  "ceiling": json.loads(CEILING_TABLE.read_text())["mean_sd"]["noisy_nocue"]}
    fig1(round_table["mean_sd"], references)
    fig2(json.loads(DYNAMICS.read_text()))
    fig3(json.loads(PROBE_TABLE.read_text()))


if __name__ == "__main__":
    main()
