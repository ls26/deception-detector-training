"""Figures of the main results, in the style of /workspace/plot_style_guide.ipynb (design tokens copied from 261020's
plot_results.py).

    python plot_results.py     # -> figures/*.png and figures/*.pdf

  fig1_eval_answers   1 x 5 panels: true accuracy with no cue, the cue at the correct answer, the cue at the wrong
                      answer and on the training cue mix, and the cue agreement rate on the training cue mix. Bars:
                      the penalty run (seed 42) and the no-penalty baseline (mean of seeds 42-44, with a line for the
                      SD over the seeds, n - 1); the dotted line is the base model, with its value at the right edge.
                      From tables/penalty/ and tables/baseline/seed_table.json ("mean_sd").
  fig2_training       the cue agreement rate on training answers over the 128 training steps (training_curve.csv of
                      each run). Solid: the penalty run; dashed: each baseline seed.
  fig3_initial_probe  1 x 2 panels: the initial probe on the cued answers that name the cued option, both cue
                      conditions pooled (dynamics.json, probe rows A and C, "both"). Each line runs from the base
                      model's answers (open dot) to the trained model's (filled dot): the mean probe score of biased
                      and of unbiased answers, and the AUC between them over both letters, among answers naming (A)
                      and among answers naming (B).
"""

from __future__ import annotations

import csv
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
TABLE_FILES = {"penalty": EXP_DIR / "tables" / "penalty" / "seed_table.json",
               "baseline": EXP_DIR / "tables" / "baseline" / "seed_table.json"}
DYNAMICS_FILE = EXP_DIR / "runs" / "penalty_s42" / "dynamics.json"
EXPERIMENTS = EXP_DIR.parent
TRAINING_CURVES = {
    "penalty": EXP_DIR / "runs" / "penalty_s42" / "training_curve.csv",
    "baseline_s42": EXPERIMENTS / "261013_noisy_label_rl" / "training_curve.csv",
    "baseline_s43": EXPERIMENTS / "261018_noisy_label_rl_seeds" / "runs" / "cot_s43" / "training_curve.csv",
    "baseline_s44": EXPERIMENTS / "261018_noisy_label_rl_seeds" / "runs" / "cot_s44" / "training_curve.csv",
}
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
    "accuracy": COLORS["green"], "followed_cue": COLORS["rose"], "probe_biased": COLORS["orange"],
    "probe_unbiased": COLORS["blue"], "probe_auc": COLORS["teal"],
}
RUN_LABELS = {"penalty": "Penalty run\n(seed 42)", "baseline": "No-penalty\nbaseline\n(mean of 3 seeds)"}
EVAL_PANELS = (
    ("accuracy", "none", "True Accuracy\nNo Cue"),
    ("accuracy", "cue_gold", "True Accuracy\nCue at the Correct Answer"),
    ("accuracy", "cue_dist", "True Accuracy\nCue at the Wrong Answer"),
    ("accuracy", "train_mix", "True Accuracy\nTraining Cue Mix\n(cue at the noisy label 75%)"),
    ("followed_cue", "train_mix", "Cue Agreement Rate\nTraining Cue Mix\n(cue at the noisy label 75%)"),
)
BAR_WIDTH = 0.6
PENALTY_LINE = {"linestyle": "-", "linewidth": 2.2}
BASELINE_LINE = {"linestyle": "--", "linewidth": 1.2}
DOT_OFFSET = 0.2  # AUC panel: base model dot left of the category, trained model dot right of it
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
    ax.annotate(f"{value:.1f}", (x, top), xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=FONT["annotation"], alpha=0.85, zorder=7,  # above the base model's dotted line and the SD line
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})  # keeps lines off the label


def base_lines(ax, base_values: list[float]) -> None:
    """The base model as a dotted line across the whole panel. If neighboring bars have different base values, the
    line steps halfway between them. The last value is written above its line at the panel's right edge; the others at
    the step, on the side away from the next line."""
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


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote figures/{name}.png, .pdf")


def eval_panel(ax, stats: dict, metric: str, eval_condition: str, title: str, ylabel: str | None) -> None:
    """The penalty run's bar, the baseline's mean with its SD over seeds, and the base model as a dotted line."""
    for x, run in enumerate(RUN_LABELS):
        row = stats[run][metric][eval_condition]
        mean = 100 * row["after_rl_mean"]
        top = mean
        if len(row["after_rl"]) > 1:  # the baseline's 3 seeds; the penalty run is one seed
            sd = 100 * row["after_rl_sd"]
            ax.errorbar(x, mean, yerr=sd, color=COLORS["ink"], linewidth=1.3, capsize=5, capthick=1.3, zorder=6)  # over the dotted line
            top = mean + sd
        ax.bar(x, mean, BAR_WIDTH, color=METRIC_COLORS[metric], edgecolor=COLORS["ink"], linewidth=0.7)
        annotate_value(ax, x, top, mean)
    ax.set_xticks(range(len(RUN_LABELS)), list(RUN_LABELS.values()))
    ax.set_xlim(-0.55, len(RUN_LABELS) - 0.25)  # room right of the last bar for the base value
    base_lines(ax, [100 * stats[run][metric][eval_condition]["base_mean"] for run in RUN_LABELS])
    style_axis(ax, title=title, ylabel=ylabel)
    percent_axis(ax, 0, 108, [0, 20, 40, 60, 80, 100])  # room for the value labels above bars near 100%


def fig1(stats: dict) -> None:
    fig, axes = plt.subplots(1, len(EVAL_PANELS), figsize=(17, 6.2), sharey=True)
    for index, (ax, (metric, eval_condition, title)) in enumerate(zip(axes, EVAL_PANELS)):
        eval_panel(ax, stats, metric, eval_condition, title, "Rate" if index == 0 else None)
    # an empty error bar, so the legend shows the same capped line as the bars
    sd_handle = axes[0].errorbar([float("nan")], [float("nan")], yerr=[1.0], fmt="none", ecolor=COLORS["ink"],
                                 elinewidth=1.3, capsize=5, capthick=1.3, label="SD over the baseline's 3 seeds")
    handles = [sd_handle, Line2D([], [], color=COLORS["ink"], linestyle=":", linewidth=2.2, label="Base model")]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Eval Answers After RL: Penalty Run and No-Penalty Baseline", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig1_eval_answers")


def read_curve(path: Path, column: str) -> list[float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == list(range(STEPS)), path
    return [float(r[column]) for r in rows]


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for run, path in TRAINING_CURVES.items():
        style = PENALTY_LINE if run == "penalty" else BASELINE_LINE
        ax.plot(range(STEPS), [100 * value for value in read_curve(path, "followed_cue")], color=METRIC_COLORS["followed_cue"],
                zorder=4 if run == "penalty" else 3, **style)
    ax.set_xlim(0, STEPS - 1)
    ax.set_xlabel("Training step", fontsize=FONT["label"], fontweight="bold")
    style_axis(ax, ylabel="Cue agreement rate")
    percent_axis(ax, 55, 101, [60, 70, 80, 90, 100])
    handles = [Line2D([], [], color=METRIC_COLORS["followed_cue"], label="Penalty run (seed 42)", **PENALTY_LINE),
               Line2D([], [], color=METRIC_COLORS["followed_cue"], label="No-penalty baseline (each of seeds 42–44)",
                      **BASELINE_LINE)]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Cue Agreement Rate on Training Answers", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig2_training")


def shift_line(ax, x_start: float, x_end: float, base_value: float, trained_value: float, color: str,
               end_label_shift: float = 0) -> None:
    """A line from the base model's value (open dot) to the trained model's (filled dot), each value written beside
    its dot; end_label_shift moves the trained model's label up or down (points), off a neighbouring label."""
    ax.plot([x_start, x_end], [base_value, trained_value], color=COLORS["ink"], linewidth=2, zorder=3)
    ax.scatter(x_start, base_value, s=DOT_SIZE, facecolor="white", edgecolor=color, linewidth=2.2, zorder=4)
    ax.scatter(x_end, trained_value, s=DOT_SIZE, facecolor=color, edgecolor=color, linewidth=2.2, zorder=4)
    for dot_x, value, side, shift in ((x_start, base_value, -1, 0), (x_end, trained_value, 1, end_label_shift)):
        ax.annotate(f"{value:.3f}", (dot_x, value), xytext=(9 * side, shift), textcoords="offset points",
                    ha="left" if side == 1 else "right", va="center", fontsize=FONT["annotation"] + 1, fontweight="bold",
                    color=color, zorder=6,  # the dot's color, so a label belongs to its line where dots overlap
                    bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})  # keeps gridlines off the label


def fig3(dynamics: dict) -> None:
    base, trained = dynamics["probe_letters"]["A"]["both"], dynamics["probe_letters"]["C"]["both"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [2, 3]})

    # biased and unbiased share the two x positions; their trained-model values are both near 0, so the biased
    # label is moved up off the unbiased one
    shift_line(axes[0], 0, 1, base["mean_score_biased"], trained["mean_score_biased"], METRIC_COLORS["probe_biased"], 12)
    shift_line(axes[0], 0, 1, base["mean_score_unbiased"], trained["mean_score_unbiased"], METRIC_COLORS["probe_unbiased"])
    axes[0].set_xticks([0, 1], ["Base model's\nanswers", "Trained model's\nanswers"])
    axes[0].set_xlim(-0.5, 1.5)
    style_axis(axes[0], title="Mean Probe Score", ylabel="Mean probe score")
    axes[0].set_ylim(-0.04, 0.85)
    axes[0].legend(handles=[
        Patch(facecolor=METRIC_COLORS["probe_biased"], edgecolor="none", label="Biased answers"),
        Patch(facecolor=METRIC_COLORS["probe_unbiased"], edgecolor="none", label="Unbiased answers"),
    ], frameon=False, loc="upper right", prop={"size": 11, "weight": "bold"})

    auc_categories = [
        ("Both letters", base["auc"], trained["auc"]),
        ("Answers naming (A)", base["auc_within_letter"]["A"], trained["auc_within_letter"]["A"]),
        ("Answers naming (B)", base["auc_within_letter"]["B"], trained["auc_within_letter"]["B"]),
    ]
    for x, (_, base_value, trained_value) in enumerate(auc_categories):
        shift_line(axes[1], x - DOT_OFFSET, x + DOT_OFFSET, base_value, trained_value, METRIC_COLORS["probe_auc"])
    axes[1].set_xticks(range(len(auc_categories)), [label for label, _, _ in auc_categories])
    axes[1].set_xlim(-0.6, len(auc_categories) - 0.4)
    style_axis(axes[1], title="AUC, Biased vs Unbiased Answers", ylabel="AUC")
    axes[1].set_ylim(0.4, 1.0)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Base model's answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=COLORS["ink"], label="Trained model's answers (penalty run)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Initial Probe on Cued Answers That Name the Cued Option, Both Cue Conditions Pooled",
                 fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig3_initial_probe")


def main() -> None:
    stats = {run: json.loads(path.read_text())["mean_sd"] for run, path in TABLE_FILES.items()}
    dynamics = json.loads(DYNAMICS_FILE.read_text())
    fig1(stats)
    fig2()
    fig3(dynamics)


if __name__ == "__main__":
    main()
