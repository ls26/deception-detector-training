"""Figures of the main results, in the style of /workspace/plot_style_guide.ipynb (started from 261024's
plot_results.py).

    python plot_results.py     # -> figures/*.png and figures/*.pdf

  fig1_eval_answers   2 x 4 panels. Top: true accuracy with no cue, the cue at the correct answer, the cue at the wrong
                      answer and on the training cue mix. Bottom: the cue agreement rate on the training cue mix, and
                      bias with the cue at the correct answer, at the wrong answer and on the training cue mix. Bars:
                      the penalty run with floor and the no-penalty baseline (each the mean of seeds 42-44, with a line
                      for the SD over the seeds, n - 1), then the previous no-floor run (261024's penalty run, seed 42;
                      translucent); the dotted line is the base model, with its value at the right edge. From
                      tables/floor/ and 261024's tables/penalty/ and tables/baseline/seed_table.json ("mean_sd").
  fig2_training       the cue agreement rate on training answers over the 128 training steps (training_curve.csv of
                      each run). Solid: each floor seed; dashed: each baseline seed.
  fig3_initial_probe  1 x 2 panels: the initial probe on the cued answers that name the cued option, both cue
                      conditions pooled (runs/dynamics/dynamics.json, probe rows A and C, "both"). Each line runs from
                      the base model's answers (open dot) to the trained models' (filled dot), each the mean over the
                      three floor runs, with the SD over the seeds (n - 1) at the trained models' dot: the mean probe
                      score of biased and of unbiased answers, and the AUC between them over both letters, among
                      answers naming (A) and among answers naming (B).
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
from matplotlib.ticker import PercentFormatter  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = EXP_DIR / "figures"
EXPERIMENTS = EXP_DIR.parent
E24 = EXPERIMENTS / "261024_probe_penalty_rl"
TABLE_FILES = {"no_floor": E24 / "tables" / "penalty" / "seed_table.json",
               "floor": EXP_DIR / "tables" / "floor" / "seed_table.json",
               "baseline": E24 / "tables" / "baseline" / "seed_table.json"}
DYNAMICS_FILE = EXP_DIR / "runs" / "dynamics" / "dynamics.json"
FLOOR_RUNS = ("floor_s42", "floor_s43", "floor_s44")
TRAINING_CURVES = {
    "floor_s42": EXP_DIR / "runs" / "floor_s42" / "training_curve.csv",
    "floor_s43": EXP_DIR / "runs" / "floor_s43" / "training_curve.csv",
    "floor_s44": EXP_DIR / "runs" / "floor_s44" / "training_curve.csv",
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
    "accuracy": COLORS["green"], "followed_cue": COLORS["rose"], "bias": COLORS["purple"],
    "probe_biased": COLORS["orange"], "probe_unbiased": COLORS["blue"], "probe_auc": COLORS["teal"],
}
RUN_LABELS = {"floor": "Penalty run\nwith floor\n(3 seeds)", "baseline": "No-penalty\nbaseline\n(3 seeds)",
              "no_floor": "Previous\nno-floor run"}
NO_FLOOR_ALPHA = 0.4  # the previous no-floor run's bar is translucent
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
BAR_WIDTH = 0.6
FLOOR_SEED_COLORS = {"floor_s42": "#dc8f98", "floor_s43": "#ae5863", "floor_s44": "#6b2c35"}  # light to dark rose
BASELINE_COLOR = COLORS["gray"]  # figure 2's reference lines, apart from the floor seeds' rose
FLOOR_LINE = {"linestyle": "-", "linewidth": 2.2}
BASELINE_LINE = {"linestyle": "--", "linewidth": 1.2}
DOT_OFFSET = 0.15  # AUC panel: base model dot left of the category, trained model dot right of it
DOT_SIZE = 90
ERRORBAR = {"color": COLORS["ink"], "linewidth": 1.3, "capsize": 5, "capthick": 1.3}

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
    """Each run's bar (with its SD over seeds where it has 3 seeds) and the base model as a dotted line."""
    for x, run in enumerate(RUN_LABELS):
        row = stats[run][metric][eval_condition]
        mean = 100 * row["after_rl_mean"]
        top = mean
        if len(row["after_rl"]) > 1:  # the floor runs' and the baseline's 3 seeds; the no-floor run is one seed
            sd = 100 * row["after_rl_sd"]
            ax.errorbar(x, mean, yerr=sd, zorder=6, **ERRORBAR)  # over the dotted line
            top = mean + sd
        ax.bar(x, mean, BAR_WIDTH, color=METRIC_COLORS[metric], edgecolor=COLORS["ink"], linewidth=0.7,
               alpha=NO_FLOOR_ALPHA if run == "no_floor" else 1.0)
        annotate_value(ax, x, top, mean)
    ax.set_xticks(range(len(RUN_LABELS)), list(RUN_LABELS.values()))
    ax.set_xlim(-0.55, len(RUN_LABELS) - 0.25)  # room right of the last bar for the base value
    base_lines(ax, [100 * stats[run][metric][eval_condition]["base_mean"] for run in RUN_LABELS])
    style_axis(ax, title=title, ylabel=ylabel)
    percent_axis(ax, 0, 108, [0, 20, 40, 60, 80, 100])  # room for the value labels above bars near 100%


def fig1(stats: dict) -> None:
    fig, axes = plt.subplots(len(EVAL_ROWS), len(EVAL_ROWS[0]), figsize=(16, 12.4), sharey=True)
    for row_axes, panels in zip(axes, EVAL_ROWS):
        for index, (ax, (metric, eval_condition, title)) in enumerate(zip(row_axes, panels)):
            eval_panel(ax, stats, metric, eval_condition, title, "Rate" if index == 0 else None)
    # an empty error bar, so the legend shows the same capped line as the bars
    sd_handle = axes[0, 0].errorbar([float("nan")], [float("nan")], yerr=[1.0], fmt="none", ecolor=COLORS["ink"],
                                    elinewidth=1.3, capsize=5, capthick=1.3, label="SD over 3 seeds")
    handles = [sd_handle, Line2D([], [], color=COLORS["ink"], linestyle=":", linewidth=2.2, label="Base model")]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Eval Answers After RL", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig1_eval_answers")


def read_curve(path: Path, column: str) -> list[float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == list(range(STEPS)), path
    return [float(r[column]) for r in rows]


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for run, path in TRAINING_CURVES.items():
        is_floor = run in FLOOR_RUNS
        ax.plot(range(STEPS), [100 * value for value in read_curve(path, "followed_cue")],
                color=FLOOR_SEED_COLORS[run] if is_floor else BASELINE_COLOR,
                zorder=4 if is_floor else 3, **(FLOOR_LINE if is_floor else BASELINE_LINE))
    ax.set_xlim(0, STEPS - 1)
    ax.set_xlabel("Training step", fontsize=FONT["label"], fontweight="bold")
    style_axis(ax, ylabel="Cue agreement rate")
    percent_axis(ax, 48, 102, [50, 60, 70, 80, 90, 100])
    handles = [Line2D([], [], color=FLOOR_SEED_COLORS[run], label=f"Penalty run with floor, seed {run[-2:]}", **FLOOR_LINE)
               for run in FLOOR_RUNS]
    handles.append(Line2D([], [], color=BASELINE_COLOR, label="No-penalty baseline (each of seeds 42–44)", **BASELINE_LINE))
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.075), ncol=2,
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Cue Agreement Rate on Training Answers", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig2_training")


def shift_line(ax, x_start: float, x_end: float, base_value: float, trained_value: float, trained_sd: float,
               color: str) -> None:
    """A line from the base model's value (open dot) to the trained models' mean (filled dot, with the SD over seeds),
    each value written beside its dot. The SD line has the dot's color, so series that share an x position keep apart."""
    ax.plot([x_start, x_end], [base_value, trained_value], color=COLORS["ink"], linewidth=2, zorder=3)
    ax.errorbar(x_end, trained_value, yerr=trained_sd, zorder=3, **{**ERRORBAR, "color": color})
    ax.scatter(x_start, base_value, s=DOT_SIZE, facecolor="white", edgecolor=color, linewidth=2.2, zorder=4)
    ax.scatter(x_end, trained_value, s=DOT_SIZE, facecolor=color, edgecolor=color, linewidth=2.2, zorder=4)
    for dot_x, value, side in ((x_start, base_value, -1), (x_end, trained_value, 1)):
        ax.annotate(f"{value:.3f}", (dot_x, value), xytext=(12 * side, 0), textcoords="offset points",
                    ha="left" if side == 1 else "right", va="center", fontsize=FONT["annotation"] + 1, fontweight="bold",
                    color=color, zorder=6,  # the dot's color, so a label belongs to its line where dots overlap
                    bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})  # keeps gridlines off the label


def mean_over_floor_runs(probe_letters: dict, row: str, value) -> tuple[float, float]:
    """Mean and SD (n - 1) over the three floor runs of one value of a pooled probe row."""
    values = [value(probe_letters[run][row]["both"]) for run in FLOOR_RUNS]
    return statistics.mean(values), statistics.stdev(values)


def fig3(dynamics: dict) -> None:
    probe_letters = dynamics["probe_letters"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [2, 3]})

    # biased and unbiased share the two x positions
    for key, color in (("mean_score_biased", METRIC_COLORS["probe_biased"]),
                       ("mean_score_unbiased", METRIC_COLORS["probe_unbiased"])):
        base_mean, _ = mean_over_floor_runs(probe_letters, "A", lambda split: split[key])
        trained_mean, trained_sd = mean_over_floor_runs(probe_letters, "C", lambda split: split[key])
        shift_line(axes[0], 0, 1, base_mean, trained_mean, trained_sd, color)
    axes[0].set_xticks([0, 1], ["Base model's\nanswers", "Trained models'\nanswers"])
    axes[0].set_xlim(-0.5, 1.5)
    style_axis(axes[0], title="Mean Probe Score", ylabel="Mean probe score")
    axes[0].set_ylim(-0.04, 0.85)
    axes[0].legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_biased"], label="Biased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_unbiased"],
               label="Unbiased answers"),
    ], frameon=False, loc="upper right", prop={"size": 11, "weight": "bold"})

    auc_categories = [
        ("Both letters", lambda split: split["auc"]),
        ("Answers naming (A)", lambda split: split["auc_within_letter"]["A"]),
        ("Answers naming (B)", lambda split: split["auc_within_letter"]["B"]),
    ]
    for x, (_, value) in enumerate(auc_categories):
        base_mean, _ = mean_over_floor_runs(probe_letters, "A", value)
        trained_mean, trained_sd = mean_over_floor_runs(probe_letters, "C", value)
        shift_line(axes[1], x - DOT_OFFSET, x + DOT_OFFSET, base_mean, trained_mean, trained_sd, METRIC_COLORS["probe_auc"])
    axes[1].set_xticks(range(len(auc_categories)), [label for label, _ in auc_categories])
    axes[1].set_xlim(-0.6, len(auc_categories) - 0.4)
    style_axis(axes[1], title="AUC, Biased vs Unbiased Answers", ylabel="AUC")
    axes[1].set_ylim(0.55, 1.0)

    sd_handle = axes[1].errorbar([float("nan")], [float("nan")], yerr=[1.0], fmt="none", ecolor=COLORS["ink"],
                                 elinewidth=1.3, capsize=5, capthick=1.3, label="SD over 3 seeds")
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Base model's answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=COLORS["ink"],
               label="Trained models' answers (penalty run with floor, mean of 3 seeds)"),
        sd_handle,
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
