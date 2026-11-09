"""Figures of the main results over the five rounds of the seed-42 and seed-43 lines, in the style of
/workspace/plot_style_guide.ipynb (261102's plot_results.py, its figures 2 and 3 swapped; seeds drawn as in 261103's).

    python plot_results.py     # -> figures/*.png and figures/*.pdf

  fig1_eval_answers  2 x 4 panels. Top: true accuracy with no cue, the cue at the correct answer, the cue at the wrong
                     answer and on the training cue mix. Bottom: the cue agreement rate on the training cue mix, and
                     bias with the cue at the correct answer, at the wrong answer and on the training cue mix. Bars:
                     the base model (hatched and lighter; on the bias rows, its mean over the two lines' ten question
                     sets, as in the README), then each round's final checkpoint, the mean over the two lines with a
                     dot per line (seed_4X/tables/rounds/seed_table.json, "mean_sd" "base" and "after_rl"). Lines, each
                     with its value at the right edge: the mean over seeds 42-44 of 261103's no-penalty runs at KL 0.01
                     (dashed; no_penalty/tables/kl_no_penalty/seed_table.json) and of the no-cue ceiling (dash-dot;
                     261020's cot/condition_table.json, "noisy_nocue").
  fig2_round_probe   2 x 2 panels, one row per line: round R's probe on the cued eval answers that name the cued option,
                     both cue conditions pooled (seed_4X/probe_table.json): the mean probe score of biased and of
                     unbiased answers, and the AUC between them. The x axis is the model whose eval answers are scored.
                     Round R's probe is a line from the Initial value (open dot: its own test set, the eval answers of
                     the model the round starts from) to the Final value (filled dot: round R's final-eval answers), so
                     round R's open dot sits at the same x as round R-1's filled dot: two probes on the same answers.
  fig3_training      2 x 5 panels, one column per round, over its 128 training steps, one line per line of rounds
                     (seed 42 lighter and thinner, on top). Top: the cue agreement rate on training answers
                     (training_curve.csv, "followed_cue": round 1's in 261103's weight_1_0/runs/kl_weight_1_0_s4X/,
                     rounds 2-5's in seed_4X/runs/s4X_round<R>/). Bottom: W&B's mean answer length in tokens
                     (seed_4X/runs/dynamics/dynamics.json, "per_step" "response_length_mean").
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
SEEDS = (42, 43)
LINE_DIRS = {seed: EXP_DIR / f"seed_{seed}" for seed in SEEDS}
ROUNDS = ("round1", "round2", "round3", "round4", "round5")
ROUND_LABELS = {"round1": "Round 1", "round2": "Round 2", "round3": "Round 3", "round4": "Round 4", "round5": "Round 5"}
ROUND_TABLES = {seed: LINE_DIRS[seed] / "tables" / "rounds" / "seed_table.json" for seed in SEEDS}
NO_PENALTY_TABLE = EXP_DIR.parent / "261103_kl_0_01" / "no_penalty" / "tables" / "kl_no_penalty" / "seed_table.json"
CEILING_TABLE = EXP_DIR.parent / "261020_reference_rl" / "cot" / "condition_table.json"  # "noisy_nocue": the ceiling
PROBE_TABLES = {seed: LINE_DIRS[seed] / "probe_table.json" for seed in SEEDS}
DYNAMICS = {seed: LINE_DIRS[seed] / "runs" / "dynamics" / "dynamics.json" for seed in SEEDS}
TRAINING_RUNS = {  # fig3: per line, each round's training_curve.csv and its run name in dynamics.json
    seed: {
        "round1": (EXP_DIR.parent / "261103_kl_0_01" / "weight_1_0" / "runs" / f"kl_weight_1_0_s{seed}" / "training_curve.csv",
                   f"kl_weight_1_0_s{seed}"),
        **{f"round{r}": (LINE_DIRS[seed] / "runs" / f"s{seed}_round{r}" / "training_curve.csv", f"s{seed}_round{r}")
           for r in (2, 3, 4, 5)},
    }
    for seed in SEEDS
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
SEED_DOT_OFFSETS = (-0.14, 0.14)  # fig1: seed 42 left, seed 43 right on each round's bar
SEED_DOT = {"s": 22, "color": COLORS["ink"], "edgecolor": "white", "linewidth": 0.8, "zorder": 7, "clip_on": False}
REFERENCE_LINES = {  # fig1: the means over 3 seeds of two earlier sets of runs, told apart by color and dash pattern
    "no_penalty": {"color": COLORS["ink"], "linestyle": "--",
                   "label": "No penalty: mean of 3 runs trained with the cue, no probe penalty (KL 0.01)"},
    "ceiling": {"color": COLORS["gray"], "linestyle": "-.",
                "label": "No-cue ceiling: mean of 3 runs trained without the cue (KL 0.001)"},
}
DOT_SIZE = 90
SEED_CURVES = {  # fig3: per line, a lighter and darker shade of the measure's color; seed 42 thinner, drawn on top
    42: {"followed_cue": "#e2b0b6", "answer_length": "#a7b3bf", "linewidth": 1.3},
    43: {"followed_cue": "#9a4a55", "answer_length": "#46525e", "linewidth": 2.2},
}

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
    """A bar's value above it (above its highest seed dot), over the reference lines: its box cuts a reference line
    behind it, so the number reads cleanly."""
    ax.annotate(f"{value:.1f}", (x, top), xytext=(0, 5), textcoords="offset points", ha="center",
                fontsize=FONT["annotation"], zorder=8,
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})


def reference_line(ax, value: float, style: dict, label_above: bool) -> None:
    """A reference run's mean as a horizontal line across the whole panel, its value written at the right edge in the
    line's color, above the line or below it."""
    x_min, x_max = ax.get_xlim()
    ax.plot([x_min, x_max], [value, value], color=style["color"], linestyle=style["linestyle"], linewidth=2.2, zorder=5)
    ax.annotate(f"{value:.1f}", (x_max - 0.1, value), xytext=(0, 3 if label_above else -3), textcoords="offset points",
                ha="right", va="bottom" if label_above else "top", fontsize=FONT["annotation"], fontstyle="italic",
                fontweight="bold", color=style["color"], zorder=6,
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote figures/{name}.png, .pdf")


def eval_panel(ax, round_tables: dict, references: dict, metric: str, eval_condition: str, title: str,
               ylabel: str | None) -> None:
    """The base model's bar (hatched, lighter), one bar per round (the mean over the two lines, with a dot per line),
    and the no-penalty runs' and the no-cue ceiling's means over their 3 seeds as horizontal lines."""
    rows = {seed: round_tables[seed][metric][eval_condition] for seed in SEEDS}
    for seed, row in rows.items():
        assert len(row["after_rl"]) == len(ROUNDS), (seed, metric, eval_condition)
    # the base model: one eval; on the bias rows, its mean over the two lines' ten question sets (README main table)
    base_values = [value for row in rows.values() for value in (row["base"] if isinstance(row["base"], list) else [row["base"]])]
    base = 100 * statistics.mean(base_values)
    ax.bar(BASE_X, base, BAR_WIDTH, color=METRIC_COLORS[metric], alpha=BASE_ALPHA, hatch=BASE_HATCH,
           edgecolor=COLORS["ink"], linewidth=0.7)
    annotate_value(ax, BASE_X, base, base)
    for index, x in enumerate(ROUND_X):
        seed_values = [100 * rows[seed]["after_rl"][index] for seed in SEEDS]
        mean = statistics.mean(seed_values)
        ax.bar(x, mean, BAR_WIDTH, color=METRIC_COLORS[metric], edgecolor=COLORS["ink"], linewidth=0.7)
        ax.scatter([x + offset for offset in SEED_DOT_OFFSETS], seed_values, **SEED_DOT)
        annotate_value(ax, x, max(mean, *seed_values), mean)
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


def fig1(round_tables: dict, references: dict) -> None:
    fig, axes = plt.subplots(len(EVAL_ROWS), len(EVAL_ROWS[0]), figsize=(22, 12.6), sharey=True)
    for row_axes, panels in zip(axes, EVAL_ROWS):
        for index, (ax, (metric, eval_condition, title)) in enumerate(zip(row_axes, panels)):
            eval_panel(ax, round_tables, references, metric, eval_condition, title, "Rate" if index == 0 else None)
    handles = [
        Patch(facecolor="white", edgecolor=COLORS["ink"], linewidth=0.7, hatch=BASE_HATCH, label="Base model"),
        Patch(facecolor="white", edgecolor=COLORS["ink"], linewidth=0.7,
              label="Rounds 1–5, mean of seeds 42 and 43: each round keeps training the model from the round before"),
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=COLORS["ink"], markeredgecolor="white",
               label="Each seed (42 left, 43 right)"),
        *(Line2D([], [], color=style["color"], linestyle=style["linestyle"], linewidth=2.2, label=style["label"])
          for style in REFERENCE_LINES.values()),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=2,
               handlelength=4.5, prop={"size": 12, "weight": "bold"})  # line samples long enough to show the dashes
    fig.suptitle("Eval Answers After Each Round of RL Against the Bias Probe", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    save(fig, "fig1_eval_answers")


def shift_line(ax, x_start: float, x_end: float, start_value: float, end_value: float, color: str) -> None:
    """A line from the Initial value (open dot) to the Final value (filled dot). Each value is written in the dot's
    color on its own line's side of the dot (Initial right, Final left), moved up or down away from the line, so the
    labels of two rounds' dots at one x sit on opposite sides."""
    ax.plot([x_start, x_end], [start_value, end_value], color=COLORS["ink"], linewidth=2, zorder=3)
    ax.scatter(x_start, start_value, s=DOT_SIZE, facecolor="white", edgecolor=color, linewidth=2.2, zorder=4)
    ax.scatter(x_end, end_value, s=DOT_SIZE, facecolor=color, edgecolor=color, linewidth=2.2, zorder=5)
    away_from_line = 9 if end_value < start_value else -9  # points; the Initial label goes up when the line goes down
    for dot_x, value, offset, dy in ((x_start, start_value, 12, away_from_line), (x_end, end_value, -12, -away_from_line)):
        ax.annotate(f"{value:.3f}", (dot_x, value), xytext=(offset, dy), textcoords="offset points",
                    ha="left" if offset > 0 else "right", va="center", fontsize=FONT["annotation"] + 1,
                    fontweight="bold", color=color, zorder=6,
                    bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})  # keeps gridlines off the label


def fig2(probe_tables: dict) -> None:
    fig, axes = plt.subplots(len(SEEDS), 2, figsize=(20, 17))
    model_positions = range(len(ROUNDS) + 1)  # the base model, then each round's final model
    for seed, row_axes in zip(SEEDS, axes):
        probe_table = probe_tables[seed]
        for previous_run, run in zip(ROUNDS, ROUNDS[1:]):  # round R's Initial answers are round R-1's Final answers
            initial = probe_table[run]["scores"]["own_test_set"]
            previous_final = probe_table[previous_run]["scores"]["round_probe_round_answers"]
            assert (initial["n_biased"], initial["n_unbiased"]) == (previous_final["n_biased"], previous_final["n_unbiased"]), (seed, run)
        panels = (
            (row_axes[0], (("mean_biased", METRIC_COLORS["probe_biased"]), ("mean_unbiased", METRIC_COLORS["probe_unbiased"]))),
            (row_axes[1], (("auc", METRIC_COLORS["probe_auc"]),)),
        )
        for ax, series in panels:
            for round_index, run in enumerate(ROUNDS):  # round R's line runs from model R-1 to model R
                initial = probe_table[run]["scores"]["own_test_set"]
                final = probe_table[run]["scores"]["round_probe_round_answers"]
                for key, color in series:  # biased and unbiased share the x positions
                    shift_line(ax, round_index, round_index + 1, initial[key], final[key], color)
            ax.set_xticks(model_positions, ["Base\nmodel", *(f"{ROUND_LABELS[run]}\nmodel" for run in ROUNDS)])
            ax.set_xlim(-0.4, len(ROUNDS) + 0.4)
            ax.set_xlabel("Model whose eval answers are scored", fontsize=FONT["label"], fontweight="bold")
            probe_axis = ax.secondary_xaxis("top")  # names each round's line, between the two models it joins
            probe_axis.set_xticks([index + 0.5 for index in range(len(ROUNDS))], [f"{ROUND_LABELS[run]} probe" for run in ROUNDS])
            probe_axis.tick_params(length=0, labelsize=FONT["tick"], pad=4)
            probe_axis.spines["top"].set_visible(False)
        style_axis(row_axes[0], ylabel="Mean probe score (0 to 1)")
        row_axes[0].set_title(f"Seed {seed}: Mean Probe Score", fontsize=FONT["panel_title"], fontweight="bold", y=1.0,
                              pad=30)  # fixed y: above the probe labels
        row_axes[0].set_ylim(-0.04, 0.85)
        style_axis(row_axes[1], ylabel="AUC")
        row_axes[1].set_title(f"Seed {seed}: AUC, Biased vs Unbiased Answers", fontsize=FONT["panel_title"], fontweight="bold",
                              y=1.0, pad=30)
        row_axes[1].set_ylim(0.45, 0.95)  # 261102's range, so the two experiments' figures compare directly
    handles = [  # below the panels, off the data
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_biased"], label="Biased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_unbiased"],
               label="Unbiased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_auc"],
               label="AUC between biased and unbiased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Initial: start of the round, eval answers of the model the round starts from"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=COLORS["ink"],
               label="Final: end of the round, eval answers of the round's final model"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.062), ncol=2,
               prop={"size": 12, "weight": "bold"})  # column 1: the colours; column 2: open and filled dots
    fig.suptitle("Each Round's Probe at the Start and End of the Round", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.068, 1, 1))
    save(fig, "fig2_round_probe")


def read_curve(path: Path, column: str) -> list[float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == list(range(STEPS)), path
    return [float(r[column]) for r in rows]


def fig3(dynamics: dict) -> None:
    fig, axes = plt.subplots(2, len(ROUNDS), figsize=(22, 9.5), sharey="row")
    for column, run in enumerate(ROUNDS):
        top, bottom = axes[0, column], axes[1, column]
        for seed in reversed(SEEDS):  # seed 43 first, seed 42 on top
            curve_path, dynamics_name = TRAINING_RUNS[seed][run]
            cue_agreement = [100 * value for value in read_curve(curve_path, "followed_cue")]
            per_step = dynamics[seed]["training"][dynamics_name]["per_step"]
            assert len(per_step) == STEPS, (seed, run)
            lengths = [step["response_length_mean"] for step in per_step]
            style = SEED_CURVES[seed]
            top.plot(range(STEPS), cue_agreement, color=style["followed_cue"], linewidth=style["linewidth"])
            bottom.plot(range(STEPS), lengths, color=style["answer_length"], linewidth=style["linewidth"])
        style_axis(top, title=ROUND_LABELS[run], ylabel="Cue agreement rate" if column == 0 else None)
        percent_axis(top, -2, 102, [0, 20, 40, 60, 80, 100])
        style_axis(bottom, ylabel="Mean answer length, tokens" if column == 0 else None)
        bottom.set_ylim(0, 290)
        bottom.set_xlabel("Training step", fontsize=FONT["label"], fontweight="bold")
        for ax in (top, bottom):
            ax.set_xlim(-3, STEPS + 2)  # the first and last steps clear of the axes
            ax.tick_params(labelleft=True)  # each round's values readable in its own panel
    handles = [Line2D([], [], color=SEED_CURVES[seed][measure], linewidth=SEED_CURVES[seed]["linewidth"],
                      label=f"{name}, seed {seed}")
               for measure, name in (("followed_cue", "Cue agreement rate"), ("answer_length", "Answer length"))
               for seed in SEEDS]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Training Answers in Each Round of RL Against the Bias Probe", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, "fig3_training")


def main() -> None:
    round_tables = {}
    for seed in SEEDS:
        table = json.loads(ROUND_TABLES[seed].read_text())
        assert list(table["runs"]) == list(ROUNDS), (seed, list(table["runs"]))  # "after_rl" lists are in this order
        round_tables[seed] = table["mean_sd"]
    references = {"no_penalty": json.loads(NO_PENALTY_TABLE.read_text())["mean_sd"],
                  "ceiling": json.loads(CEILING_TABLE.read_text())["mean_sd"]["noisy_nocue"]}
    fig1(round_tables, references)
    fig2({seed: json.loads(PROBE_TABLES[seed].read_text()) for seed in SEEDS})
    fig3({seed: json.loads(DYNAMICS[seed].read_text()) for seed in SEEDS})


if __name__ == "__main__":
    main()
