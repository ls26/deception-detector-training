"""Figures of the main results across the penalty weights, in the style of /workspace/plot_style_guide.ipynb (started
from 261025's plot_results.py).

    python plot_results.py     # -> figures/*.png and figures/*.pdf

  fig1_eval_answers   2 x 4 panels. Top: true accuracy with no cue, the cue at the correct answer, the cue at the wrong
                      answer and on the training cue mix. Bottom: the cue agreement rate on the training cue mix, and
                      bias with the cue at the correct answer, at the wrong answer and on the training cue mix. Bars:
                      the no-penalty baseline and the penalty runs with the floor at weights 0.3, 0.6 and 1.0 (each the
                      mean of seeds 42-44, with a line for the SD over the seeds, n - 1); the dotted line is the base
                      model, with its value at the right edge (on the bias rows, its mean over the six weight-0.6 and
                      weight-1.0 runs' question sets, as in the README). From 261024's tables/baseline/, 261025's
                      tables/floor/ and weight_<w>/tables/weight_<w>/seed_table.json ("mean_sd").
  fig2_training       1 x 4 panels: the cue agreement rate on training answers over the 128 training steps
                      (training_curve.csv of each run), one panel per run (no penalty, weights 0.3, 0.6, 1.0), one line
                      per seed.
  fig3_initial_probe  1 x 2 panels: the initial probe on the cued answers that name the cued option, both cue
                      conditions pooled (each weight's runs/dynamics/dynamics.json, probe rows A and C, "both"). At each
                      weight, a line runs from the base model's answers (open dot) to the trained models' (filled dot),
                      each the mean over that weight's three runs, with the SD over the seeds (n - 1) at the trained
                      models' dot: the mean probe score of biased and of unbiased answers, and the AUC between them.
  fig3a_own_probe     fig3 as 2 x 1 panels (mean probe score above, AUC below) with two additions, from
                      261028_own_answer_probes/probe_table.json (its initial-probe values equal dynamics.json's rows A
                      and C): the probe retrained on each trained model's own answers and activations, on the same
                      answers (open diamond, just right of the trained models' dot, no line), and the no-penalty runs
                      at weight 0.0.
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
E25 = EXPERIMENTS / "261025_probe_penalty_sd_floor"
SEEDS = (42, 43, 44)
TABLE_FILES = {"baseline": E24 / "tables" / "baseline" / "seed_table.json",
               "weight_0_3": E25 / "tables" / "floor" / "seed_table.json",
               "weight_0_6": EXP_DIR / "weight_0_6" / "tables" / "weight_0_6" / "seed_table.json",
               "weight_1_0": EXP_DIR / "weight_1_0" / "tables" / "weight_1_0" / "seed_table.json"}
TRAINING_CURVES = {  # per run, the three seeds' training_curve.csv
    "baseline": [EXPERIMENTS / "261013_noisy_label_rl" / "training_curve.csv",
                 EXPERIMENTS / "261018_noisy_label_rl_seeds" / "runs" / "cot_s43" / "training_curve.csv",
                 EXPERIMENTS / "261018_noisy_label_rl_seeds" / "runs" / "cot_s44" / "training_curve.csv"],
    "weight_0_3": [E25 / "runs" / f"floor_s{seed}" / "training_curve.csv" for seed in SEEDS],
    "weight_0_6": [EXP_DIR / "weight_0_6" / "runs" / f"weight_0_6_s{seed}" / "training_curve.csv" for seed in SEEDS],
    "weight_1_0": [EXP_DIR / "weight_1_0" / "runs" / f"weight_1_0_s{seed}" / "training_curve.csv" for seed in SEEDS],
}
DYNAMICS = {  # per weight, its dynamics.json and the names of its three runs there
    "weight_0_3": (E25 / "runs" / "dynamics" / "dynamics.json", [f"floor_s{seed}" for seed in SEEDS]),
    "weight_0_6": (EXP_DIR / "weight_0_6" / "runs" / "dynamics" / "dynamics.json", [f"weight_0_6_s{seed}" for seed in SEEDS]),
    "weight_1_0": (EXP_DIR / "weight_1_0" / "runs" / "dynamics" / "dynamics.json", [f"weight_1_0_s{seed}" for seed in SEEDS]),
}
PROBE_TABLE = EXPERIMENTS / "261028_own_answer_probes" / "probe_table.json"
PROBE_TABLE_MODELS = {  # per run, its three models in the probe table (floor_s4X: 261025's weight-0.3 runs)
    "baseline": [f"baseline_s{seed}" for seed in SEEDS], "weight_0_3": [f"floor_s{seed}" for seed in SEEDS],
    "weight_0_6": [f"w06_s{seed}" for seed in SEEDS], "weight_1_0": [f"w10_s{seed}" for seed in SEEDS],
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
RUN_LABELS = {"baseline": "No penalty", "weight_0_3": "Weight 0.3", "weight_0_6": "Weight 0.6",
              "weight_1_0": "Weight 1.0"}
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
SEED_COLORS = ("#dc8f98", "#ae5863", "#6b2c35")  # seeds 42, 43, 44: light to dark rose
SEED_LINE = {"linestyle": "-", "linewidth": 1.6}
DOT_OFFSET = 0.15  # figure 3: base model dot left of the weight, trained model dot right of it
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


def base_line(ax, value: float) -> None:
    """The base model as a dotted line across the whole panel, its value written above the line at the right edge."""
    x_min, x_max = ax.get_xlim()
    ax.plot([x_min, x_max], [value, value], linestyle=":", color=COLORS["ink"], linewidth=2.2, zorder=5)
    ax.annotate(f"{value:.1f}", (x_max - 0.02, value), xytext=(0, 3), textcoords="offset points", ha="right",
                va="bottom", fontsize=FONT["annotation"], fontstyle="italic", color=COLORS["ink"], zorder=6,
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})


def base_value(stats: dict, metric: str, eval_condition: str) -> float:
    """The base model's value as in the README's main table: on the bias rows, which count each run's own questions,
    the mean over the six weight-0.6 and weight-1.0 runs' question sets."""
    values = stats["weight_0_6"][metric][eval_condition]["base"] + stats["weight_1_0"][metric][eval_condition]["base"]
    return 100 * statistics.mean(values)


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote figures/{name}.png, .pdf")


def eval_panel(ax, stats: dict, metric: str, eval_condition: str, title: str, ylabel: str | None) -> None:
    """Each run's bar with its SD over the 3 seeds, and the base model as a dotted line."""
    for x, run in enumerate(RUN_LABELS):
        row = stats[run][metric][eval_condition]
        assert len(row["after_rl"]) == 3, (run, metric, eval_condition)
        mean = 100 * row["after_rl_mean"]
        sd = 100 * row["after_rl_sd"]
        ax.errorbar(x, mean, yerr=sd, zorder=6, **ERRORBAR)  # over the dotted line
        ax.bar(x, mean, BAR_WIDTH, color=METRIC_COLORS[metric], edgecolor=COLORS["ink"], linewidth=0.7)
        annotate_value(ax, x, mean + sd, mean)
    ax.set_xticks(range(len(RUN_LABELS)), list(RUN_LABELS.values()))
    ax.set_xlim(-0.55, len(RUN_LABELS) - 0.25)  # room right of the last bar for the base value
    base_line(ax, base_value(stats, metric, eval_condition))
    style_axis(ax, title=title, ylabel=ylabel)
    percent_axis(ax, 0, 108, [0, 20, 40, 60, 80, 100])  # room for the value labels above bars near 100%


def fig1(stats: dict) -> None:
    fig, axes = plt.subplots(len(EVAL_ROWS), len(EVAL_ROWS[0]), figsize=(18, 12.4), sharey=True)
    for row_axes, panels in zip(axes, EVAL_ROWS):
        for index, (ax, (metric, eval_condition, title)) in enumerate(zip(row_axes, panels)):
            eval_panel(ax, stats, metric, eval_condition, title, "Rate" if index == 0 else None)
    # an empty error bar, so the legend shows the same capped line as the bars
    sd_handle = axes[0, 0].errorbar([float("nan")], [float("nan")], yerr=[1.0], fmt="none", ecolor=COLORS["ink"],
                                    elinewidth=1.3, capsize=5, capthick=1.3, label="SD over 3 seeds")
    handles = [sd_handle, Line2D([], [], color=COLORS["ink"], linestyle=":", linewidth=2.2, label="Base model")]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Eval Answers After RL: No-Penalty Runs and Penalty Runs with the SD Floor", fontsize=FONT["title"],
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig1_eval_answers")


def read_curve(path: Path, column: str) -> list[float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == list(range(STEPS)), path
    return [float(r[column]) for r in rows]


def fig2() -> None:
    fig, axes = plt.subplots(1, len(TRAINING_CURVES), figsize=(20, 5.8), sharey=True)
    for index, (ax, (run, paths)) in enumerate(zip(axes, TRAINING_CURVES.items())):
        for color, path in zip(SEED_COLORS, paths):
            ax.plot(range(STEPS), [100 * value for value in read_curve(path, "followed_cue")], color=color, **SEED_LINE)
        ax.set_xlim(-3, STEPS + 2)  # the first and last steps clear of the axes
        ax.set_xlabel("Training step", fontsize=FONT["label"], fontweight="bold")
        style_axis(ax, title=RUN_LABELS[run].replace("\n", " "), ylabel="Cue agreement rate" if index == 0 else None)
        percent_axis(ax, 48, 102, [50, 60, 70, 80, 90, 100])
    handles = [Line2D([], [], color=color, label=f"Seed {seed}", **SEED_LINE) for color, seed in zip(SEED_COLORS, SEEDS)]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Cue Agreement Rate on Training Answers: No-Penalty Runs and Penalty Runs with the SD Floor",
                 fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig2_training")


def shift_line(ax, x_start: float, x_end: float, base_value: float, trained_value: float, trained_sd: float,
               color: str, trained_label_offset: float = 12) -> None:
    """A line from the base model's value (open dot) to the trained models' mean (filled dot, with the SD over seeds),
    each value written beside its dot (the trained value trained_label_offset points to the right). The SD line has the
    dot's color, so series that share an x position keep apart."""
    ax.plot([x_start, x_end], [base_value, trained_value], color=COLORS["ink"], linewidth=2, zorder=3)
    ax.errorbar(x_end, trained_value, yerr=trained_sd, zorder=3, **{**ERRORBAR, "color": color})
    ax.scatter(x_start, base_value, s=DOT_SIZE, facecolor="white", edgecolor=color, linewidth=2.2, zorder=4)
    ax.scatter(x_end, trained_value, s=DOT_SIZE, facecolor=color, edgecolor=color, linewidth=2.2, zorder=4)
    for dot_x, value, offset in ((x_start, base_value, -12), (x_end, trained_value, trained_label_offset)):
        side = 1 if offset > 0 else -1
        ax.annotate(f"{value:.3f}", (dot_x, value), xytext=(offset, 0), textcoords="offset points",
                    ha="left" if side == 1 else "right", va="center", fontsize=FONT["annotation"] + 1, fontweight="bold",
                    color=color, zorder=6,  # the dot's color, so a label belongs to its line where dots overlap
                    bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})  # keeps gridlines off the label


def mean_over_runs(dynamics: dict, weight: str, row: str, key: str) -> tuple[float, float]:
    """Mean and SD (n - 1) over one weight's three runs of one value of a pooled probe row."""
    values = [dynamics[weight]["probe_letters"][run][row]["both"][key] for run in DYNAMICS[weight][1]]
    return statistics.mean(values), statistics.stdev(values)


def fig3(dynamics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    panels = (
        (axes[0], (("mean_score_biased", METRIC_COLORS["probe_biased"]),
                   ("mean_score_unbiased", METRIC_COLORS["probe_unbiased"]))),
        (axes[1], (("auc", METRIC_COLORS["probe_auc"]),)),
    )
    for ax, series in panels:
        for x, weight in enumerate(DYNAMICS):
            for key, color in series:  # biased and unbiased share the x positions
                base_mean, _ = mean_over_runs(dynamics, weight, "A", key)
                trained_mean, trained_sd = mean_over_runs(dynamics, weight, "C", key)
                shift_line(ax, x - DOT_OFFSET, x + DOT_OFFSET, base_mean, trained_mean, trained_sd, color)
        ax.set_xticks(range(len(DYNAMICS)), [RUN_LABELS[weight].removeprefix("Weight ") for weight in DYNAMICS])
        ax.set_xlim(-0.6, len(DYNAMICS) - 0.4)
        ax.set_xlabel("Probe penalty weight", fontsize=FONT["label"], fontweight="bold")
    style_axis(axes[0], title="Mean Probe Score", ylabel="Mean probe score")
    axes[0].set_ylim(-0.04, 0.85)
    style_axis(axes[1], title="AUC, Biased vs Unbiased Answers", ylabel="AUC")
    axes[1].set_ylim(0.65, 0.95)

    sd_handle = axes[1].errorbar([float("nan")], [float("nan")], yerr=[1.0], fmt="none", ecolor=COLORS["ink"],
                                 elinewidth=1.3, capsize=5, capthick=1.3, label="SD over 3 seeds")
    handles = [  # below the panels, off the data
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_biased"], label="Biased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_unbiased"],
               label="Unbiased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Base model's answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=COLORS["ink"],
               label="Trained models' answers (mean of 3 seeds)"),
        sd_handle,
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.07), ncol=3,
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Initial Probe on Cued Answers That Name the Cued Option, Both Cue Conditions Pooled\n"
                 "Penalty Runs with the SD Floor", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "fig3_initial_probe")


def table_mean(probe_table: dict, run: str, row: str, key: str) -> tuple[float, float]:
    """Mean and SD (n - 1) over one run's three models of one value of a probe-table row."""
    values = [probe_table[model][row][key] for model in PROBE_TABLE_MODELS[run]]
    return statistics.mean(values), statistics.stdev(values)


def own_probe_dot(ax, x: float, value: float, sd: float, color: str) -> None:
    """The own-answer probe on the trained models' answers: an open diamond with its SD, its value to the right."""
    ax.errorbar(x, value, yerr=sd, zorder=3, **{**ERRORBAR, "color": color})
    ax.scatter(x, value, s=DOT_SIZE, marker="D", facecolor="white", edgecolor=color, linewidth=2.2, zorder=4)
    ax.annotate(f"{value:.3f}", (x, value), xytext=(12, 0), textcoords="offset points", ha="left", va="center",
                fontsize=FONT["annotation"] + 1, fontweight="bold", color=color, zorder=6,
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})


def fig3a(probe_table: dict) -> None:
    # at each weight: base model's answers left of it, the trained models' answers right of it (initial probe: filled
    # dot, own-answer probe: open diamond), with room between the weights for the value labels on either side
    spacing, offset = 1.4, 0.15
    own_nudge = 0.09  # the diamond just right of the filled dot, so their SD bars and caps stay apart; the filled
    # dot's value label moves right past the diamond (28 points instead of 12)
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))  # mean probe score on top, AUC below
    panels = (
        (axes[0], (("mean_biased", METRIC_COLORS["probe_biased"]), ("mean_unbiased", METRIC_COLORS["probe_unbiased"]))),
        (axes[1], (("auc", METRIC_COLORS["probe_auc"]),)),
    )
    positions = [spacing * index for index in range(len(PROBE_TABLE_MODELS))]
    for ax, series in panels:
        for x, run in zip(positions, PROBE_TABLE_MODELS):
            for key, color in series:  # biased and unbiased share the x positions
                base_mean, _ = table_mean(probe_table, run, "initial_base", key)
                initial_mean, initial_sd = table_mean(probe_table, run, "initial_model", key)
                own_mean, own_sd = table_mean(probe_table, run, "own_model", key)
                shift_line(ax, x - offset, x + offset, base_mean, initial_mean, initial_sd, color, trained_label_offset=28)
                own_probe_dot(ax, x + offset + own_nudge, own_mean, own_sd, color)
        ax.set_xticks(positions, ["0.0\n(no penalty)", "0.3", "0.6", "1.0"])
        ax.set_xlim(positions[0] - 0.75, positions[-1] + 0.75)  # room for the outermost value labels
    axes[1].set_xlabel("Probe penalty weight", fontsize=FONT["label"], fontweight="bold")
    style_axis(axes[0], title="Mean Probe Score", ylabel="Mean probe score")
    axes[0].set_ylim(-0.04, 0.95)
    style_axis(axes[1], title="AUC, Biased vs Unbiased Answers", ylabel="AUC")
    axes[1].set_ylim(0.69, 1.0)

    sd_handle = axes[1].errorbar([float("nan")], [float("nan")], yerr=[1.0], fmt="none", ecolor=COLORS["ink"],
                                 elinewidth=1.3, capsize=5, capthick=1.3, label="SD over 3 seeds")
    handles = [  # below the panels, off the data
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_biased"], label="Biased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=METRIC_COLORS["probe_unbiased"],
               label="Unbiased answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Initial probe, base model's answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=COLORS["ink"],
               label="Initial probe, trained models' answers"),
        Line2D([], [], marker="D", linestyle="", markersize=8, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Probe retrained on each trained model's answers"),
        sd_handle,
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=3,
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Initial vs Retrained Bias Probe", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    save(fig, "fig3a_own_probe")


def main() -> None:
    stats = {run: json.loads(path.read_text())["mean_sd"] for run, path in TABLE_FILES.items()}
    dynamics = {weight: json.loads(path.read_text()) for weight, (path, _) in DYNAMICS.items()}
    fig1(stats)
    fig2()
    fig3(dynamics)
    fig3a(json.loads(PROBE_TABLE.read_text()))


if __name__ == "__main__":
    main()
