"""Figures of the main results of the KL 0.01 runs, in the style of /workspace/plot_style_guide.ipynb (design tokens and
helpers from 261031's plot_results.py).

    python plot_results.py     # -> figures/*.png and figures/*.pdf

  fig1_answer_length  1 x 2 panels: the mean length of the training answers (W&B response_length_mean, tokens) at
                      each of the 128 training steps, weight 1.0 at KL 0.001 (261027) and at KL 0.01 (this folder),
                      one line per seed (seed 42 thinnest and on top). From each run's per-step values in 261027's
                      weight_1_0/runs/dynamics/dynamics.json and this folder's weight_1_0/runs/dynamics/dynamics.json
                      ("training").
  fig2_eval_answers   1 x 3 panels: the cue agreement rate and bias on the training cue mix, and true accuracy with the
                      cue at the wrong answer. Bars: no penalty and weight 1.0, each at KL 0.001 and KL 0.01 (each the
                      mean of seeds 42-44, with a dot per seed, seeds 42, 43, 44 left to right); the dotted line is the base model, with its value at the right edge (on
                      the bias panel, its mean over the six KL 0.01 runs' question sets). From each condition's
                      seed_table.json ("mean_sd"): 261024's tables/baseline/, 261027's weight_1_0/tables/weight_1_0/ and
                      this folder's no_penalty/tables/kl_no_penalty/ and weight_1_0/tables/kl_weight_1_0/.
  fig3_initial_probe  1 x 2 panels: the initial probe on the cued answers that name the cued option, both cue
                      conditions pooled (weight_1_0/runs/dynamics/dynamics.json, probe rows A and C, "both"). For each
                      weight-1.0 run at KL 0.01, a line in the series' color runs from the base model's answers on that run's
                      questions (open dot) to the trained model's (filled dot): the mean probe score of biased and of
                      unbiased answers, and the AUC between them.
  fig3a_fresh_probe   fig3 with the fresh probe, fit on each weight-1.0 model's own answers and activations, on the same
                      answers (open diamond, just right of the trained model's dot, no line). From
                      own_answer/probe_table.json (its initial-probe rows equal dynamics.json's rows A and C).
"""

from __future__ import annotations

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
E27 = EXPERIMENTS / "261027_probe_penalty_weight_sweep"
SEEDS = (42, 43, 44)
STEPS = 128
TABLE_FILES = {  # fig2's bars, left to right
    "np_kl001": E24 / "tables" / "baseline" / "seed_table.json",
    "np_kl01": EXP_DIR / "no_penalty" / "tables" / "kl_no_penalty" / "seed_table.json",
    "w10_kl001": E27 / "weight_1_0" / "tables" / "weight_1_0" / "seed_table.json",
    "w10_kl01": EXP_DIR / "weight_1_0" / "tables" / "kl_weight_1_0" / "seed_table.json",
}
BAR_LABELS = {"np_kl001": "No penalty\nKL 0.001", "np_kl01": "No penalty\nKL 0.01",
              "w10_kl001": "Weight 1.0\nKL 0.001", "w10_kl01": "Weight 1.0\nKL 0.01"}
BAR_POSITIONS = {"np_kl001": 0.0, "np_kl01": 1.0, "w10_kl001": 2.4, "w10_kl01": 3.4}  # a gap between the conditions
LENGTH_RUNS = {  # fig1's panels: dynamics.json and the run-name prefix there
    "KL 0.001": (E27 / "weight_1_0" / "runs" / "dynamics" / "dynamics.json", "weight_1_0_s"),
    "KL 0.01": (EXP_DIR / "weight_1_0" / "runs" / "dynamics" / "dynamics.json", "kl_weight_1_0_s"),
}
PROBE_RUNS = {  # fig3's groups: dynamics.json and the run-name prefix there
    "Weight 1.0, KL 0.01": (EXP_DIR / "weight_1_0" / "runs" / "dynamics" / "dynamics.json", "kl_weight_1_0_s"),
}

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
EVAL_PANELS = (  # metric, eval condition, panel title, y-axis top and ticks
    ("followed_cue", "train_mix", "Cue Agreement Rate\nTraining Cue Mix", 110, [0, 20, 40, 60, 80, 100]),
    ("bias", "train_mix", "Bias\nTraining Cue Mix", 55, [0, 10, 20, 30, 40, 50]),
    ("accuracy", "cue_dist", "True Accuracy\nCue at the Wrong Answer", 80, [0, 20, 40, 60, 80]),
)
BAR_WIDTH = 0.66
SEED_DOT_OFFSETS = (-0.19, 0.0, 0.19)  # fig2: seeds 42, 43, 44 left to right on each bar
SEED_DOT = {"s": 22, "color": COLORS["ink"], "edgecolor": "white", "linewidth": 0.8, "zorder": 7, "clip_on": False}
SEED_COLORS = ("#e3a24f", "#a8641f", "#4f310d")  # fig1: seeds 42, 43, 44, light to dark gold
SEED_LINE_WIDTHS = (1.3, 2.0, 2.8)  # fig1: seed 42 thinnest, drawn last, so overlapping lines all show
LENGTH_TICKS = (0, 50, 100, 150, 200, 250, 300)
FRESH_PROBE_TABLE = EXP_DIR / "own_answer" / "probe_table.json"  # fig3a: the three weight-1.0 models' probe rows
DOT_OFFSET = 0.15  # fig3: base model dot left of the run, trained model dot right of it
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


def annotate_value(ax, x: float, value: float) -> None:
    """The bar's value, inside the bar near its base, clear of the seed dots and the base model's line."""
    ax.annotate(f"{value:.1f}", (x, 0), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                fontsize=FONT["annotation"] + 1, fontweight="bold", color=COLORS["ink"], zorder=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.8})


def base_line(ax, value: float) -> None:
    """The base model as a dotted line across the whole panel, its value written above the line at the right edge."""
    x_min, x_max = ax.get_xlim()
    ax.plot([x_min, x_max], [value, value], linestyle=":", color=COLORS["ink"], linewidth=2.2, zorder=5)
    ax.annotate(f"{value:.1f}", (x_max - 0.02, value), xytext=(0, 3), textcoords="offset points", ha="right",
                va="bottom", fontsize=FONT["annotation"], fontstyle="italic", color=COLORS["ink"], zorder=6,
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote figures/{name}.png, .pdf")


def fig1() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for index, (ax, (title, (path, prefix))) in enumerate(zip(axes, LENGTH_RUNS.items())):
        training = json.loads(path.read_text())["training"]
        for color, width, seed in reversed(list(zip(SEED_COLORS, SEED_LINE_WIDTHS, SEEDS))):  # seed 44 first, 42 on top
            per_step = training[f"{prefix}{seed}"]["per_step"]
            assert len(per_step) == STEPS, (title, seed)
            ax.plot(range(STEPS), [row["response_length_mean"] for row in per_step], color=color, linewidth=width)
        ax.set_xlim(-3, STEPS + 2)  # the first and last steps clear of the axes
        ax.set_xlabel("Training step", fontsize=FONT["label"], fontweight="bold")
        style_axis(ax, title=title, ylabel="Mean length of the step's 512 answers\n(tokens)" if index == 0 else None)
        ax.set_ylim(-15, 320)  # the ~5-token lines clear of the axis line
        ax.set_yticks(LENGTH_TICKS)
    handles = [Line2D([], [], color=color, linewidth=width, label=f"Seed {seed}")
               for color, width, seed in zip(SEED_COLORS, SEED_LINE_WIDTHS, SEEDS)]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.025), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Training-Answer Length at Probe Penalty Weight 1.0", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig1_answer_length")


def base_value(stats: dict, metric: str, eval_condition: str) -> float:
    """The base model's value: on the bias panel, which counts each run's own questions, the mean over the six KL 0.01
    runs' question sets."""
    values = stats["np_kl01"][metric][eval_condition]["base"] + stats["w10_kl01"][metric][eval_condition]["base"]
    return 100 * statistics.mean(values)


def eval_panel(ax, stats: dict, metric: str, eval_condition: str, title: str, ylabel: str | None, y_top: float,
               y_ticks: list[float]) -> None:
    """Each condition's bar (the mean over the 3 seeds) with a dot per seed, and the base model as a dotted line."""
    for run, x in BAR_POSITIONS.items():
        row = stats[run][metric][eval_condition]
        assert len(row["after_rl"]) == 3, (run, metric, eval_condition)
        mean = 100 * row["after_rl_mean"]
        ax.bar(x, mean, BAR_WIDTH, color=METRIC_COLORS[metric], edgecolor=COLORS["ink"], linewidth=0.7)
        seed_values = [100 * value for value in row["after_rl"]]
        ax.scatter([x + offset for offset in SEED_DOT_OFFSETS], seed_values, **SEED_DOT)
        annotate_value(ax, x, mean)
    ax.set_xticks(list(BAR_POSITIONS.values()), [BAR_LABELS[run] for run in BAR_POSITIONS])
    ax.set_xlim(-0.55, max(BAR_POSITIONS.values()) + 0.75)  # room right of the last bar for the base value
    base_line(ax, base_value(stats, metric, eval_condition))
    style_axis(ax, title=title, ylabel=ylabel)
    percent_axis(ax, 0, y_top, y_ticks)


def fig2(stats: dict) -> None:
    fig, axes = plt.subplots(1, len(EVAL_PANELS), figsize=(19, 6.6))
    for index, (ax, (metric, eval_condition, title, y_top, y_ticks)) in enumerate(zip(axes, EVAL_PANELS)):
        eval_panel(ax, stats, metric, eval_condition, title, "Rate" if index == 0 else None, y_top, y_ticks)
    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=11, markerfacecolor="white",
               markeredgecolor=COLORS["ink"], label="Bar: mean over 3 seeds (value at its base)"),
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=COLORS["ink"], markeredgecolor="white",
               label="Each seed (42, 43, 44 left to right)"),
        Line2D([], [], color=COLORS["ink"], linestyle=":", linewidth=2.2, label="Base model"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Eval Answers After RL, KL 0.001 and KL 0.01", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig2_eval_answers")


def shift_line(ax, x_start: float, x_end: float, base_value: float, trained_value: float, color: str,
               trained_label_dy: float = 0, trained_label_offset: float = 12) -> None:
    """A line from the base model's value (open dot) to the trained model's (filled dot), both in the series' color,
    each value written beside its dot in that color."""
    ax.plot([x_start, x_end], [base_value, trained_value], color=color, linewidth=2, zorder=3)
    ax.scatter(x_start, base_value, s=DOT_SIZE, facecolor="white", edgecolor=color, linewidth=2.2, zorder=4)
    ax.scatter(x_end, trained_value, s=DOT_SIZE, facecolor=color, edgecolor=color, linewidth=2.2, zorder=4)
    for dot_x, value, offset, dy in ((x_start, base_value, -12, 0),
                                     (x_end, trained_value, trained_label_offset, trained_label_dy)):
        ax.annotate(f"{value:.3f}", (dot_x, value), xytext=(offset, dy), textcoords="offset points",
                    ha="left" if offset > 0 else "right", va="center", fontsize=FONT["annotation"] + 1,
                    fontweight="bold", color=color, zorder=6,
                    bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})  # keeps gridlines off the label


def fig3() -> None:
    dynamics = {group: json.loads(path.read_text())["probe_letters"] for group, (path, _) in PROBE_RUNS.items()}
    spacing, group_gap = 2.2, 1.0
    positions = []  # (x, group, seed)
    for group_index, group in enumerate(PROBE_RUNS):
        for seed_index, seed in enumerate(SEEDS):
            positions.append((group_index * (len(SEEDS) * spacing + group_gap) + seed_index * spacing, group, seed))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    panels = (
        (axes[0], (("mean_score_biased", METRIC_COLORS["probe_biased"]),
                   ("mean_score_unbiased", METRIC_COLORS["probe_unbiased"]))),
        (axes[1], (("auc", METRIC_COLORS["probe_auc"]),)),
    )
    for ax, series in panels:
        for x, group, seed in positions:
            _, prefix = PROBE_RUNS[group]
            rows = dynamics[group][f"{prefix}{seed}"]
            for key, color in series:  # biased and unbiased share the x positions
                shift_line(ax, x - DOT_OFFSET, x + DOT_OFFSET, rows["A"]["both"][key], rows["C"]["both"][key], color)
        ax.set_xticks([x for x, _, _ in positions],
                      [f"Seed {seed}" for _, _, seed in positions])
        ax.set_xlim(positions[0][0] - 1.0, positions[-1][0] + 1.0)  # room for the outer value labels
    style_axis(axes[0], title="Mean Probe Score", ylabel="Mean probe score")
    axes[0].set_ylim(-0.02, 0.9)
    style_axis(axes[1], title="AUC, Biased vs Unbiased Answers", ylabel="AUC")
    axes[1].set_ylim(0.75, 0.95)
    handles = [  # below the panels, off the data
        *color_handles(),
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Base model's answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=COLORS["ink"], label="Trained model's answers"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=len(handles),
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Initial Probe on Answers Before and After RL, Probe Penalty Weight 1.0, KL 0.01", fontsize=FONT["title"],
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig3_initial_probe")


def color_handles() -> list:
    """Legend squares for the three series colors of fig3 and fig3a."""
    return [Line2D([], [], marker="s", linestyle="", markersize=10, color=METRIC_COLORS[key], label=label)
            for key, label in (("probe_biased", "Biased answers"), ("probe_unbiased", "Unbiased answers"),
                               ("probe_auc", "AUC"))]


def fresh_probe_dot(ax, x: float, value: float, color: str) -> None:
    """The fresh probe on the trained model's answers: an open diamond, its value to the right."""
    ax.scatter(x, value, s=DOT_SIZE, marker="D", facecolor="white", edgecolor=color, linewidth=2.2, zorder=4)
    ax.annotate(f"{value:.3f}", (x, value), xytext=(12, 0), textcoords="offset points", ha="left", va="center",
                fontsize=FONT["annotation"] + 1, fontweight="bold", color=color, zorder=6,
                bbox={"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.8})


def fig3a() -> None:
    """fig3 with the fresh probe on the same answers: an open diamond just right of the trained model's dot, whose value
    label moves right past the diamond."""
    probe_table = json.loads(FRESH_PROBE_TABLE.read_text())
    spacing, fresh_nudge = 2.2, 0.09
    positions = [spacing * index for index in range(len(SEEDS))]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    panels = (
        (axes[0], (("mean_biased", METRIC_COLORS["probe_biased"]), ("mean_unbiased", METRIC_COLORS["probe_unbiased"]))),
        (axes[1], (("auc", METRIC_COLORS["probe_auc"]),)),
    )
    for ax, series in panels:
        for x, seed in zip(positions, SEEDS):
            rows = probe_table[f"w10_s{seed}"]
            for key, color in series:  # biased and unbiased share the x positions
                shift_line(ax, x - DOT_OFFSET, x + DOT_OFFSET, rows["initial_base"][key], rows["initial_model"][key], color,
                           trained_label_offset=28)
                fresh_probe_dot(ax, x + DOT_OFFSET + fresh_nudge, rows["own_model"][key], color)
        ax.set_xticks(positions, [f"Seed {seed}" for seed in SEEDS])
        ax.set_xlim(positions[0] - 1.0, positions[-1] + 1.1)  # room for the outer value labels
    style_axis(axes[0], title="Mean Probe Score", ylabel="Mean probe score")
    axes[0].set_ylim(-0.02, 0.9)
    style_axis(axes[1], title="AUC, Biased vs Unbiased Answers", ylabel="AUC")
    axes[1].set_ylim(0.75, 0.95)
    colors = color_handles()
    markers = [
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Initial probe, base model's answers"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=COLORS["ink"],
               label="Initial probe, trained model's answers"),
        Line2D([], [], marker="D", linestyle="", markersize=8, markerfacecolor="white", markeredgecolor=COLORS["ink"],
               markeredgewidth=2.2, label="Fresh probe, trained model's answers"),
    ]
    handles = [handle for pair in zip(colors, markers) for handle in pair]  # columns fill top-down: colors on row 1
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3,
               prop={"size": 12, "weight": "bold"})
    fig.suptitle("Initial and Fresh Probe on Answers Before and After RL, Probe Penalty Weight 1.0, KL 0.01",
                 fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig3a_fresh_probe")


def main() -> None:
    stats = {run: json.loads(path.read_text())["mean_sd"] for run, path in TABLE_FILES.items()}
    fig1()
    fig2(stats)
    fig3()
    fig3a()


if __name__ == "__main__":
    main()
