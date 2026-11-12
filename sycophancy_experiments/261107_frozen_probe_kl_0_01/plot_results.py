"""Figures of the main results: the frozen probe (this experiment) against the retrained probe (261104) over the five
rounds of the seed-42 and seed-43 lines, in the style of /workspace/plot_style_guide.ipynb (261104's plot_results.py).
Color means the probe in every figure: teal = frozen, orange = retrained.

    python plot_results.py     # -> figures/*.png and figures/*.pdf

  fig1_eval_answers   1 x 5 panels: true accuracy with no cue, with the cue at the wrong answer and on the training cue
                      mix; the cue agreement rate and bias on the training cue mix. Lines over rounds 1-5, one per probe,
                      each point the mean over the two lines with a vertical line for ± 1 SD (seed_4X/tables/rounds/seed_table.json of
                      this folder and of 261104, "mean_sd" "after_rl"). Round 1 is one point, the same run in both. The
                      base model is a gray diamond (on the bias panel, its mean over the frozen lines' ten question sets).
                      Every panel spans 60 points of its own range, so slopes compare across panels.
  fig2_round_probe    2 x 2 panels, one row per probe: the round's penalty probe on the cued eval answers that name the
                      cued option, both cue conditions pooled: the mean probe score of biased and of unbiased answers
                      (darker and lighter shade of the probe's color), and the AUC between them (purple). Round R is a line from its start (open dot:
                      the eval answers of the model the round starts from) to its end (filled dot: round R's final-eval
                      answers); the x axis is the model whose answers are scored, so round R's start sits at round R-1's (in the
                      retrained row a little to its right, so the two dots do not cover each other)
                      end. Each point is the mean over the two lines, with a vertical line for ± 1 SD. Frozen: seed_4X/probe_table.json
                      (start of round 1: round 1's "base_answers"; every other point: "round_answers"; its start in round
                      R is its end in round R-1). Retrained: 261104's seed_4X/probe_table.json ("own_test_set" for the
                      start of rounds 2-5, "round_probe_round_answers" for the end; round 1 is this folder's).
  fig4_training_label 1 x 2 panels, unflipped and flipped training questions: the share of training answers that give the
                      training label, steps 60-127, with the cue at the label (solid, circles) or at the other option
                      (dashed, squares), over rounds 1-5 of the frozen probe; mean over the two lines with a vertical
                      line for ± 1 SD (runs/memorization/memorization.json). SD over the two lines, with n - 1, in all figures.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = EXP_DIR / "figures"
RETRAINED_DIR = EXP_DIR.parent / "261104_iterative_kl_0_01"
SEEDS = (42, 43)
ROUNDS = ("round1", "round2", "round3", "round4", "round5")
PROBES = ("frozen", "retrained")
ROUND_TABLES = {
    probe: {seed: folder / f"seed_{seed}" / "tables" / "rounds" / "seed_table.json" for seed in SEEDS}
    for probe, folder in (("frozen", EXP_DIR), ("retrained", RETRAINED_DIR))
}
PROBE_TABLES = {
    probe: {seed: folder / f"seed_{seed}" / "probe_table.json" for seed in SEEDS}
    for probe, folder in (("frozen", EXP_DIR), ("retrained", RETRAINED_DIR))
}
MEMORIZATION = EXP_DIR / "runs" / "memorization" / "memorization.json"

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

PROBE_STYLE = {  # color = the probe; the darker shade marks biased answers in fig2, the lighter unbiased answers
    "frozen": {"color": "#3f8f84", "light": "#93cbc2", "label": "Frozen probe (round 1's probe in every round)"},
    "retrained": {"color": "#c96f36", "light": "#eab089", "label": "Retrained probe (a new probe each round)"},
}
SHARED_COLOR = COLORS["ink"]  # round 1: the same run for both probes
AUC_COLOR = "#8a62a8"  # fig2: the AUC, in both rows, apart from the probes' colors
START_SHIFT = 0.14  # fig2, retrained row: start dots just right of the previous round's end dots at the same x
BASE_COLOR = COLORS["gray"]
MEAN_DOT = 70
EVAL_PANELS = (
    ("accuracy", "none", "True Accuracy\nNo Cue"),
    ("accuracy", "cue_dist", "True Accuracy\nCue at the Wrong Answer"),
    ("accuracy", "train_mix", "True Accuracy\nTraining Cue Mix"),
    ("followed_cue", "train_mix", "Cue Agreement Rate\nTraining Cue Mix"),
    ("bias", "train_mix", "Bias\nTraining Cue Mix"),
)
LABEL_CELLS = (  # fig4: panel, then (memorization cell, line style, marker) per line: cue at the label, at the other option
    ("Unflipped Questions\n(training label = the correct answer)",
     (("label true, cue at the true answer", "-", "o"), ("label true, cue at the wrong answer", "--", "s"))),
    ("Flipped Questions\n(training label = the wrong answer)",
     (("label flipped, cue at the wrong answer", "-", "o"), ("label flipped, cue at the true answer", "--", "s"))),
)

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


PANEL_SPAN = 60  # fig1: every panel spans 60 points, placed over its own values, so slopes compare across panels


def percent_axis(ax, values: list[float]) -> None:
    """A 60-point range starting at the tens below the panel's lowest value, moved down if it would pass 100%."""
    low = max(0, 10 * ((min(values) - 4) // 10))
    low = min(low, 100 - PANEL_SPAN)
    bottom = low - 4 if low == 0 else low  # a little room under 0% for values written below a point
    top = bottom + PANEL_SPAN  # every panel spans exactly 60 points
    assert max(values) + 4 <= top, (min(values), max(values))
    ax.set_ylim(bottom, top)
    ax.set_yticks(range(int(low), int(top) + 1, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def write_value(ax, x: float, y: float, text: str, color: str, above: bool, box: bool = False) -> None:
    """A point's value above or below y, in its color; with box, on a tight, slightly see-through box of the panel
    color, so a line passing behind the text does not cross the digits."""
    bbox = {"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.3, "alpha": 0.85} if box else None
    ax.annotate(text, (x, y), xytext=(0, 9 if above else -13), textcoords="offset points", ha="center",
                va="bottom" if above else "top", fontsize=FONT["annotation"] + 1, fontweight="bold", color=color, zorder=8,
                bbox=bbox)


def spread(seed_values: list[float]) -> tuple[float, float]:
    """The mean and the SD (n - 1) over the seeds."""
    return statistics.mean(seed_values), statistics.stdev(seed_values)


SD_STYLE = {"fmt": "none", "elinewidth": 1.8, "capsize": 3, "capthick": 1.8}


def mean_point(ax, x: float, seed_values: list[float], color: str, *, filled: bool = True, marker: str = "o",
               sd_on_top: bool = False) -> float:
    """The mean over the seeds as a large marker (open or filled), with a capped vertical line from mean - SD to mean + SD;
    sd_on_top draws the line over another point's line at the same x (the shorter of two, so neither hides the other)."""
    mean, sd = spread(seed_values)
    ax.errorbar(x, mean, yerr=sd, ecolor=color, zorder=5.5 if sd_on_top else 5, **SD_STYLE)
    ax.scatter(x, mean, s=MEAN_DOT, marker=marker, facecolor=color if filled else "white", edgecolor=color, linewidth=2.2,
               zorder=6)
    return mean


def sd_handle(ax, color: str, label: str):
    """A legend entry drawn like the plot's SD lines: an empty errorbar (nothing appears in the panel)."""
    handle = ax.errorbar([float("nan")], [float("nan")], yerr=[1.0], ecolor=color, **SD_STYLE)
    handle.set_label(label)
    return handle


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote figures/{name}.png, .pdf")


def eval_panel(ax, tables: dict, metric: str, eval_condition: str, title: str, ylabel: str | None) -> None:
    """The base model (gray diamond) joined to round 1 (one point, the same run for both probes), then one line per probe over
    rounds 2-5; each point the mean over the two lines with a line for ± 1 SD, its value written above or below."""
    rows = {probe: {seed: tables[probe][seed][metric][eval_condition] for seed in SEEDS} for probe in PROBES}
    for seed in SEEDS:  # round 1 is the same run in both experiments
        assert rows["frozen"][seed]["after_rl"][0] == rows["retrained"][seed]["after_rl"][0], (metric, eval_condition, seed)
    base_values = [value for row in rows["frozen"].values() for value in (row["base"] if isinstance(row["base"], list) else [row["base"]])]
    base = 100 * statistics.mean(base_values)
    plotted = [base] + [100 * row["after_rl"][index] for probe in PROBES for row in rows[probe].values()
                        for index in range(len(ROUNDS))]
    ax.scatter(0, base, s=MEAN_DOT + 20, marker="D", color=BASE_COLOR, zorder=6)
    seeds = {probe: [[100 * rows[probe][seed]["after_rl"][index] for seed in SEEDS] for index in range(len(ROUNDS))]
             for probe in PROBES}
    round1 = mean_point(ax, 1, seeds["frozen"][0], SHARED_COLOR)
    means = {probe: [round1] for probe in PROBES}
    for index in range(1, len(ROUNDS)):
        for probe in PROBES:
            shorter = min(PROBES, key=lambda name: spread(seeds[name][index])[1])
            means[probe].append(mean_point(ax, index + 1, seeds[probe][index], PROBE_STYLE[probe]["color"],
                                           sd_on_top=probe == shorter))
    ax.plot([0, 1], [base, round1], color=SHARED_COLOR, linewidth=2.4, zorder=4)  # the base model to round 1
    write_value(ax, 0, base, f"{base:.1f}", BASE_COLOR, above=round1 < base, box=True)  # off the line to round 1
    for probe in PROBES:  # the frozen line over the retrained one where they share round 1
        xs = list(range(1, len(ROUNDS) + 1))
        ax.plot(xs, means[probe], color=PROBE_STYLE[probe]["color"], linewidth=2.4, zorder=4.5 if probe == "frozen" else 4)
    write_value(ax, 1, sum(spread(seeds["frozen"][0])), f"{round1:.1f}", SHARED_COLOR, above=True, box=True)
    for index in range(1, len(ROUNDS)):  # the higher probe's value above the column's highest SD end, the other's below the lowest
        higher = max(PROBES, key=lambda probe: means[probe][index])
        for probe in PROBES:
            above = probe == higher
            column = [value for name in PROBES for mean, sd in [spread(seeds[name][index])] for value in (mean - sd, mean + sd)]
            anchor = max(column) if above else min(column)
            write_value(ax, index + 1, anchor, f"{means[probe][index]:.1f}", PROBE_STYLE[probe]["color"],
                        above=above, box=True)
    ax.set_xticks(range(len(ROUNDS) + 1), ["Base\nmodel", *(f"Round\n{r}" for r in range(1, len(ROUNDS) + 1))])
    ax.set_xlim(-0.5, len(ROUNDS) + 0.5)
    style_axis(ax, title=title, ylabel=ylabel)
    percent_axis(ax, plotted)


def fig1() -> None:
    tables = {probe: {seed: json.loads(ROUND_TABLES[probe][seed].read_text())["mean_sd"] for seed in SEEDS} for probe in PROBES}
    fig, axes = plt.subplots(1, len(EVAL_PANELS), figsize=(27, 7.5))  # each panel its own range
    for index, (ax, (metric, eval_condition, title)) in enumerate(zip(axes, EVAL_PANELS)):
        eval_panel(ax, tables, metric, eval_condition, title, "Rate (%)" if index == 0 else None)
    handles = [
        Line2D([], [], marker="D", linestyle="", markersize=9, color=BASE_COLOR, label="Base model"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=SHARED_COLOR,
               label="Round 1: the same run for both probes"),
        *(Line2D([], [], marker="o", color=PROBE_STYLE[probe]["color"], linewidth=2.4, markersize=9,
                 label=PROBE_STYLE[probe]["label"]) for probe in PROBES),
        sd_handle(axes[0], COLORS["ink"], "± 1 SD over seeds 42 and 43; dots: their mean"),
    ]
    fig.legend(handles=handles, labels=[handle.get_label() for handle in handles], frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.075), ncol=3, prop={"size": 12, "weight": "bold"})
    fig.suptitle("Eval Answers After Each Round: Frozen vs Retrained Probe", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "fig1_eval_answers")


def probe_points(probe: str, tables: dict) -> list[tuple[dict, dict]]:
    """Per round, (start, end) per seed: the round's penalty probe on the start model's and on its own eval answers."""
    frozen = tables["frozen"]
    points = []
    for round_index, run in enumerate(ROUNDS):
        start, end = {}, {}
        for seed in SEEDS:
            if probe == "frozen":
                start[seed] = frozen[seed]["round1"]["base_answers"] if run == "round1" else frozen[seed][ROUNDS[round_index - 1]]["round_answers"]
                end[seed] = frozen[seed][run]["round_answers"]
            else:
                retrained = tables["retrained"][seed][run]["scores"]
                start[seed] = frozen[seed]["round1"]["base_answers"] if run == "round1" else retrained["own_test_set"]
                end[seed] = retrained["round_probe_round_answers"]
                if run != "round1":  # the start answers are the previous round's end answers
                    previous_end = tables["retrained"][seed][ROUNDS[round_index - 1]]["scores"]["round_probe_round_answers"]
                    assert (start[seed]["n_biased"], start[seed]["n_unbiased"]) == (previous_end["n_biased"], previous_end["n_unbiased"]), (seed, run)
        points.append((start, end))
    return points


def probe_panel(ax, points: list[tuple[dict, dict]], probe: str, series: tuple[tuple[str, str], ...], decimals: int,
                label_starts: bool) -> None:
    """Round R: a line from its start (open dot, x = R-1) to its end (filled dot, x = R), per series. The start of the
    frozen probe's rounds 2-5 equals the previous end, so only round 1's start is written for it."""
    for round_index, (start, end) in enumerate(points):
        for key, color in series:
            start_mean = statistics.mean(start[seed][key] for seed in SEEDS)
            end_mean = statistics.mean(end[seed][key] for seed in SEEDS)
            # off the previous round's end dot at the same x (round 1's start has none there)
            start_x = round_index + (START_SHIFT if probe == "retrained" and round_index > 0 else 0)
            ax.plot([start_x, round_index + 1], [start_mean, end_mean], color=color, linewidth=2.2, zorder=3)
            if label_starts or round_index == 0:  # the frozen probe's later starts are the previous filled end dots
                mean_point(ax, start_x, [start[seed][key] for seed in SEEDS], color, filled=False)
            mean_point(ax, round_index + 1, [end[seed][key] for seed in SEEDS], color)
            falling = end_mean < start_mean  # 261104's placement: start right of its dot, end left of its dot, both off the line
            if label_starts or round_index == 0:
                side_value(ax, start_x, start_mean, f"{start_mean:.{decimals}f}", color, right=True, up=falling)
            side_value(ax, round_index + 1, end_mean, f"{end_mean:.{decimals}f}", color, right=False, up=not falling)
    ax.set_xticks(range(len(ROUNDS) + 1), ["Base\nmodel", *(f"Round {r}\nmodel" for r in range(1, len(ROUNDS) + 1))])
    ax.set_xlim(-0.4, len(ROUNDS) + 0.4)
    round_axis = ax.secondary_xaxis("top")  # names each round's line, between the two models it joins
    round_axis.set_xticks([index + 0.5 for index in range(len(ROUNDS))], [f"Round {r}" for r in range(1, len(ROUNDS) + 1)])
    round_axis.tick_params(length=0, labelsize=FONT["tick"], pad=4)
    round_axis.spines["top"].set_visible(False)


def side_value(ax, x: float, y: float, text: str, color: str, right: bool, up: bool) -> None:
    """A dot's value beside it (a start dot's to the right, an end dot's to the left), moved up or down away from the
    round's line, so the labels of two rounds' dots at one x sit on opposite sides."""
    ax.annotate(text, (x, y), xytext=(18 if right else -18, 9 if up else -9), textcoords="offset points",
                ha="left" if right else "right", va="center", fontsize=FONT["annotation"] + 1, fontweight="bold", color=color,
                zorder=8)


def fig2() -> None:
    tables = {probe: {seed: json.loads(PROBE_TABLES[probe][seed].read_text()) for seed in SEEDS} for probe in PROBES}
    tables["frozen"] = {seed: table["rounds"] for seed, table in tables["frozen"].items()}
    fig, axes = plt.subplots(len(PROBES), 2, figsize=(20, 16))
    for probe, row_axes in zip(PROBES, axes):
        points = probe_points(probe, tables)
        style = PROBE_STYLE[probe]
        name = "Frozen Probe" if probe == "frozen" else "Retrained Probe"
        probe_panel(row_axes[0], points, probe, (("mean_biased", style["color"]), ("mean_unbiased", style["light"])), 2,
                    label_starts=probe == "retrained")
        probe_panel(row_axes[1], points, probe, (("auc", AUC_COLOR),), 3, label_starts=probe == "retrained")
        style_axis(row_axes[0], ylabel="Mean probe score (0 to 1)")
        row_axes[0].set_title(f"{name}: Mean Probe Score", fontsize=FONT["panel_title"], fontweight="bold", y=1.0, pad=30)
        row_axes[0].set_ylim(-0.04, 0.85)
        style_axis(row_axes[1], ylabel="AUC")
        row_axes[1].set_title(f"{name}: AUC, Biased vs Unbiased Answers", fontsize=FONT["panel_title"], fontweight="bold",
                              y=1.0, pad=30)
        row_axes[1].set_ylim(0.65, 0.95)
    def dot(color, **style):
        return Line2D([], [], marker="o", linestyle="", markersize=9, color=color, **style)

    entries = [  # filled column by column: biased over unbiased, AUC over the SD line, start over end
        ((dot(PROBE_STYLE["frozen"]["color"]), dot(PROBE_STYLE["retrained"]["color"])), "Biased answers (teal: frozen probe, orange: retrained probe)"),
        ((dot(PROBE_STYLE["frozen"]["light"]), dot(PROBE_STYLE["retrained"]["light"])), "Unbiased answers (lighter shade of the same colors)"),
        (dot(AUC_COLOR), "AUC, biased vs unbiased answers"),
        (sd_handle(axes[0][0], COLORS["ink"], ""), "± 1 SD over seeds 42 and 43"),
        (dot(COLORS["ink"], markerfacecolor="white", markeredgewidth=2.2),
         "Start of round"),
        (dot(COLORS["ink"]), "End of round"),
    ]
    fig.legend(handles=[handle for handle, _ in entries], labels=[label for _, label in entries],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.06), ncol=3, handlelength=2.6, prop={"size": 12, "weight": "bold"})
    fig.suptitle("The Penalty Probe at the Start and End of Each Round", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig2_round_probe")


def fig4() -> None:
    training = json.loads(MEMORIZATION.read_text())["training"]
    runs = {seed: training[f"frozen s{seed}"] for seed in SEEDS}
    fig, axes = plt.subplots(1, len(LABEL_CELLS), figsize=(16, 7.5), sharey=True)
    color = PROBE_STYLE["frozen"]["color"]
    for index, (ax, (title, cells)) in enumerate(zip(axes, LABEL_CELLS)):
        means = {}
        seed_values = {cell: [[100 * runs[seed][round_index]["windows"]["60-127"]["cells"][cell]["share_naming_label"]
                               for seed in SEEDS] for round_index in range(len(ROUNDS))] for cell, _, _ in cells}
        for cell, linestyle, marker in cells:
            means[cell] = [mean_point(ax, round_index + 1, values, color, marker=marker,
                                      sd_on_top=cell == min(seed_values, key=lambda name: spread(seed_values[name][round_index])[1]))
                           for round_index, values in enumerate(seed_values[cell])]
            ax.plot(range(1, len(ROUNDS) + 1), means[cell], color=color, linestyle=linestyle, linewidth=2.4, zorder=4)
        for round_index in range(len(ROUNDS)):  # beside the point, off its seed markers: higher line upper left, lower right
            higher = max(means, key=lambda cell: means[cell][round_index])
            for cell, _, _ in cells:
                side_value(ax, round_index + 1, means[cell][round_index], f"{means[cell][round_index]:.1f}", color,
                           right=cell != higher, up=cell == higher)
        ax.set_xticks(range(1, len(ROUNDS) + 1), [f"Round {r}" for r in range(1, len(ROUNDS) + 1)])
        ax.set_xlim(0.3, len(ROUNDS) + 0.6)
        style_axis(ax, title=title,
                   ylabel="Training answers that give the training label" if index == 0 else None)
        ax.set_ylim(0, 105)  # both panels on one 0-100% scale: the shares are compared across panels
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    handles = [
        Line2D([], [], color=color, linestyle="-", marker="o", markersize=9, linewidth=2.4,
               label="Cue at the training label (unflipped: the correct answer; flipped: the wrong answer)"),
        Line2D([], [], color=color, linestyle="--", marker="s", markersize=9, linewidth=2.4,
               label="Cue at the other option (unflipped: the wrong answer; flipped: the correct answer)"),
        sd_handle(axes[0], color, "± 1 SD over seeds 42 and 43; markers: their mean"),
    ]
    fig.legend(handles=handles, labels=[handle.get_label() for handle in handles], frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.115), ncol=1,
               handlelength=4, prop={"size": 12, "weight": "bold"})
    fig.suptitle("Training Answers That Give the Training Label, Frozen Probe", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig4_training_label")


def main() -> None:
    fig1()
    fig2()
    fig4()


if __name__ == "__main__":
    main()
