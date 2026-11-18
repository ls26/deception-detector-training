"""Figures of the main results: the frozen probe against the retrained probe over five rounds, with the training labels
redrawn every round, seeds 42, 43, 45 and 46 together and seed 45 alone, in the style of
/workspace/plot_style_guide.ipynb (261107's plot_results.py). Color means the probe in every figure: teal = frozen,
orange = retrained.

    /workspace/obfuscation-atlas/.venv/bin/python plot_results.py     # -> figures/*.png and figures/*.pdf

All values come from runs/comparison_four_seeds/comparison.json (memorization.py; seeds 45 and 46 from this folder,
seeds 42 and 43 from 261109 (retrained) and 261108 (frozen); round 1 is the same run for both probes; under
"fixed_labels", 261107's frozen and 261104's retrained runs of seeds 42 and 43, which trained every round on the original
labels), apart from the base model's values (each run's seed_table.json). The four-seed figures show each point as the
mean over the seeds with a capped vertical line for ± 1 SD (n - 1). The seed-45 figures show seed 45's values (solid
lines, values written, no SD lines) against the mean ± 1 SD of seeds 42, 43 and 46 (dotted lines in the same colors,
smaller markers, thinner and lighter SD lines, no values written).

  fig1_eval_answers         2 x 3 panels. Top: bias, cue agreement rate and true accuracy on the training cue mix.
  fig1_eval_answers_seed45  Bottom: true accuracy with no cue, with the cue at the correct answer and at the wrong answer.
                            Lines over rounds 1-5, one per probe; round 1 is one point (the same run for both probes),
                            joined to the base model (gray diamond: the mean of the frozen runs' "base" values in their
                            seed_table.json, memorization.seed_table_path, over the solid lines' seeds; on the bias panel
                            one value per run). The dotted lines of the seed-45 figure run over rounds 1-5 only. Every
                            panel spans 60 points of its own range, so slopes compare across panels.
  fig2_round_probe          2 x 2 panels, one row per probe: the round's penalty probe on the cued eval answers that
  fig2_round_probe_seed45   name the cued option, both cue conditions pooled: the mean probe score of biased and of
                            unbiased answers (darker and lighter shade of the probe's color) and the AUC between them
                            (purple). Round R is a line from its start (open dot: the eval answers of the model the round
                            starts from) to its end (filled dot: round R's own eval answers); the x axis is the model
                            whose answers are scored, so round R's start sits at round R-1's end (in the retrained row a
                            little to its right). Round 1's start, in both rows: ROUND1_START_ROWS (the "penalty probe,
                            Initial" round-1 values: round 1's probe on the base model's eval answers, as in 261107; see
                            memorization.py). Retrained: the start of rounds 2-5 = the "penalty probe, Initial" rows (round
                            R's probe on round R-1's eval answers), end = the "penalty probe, Final" rows. Frozen: every
                            end = the Final rows; the start of rounds 2-5 is the previous end (the same probe on the same
                            answers), so only round 1's start is drawn.
  fig3_training_label       1 panel: on flipped questions (training label = the wrong answer) with the cue at the
                            correct answer, the share of training answers that give the training label, all 128 steps,
                            over rounds 1-5, for both probes with labels redrawn every round (solid, seeds 42, 43, 45 and
                            46) and with labels fixed in every round (dashed, open squares, seeds 42 and 43). Each pair
                            shares its round 1 (one run per seed): an ink dot for the redrawn lines, an ink open square
                            for the fixed ones (the same runs as two of the four seeds, so it differs from the dot only
                            by the seeds averaged; drawn over the dot and larger, so both show).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
import memorization  # noqa: E402

FIG_DIR = EXP_DIR / "figures"
COMPARISON = EXP_DIR / "runs" / "comparison_four_seeds" / "comparison.json"
FOUR_SEEDS = (42, 43, 45, 46)
SEED_45 = (45,)
OTHER_SEEDS = (42, 43, 46)  # the seed-45 figures' dotted lines
FIXED_SEEDS = (42, 43)  # comparison.json's "fixed_labels"
ROUNDS = (1, 2, 3, 4, 5)
PROBES = ("frozen", "refit")  # comparison.json's conditions; "refit" is drawn as the retrained probe
MEMORIZATION_ROW = memorization.MEMORIZATION_ROW

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
    "frozen": {"color": "#3f8f84", "light": "#93cbc2", "label": "Frozen probe (round 1's probe in every round)",
               "name": "Frozen Probe"},
    "refit": {"color": "#c96f36", "light": "#eab089", "label": "Retrained probe (a new probe each round)",
              "name": "Retrained Probe"},
}
SHARED_COLOR = COLORS["ink"]  # round 1: the same run for both probes
AUC_COLOR = "#8a62a8"  # fig2: the AUC, in both rows, apart from the probes' colors
START_SHIFT = 0.14  # fig2, retrained row: start dots just right of the previous round's end dots at the same x
BASE_COLOR = COLORS["gray"]
MEAN_DOT = 70
EVAL_PANELS = (  # (comparison.json row, seed_table.json metric and cue condition for the base model, panel title)
    (("bias, training cue mix", "bias", "train_mix", "Bias\nTraining Cue Mix"),
     ("cue agreement rate, training cue mix", "followed_cue", "train_mix", "Cue Agreement Rate\nTraining Cue Mix"),
     ("true accuracy, training cue mix", "accuracy", "train_mix", "True Accuracy\nTraining Cue Mix")),
    (("true accuracy, no cue", "accuracy", "none", "True Accuracy\nNo Cue"),
     ("true accuracy, cue at the correct answer", "accuracy", "cue_gold", "True Accuracy\nCue at the Correct Answer"),
     ("true accuracy, cue at the wrong answer", "accuracy", "cue_dist", "True Accuracy\nCue at the Wrong Answer")),
)
PROBE_ROWS = {  # fig2: key -> (Initial row, Final row) of comparison.json
    "biased": ("penalty probe, Initial: mean score of biased answers", "penalty probe, Final: mean score of biased answers"),
    "unbiased": ("penalty probe, Initial: mean score of unbiased answers",
                 "penalty probe, Final: mean score of unbiased answers"),
    "auc": ("penalty probe, Initial: AUC biased vs unbiased", "penalty probe, Final: AUC biased vs unbiased"),
}
# fig2: round 1's start (round 1's probe before round 1), the same for both probes: key -> comparison.json row, read at
# round 1: the Initial rows, round 1's probe on the base model's eval answers (memorization.py).
ROUND1_START_ROWS = {key: rows[0] for key, rows in PROBE_ROWS.items()}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT["tick"],
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "savefig.dpi": EXPORT_DPI,
})


def seed_words(seeds: tuple[int, ...]) -> str:
    """'seed 45' or 'seeds 42, 43, 45 and 46'."""
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    return "seeds " + ", ".join(str(seed) for seed in seeds[:-1]) + f" and {seeds[-1]}"


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
    """A 60-point range starting at the tens below the panel's lowest value, moved down if it would pass 100%; when a
    value is within 4 points of 100% the range ends at 104% instead, so the value keeps room for its label."""
    low = max(0, 10 * ((min(values) - 4) // 10))
    low = min(low, 100 - PANEL_SPAN)
    bottom = low - 4 if low == 0 else low  # a little room under 0% for values written below a point
    if max(values) + 4 > 100:
        bottom = 104 - PANEL_SPAN
        low = 10 * (-(-bottom // 10))  # first tick: the tens at or above the bottom
    top = bottom + PANEL_SPAN  # every panel spans exactly 60 points
    assert min(values) - 2 >= bottom and max(values) + 4 <= top, (min(values), max(values), bottom, top)
    ax.set_ylim(bottom, top)
    ax.set_yticks(range(int(low), int(min(top, 100)) + 1, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def write_value(ax, x: float, y: float, text: str, color: str, above: bool, box: bool = False, extra: float = 0) -> None:
    """A point's value above or below y, in its color; with box, on a tight, slightly see-through box of the panel
    color, so a line passing behind the text does not cross the digits; extra: that many points farther from y."""
    bbox = {"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.3, "alpha": 0.85} if box else None
    ax.annotate(text, (x, y), xytext=(0, 9 + extra if above else -13 - extra), textcoords="offset points", ha="center",
                va="bottom" if above else "top", fontsize=FONT["annotation"] + 1, fontweight="bold", color=color, zorder=8,
                bbox=bbox)


def spread(seed_values: list[float]) -> tuple[float, float]:
    """The mean and the SD (n - 1) over the seeds; one seed has no SD (0 here, and no SD line is drawn)."""
    if len(seed_values) == 1:
        return seed_values[0], 0.0
    return statistics.mean(seed_values), statistics.stdev(seed_values)


SD_STYLE = {"fmt": "none", "elinewidth": 1.8, "capsize": 3, "capthick": 1.8}
# the seed-45 figures' comparison seeds: dotted lines, smaller markers, thinner and lighter SD lines, under seed 45's
FAINT_SD_STYLE = {"fmt": "none", "elinewidth": 1.0, "capsize": 2, "capthick": 1.0, "alpha": 0.5}
FAINT_DOT = 28
FAINT_LINE = {"linestyle": ":", "linewidth": 2.0, "alpha": 0.8, "zorder": 2}


def mean_point(ax, x: float, seed_values: list[float], color: str, *, filled: bool = True, marker: str = "o",
               sd_on_top: bool = False, faint: bool = False) -> float:
    """The mean over the seeds as a large marker (open or filled), with a capped vertical line from mean - SD to mean + SD
    when there is more than one seed; sd_on_top draws the line over another point's line at the same x. faint: the
    comparison seeds' smaller marker and thinner, lighter SD line, under the main points."""
    mean, sd = spread(seed_values)
    if faint:
        ax.errorbar(x, mean, yerr=sd, ecolor=color, zorder=2.5, **FAINT_SD_STYLE)
        ax.scatter(x, mean, s=FAINT_DOT, marker=marker, facecolor=color if filled else "white", edgecolor=color,
                   linewidth=1.4, alpha=0.8, zorder=2.6)
        return mean
    if len(seed_values) > 1:
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


def seed_rows(comparison: dict, probe: str, row: str, seeds: tuple[int, ...]) -> list[list[float]]:
    """Per round, the seeds' values of a comparison.json row."""
    return [[comparison[probe][f"s{seed}"]["rows"][row][index] for seed in seeds] for index in range(len(ROUNDS))]


def base_model_value(metric: str, eval_condition: str, seeds: tuple[int, ...]) -> float:
    """The mean of the frozen runs' base-model values (one per run on the bias rows, one per seed table elsewhere)."""
    values = []
    for seed in seeds:
        base = json.loads(memorization.seed_table_path("frozen", seed).read_text())["mean_sd"][metric][eval_condition]["base"]
        values += base if isinstance(base, list) else [base]
    return statistics.mean(values)


# Seed-45 figures: each of seed 45's values goes on its preferred side unless that side's box overlaps a mark more than
# the other side's (marks: the comparison seeds' dotted points, SD lines and lines; seed 45's own points and lines; the
# values already placed). Sizes in points, converted with the axes' size before tight_layout (close enough to pick a side).
VALUE_HEIGHT = 12  # points, a written value's box
VALUE_CHARACTER_WIDTH = 6.8  # points per character


def points_scale(ax) -> tuple[float, float]:
    """Data units per point along x and y."""
    width, height = ax.figure.get_size_inches()
    position = ax.get_position()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x1 - x0) / (position.width * width * 72), (y1 - y0) / (position.height * height * 72)


def point_mark(ax, x: float, y: float, sd: float, radius: float) -> tuple:
    """A marker of the given radius (points) with its SD line, as a box (x0, x1, y0, y1) in data units."""
    unit_x, unit_y = points_scale(ax)
    return (x - radius * unit_x, x + radius * unit_x, y - sd - radius * unit_y, y + sd + radius * unit_y)


def line_marks(ax, xs: list[float], ys: list[float], half_width: float = 2.0, steps: int = 20) -> list[tuple]:
    """A line through the points, as small boxes along it."""
    unit_x, unit_y = points_scale(ax)
    marks = []
    for (x_a, y_a), (x_b, y_b) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
        for step in range(steps + 1):
            x, y = x_a + (x_b - x_a) * step / steps, y_a + (y_b - y_a) * step / steps
            marks.append((x - half_width * unit_x, x + half_width * unit_x, y - half_width * unit_y, y + half_width * unit_y))
    return marks


def value_box(ax, x: float, y: float, text: str, side: str, up: bool, extra: float) -> tuple:
    """The box a written value covers: side "center" for write_value (above: from 9 + extra points over y; below: from
    13 + extra points under y), "right" or "left" for side_value (18 points beside x, centered 9 + extra points over or
    under y)."""
    unit_x, unit_y = points_scale(ax)
    width, height = VALUE_CHARACTER_WIDTH * len(text) * unit_x, VALUE_HEIGHT * unit_y
    if side == "center":
        x0 = x - width / 2
        y0 = y + (9 + extra) * unit_y if up else y - (13 + extra) * unit_y - height
    else:
        x0 = x + 18 * unit_x if side == "right" else x - 18 * unit_x - width
        y0 = y + (9 + extra if up else -9 - extra) * unit_y - height / 2
    return (x0, x0 + width, y0, y0 + height)


def overlap(box: tuple, marks: list[tuple]) -> float:
    return sum(max(0.0, min(box[1], mark[1]) - max(box[0], mark[0])) * max(0.0, min(box[3], mark[3]) - max(box[2], mark[2]))
               for mark in marks)


EXTRA_DISTANCES = (0, 12, 24)  # points: a value moves this much farther from its point when nearer places are taken


def place_value(ax, x: float, y: float, text: str, side: str, preferred_up: bool, marks: list[tuple]) -> tuple[bool, float]:
    """(up, extra): the first place clear of the marks, trying the preferred side, then the other, at each distance in
    turn; if none is clear, the one overlapping least. The chosen box joins the marks, so later values keep off it."""
    places = [(up, extra) for extra in EXTRA_DISTANCES for up in (preferred_up, not preferred_up)]
    boxes = {place: value_box(ax, x, y, text, side, *place) for place in places}
    clear = [place for place in places if overlap(boxes[place], marks) == 0]
    place = clear[0] if clear else min(places, key=lambda place: overlap(boxes[place], marks))
    marks.append(boxes[place])
    return place


def faint_values(comparison: dict, row: str, others: tuple[int, ...]) -> dict:
    """The comparison seeds' values in %, per probe and round."""
    values = {probe: [[100 * value for value in round_values] for round_values in seed_rows(comparison, probe, row, others)]
              for probe in PROBES}
    assert values["frozen"][0] == values["refit"][0], (row, "round 1 differs between the probes")
    return values


def faint_lines(ax, values: dict) -> list[tuple]:
    """The comparison seeds over rounds 1-5: round 1 (ink, the same run for both probes), then one dotted line per probe;
    mean ± 1 SD, no values written. Returns the marks they make."""
    round1 = mean_point(ax, 1, values["frozen"][0], SHARED_COLOR, faint=True)
    marks = [point_mark(ax, 1, round1, spread(values["frozen"][0])[1], 3.5)]
    for probe in PROBES:
        color = PROBE_STYLE[probe]["color"]
        means = [round1] + [mean_point(ax, index + 1, values[probe][index], color, faint=True)
                            for index in range(1, len(ROUNDS))]
        ax.plot(range(1, len(ROUNDS) + 1), means, color=color, **FAINT_LINE)
        marks += [point_mark(ax, index + 1, means[index], spread(values[probe][index])[1], 3.5)
                  for index in range(1, len(ROUNDS))]
        marks += line_marks(ax, list(range(1, len(ROUNDS) + 1)), means)
    return marks


def faint_handle(others: tuple[int, ...]):
    """The legend entry of the comparison seeds' dotted lines: (handles, label)."""
    lines = tuple(Line2D([], [], color=PROBE_STYLE[probe]["color"], marker="o", markersize=5, **FAINT_LINE)
                  for probe in PROBES)
    return lines, f"{seed_words(others).capitalize()}: mean (dotted, same colors) ± 1 SD"


def eval_panel(ax, comparison: dict, seeds: tuple[int, ...], panel: tuple, ylabel: str | None,
               others: tuple[int, ...] = ()) -> None:
    """The base model (gray diamond) joined to round 1 (one point, the same run for both probes), then one line per probe
    over rounds 2-5; each point the mean over the seeds with a line for ± 1 SD, its value written above or below. With
    others, their mean ± 1 SD as dotted lines over rounds 1-5, without values, and seed 45's values placed off them."""
    row, metric, eval_condition, title = panel
    seed_values = {probe: [[100 * value for value in values] for values in seed_rows(comparison, probe, row, seeds)]
                   for probe in PROBES}
    assert seed_values["frozen"][0] == seed_values["refit"][0], (row, "round 1 differs between the probes")
    base = 100 * base_model_value(metric, eval_condition, seeds)
    plotted = [base] + [value for probe in PROBES for values in seed_values[probe] for value in values]
    other_values = faint_values(comparison, row, others) if others else {}
    plotted += [value for probe in other_values for values in other_values[probe] for value in values]
    ax.set_xticks(range(len(ROUNDS) + 1), ["Base\nmodel", *(f"Round\n{r}" for r in ROUNDS)])
    ax.set_xlim(-0.5, len(ROUNDS) + 0.5)
    style_axis(ax, title=title, ylabel=ylabel)
    percent_axis(ax, plotted)  # the limits first: the seed-45 figure places its values in points
    marks = faint_lines(ax, other_values) if others else []
    ax.scatter(0, base, s=MEAN_DOT + 20, marker="D", color=BASE_COLOR, zorder=6)
    round1 = mean_point(ax, 1, seed_values["frozen"][0], SHARED_COLOR)
    means = {probe: [round1] for probe in PROBES}
    for index in range(1, len(ROUNDS)):
        shorter = min(PROBES, key=lambda name: spread(seed_values[name][index])[1])
        for probe in PROBES:
            means[probe].append(mean_point(ax, index + 1, seed_values[probe][index], PROBE_STYLE[probe]["color"],
                                           sd_on_top=probe == shorter))
    ax.plot([0, 1], [base, round1], color=SHARED_COLOR, linewidth=2.4, zorder=4)  # the base model to round 1
    write_value(ax, 0, base, f"{base:.1f}", BASE_COLOR, above=round1 < base, box=True)  # off the line to round 1
    for probe in PROBES:  # the frozen line over the retrained one where they share round 1
        ax.plot(range(1, len(ROUNDS) + 1), means[probe], color=PROBE_STYLE[probe]["color"], linewidth=2.4,
                zorder=4.5 if probe == "frozen" else 4)
    if len(seeds) > 1:  # above round 1's SD line
        write_value(ax, 1, sum(spread(seed_values["frozen"][0])), f"{round1:.1f}", SHARED_COLOR, above=True, box=True)
        for index in range(1, len(ROUNDS)):  # the higher probe's value above the column's highest SD end, the other's below the lowest
            higher = max(PROBES, key=lambda probe: means[probe][index])
            column = [value for name in PROBES for mean, sd in [spread(seed_values[name][index])] for value in (mean - sd, mean + sd)]
            for probe in PROBES:
                above = probe == higher
                write_value(ax, index + 1, max(column) if above else min(column), f"{means[probe][index]:.1f}",
                            PROBE_STYLE[probe]["color"], above=above, box=True)
        return
    # one seed: its points and lines are marks too; round 1's value left of the dot (away from the two lines leaving it),
    # then per round the higher probe's value (preferably above) and the lower one's (preferably below)
    marks += [point_mark(ax, x, y, 0.0, 5.0) for x, y in [(0, base), (1, round1)]]
    marks += line_marks(ax, [0, 1], [base, round1])
    for probe in PROBES:
        marks += [point_mark(ax, index + 1, means[probe][index], 0.0, 5.0) for index in range(1, len(ROUNDS))]
        marks += line_marks(ax, list(range(1, len(ROUNDS) + 1)), means[probe])
    marks.append(value_box(ax, 0, base, f"{base:.1f}", "center", round1 < base, 0))
    text = f"{round1:.1f}"
    up, extra = place_value(ax, 1, round1, text, "left", base < round1, marks)
    side_value(ax, 1, round1, text, SHARED_COLOR, right=False, up=up, box=True, extra=extra)
    for index in range(1, len(ROUNDS)):
        for probe in sorted(PROBES, key=lambda name: -means[name][index]):  # the higher first
            text = f"{means[probe][index]:.1f}"
            higher = probe == max(PROBES, key=lambda name: means[name][index])
            above, extra = place_value(ax, index + 1, means[probe][index], text, "center", higher, marks)
            write_value(ax, index + 1, means[probe][index], text, PROBE_STYLE[probe]["color"], above=above, box=True,
                        extra=extra)


def fig1(comparison: dict, seeds: tuple[int, ...], name: str, title: str, others: tuple[int, ...] = ()) -> None:
    fig, axes = plt.subplots(len(EVAL_PANELS), len(EVAL_PANELS[0]), figsize=(19, 15))  # each panel its own range
    for row_axes, panels in zip(axes, EVAL_PANELS):
        for index, (ax, panel) in enumerate(zip(row_axes, panels)):
            eval_panel(ax, comparison, seeds, panel, "Rate (%)" if index == 0 else None, others)
    entries = [
        (Line2D([], [], marker="D", linestyle="", markersize=9, color=BASE_COLOR), "Base model"),
        (Line2D([], [], marker="o", linestyle="", markersize=9, color=SHARED_COLOR),
         "Round 1: the same run for both probes" + (f", {seed_words(seeds)}" if others else "")),
        *((Line2D([], [], marker="o", color=PROBE_STYLE[probe]["color"], linewidth=2.4, markersize=9),
           PROBE_STYLE[probe]["label"] + (f", {seed_words(seeds)}" if others else "")) for probe in PROBES),
    ]
    if len(seeds) > 1:
        entries.append((sd_handle(axes[0][0], COLORS["ink"], ""), f"± 1 SD over {seed_words(seeds)}; dots: their mean"))
    if others:
        entries.append(faint_handle(others))
        entries.append((Line2D([], [], marker="o", linestyle="", markersize=5, color=SHARED_COLOR, alpha=0.8),
                        f"Round 1, {seed_words(others)}: mean ± 1 SD"))
    fig.legend(handles=[handle for handle, _ in entries], labels=[label for _, label in entries],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.06), ncol=3, handlelength=2.6 if others else 2.0, prop={"size": 12, "weight": "bold"})
    fig.suptitle(title, fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, name)


def probe_points(comparison: dict, probe: str, seeds: tuple[int, ...]) -> list[tuple[dict, dict]]:
    """Per round, (start, end) per seed as {key: value}: the round's penalty probe on the start model's and on its own
    eval answers."""
    def value(condition: str, rows: dict, seed: int, round_index: int) -> dict:
        values = {key: comparison[condition][f"s{seed}"]["rows"][row][round_index] for key, row in rows.items()}
        assert None not in values.values(), (condition, seed, round_index, rows)
        return values

    initial = {key: rows[0] for key, rows in PROBE_ROWS.items()}
    final = {key: rows[1] for key, rows in PROBE_ROWS.items()}
    points = []
    for round_index in range(len(ROUNDS)):
        if round_index == 0:  # round 1's probe before round 1, the same for both probes
            start = {seed: value(probe, ROUND1_START_ROWS, seed, 0) for seed in seeds}
        elif probe == "frozen":  # the same probe on the same answers as the previous round's end
            start = {seed: value("frozen", final, seed, round_index - 1) for seed in seeds}
        else:
            start = {seed: value("refit", initial, seed, round_index) for seed in seeds}
        end = {seed: value(probe, final, seed, round_index) for seed in seeds}
        points.append((start, end))
    return points


def side_value(ax, x: float, y: float, text: str, color: str, right: bool, up: bool, box: bool = False,
               extra: float = 0) -> None:
    """A dot's value beside it (a start dot's to the right, an end dot's to the left), moved up or down away from the
    round's line, so the labels of two rounds' dots at one x sit on opposite sides; with box, on write_value's box (the
    seed-45 figures, where the comparison seeds' dotted lines pass behind the values); extra: that many points farther
    up or down."""
    bbox = {"facecolor": PANEL_BACKGROUND, "edgecolor": "none", "pad": 0.3, "alpha": 0.85} if box else None
    ax.annotate(text, (x, y), xytext=(18 if right else -18, 9 + extra if up else -9 - extra), textcoords="offset points",
                ha="left" if right else "right", va="center", fontsize=FONT["annotation"] + 1, fontweight="bold", color=color,
                zorder=8, bbox=bbox)


def probe_panel(ax, points: list[tuple[dict, dict]], probe: str, series: tuple[tuple[str, str], ...], decimals: int,
                seeds: tuple[int, ...], faint: bool = False, marks: list[tuple] | None = None) -> list[tuple]:
    """Round R: a line from its start (open dot, x = R-1) to its end (filled dot, x = R), per series. The start of the
    frozen probe's rounds 2-5 equals the previous end, so only round 1's start is drawn for it. faint: the comparison
    seeds, dotted, without values. marks (the seed-45 figure): what the values keep off (place_value), together with this
    call's own points and lines; the values then go on the panel-colored box. Returns the marks this call drew."""
    draws_starts = probe == "refit"
    drawn, values = [], []  # values: (x, y, text, color, right, preferred up)
    for round_index, (start, end) in enumerate(points):
        for key, color in series:
            start_values = [start[seed][key] for seed in seeds]
            end_values = [end[seed][key] for seed in seeds]
            start_mean, end_mean = spread(start_values)[0], spread(end_values)[0]
            # off the previous round's end dot at the same x (round 1's start has none there)
            start_x = round_index + (START_SHIFT if draws_starts and round_index > 0 else 0)
            line_style = FAINT_LINE if faint else {"linewidth": 2.2, "zorder": 3}
            ax.plot([start_x, round_index + 1], [start_mean, end_mean], color=color, **line_style)
            drawn += line_marks(ax, [start_x, round_index + 1], [start_mean, end_mean])
            radius = 3.5 if faint else 5.0
            if draws_starts or round_index == 0:
                mean_point(ax, start_x, start_values, color, filled=False, faint=faint)
                drawn.append(point_mark(ax, start_x, start_mean, spread(start_values)[1], radius))
            mean_point(ax, round_index + 1, end_values, color, faint=faint)
            drawn.append(point_mark(ax, round_index + 1, end_mean, spread(end_values)[1], radius))
            if faint:
                continue
            falling = end_mean < start_mean  # 261104's placement: start right of its dot, end left of its dot, both off the line
            if draws_starts or round_index == 0:
                values.append((start_x, start_mean, f"{start_mean:.{decimals}f}", color, True, falling))
            values.append((round_index + 1, end_mean, f"{end_mean:.{decimals}f}", color, False, not falling))
    keep_off = None if marks is None else marks + drawn
    for x, y, text, color, right, up in values:
        extra = 0
        if keep_off is not None:
            up, extra = place_value(ax, x, y, text, "right" if right else "left", up, keep_off)
        side_value(ax, x, y, text, color, right=right, up=up, box=keep_off is not None, extra=extra)
    return drawn


def probe_axis(ax) -> None:
    """fig2's x axis: the model whose answers are scored, and above the panel each round's name between its two models."""
    ax.set_xticks(range(len(ROUNDS) + 1), ["Base\nmodel", *(f"Round {r}\nmodel" for r in ROUNDS)])
    ax.set_xlim(-0.4, len(ROUNDS) + 0.4)
    round_axis = ax.secondary_xaxis("top")  # names each round's line, between the two models it joins
    round_axis.set_xticks([index + 0.5 for index in range(len(ROUNDS))], [f"Round {r}" for r in ROUNDS])
    round_axis.tick_params(length=0, labelsize=FONT["tick"], pad=4)
    round_axis.spines["top"].set_visible(False)


def fig2(comparison: dict, seeds: tuple[int, ...], name: str, title: str, others: tuple[int, ...] = ()) -> None:
    fig, axes = plt.subplots(len(PROBES), 2, figsize=(20, 16))
    for probe, row_axes in zip(PROBES, axes):
        style = PROBE_STYLE[probe]
        for ax in row_axes:  # the limits first: the seed-45 figure places its values in points
            probe_axis(ax)
        style_axis(row_axes[0], ylabel="Mean probe score (0 to 1)")
        row_axes[0].set_title(f"{style['name']}: Mean Probe Score", fontsize=FONT["panel_title"], fontweight="bold",
                              y=1.0, pad=30)
        row_axes[0].set_ylim(-0.04, 0.9)
        style_axis(row_axes[1], ylabel="AUC")
        row_axes[1].set_title(f"{style['name']}: AUC, Biased vs Unbiased Answers", fontsize=FONT["panel_title"],
                              fontweight="bold", y=1.0, pad=30)
        row_axes[1].set_ylim(0.62, 0.97)
        panel_series = ((("biased", style["color"]), ("unbiased", style["light"])), (("auc", AUC_COLOR),))
        faint_marks = [None, None]  # per panel, the comparison seeds' marks (the seed-45 figure)
        if others:
            other_points = probe_points(comparison, probe, others)
            faint_marks = [probe_panel(ax, other_points, probe, series, decimals, others, faint=True)
                           for ax, series, decimals in zip(row_axes, panel_series, (2, 3))]
        points = probe_points(comparison, probe, seeds)
        for ax, series, decimals, marks in zip(row_axes, panel_series, (2, 3), faint_marks):
            probe_panel(ax, points, probe, series, decimals, seeds, marks=marks)

    def dot(color, **style):
        return Line2D([], [], marker="o", linestyle="", markersize=9, color=color, **style)

    entries = [  # filled column by column: biased over unbiased, AUC over the SD line, start over end
        ((dot(PROBE_STYLE["frozen"]["color"]), dot(PROBE_STYLE["refit"]["color"])),
         "Biased answers (teal: frozen probe, orange: retrained probe)"),
        ((dot(PROBE_STYLE["frozen"]["light"]), dot(PROBE_STYLE["refit"]["light"])),
         "Unbiased answers (lighter shade of the same colors)"),
        (dot(AUC_COLOR), "AUC, biased vs unbiased answers"),
    ]
    if len(seeds) > 1:
        entries.append((sd_handle(axes[0][0], COLORS["ink"], ""), f"± 1 SD over {seed_words(seeds)}"))
    if others:
        entries.append(faint_handle(others))
    entries += [
        (dot(COLORS["ink"], markerfacecolor="white", markeredgewidth=2.2), "Start of round"),
        (dot(COLORS["ink"]), "End of round"),
    ]
    fig.legend(handles=[handle for handle, _ in entries], labels=[label for _, label in entries],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.06), ncol=3 if len(entries) == 6 else 2,  # start and end in one column
               handlelength=2.6, prop={"size": 12, "weight": "bold"})
    fig.suptitle(title, fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, name)


def pair_values(ax, means: dict, colors: dict) -> None:
    """Two lines' values at rounds 2-5, beside their points (as 261107's fig4), which keeps low values off the x axis:
    the higher line's upper left; the lower line's below, on the side where its line rises away from the label (the
    next round's value above the previous round's, or the last round: right)."""
    for index in range(1, len(ROUNDS)):
        higher = max(means, key=lambda line: means[line][index])
        for line, line_means in means.items():
            if line == higher:
                right = False
            else:
                right = index + 1 == len(ROUNDS) or line_means[index + 1] > line_means[index - 1]
            side_value(ax, index + 1, line_means[index], f"{line_means[index]:.1f}", colors[line], right=right,
                       up=line == higher)


def fig3(comparison: dict) -> None:
    def shares(group: dict, seeds: tuple[int, ...]) -> dict:  # probe -> per round, the seeds' shares in %
        return {probe: [[100 * group[probe][f"s{seed}"]["rows"][MEMORIZATION_ROW][index] for seed in seeds]
                        for index in range(len(ROUNDS))] for probe in PROBES}

    redrawn = shares(comparison, FOUR_SEEDS)
    fixed = shares(comparison["fixed_labels"], FIXED_SEEDS)
    for group in (redrawn, fixed):  # each pair's round 1 is one run per seed
        assert group["frozen"][0] == group["refit"][0]
    for position, seed in enumerate(FIXED_SEEDS):  # the fixed lines' round 1 is the same runs as the redrawn lines' round 1
        assert fixed["frozen"][0][position] == redrawn["frozen"][0][FOUR_SEEDS.index(seed)], seed
    fig, ax = plt.subplots(figsize=(14, 9))
    xs = list(range(1, len(ROUNDS) + 1))
    colors = {probe: PROBE_STYLE[probe]["color"] for probe in PROBES}
    lines = {}  # (labels, probe) -> means over rounds 1-5
    for labels, group, marker, filled, linestyle in (("redrawn", redrawn, "o", True, "-"), ("fixed", fixed, "s", False, "--")):
        for probe in PROBES:
            means = [spread(group[probe][0])[0]]
            for index in range(1, len(ROUNDS)):
                shorter = min(PROBES, key=lambda name: spread(group[name][index])[1])
                means.append(mean_point(ax, index + 1, group[probe][index], colors[probe], marker=marker, filled=filled,
                                        sd_on_top=probe == shorter))
            ax.plot(xs, means, color=colors[probe], linestyle=linestyle, linewidth=2.4,
                    zorder=4.5 if probe == "frozen" else 4)
            lines[labels, probe] = means
    # round 1: the redrawn lines' ink dot, then the fixed lines' ink open square over it, larger, so both show
    redrawn_round1 = mean_point(ax, 1, redrawn["frozen"][0], SHARED_COLOR)
    fixed_round1, fixed_sd = spread(fixed["frozen"][0])
    ax.errorbar(1, fixed_round1, yerr=fixed_sd, ecolor=SHARED_COLOR, zorder=5, **SD_STYLE)
    ax.scatter(1, fixed_round1, s=MEAN_DOT * 3, marker="s", facecolor="none", edgecolor=SHARED_COLOR, linewidth=2.2, zorder=7)
    # each round-1 value on its own marker's side: the square's to its left, the dot's below its SD line
    ax.annotate(f"{fixed_round1:.1f}", (1, fixed_round1), xytext=(-14, 0), textcoords="offset points", ha="right",
                va="center", fontsize=FONT["annotation"] + 1, fontweight="bold", color=SHARED_COLOR, zorder=8)
    write_value(ax, 1, redrawn_round1 - spread(redrawn["frozen"][0])[1], f"{redrawn_round1:.1f}", SHARED_COLOR,
                above=False, box=True)
    for labels in ("redrawn", "fixed"):
        pair_values(ax, {probe: lines[labels, probe] for probe in PROBES}, colors)
    ax.set_xticks(xs, [f"Round {r}" for r in ROUNDS])
    ax.set_xlim(0.5, len(ROUNDS) + 0.5)
    style_axis(ax, title="Flipped Questions (training label = the wrong answer), Cue at the Correct Answer",
               ylabel="Training answers that give the flipped label")
    ax.set_ylim(-4, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    def line_handle(probe: str, marker: str, filled: bool, linestyle: str):
        return Line2D([], [], color=colors[probe], linestyle=linestyle, marker=marker, markersize=9, linewidth=2.4,
                      markerfacecolor=colors[probe] if filled else "white")

    entries = [  # two columns: the fixed-label lines and the SD line, then the redrawn-label lines
        (line_handle("frozen", "s", False, "--"), f"Frozen probe, labels fixed in every round ({seed_words(FIXED_SEEDS)})"),
        (line_handle("refit", "s", False, "--"), f"Retrained probe, labels fixed in every round ({seed_words(FIXED_SEEDS)})"),
        (Line2D([], [], marker="s", linestyle="", markersize=11, color=SHARED_COLOR, markerfacecolor="none",
                markeredgewidth=2.2), f"Round 1, {seed_words(FIXED_SEEDS)} (shared by the fixed-label lines)"),
        (sd_handle(ax, COLORS["ink"], ""), "± 1 SD over each line's seeds; markers: their mean"),
        (line_handle("frozen", "o", True, "-"), f"Frozen probe, labels redrawn every round ({seed_words(FOUR_SEEDS)})"),
        (line_handle("refit", "o", True, "-"), f"Retrained probe, labels redrawn every round ({seed_words(FOUR_SEEDS)})"),
        (Line2D([], [], marker="o", linestyle="", markersize=9, color=SHARED_COLOR),
         f"Round 1, {seed_words(FOUR_SEEDS)} (shared by the redrawn-label lines)"),
    ]
    fig.legend(handles=[handle for handle, _ in entries], labels=[label for _, label in entries], frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.12), ncol=2, handlelength=4, prop={"size": 12, "weight": "bold"})
    fig.suptitle("Training Answers That Give the Flipped Label", fontsize=FONT["title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig3_training_label")


def main() -> None:
    comparison = json.loads(COMPARISON.read_text())
    fig1(comparison, FOUR_SEEDS, "fig1_eval_answers", "Eval Answers After Each Round: Frozen vs Retrained Probe")
    fig2(comparison, FOUR_SEEDS, "fig2_round_probe", "The Penalty Probe at the Start and End of Each Round")
    fig3(comparison)
    fig1(comparison, SEED_45, "fig1_eval_answers_seed45", "Eval Answers After Each Round: Seed 45 Against Seeds 42, 43 and 46",
         OTHER_SEEDS)
    fig2(comparison, SEED_45, "fig2_round_probe_seed45", "The Penalty Probe in Each Round: Seed 45 Against Seeds 42, 43 and 46",
         OTHER_SEEDS)


if __name__ == "__main__":
    main()
