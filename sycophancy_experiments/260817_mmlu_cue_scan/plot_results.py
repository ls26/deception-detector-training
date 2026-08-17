# Plot per-category results from the two-option MMLU-Pro cue scan.
#
# Interactive notebook (`# %%` cells). Run cell-by-cell in VS Code / Cursor,
# or `python plot_results.py` for a non-interactive save.
#
# All three charts use the default reporting view: one random ordering per
# question, individual samples (no majority vote).
#
# Chart 1: the three bias metrics per category, against the two sampling-noise
#          nulls (cued and no-cue).
# Chart 2: accuracy per condition per category, against the 0.5 chance line and
#          the overall no-cue rate.
# Chart 3: order sensitivity per condition -- the only use of the second ordering.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
METRICS_CSV = EXP_DIR / "category_metrics.csv"

OUTPUT_FIG_BIAS = EXP_DIR / "bias_by_category.png"
OUTPUT_FIG_ACC = EXP_DIR / "accuracy_by_condition.png"
OUTPUT_FIG_ORDER = EXP_DIR / "order_sensitivity.png"

VOTE = "single"  # the default reporting view

SAVE_FIG = True  # set False to skip writing the PNGs
DPI = 150
CHANCE = 0.5  # two options

# Categorical slots of the dataviz reference palette, in fixed order. Colour
# follows the *condition*, so a condition keeps its hue across every chart:
# slot 1 = no cue, slot 2 = cue at the correct option, slot 3 = cue at the
# distractor. Slot 4 is the one series that is not a single condition --
# cue-switch pairs the two cued arms -- so it does not borrow a condition's hue.
C_NONE = "#2a78d6"
C_GOLD = "#eb6834"
C_DIST = "#1baf7a"
C_SWITCH = "#eda100"
C_INK = "#333333"
C_MUTED = "#777777"

WIDTH = 0.27
# The white bbox knocks the label out of any reference line running behind it.
BAR_LABEL_KW = dict(
    fmt="%.2f",
    fontsize=7,
    padding=2,
    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
)
LEGEND_KW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=9, frameon=False)

# %%
df = pd.read_csv(METRICS_CSV)
df = df[df["vote"] == VOTE]
overall = df[df["category"] == "OVERALL"].iloc[0]
cats = df[df["category"] != "OVERALL"].copy()

# The null for the directional bias metrics: the same directional event measured
# across two samples of a single condition. Shown as two markers -- the cued
# conditions (the null for cue-switch, which pairs them) and the no-cue condition
# (which the two cue-removal metrics each pair against).
cats["noise_dir_cued"] = (cats["noise_dir_gold"] + cats["noise_dir_dist"]) / 2
overall_noise_dir_cued = (overall["noise_dir_gold"] + overall["noise_dir_dist"]) / 2

n_questions = int(overall["n_questions"])
print(f"Loaded {len(cats)} categories (vote={VOTE}, {n_questions} questions)")
print(
    f"OVERALL  switch={overall['bias_cue_switch']:.3f}  "
    f"rm_dist={overall['bias_removal_dist']:.3f}  rm_gold={overall['bias_removal_gold']:.3f}  "
    f"null_cued={overall_noise_dir_cued:.3f}  null_none={overall['noise_dir_none']:.3f}  "
    f"contra={overall['contrarian_switch']:.3f}"
)


def grouped_bars(ax, frame, columns, colors, labels):
    """Three grouped bars per category; direct-label the first series only."""
    x = range(len(frame))
    offsets = [-WIDTH, 0.0, WIDTH]
    bars = []
    for column, color, label, offset in zip(columns, colors, labels, offsets):
        bars.append(
            ax.bar([i + offset for i in x], frame[column], WIDTH, label=label, color=color)
        )
    ax.bar_label(bars[0], **BAR_LABEL_KW)
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["category"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    return bars


# %%
# Chart 1: the three bias metrics, sorted by the headline cue-switch rate.
bias = cats.sort_values("bias_cue_switch", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 6.5))
grouped_bars(
    ax,
    bias,
    ["bias_cue_switch", "bias_removal_dist", "bias_removal_gold"],
    [C_SWITCH, C_DIST, C_GOLD],
    [
        "Cue-switch: follows the cue as the cue moves",
        "Cue-removal (distractor): cue is incorrect, no cue -> answer is correct",
        "Cue-removal (gold): cue is correct, no cue -> answer is incorrect",
    ],
)

# Filled vs open marker, not colour, separates the two nulls -- the three hues
# are spoken for by the bars, and shape survives greyscale and CVD.
ax.plot(
    list(range(len(bias))),
    bias["noise_dir_cued"],
    marker="D",
    linestyle="none",
    markersize=7,
    color=C_INK,
    markeredgecolor="white",
    markeredgewidth=1.2,
    label="Sampling noise, with cue",
)
ax.plot(
    list(range(len(bias))),
    bias["noise_dir_none"],
    marker="D",
    linestyle="none",
    markersize=7,
    markerfacecolor="white",
    markeredgecolor=C_INK,
    markeredgewidth=1.4,
    label="Sampling noise, no cue",
)

ax.set_ylabel("Rate")
ax.set_title("Cue bias by category")
ax.legend(**LEGEND_KW)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(OUTPUT_FIG_BIAS, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_BIAS}")
plt.show()

# %%
# Chart 2: accuracy per condition, sorted by the no-cue baseline.
acc = cats.sort_values("acc_none", ascending=False).reset_index(drop=True)

fig2, ax2 = plt.subplots(figsize=(13, 6.5))
grouped_bars(
    ax2,
    acc,
    ["acc_none", "acc_cue_gold", "acc_cue_dist"],
    [C_NONE, C_GOLD, C_DIST],
    ["No cue", "Cue points at the correct option", "Cue points at the distractor"],
)

ax2.axhline(
    overall["acc_none"],
    linestyle="--",
    linewidth=1.2,
    color=C_INK,
    label=f"Overall no-cue accuracy ({overall['acc_none']:.3f})",
)
ax2.axhline(
    CHANCE, linestyle=":", linewidth=1.2, color=C_MUTED, label=f"Random guess ({CHANCE:.3f})"
)

ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy by cue condition")
ax2.legend(**LEGEND_KW)
fig2.tight_layout()
if SAVE_FIG:
    fig2.savefig(OUTPUT_FIG_ACC, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_ACC}")
plt.show()

# %%
# Chart 3: order sensitivity -- how often swapping A/B changes the answer.
order = cats.sort_values("order_flip_none", ascending=False).reset_index(drop=True)

fig3, ax3 = plt.subplots(figsize=(13, 6.5))
grouped_bars(
    ax3,
    order,
    ["order_flip_none", "order_flip_gold", "order_flip_dist"],
    [C_NONE, C_GOLD, C_DIST],
    ["No cue", "Cue points at the correct option", "Cue points at the distractor"],
)

ax3.axhline(
    overall["order_flip_none"],
    linestyle="--",
    linewidth=1.2,
    color=C_INK,
    label=f"Overall no-cue order flip ({overall['order_flip_none']:.3f})",
)

ax3.set_ylabel("P(answer changes when option order is swapped)")
ax3.set_title("Order sensitivity by cue condition")
ax3.legend(**LEGEND_KW)
fig3.tight_layout()
if SAVE_FIG:
    fig3.savefig(OUTPUT_FIG_ORDER, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_ORDER}")
plt.show()

# %%
