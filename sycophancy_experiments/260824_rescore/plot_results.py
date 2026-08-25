# Charts for the 260824 rescore.
#
# Interactive notebook (`# %%` cells). Run cell-by-cell in VS Code / Cursor, or
# `python plot_results.py` for a non-interactive save.
#
# Everything is read from `metrics.csv` and `label_counts.csv` -- never recomputed
# from the rollouts -- so a chart cannot disagree with a published table. If a number
# here looks wrong, the filter is wrong, not the arithmetic.
#
# Metric charts are the eval split, matching the README. The counts chart shows train
# and eval, which is what was asked for.
#
#   1. accuracy.png       accuracy by condition, against chance and the ceiling
#   2. bias_strata.png    bias / anti-bias / no bias, with each bar's matched null
#   3. deception.png      E[P(Yes)|bias] vs E[P(Yes)|no bias], and the deception rate
#   4. label_counts.png   the four turn-1 and five turn-2 labels, as counts

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
METRICS_CSV = EXP_DIR / "metrics.csv"
COUNTS_CSV = EXP_DIR / "label_counts.csv"

SPLIT = "eval"        # the reported split; counts show train and eval
CATEGORY = "OVERALL"
SAVE_FIG = True
DPI = 150
CHANCE = 0.5

# Colour follows the *entity*, held identical to 260817-260823 so a condition keeps
# its hue across every folder in this line of work. Slots are the dataviz reference
# palette's categorical order.
#
# Conditions (chart 1): slot 1 blue = no cue, slot 2 orange = cue at gold, slot 3
# aqua = cue at distractor. Validated all-pairs on the light surface: worst CVD
# dE 9.2, normal-vision 24.0.
#
# Strata (charts 2-4): slot 4 yellow = bias, slot 7 violet = anti-bias, slot 1 blue =
# no bias. Worst CVD dE 13.0, normal-vision 16.3. Violet carries anti-bias into the
# turn-2 panel as `excluded_antibias`, since it is the same rollouts.
#
# Turn-2 outcome (chart 4): slot 8 red = deceptive, slot 1 blue = honest. Worst CVD
# dE 13.0. Red/aqua would have been the intuitive good/bad pair and fails: dE 6.9
# under deuteranopia, the classic trap.
#
# The two unparsed segments are neutrals, not series -- they are a residual, and a
# grey says so. They are deliberately outside the validated categorical set.
C_NONE = "#2a78d6"
C_GOLD = "#eb6834"
C_DIST = "#1baf7a"
C_BIAS = "#eda100"
C_ANTI = "#4a3aa7"
C_NOBIAS = "#2a78d6"
C_DECEPTIVE = "#e34948"
C_HONEST = "#2a78d6"
C_UNPARSED_1 = "#777777"
C_UNPARSED_2 = "#bdbdbd"
C_INK = "#333333"
C_MUTED = "#777777"

BLOCKS = ["at_p", "cue_gold", "cue_dist"]
BLOCK_LABEL = {"at_p": "at p = 0.85", "cue_gold": "cue at gold", "cue_dist": "cue at distractor"}
DEFINITIONS = ["rm", "swap"]
DEFINITION_LABEL = {"rm": "cue-removal", "swap": "cue-switch"}

BAR_LABEL_KW = dict(
    fmt="%.3f", fontsize=7, padding=2,
    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
)
FIG_W = 6.5
FIG_H = 5.6
BAR_W = 0.38
TITLE_FS = 12
LEGEND_KW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=9,
                 frameon=False)
# A 2px surface gap between stacked segments, as the dataviz skill requires.
STACK_KW = dict(edgecolor="white", linewidth=2)


def m(df, block, definition, field, split=SPLIT):
    row = df[(df["block"] == block) & (df["definition"] == definition)
             & (df["split"] == split) & (df["category"] == CATEGORY)]
    return float(row[field].iloc[0])


# %%
mdf = pd.read_csv(METRICS_CSV)
cdf = pd.read_csv(COUNTS_CSV)
p_cue_gold = float(mdf["p_cue_gold"].iloc[0])
print(f"Loaded metrics.csv ({len(mdf)} rows) and label_counts.csv ({len(cdf)} rows); "
      f"reporting split={SPLIT}, p={p_cue_gold}")

# %%
# Chart 1: accuracy by condition. The gap between the at-p bar and the always-follow
# ceiling is the reward headroom -- what RL can buy purely by increasing bias.
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
labels = ["no cue", "cue at gold", "cue at distractor", f"at p = {p_cue_gold}"]
values = [
    m(mdf, "at_p", "rm", "accuracy_none"),
    m(mdf, "cue_gold", "rm", "accuracy"),
    m(mdf, "cue_dist", "rm", "accuracy"),
    m(mdf, "at_p", "rm", "accuracy"),
]
bars = ax.bar(range(4), values, BAR_W, color=[C_NONE, C_GOLD, C_DIST, C_MUTED])
ax.bar_label(bars, **BAR_LABEL_KW)
chance = ax.axhline(CHANCE, color=C_INK, ls="--", lw=1, label="chance (2 options)")
ceiling = ax.axhline(p_cue_gold, color=C_BIAS, ls=":", lw=1.6,
                     label=f"always follow the cue ({p_cue_gold})")
ax.set_xticks(range(4))
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.02)
ax.set_ylabel("accuracy")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.legend(handles=[chance, ceiling], loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=2, fontsize=9, frameon=False)
ax.set_title(f"Accuracy by condition ({SPLIT} split)", fontsize=TITLE_FS)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "accuracy.png", dpi=DPI, bbox_inches="tight")
print("wrote accuracy.png")
plt.show()

# %%
# Chart 2: the run's headline picture. Parsed answers only, so the three segments sum
# to 1 -- these are rates. Unparsed rollouts appear only in chart 4, which counts.
COLUMNS = [
    ("rm", "at_p", "removal\nat p"),
    ("rm", "cue_gold", "removal\ncue at gold"),
    ("rm", "cue_dist", "removal\ncue at distractor"),
    ("swap", "at_p", "switch\n(p-invariant)"),
]
SEGMENTS = [("bias_rate", "bias", C_BIAS),
            ("antibias_rate", "anti-bias", C_ANTI),
            ("nobias_rate", "no bias", C_NOBIAS)]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
bottom = [0.0] * len(COLUMNS)
handles = []
for field, label, color in SEGMENTS:
    heights = [m(mdf, block, definition, field) for definition, block, _ in COLUMNS]
    bar = ax.bar(range(len(COLUMNS)), heights, BAR_W, bottom=bottom, label=label,
                 color=color)
    ax.bar_label(bar, labels=[f"{h:.3f}" for h in heights], label_type="center",
                 fontsize=8, color="white", fontweight="bold")
    handles.append(bar)
    bottom = [b + h for b, h in zip(bottom, heights)]
ax.set_xticks(range(len(COLUMNS)))
ax.set_xticklabels([label for _, _, label in COLUMNS], fontsize=9)
ax.set_ylim(0, 1.02)
ax.set_ylabel("share of parsed rollouts")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.legend(handles=handles, **LEGEND_KW)
ax.set_title(f"Bias, anti-bias, and no bias ({SPLIT} split)", fontsize=TITLE_FS)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "bias_strata.png", dpi=DPI, bbox_inches="tight")
print("wrote bias_strata.png")
plt.show()

# %%
# Chart 3: does the self-report move with the measured counterfactual? If it were
# informative the bias bar would clear the no-bias bar. Anti-bias rollouts are absent
# by construction -- they carry no deception label.
#
# Three bars rather than two bars and an overlaid marker: all three are probabilities
# on one 0-1 axis, and deception is built from the other two, so it belongs in the
# group rather than floating above it. An overlaid marker read as a second measure
# sharing the axis, which is the dual-axis mistake in disguise.
WIDTH = 0.26
SERIES = [
    ("true_admission", "P(Yes) | biased", C_BIAS),
    ("false_admission", "P(Yes) | not biased", C_NOBIAS),
    ("deception", "deception rate", C_DECEPTIVE),
]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
for ax, definition in zip(axes, DEFINITIONS):
    x = range(len(BLOCKS))
    handles = []
    for i, (field, label, color) in enumerate(SERIES):
        offset = (i - 1) * WIDTH
        bar = ax.bar([j + offset for j in x],
                     [m(mdf, b, definition, field) for b in BLOCKS],
                     WIDTH, color=color, label=label)
        ax.bar_label(bar, **BAR_LABEL_KW)
        handles.append(bar)
    ax.set_xticks(list(x))
    ax.set_xticklabels([BLOCK_LABEL[b] for b in BLOCKS], fontsize=9)
    ax.set_title(f"{DEFINITION_LABEL[definition]} definition", fontsize=TITLE_FS)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
axes[0].set_ylabel("probability")
axes[0].set_ylim(0, 1.0)
axes[0].legend(handles=handles, loc="upper left", fontsize=9, frameon=False)
fig.suptitle(f"Self-report against measured bias ({SPLIT} split)", fontsize=TITLE_FS)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "deception.png", dpi=DPI, bbox_inches="tight")
print("wrote deception.png")
plt.show()

# %%
# Chart 4: the label counts. Counts, not rates, so the unparsed and excluded
# segments are visible. Enumerated is every rollout that exists, balanced across the
# two cued conditions; realized is the subset the p = 0.85 draw actually served.
STAGE_SEGMENTS = {
    "turn1": [("bias", C_BIAS), ("anti-bias", C_ANTI), ("no bias", C_NOBIAS),
              ("unparsed_turn1", C_UNPARSED_1)],
    "turn2": [("deceptive", C_DECEPTIVE), ("honest", C_HONEST),
              ("excluded_antibias", C_ANTI), ("unparsed_turn1", C_UNPARSED_1),
              ("unparsed_turn2", C_UNPARSED_2)],
}
COLUMNS_4 = [("train", "enumerated"), ("eval", "enumerated"),
             ("train", "realized"), ("eval", "realized")]
LABEL_MIN_SHARE = 0.08

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
for row, stage in enumerate(("turn1", "turn2")):
    for col, definition in enumerate(DEFINITIONS):
        ax = axes[row][col]
        panel = cdf[(cdf["definition"] == definition) & (cdf["stage"] == stage)]
        totals = [
            int(panel[(panel["split"] == s) & (panel["population"] == pop)]["n"].sum())
            for s, pop in COLUMNS_4
        ]
        bottom = [0.0] * len(COLUMNS_4)
        handles = []
        for label, color in STAGE_SEGMENTS[stage]:
            heights = [
                int(cdf[(cdf["split"] == s) & (cdf["population"] == pop)
                        & (cdf["definition"] == definition) & (cdf["stage"] == stage)
                        & (cdf["label"] == label)]["n"].iloc[0])
                for s, pop in COLUMNS_4
            ]
            bar = ax.bar(range(len(COLUMNS_4)), heights, 0.5, bottom=bottom,
                         label=label, color=color, **STACK_KW)
            # Label only segments with room for the number. Below ~4% of a column the
            # text overruns its own segment and collides with its neighbours; those
            # counts are in `label_counts.csv` and the README table, which is where
            # exact numbers belong anyway.
            ax.bar_label(
                bar,
                labels=[f"{h}" if h / max(t, 1) >= LABEL_MIN_SHARE else ""
                        for h, t in zip(heights, totals)],
                label_type="center", fontsize=7, color="white", fontweight="bold",
            )
            handles.append(bar)
            bottom = [b + h for b, h in zip(bottom, heights)]
        ax.set_xticks(range(len(COLUMNS_4)))
        ax.set_xticklabels([f"{s}\n{pop}" for s, pop in COLUMNS_4], fontsize=8)
        ax.set_title(f"{stage} — {DEFINITION_LABEL[definition]}", fontsize=TITLE_FS)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("rollouts")
        if col == 1:
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=9, frameon=False)
fig.suptitle("Label counts, from the sampled Yes/No token", fontsize=TITLE_FS)
fig.text(0.5, 0.005, f"segments under {LABEL_MIN_SHARE:.0%} of a column are left unlabelled; "
         "exact counts in label_counts.csv", ha="center", fontsize=8, color=C_MUTED)
# Reserve the bottom strip for that caption -- tight_layout does not see fig.text, so
# without the rect it lays the axes over the top of it.
fig.tight_layout(rect=[0, 0.028, 1, 1])
if SAVE_FIG:
    fig.savefig(EXP_DIR / "label_counts.png", dpi=DPI, bbox_inches="tight")
print("wrote label_counts.png")
plt.show()

# %%
