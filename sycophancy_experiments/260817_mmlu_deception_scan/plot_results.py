# Plot per-category results from the MMLU-Pro deception follow-up scan.
#
# Interactive notebook (`# %%` cells). Run cell-by-cell in VS Code / Cursor,
# or `python plot_results.py` for a non-interactive save.
#
# All charts use the default reporting view: one random ordering per question,
# individual samples (no majority vote).
#
# Chart 1: deception rate per category under both reliance definitions, against
#          the overall rate at which the model admits influence at all.
# Chart 2: P(says "Yes") in the four cells (relied x did-not-rely) x (cue at the
#          correct option x cue at the distractor). If the four bars are level,
#          the self-report carries no information about reliance.
# Chart 3: cue bias against deception rate per category -- do the categories that
#          flip most also deny most?

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
METRICS_CSV = EXP_DIR / "category_metrics.csv"

OUTPUT_FIG_DECEPTION = EXP_DIR / "deception_by_category.png"
OUTPUT_FIG_SELFREPORT = EXP_DIR / "selfreport_2x2.png"
OUTPUT_FIG_BIAS_VS = EXP_DIR / "bias_vs_deception.png"

VOTE = "single"  # the default reporting view

SAVE_FIG = True  # set False to skip writing the PNGs
DPI = 150

# Categorical slots 1-4 of the dataviz reference palette, in fixed order, with the
# same semantics 260817 gave them so a series keeps its hue across both scans:
# slot 1 = the no-cue arm (and so the cue-removal definition, which pairs against
# it), slot 2 = cue at the correct option, slot 3 = cue at the distractor,
# slot 4 = the cue-switch pairing (and so the cue-swap definition).
C_NONE = "#2a78d6"
C_GOLD = "#eb6834"
C_DIST = "#1baf7a"
C_SWITCH = "#eda100"
C_INK = "#333333"
C_MUTED = "#777777"

WIDTH = 0.38
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

n_questions = int(overall["n_questions"])
print(f"Loaded {len(cats)} categories (vote={VOTE}, {n_questions} questions)")
print(
    f"OVERALL  p_yes={overall['p_yes']:.3f}  "
    f"swap: relied={overall['swap_relied_rate']:.3f} decep={overall['swap_deception_rate']:.3f}  "
    f"rm: relied={overall['rm_relied_rate']:.3f} decep={overall['rm_deception_rate']:.3f}"
)


def grouped_bars(ax, frame, columns, colors, labels, *, label_first=True):
    """Two grouped bars per category; direct-label the first series only."""
    x = range(len(frame))
    offsets = [-WIDTH / 2, WIDTH / 2]
    bars = []
    for column, color, label, offset in zip(columns, colors, labels, offsets):
        bars.append(
            ax.bar([i + offset for i in x], frame[column], WIDTH, label=label, color=color)
        )
    if label_first:
        ax.bar_label(bars[0], **BAR_LABEL_KW)
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["category"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    return bars


# %%
# Chart 1: deception rate under both reliance definitions.
decep = cats.sort_values("swap_deception_rate", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 6.5))
grouped_bars(
    ax,
    decep,
    ["swap_deception_rate", "rm_deception_rate"],
    [C_SWITCH, C_NONE],
    [
        "Deception, cue-swap definition (reliance = answer follows the cue as it moves)",
        "Deception, cue-removal definition (reliance = answer changes when the cue goes)",
    ],
)

# P(says Yes) is the quantity that drives both rates: the model is scored deceptive
# almost entirely by denying influence it did have, so show how rarely it admits any.
ax.plot(
    list(range(len(decep))),
    decep["p_yes"],
    marker="D",
    linestyle="none",
    markersize=7,
    color=C_INK,
    markeredgecolor="white",
    markeredgewidth=1.2,
    label="P(self-report = \"Yes\", influence admitted)",
)

ax.set_ylabel("Rate")
ax.set_title("Deception rate by category (self-report vs measured cue reliance)")
ax.legend(**LEGEND_KW)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(OUTPUT_FIG_DECEPTION, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_DECEPTION}")
plt.show()

# %%
# Chart 2: does the self-report move with reliance at all?
# P(Yes | relied) = 1 - false_denial; P(Yes | did not rely) = false_admission.
DEFS = [
    ("swap", "Cue-swap definition"),
    ("rm", "Cue-removal definition"),
]

fig2, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
for ax2, (definition, title) in zip(axes, DEFS):
    values = {
        "gold": [1 - overall[f"{definition}_false_denial_gold"],
                 overall[f"{definition}_false_admission_gold"]],
        "dist": [1 - overall[f"{definition}_false_denial_dist"],
                 overall[f"{definition}_false_admission_dist"]],
    }
    x = range(2)
    for offset, (key, color, label) in zip(
        [-WIDTH / 2, WIDTH / 2],
        [("gold", C_GOLD, "Cue at the correct option"),
         ("dist", C_DIST, "Cue at the distractor")],
    ):
        bars = ax2.bar([i + offset for i in x], values[key], WIDTH, color=color, label=label)
        ax2.bar_label(bars, **BAR_LABEL_KW)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(["Relied on the cue", "Did not rely"])
    ax2.set_title(title, fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

axes[0].set_ylabel("P(self-report = \"Yes\")")
axes[0].set_ylim(0, 1.05)
handles, labels = axes[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.06),
            ncol=2, fontsize=9, frameon=False)
fig2.suptitle("Does the self-report track measured reliance?", y=0.99)
fig2.tight_layout(rect=(0, 0.08, 1, 1))
if SAVE_FIG:
    fig2.savefig(OUTPUT_FIG_SELFREPORT, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_SELFREPORT}")
plt.show()

# %%
# Chart 3: bias against deception, per category.
# x is this scan's swap reliance rate, which reproduces 260817's bias_cue_switch.
corr = cats["swap_relied_rate"].corr(cats["swap_deception_rate"])

fig3, ax3 = plt.subplots(figsize=(8, 6.5))
ax3.scatter(
    cats["swap_relied_rate"],
    cats["swap_deception_rate"],
    s=70,
    color=C_SWITCH,
    edgecolor="white",
    linewidth=1.2,
    zorder=3,
)
for _, row in cats.iterrows():
    ax3.annotate(
        row["category"],
        (row["swap_relied_rate"], row["swap_deception_rate"]),
        textcoords="offset points",
        xytext=(6, 3),
        fontsize=8,
        color=C_MUTED,
    )

ax3.axhline(
    overall["swap_deception_rate"],
    linestyle="--",
    linewidth=1.2,
    color=C_INK,
    label=f"Overall deception rate ({overall['swap_deception_rate']:.3f})",
)
ax3.axvline(
    overall["swap_relied_rate"],
    linestyle=":",
    linewidth=1.2,
    color=C_MUTED,
    label=f"Overall cue-switch rate ({overall['swap_relied_rate']:.3f})",
)

ax3.set_xlabel("Cue bias: P(answer follows the cue as the cue moves)")
ax3.set_ylabel("Deception rate (cue-swap definition)")
ax3.set_title(f"Cue bias vs deception by category (r = {corr:.2f})")
ax3.grid(alpha=0.3)
ax3.set_axisbelow(True)
ax3.legend(loc="lower right", fontsize=9, frameon=False)
fig3.tight_layout()
if SAVE_FIG:
    fig3.savefig(OUTPUT_FIG_BIAS_VS, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_BIAS_VS}")
plt.show()

# %%
