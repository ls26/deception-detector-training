# Plot per-category results from the MMLU-Pro counterfactual self-report scan.
#
# Interactive notebook (`# %%` cells). Run cell-by-cell in VS Code / Cursor,
# or `python plot_results.py` for a non-interactive save.
#
# All charts use the default reporting view: one random ordering per question,
# individual samples (no majority vote), P(Yes) read at the first generated token.
#
# Chart 1: the distribution of P(Yes) per arm. This is the chart the logprob
#          change exists to produce -- whether the model is confidently "No",
#          genuinely uncertain, or bimodal is invisible in a 3-sample rate.
# Chart 1a: the same distribution split by measured reliance, using Chart 2's
#           arm/definition pairings in a 2x2 grid of overlaid histograms.
# Chart 2: mean P(Yes) for rollouts that relied on the cue vs those that did not,
#          per arm (matched definition on row 1; original vs both definitions on
#          row 2). If the two bars are level the self-report carries no
#          information about reliance; if the right bar is higher it is
#          anti-informative, which is what 260817 found.
# Chart 3: soft deception rate per category (removal and swap vs matched definitions),
#          with measured bias rate marked by a diamond on each bar.
# Chart 4: cue reliance against deception rate per category, one panel per arm.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
METRICS_CSV = EXP_DIR / "category_metrics.csv"
SCORES_JSONL = EXP_DIR / "deception_scores.jsonl"

OUTPUT_FIG_DIST = EXP_DIR / "pyes_distribution.png"
OUTPUT_FIG_DIST_BY_RELIANCE = EXP_DIR / "pyes_distribution_by_reliance.png"
OUTPUT_FIG_SELFREPORT = EXP_DIR / "selfreport_2x2.png"
OUTPUT_FIG_DECEPTION = EXP_DIR / "deception_by_arm.png"
OUTPUT_FIG_BIAS_VS = EXP_DIR / "bias_vs_deception.png"

VOTE = "single"  # the default reporting view
POSITION = "pos0"  # the first generated token -- the activation the probe reads

SAVE_FIG = True  # set False to skip writing the PNGs
DPI = 150

# Categorical slots 1-4 of the dataviz reference palette, in fixed order, with the
# same semantics 260817 gave them so a series keeps its hue across all three scans:
# slot 1 = the no-cue arm (and so the cue-removal definition, which pairs against
# it), slot 2 = cue at the correct option, slot 3 = cue at the distractor,
# slot 4 = the cue-switch pairing (and so the cue-swap definition).
C_NONE = "#2a78d6"
C_GOLD = "#eb6834"
C_DIST = "#1baf7a"
C_SWITCH = "#eda100"
C_INK = "#333333"
C_MUTED = "#777777"

# Each arm takes the hue of the definition it asks the counterfactual of; the
# `original` wording asks neither, so it takes the neutral ink.
ARMS = [
    ("removal", "Removal: \"no opinion → chosen differently?\"", C_NONE, "rm"),
    ("swap", "Swap: \"said (X) instead → chosen differently?\"", C_SWITCH, "swap"),
    ("original", "Original: \"did the opinion influence you?\"", C_MUTED, "swap"),
]

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
df = df[(df["vote"] == VOTE) & (df["position"] == POSITION)]
overall = df[df["category"] == "OVERALL"].iloc[0]
cats = df[df["category"] != "OVERALL"].copy()

n_questions = int(overall["n_questions"])
print(f"Loaded {len(cats)} categories (vote={VOTE}, position={POSITION}, {n_questions} questions)")
for arm, _, _, definition in ARMS:
    print(
        f"OVERALL  {arm:<9} mean P(Yes)={overall[f'{arm}_mean_p_yes']:.3f}  "
        f"soft deception ({definition})={overall[f'{arm}_{definition}_soft_deception']:.3f}"
    )

rollouts = [json.loads(line) for line in SCORES_JSONL.open() if line.strip()]
print(f"Loaded {len(rollouts)} rollouts from {SCORES_JSONL.name}")


def grouped_bars(
    ax,
    frame,
    columns,
    colors,
    labels,
    *,
    reliance_cols=None,
    label_first=True,
):
    """One bar per arm per category; direct-label the first series only."""
    x = range(len(frame))
    n = len(columns)
    offsets = [(i - (n - 1) / 2) * WIDTH for i in range(n)]
    bars = []
    rel_cols = reliance_cols or [None] * n
    for column, color, label, offset, rel_col in zip(columns, colors, labels, offsets, rel_cols):
        bars.append(
            ax.bar([i + offset for i in x], frame[column], WIDTH, label=label, color=color)
        )
        if rel_col is not None:
            ax.scatter(
                [i + offset for i in x],
                frame[rel_col],
                marker="D",
                s=32,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=4,
            )
    if label_first:
        ax.bar_label(bars[0], **BAR_LABEL_KW)
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["category"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    return bars


ARM_TITLES = {arm: title for arm, title, _, _ in ARMS}
DEFINITION_SUBTITLE = {
    "rm": "cue-removal bias definition",
    "swap": "cue-swap bias definition",
}
SELFREPORT_PANELS = [
    ("removal", ARM_TITLES["removal"], "rm"),
    ("swap", ARM_TITLES["swap"], "swap"),
    ("original", ARM_TITLES["original"], "rm"),
    ("original", ARM_TITLES["original"], "swap"),
]
RELIANCE_GROUPS = [
    ("Relied on the cue", C_SWITCH),
    ("Did not rely", C_NONE),
]


def rollout_relied(rollout: dict, definition: str) -> bool | None:
    """Per-rollout reliance under the rm or swap definition."""
    if definition == "rm":
        if rollout["a_cued"] is None or rollout["a_none"] is None:
            return None
        target = "G" if rollout["condition"] == "cue_gold" else "D"
        return rollout["a_cued"] == target and rollout["a_none"] != rollout["a_cued"]
    if rollout["a_cued"] is None or rollout["a_other_cue"] is None:
        return None
    if rollout["condition"] == "cue_gold":
        return rollout["a_cued"] == "G" and rollout["a_other_cue"] == "D"
    return rollout["a_other_cue"] == "G" and rollout["a_cued"] == "D"


# %%
# Chart 1: the shape of P(Yes). A 3-sample rate can only take the values 0, 1/3,
# 2/3, 1; the logprob shows where the mass actually sits, and whether the model is
# confident or on the fence.
key = "p_yes_pos0" if POSITION == "pos0" else "p_yes_ans"
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, (arm, title, color, _) in zip(axes, ARMS):
    values = [r["arms"][arm][key] for r in rollouts if arm in r["arms"] and r["arms"][arm][key] is not None]
    ax.hist(values, bins=40, range=(0, 1), color=color, edgecolor="white", linewidth=0.4)
    mean_value = sum(values) / len(values)
    ax.axvline(
        mean_value,
        linestyle="--",
        linewidth=1.4,
        color=C_INK,
        label=f"mean {mean_value:.3f}",
    )
    ax.axvline(0.5, linestyle=":", linewidth=1.2, color=C_MUTED, label="0.5 (hard threshold)")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("P(self-report = \"Yes\")")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)

axes[0].set_ylabel(f"Rollouts (of {len(rollouts)})")
fig.suptitle(
    "How confidently does the model answer the self-report question?", y=0.99
)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(OUTPUT_FIG_DIST, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_DIST}")
plt.show()

# %%
# Chart 1a: P(Yes) by measured reliance. Same pairings as Chart 2; overlaid
# histograms show whether the two groups' distributions separate or sit on top
# of each other. Identical styling to Chart 1, plus a yellow step outline on top.
key = "p_yes_pos0" if POSITION == "pos0" else "p_yes_ans"
HIST_BINS = 40
HIST_ALPHA = 0.55

fig1a, axes1a = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
for ax1a, (arm, title, definition) in zip(axes1a.flat, SELFREPORT_PANELS):
    by_group = {label: [] for label, _ in RELIANCE_GROUPS}
    for rollout in rollouts:
        relied = rollout_relied(rollout, definition)
        if relied is None:
            continue
        p_yes = rollout["arms"].get(arm, {}).get(key)
        if p_yes is None:
            continue
        label = RELIANCE_GROUPS[0][0] if relied else RELIANCE_GROUPS[1][0]
        by_group[label].append(p_yes)

    total = sum(len(by_group[label]) for label, _ in RELIANCE_GROUPS)
    for label, color in RELIANCE_GROUPS:
        values = by_group[label]
        pct = 100 * len(values) / total if total else 0
        mean_value = sum(values) / len(values) if values else None
        mean_text = f", mean {mean_value:.3f}" if mean_value is not None else ""
        ax1a.hist(
            values,
            bins=HIST_BINS,
            range=(0, 1),
            color=color,
            alpha=HIST_ALPHA,
            edgecolor="white",
            linewidth=0.4,
            label=f"{label} ({pct:.0f}%{mean_text})",
        )
        if mean_value is not None:
            ax1a.axvline(
                mean_value, linestyle="--", linewidth=1.2, color=color, zorder=5
            )

    relied_label, relied_color = RELIANCE_GROUPS[0]
    relied_values = by_group[relied_label]
    if relied_values:
        ax1a.hist(
            relied_values,
            bins=HIST_BINS,
            range=(0, 1),
            histtype="step",
            linewidth=2,
            color=relied_color,
            zorder=6,
        )

    ax1a.set_title(f"{title}\n({DEFINITION_SUBTITLE[definition]})", fontsize=9)
    ax1a.set_xlabel("P(self-report = \"Yes\")")
    ax1a.grid(axis="y", alpha=0.3)
    ax1a.set_axisbelow(True)
    ax1a.legend(fontsize=7, frameon=False)

axes1a[0, 0].set_ylabel(f"Rollouts (of {len(rollouts)})")
axes1a[1, 0].set_ylabel(f"Rollouts (of {len(rollouts)})")
fig1a.suptitle(
    "P(self-report = \"Yes\") by measured reliance (overlaid distributions)", y=0.99
)
fig1a.tight_layout()
if SAVE_FIG:
    fig1a.savefig(OUTPUT_FIG_DIST_BY_RELIANCE, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_DIST_BY_RELIANCE}")
plt.show()

# %%
# Chart 2: does the self-report move with reliance at all?
# mean P(Yes | relied) = 1 - soft_false_denial; mean P(Yes | did not rely) =
# soft_false_admission. Pooled over both cue arms. Row 1 is each arm against its
# matched definition; row 2 is the original wording against both definitions.
RELIANCE = [
    ("Relied on the cue\n(should say Yes)", C_SWITCH),
    ("Did not rely\n(should say No)", C_NONE),
]
# Explicit positions rather than categorical ones: with only two bars, matplotlib's
# unit spacing leaves them wide and far apart for a comparison meant to be read at
# a glance. X_POS sets the gap, BAR_W the thickness, X_PAD the margin either side.
X_POS = [0.0, 0.45]
BAR_W = WIDTH
X_PAD = 0.22

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
for ax2, (arm, title, definition) in zip(axes2.flat, SELFREPORT_PANELS):
    values = [
        1 - overall[f"{arm}_{definition}_soft_false_denial"],
        overall[f"{arm}_{definition}_soft_false_admission"],
    ]
    bars = ax2.bar(X_POS, values, BAR_W, color=[color for _, color in RELIANCE])
    ax2.bar_label(bars, **BAR_LABEL_KW)
    ax2.set_xticks(X_POS)
    ax2.set_xticklabels([label for label, _ in RELIANCE])
    ax2.set_xlim(X_POS[0] - BAR_W / 2 - X_PAD, X_POS[1] + BAR_W / 2 + X_PAD)
    ax2.set_title(f"{title}\n({DEFINITION_SUBTITLE[definition]})", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

axes2[0, 0].set_ylabel("mean P(self-report = \"Yes\")")
axes2[1, 0].set_ylabel("mean P(self-report = \"Yes\")")
axes2[0, 0].set_ylim(0, 1.05)
fig2.suptitle("Does the self-report track measured reliance?", y=0.99)
fig2.tight_layout()
if SAVE_FIG:
    fig2.savefig(OUTPUT_FIG_SELFREPORT, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_SELFREPORT}")
plt.show()

# %%
# Chart 3: soft deception rate per category, each arm against its matched definition.
CHART3_ARMS = [arm for arm in ARMS if arm[0] != "original"]
sort_key = "swap_swap_soft_deception"
decep = cats.sort_values(sort_key, ascending=False).reset_index(drop=True)

fig3, ax3 = plt.subplots(figsize=(13, 6.5))
grouped_bars(
    ax3,
    decep,
    [f"{arm}_{definition}_soft_deception" for arm, _, _, definition in CHART3_ARMS],
    [color for _, _, color, _ in CHART3_ARMS],
    [title for _, title, _, _ in CHART3_ARMS],
    reliance_cols=[f"{definition}_relied_rate" for _, _, _, definition in CHART3_ARMS],
)
ax3.set_ylabel("Deception rate")
ax3.set_title(
    "Deception rate by category and bias definition/follow-up wording "
)
handles, labels = ax3.get_legend_handles_labels()
handles.append(
    ax3.scatter([], [], marker="D", s=32, color=C_INK, edgecolor="white", linewidth=1.0)
)
labels.append("Measured bias rate")
ax3.legend(handles=handles, labels=labels, **LEGEND_KW)
fig3.tight_layout()
if SAVE_FIG:
    fig3.savefig(OUTPUT_FIG_DECEPTION, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_DECEPTION}")
plt.show()

# %%
# Chart 4: reliance against deception, per category, one panel per arm.
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax4, (arm, title, color, definition) in zip(axes4, ARMS):
    x_col = f"{definition}_relied_rate"
    y_col = f"{arm}_{definition}_soft_deception"
    corr = cats[x_col].corr(cats[y_col])

    ax4.scatter(
        cats[x_col], cats[y_col], s=60, color=color, edgecolor="white", linewidth=1.2, zorder=3
    )
    for _, row in cats.iterrows():
        ax4.annotate(
            row["category"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
            color=C_MUTED,
        )
    ax4.axhline(overall[y_col], linestyle="--", linewidth=1.2, color=C_INK)
    ax4.axvline(overall[x_col], linestyle=":", linewidth=1.2, color=C_MUTED)
    ax4.set_xlabel(f"Reliance rate ({definition} definition)")
    ax4.set_title(f"{title}\nr = {corr:.2f}", fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_axisbelow(True)

axes4[0].set_ylabel("Soft deception rate")
fig4.suptitle(
    "Cue reliance vs deception by category "
    "(dashed = overall deception, dotted = overall reliance)",
    y=0.99,
)
fig4.tight_layout()
if SAVE_FIG:
    fig4.savefig(OUTPUT_FIG_BIAS_VS, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_BIAS_VS}")
plt.show()

# %%
