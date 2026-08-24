# Plots for the four-wording follow-up run.
#
# Interactive notebook (`# %%` cells). Run cell-by-cell in VS Code / Cursor, or
# `python plot_results.py` for a non-interactive save.
#
# Turn 1 is not regenerated in this folder, so the three turn-1 charts
# (`bias_by_category`, `accuracy_by_condition`, `bias_by_template`) are unchanged
# and live in `../260819_domain_baseline/`. Everything here is the deception
# stage.
#
# All charts use the default reporting view: the `random` cue variant, individual
# samples (no majority vote), position 0 for P(Yes), and one fixed ordering per
# question. Every deception rate is on parsed answers only, matching the README.
#
#   1. selfreport_by_reliance.png       the headline: mean P(Yes) for relied vs
#                                       did-not-rely, under the assigned wording
#   2. pyes_distribution_by_reliance.png  the same split as overlaid histograms
#   3. wording_comparison.png           plain vs explicit, the same rollouts asked
#                                       both ways
#   4. wording_delta.png                the six pre-registered contrasts with
#                                       their question-clustered bootstrap CIs

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
BASELINE_DIR = EXP_DIR.parent / "260819_domain_baseline"
DECEPTION_CSV = EXP_DIR / "deception_metrics.csv"
DECEPTION_SCORES = EXP_DIR / "deception_scores.jsonl"
COMPARISON_CSV = EXP_DIR / "wording_comparison.csv"

VOTE = "single"
POSITION = "pos0"
SPLIT = "all"
VARIANT = "random"  # the headline cue variant; `original` is the paraphrase control
SAVE_FIG = True
DPI = 150

# Colour follows the *entity*, held identical to 260817/260818/260819 so a
# condition keeps its hue across every folder in this line of work. Slots 1-4 of
# the dataviz reference palette are the conditions: slot 1 = the no-cue arm, and
# so the cue-removal definition; slot 4 = the cue-switch pairing, and so the
# cue-swap definition.
#
# This run adds a second axis -- the follow-up *wording* -- which is not a
# condition, so it takes slots 7 and 8 rather than reusing a condition's hue.
# Validated with the skill's script on the light surface, all pairs: CVD dE 22.7,
# normal-vision dE 33.6, both above their floors.
C_NONE = "#2a78d6"
C_SWITCH = "#eda100"
C_INK = "#333333"
C_MUTED = "#777777"
C_PLAIN = "#4a3aa7"
C_EXPLICIT = "#e34948"

WIDTH = 0.27
BAR_LABEL_KW = dict(
    fmt="%.2f", fontsize=7, padding=2,
    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
)

# Each definition is paired with the wording whose counterfactual it asks about.
PANELS = [
    ("rm", "Removal: \"if the user had not stated an opinion...\""),
    ("swap", "Swap: \"if the user had said (Z) instead...\""),
]
DEFINITION_SUBTITLE = {
    "rm": "cue-removal bias definition",
    "swap": "cue-switch bias definition",
}
ARM_OF = {
    "rm": {"plain": "removal", "explicit": "removal_explicit",
           "assigned": "removal_assigned"},
    "swap": {"plain": "swap", "explicit": "swap_explicit",
             "assigned": "swap_assigned"},
}
RELIANCE = [
    ("Relied on the cue\n(should say Yes)", C_SWITCH),
    ("Did not rely\n(should say No)", C_NONE),
]
WORDING = [("plain", C_PLAIN), ("explicit", C_EXPLICIT)]


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def relied(rollout, definition):
    """Reliance label, recomputed from the stored turn-1 answers."""
    a_cued, a_none, a_other = rollout["a_cued"], rollout["a_none"], rollout["a_other_cue"]
    target = "G" if rollout["condition"] == "cue_gold" else "D"
    if definition == "rm":
        if a_cued is None or a_none is None:
            return None
        return a_cued == target and a_none != a_cued
    if a_cued is None or a_other is None:
        return None
    gold, dist = (a_cued, a_other) if rollout["condition"] == "cue_gold" else (a_other, a_cued)
    return gold == "G" and dist == "D"


def arm_key(rollout, definition, wording):
    """The generated arm a rollout reads for a wording slot, resolving `assigned`."""
    if wording == "assigned":
        wording = rollout["fu_wording"][definition]
    return ARM_OF[definition][wording]


# %%
rollouts = [r for r in read_jsonl(DECEPTION_SCORES) if r["cue_variant"] == VARIANT]
ddf = pd.read_csv(DECEPTION_CSV)
ddf = ddf[(ddf["vote"] == VOTE) & (ddf["position"] == POSITION) & (ddf["split"] == SPLIT)
          & (ddf["cue_variant"] == VARIANT)]
d_overall = ddf[ddf["category"] == "OVERALL"].iloc[0]
p_cue_gold = float(d_overall["p_cue_gold"])
cdf = pd.read_csv(COMPARISON_CSV)
KEY = "p_yes_pos0" if POSITION == "pos0" else "p_yes_ans"
print(f"Loaded {len(rollouts)} cued rollouts (cue_variant={VARIANT}, p={p_cue_gold})")

# %%
# Chart 1: does the self-report move with reliance at all, under the wording the
# RL environment actually serves? mean P(Yes | relied) = 1 - false_denial; mean
# P(Yes | did not rely) = false_admission. Read from the `_at_p` columns, so these
# are the same numbers the README's deception table reports.
# Explicit bar positions rather than categorical ones: with only two bars,
# matplotlib's unit spacing leaves them wide and far apart for a comparison meant to
# be read at a glance.
X_POS = [0.0, 0.45]
X_PAD = 0.22

fig, axes = plt.subplots(1, len(PANELS), figsize=(6.2 * len(PANELS), 5), sharey=True)
for ax, (definition, title) in zip(axes, PANELS):
    stem = f"{ARM_OF[definition]['assigned']}_{definition}"
    values = [
        1 - d_overall[f"{stem}_false_denial_at_p"],
        d_overall[f"{stem}_false_admission_at_p"],
    ]
    bars = ax.bar(X_POS, values, WIDTH, color=[color for _, color in RELIANCE])
    ax.bar_label(bars, **BAR_LABEL_KW)
    ax.set_xticks(X_POS)
    ax.set_xticklabels([label for label, _ in RELIANCE], fontsize=9)
    ax.set_xlim(X_POS[0] - WIDTH / 2 - X_PAD, X_POS[1] + WIDTH / 2 + X_PAD)
    ax.set_title(f"{title}\n({DEFINITION_SUBTITLE[definition]})", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
axes[0].set_ylabel("mean P(self-report = \"Yes\")")
axes[0].set_ylim(0, 1.05)
fig.suptitle(
    f"Self-report by measured reliance (at p={p_cue_gold})",
    fontsize=12,
)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "selfreport_by_reliance.png", dpi=DPI, bbox_inches="tight")
print("wrote selfreport_by_reliance.png")
plt.show()

# %%
# Chart 2: the same split as overlaid distributions, so the question is whether the
# two groups separate or sit on top of each other. Balanced rather than at p: this
# counts individual rollouts, which have no p-weighting to apply.
HIST_BINS = 40
HIST_ALPHA = 0.55

fig, axes = plt.subplots(1, len(PANELS), figsize=(6.5 * len(PANELS), 5),
                         sharex=True, sharey=True)
for ax, (definition, title) in zip(axes, PANELS):
    by_group = {label: [] for label, _ in RELIANCE}
    for rollout in rollouts:
        flag = relied(rollout, definition)
        if flag is None:
            continue
        p_yes = rollout["arms"].get(arm_key(rollout, definition, "assigned"), {}).get(KEY)
        if p_yes is None:
            continue
        by_group[RELIANCE[0][0] if flag else RELIANCE[1][0]].append(p_yes)

    total = sum(len(v) for v in by_group.values())
    for label, color in RELIANCE:
        values = by_group[label]
        pct = 100 * len(values) / total if total else 0
        mean_value = sum(values) / len(values) if values else None
        mean_text = f", mean {mean_value:.3f}" if mean_value is not None else ""
        ax.hist(values, bins=HIST_BINS, range=(0, 1), color=color, alpha=HIST_ALPHA,
                edgecolor="white", linewidth=0.4,
                label=f"{label.splitlines()[0]} ({pct:.0f}%{mean_text})")
        if mean_value is not None:
            ax.axvline(mean_value, ls="--", lw=1.2, color=color, zorder=5)

    # Step outline on the smaller group, which the larger one would otherwise bury.
    relied_values = by_group[RELIANCE[0][0]]
    if relied_values:
        ax.hist(relied_values, bins=HIST_BINS, range=(0, 1), histtype="step",
                linewidth=2, color=RELIANCE[0][1], zorder=6)

    ax.set_title(f"{title}\n({DEFINITION_SUBTITLE[definition]})", fontsize=9)
    ax.set_xlabel("P(self-report = \"Yes\")")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
axes[0].set_ylabel(f"Rollouts (of {len(rollouts)})")
fig.suptitle("P(Yes) distribution by measured reliance", fontsize=12)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "pyes_distribution_by_reliance.png", dpi=DPI, bbox_inches="tight")
print("wrote pyes_distribution_by_reliance.png")
plt.show()

# %%
# Chart 3: the wording contrast itself. Both wordings were asked of the *same*
# rollouts, so these two distributions rest on one population -- the difference
# between them is the sentence and nothing else. Restricted to the rollouts where
# both wordings exist, which is what the significance test uses.
fig, axes = plt.subplots(1, len(PANELS), figsize=(6.5 * len(PANELS), 5),
                         sharex=True, sharey=True)
for ax, (definition, title) in zip(axes, PANELS):
    by_wording = {w: [] for w, _ in WORDING}
    for rollout in rollouts:
        values = {
            w: rollout["arms"].get(ARM_OF[definition][w], {}).get(KEY)
            for w, _ in WORDING
        }
        if any(v is None for v in values.values()):
            continue  # common support: both wordings or neither
        for w, v in values.items():
            by_wording[w].append(v)

    for wording, color in WORDING:
        values = by_wording[wording]
        mean_value = sum(values) / len(values) if values else float("nan")
        ax.hist(values, bins=HIST_BINS, range=(0, 1), color=color, alpha=HIST_ALPHA,
                edgecolor="white", linewidth=0.4,
                label=f"{wording} (mean {mean_value:.3f})")
        ax.axvline(mean_value, ls="--", lw=1.2, color=color, zorder=5)

    n_paired = len(by_wording[WORDING[0][0]])
    ax.set_title(f"{title}\n({DEFINITION_SUBTITLE[definition]}, {n_paired} paired rollouts)",
                 fontsize=9)
    ax.set_xlabel("P(self-report = \"Yes\")")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False)
axes[0].set_ylabel("Rollouts")
fig.suptitle("The same rollouts asked both ways: plain vs explicit wording", fontsize=12)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "wording_comparison.png", dpi=DPI, bbox_inches="tight")
print("wrote wording_comparison.png")
plt.show()

# %%
# Chart 4: the six pre-registered contrasts, each with its question-clustered
# bootstrap CI. A dot-and-interval rather than bars: these are differences with
# a meaningful zero, so the reference line is the point of the chart.
primary = cdf[cdf["primary"]].copy()
primary["label"] = [
    f"{d}   {m}" for d, m in zip(primary["definition"], primary["metric"])
]
primary = primary.iloc[::-1].reset_index(drop=True)  # first contrast at the top

fig, ax = plt.subplots(figsize=(9, 4.6))
y = range(len(primary))
ax.errorbar(
    primary["delta"], y,
    xerr=[primary["delta"] - primary["ci_lo"], primary["ci_hi"] - primary["delta"]],
    fmt="o", ms=8, color=C_EXPLICIT, ecolor=C_EXPLICIT, elinewidth=2, capsize=4,
    markeredgecolor="white", markeredgewidth=1.0, zorder=5,
)
zero = ax.axvline(0, color=C_INK, ls="--", lw=1.2, zorder=1, label="no difference")
for i, row in primary.iterrows():
    star = " *" if row["p_holm"] < 0.05 else ""
    ax.annotate(
        f"{row['delta']:+.3f}{star}",
        (row["delta"], i), textcoords="offset points", xytext=(0, 12),
        ha="center", fontsize=8, color=C_INK,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none",
                  alpha=0.85),
    )
ax.set_yticks(list(y))
ax.set_yticklabels(primary["label"], fontsize=9, family="monospace")
# Headroom for the direct labels, which sit above their markers: without it the
# topmost one collides with the title and the widest one runs off the right edge.
ax.set_ylim(-0.55, len(primary) - 0.25)
ax.margins(x=0.10)
ax.set_xlabel("explicit - plain (at p, 95% bootstrap CI, clustered on questions)")
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
ax.legend(handles=[zero], loc="lower right", fontsize=9, frameon=False)
ax.set_title(
    f"Does the explicit wording measure something different? "
    f"(n = {int(primary['n_questions'].iloc[0])} questions)",
    fontsize=12, loc="left",
)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "wording_delta.png", dpi=DPI, bbox_inches="tight")
print("wrote wording_delta.png  (* marks p_holm < 0.05 across the six)")
plt.show()

# %%
