# Plots for the sycophancy domain baseline.
#
# Interactive notebook (`# %%` cells). Run cell-by-cell in VS Code / Cursor, or
# `python plot_results.py` for a non-interactive save.
#
# One script rather than the two the ancestor folders had: this folder runs both
# a turn-1 and a deception stage, and the deception charts simply skip themselves
# if `deception_scores.jsonl` is not there yet, so it can be run after either.
#
# All charts use the default reporting view: the `random` cue variant, individual
# samples (no majority vote), position 0 for P(Yes), and one fixed ordering per
# question. Every bias and deception rate is on parsed answers only, matching the
# README.
#
# Turn-1
#   1. bias_by_category.png   the two bias definitions per category, against
#                             their matched resampling nulls
#   2. accuracy_by_condition.png
#   3. bias_by_template.png   the new axis: bias per cue paraphrase
# Deception (skipped if absent)
#   4. selfreport_by_reliance.png     mean P(Yes) for relied vs did-not-rely
#   5. pyes_distribution_by_reliance.png   the same split as overlaid histograms

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
TURN1_CSV = EXP_DIR / "turn1_metrics.csv"
TEMPLATE_CSV = EXP_DIR / "template_metrics.csv"
DECEPTION_CSV = EXP_DIR / "deception_metrics.csv"
DECEPTION_SCORES = EXP_DIR / "deception_scores.jsonl"

VOTE = "single"
POSITION = "pos0"
SPLIT = "all"
VARIANT = "random"  # the headline cue variant; `original` is the paraphrase control
SAVE_FIG = True
DPI = 150
CHANCE = 0.5

# Colour follows the *condition*, held identical to 260817/260818/260819 so a
# condition keeps its hue across every folder in this line of work. These are
# categorical slots 1-4 of the dataviz reference palette, in fixed order: slot 1 =
# the no-cue arm, and so the cue-removal definition, which pairs against it;
# slot 2 = cue at the correct option; slot 3 = cue at the distractor; slot 4 = the
# cue-switch pairing, and so the cue-swap definition.
C_NONE = "#2a78d6"
C_GOLD = "#eb6834"
C_DIST = "#1baf7a"
C_SWITCH = "#eda100"
C_INK = "#333333"
C_MUTED = "#777777"

WIDTH = 0.27
BAR_LABEL_KW = dict(
    fmt="%.2f", fontsize=7, padding=2,
    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
)
LEGEND_KW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=9, frameon=False)


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# %%
df = pd.read_csv(TURN1_CSV)
df = df[(df["vote"] == VOTE) & (df["split"] == SPLIT)]
p_cue_gold = float(df["p_cue_gold"].iloc[0])
print(f"Loaded turn1_metrics.csv (vote={VOTE}, split={SPLIT}, p={p_cue_gold})")
for variant in ("random", "original"):
    row = df[(df["category"] == "OVERALL") & (df["cue_variant"] == variant)].iloc[0]
    print(
        f"  {variant:<9} switch={row['bias_cue_switch_parsed_only']:.3f} "
        f"removal@p={row['rm_relied_parsed_only_at_p']:.3f} "
        f"null={row['noise_dir_cued']:.3f} acc_at_p={row['acc_at_p']:.3f}"
    )


def grouped_bars(ax, frame, columns, colors, labels, null_cols=None, xcol="category",
                 width=WIDTH, label_all=False):
    """One bar per series per category, with each series' matched null on top of it.

    The null rides its own bar as a diamond rather than a connected line: it is a
    per-series reference level, not a trend across the category axis, and joining
    the points implied a continuity the x-axis does not have.

    `label_all` direct-labels every bar, which is readable at two series and
    cluttered at three -- so the default labels the first series only.
    """
    x = range(len(frame))
    n = len(columns)
    offsets = [(i - (n - 1) / 2) * width for i in range(n)]
    bars = []
    nulls = null_cols or [None] * n
    for column, color, label, offset, null_col in zip(columns, colors, labels, offsets, nulls):
        bars.append(
            ax.bar([i + offset for i in x], frame[column], width, label=label, color=color)
        )
        if null_col is not None:
            ax.scatter(
                [i + offset for i in x], frame[null_col],
                marker="D", s=30, color=color, edgecolor="white", linewidth=1.0, zorder=4,
            )
    for series in (bars if label_all else bars[:1]):
        ax.bar_label(series, **BAR_LABEL_KW)
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame[xcol], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    return bars


# %%
# Chart 1: the two bias definitions per category. Cue-removal is weighted at the
# cue probability the RL environment serves; cue-switch is p-invariant (its label
# is pair-level, so it takes the same value whichever arm was served) and is
# therefore shown as measured.
fig, ax = plt.subplots(figsize=(9, 5.6))
sub = df[(df["cue_variant"] == VARIANT) & (df["category"] != "OVERALL")]
sub = sub.sort_values("bias_cue_switch_parsed_only", ascending=False).reset_index(drop=True)
bars = grouped_bars(
    ax, sub,
    ["rm_relied_parsed_only_at_p", "bias_cue_switch_parsed_only"],
    [C_NONE, C_SWITCH],
    [f"cue-removal (at p={p_cue_gold})", "cue-switch"],
    null_cols=["noise_dir_none", "noise_dir_cued"],
    width=0.34, label_all=True,
)
null_proxy = ax.scatter([], [], marker="D", s=30, color=C_MUTED, edgecolor="white",
                        linewidth=1.0, label="matched resampling null")
ax.set_ylim(0, 0.62)
ax.set_ylabel("rate")
# Explicit handles so the two bar series read before their shared null marker.
ax.legend(handles=[*bars, null_proxy], **LEGEND_KW)
ax.set_title("Bias by category", fontsize=12, loc="left")
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "bias_by_category.png", dpi=DPI, bbox_inches="tight")
print("wrote bias_by_category.png")
plt.show()

# %%
# Chart 2: accuracy per condition, with the reward headroom as the gap between the
# base-policy-at-p line and the always-follow-the-cue line.
fig, ax = plt.subplots(figsize=(9, 5.6))
sub = df[(df["cue_variant"] == VARIANT) & (df["category"] != "OVERALL")]
sub = sub.sort_values("acc_none", ascending=False).reset_index(drop=True)
overall = df[(df["category"] == "OVERALL") & (df["cue_variant"] == VARIANT)].iloc[0]
bars = grouped_bars(
    ax, sub,
    ["acc_none", "acc_cue_gold", "acc_cue_dist"],
    [C_NONE, C_GOLD, C_DIST],
    ["no cue", "cue at gold", "cue at distractor"],
)
chance = ax.axhline(CHANCE, color=C_INK, ls="--", lw=1, label="chance (2 options)")
ceiling = ax.axhline(p_cue_gold, color=C_SWITCH, ls=":", lw=1.4,
                     label=f"always follow the cue (p={p_cue_gold})")
at_p = ax.axhline(overall["acc_at_p"], color=C_MUTED, ls="-.", lw=1.4,
                  label=f"base policy at p ({overall['acc_at_p']:.3f})")
ax.set_ylim(0, 1.05)
ax.set_ylabel("accuracy")
# Explicit handles: the three conditions, then the three reference levels. The gap
# between the last two is the reward headroom.
ax.legend(handles=[*bars, chance, at_p, ceiling], **LEGEND_KW)
ax.set_title("Accuracy by condition", fontsize=12, loc="left")
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "accuracy_by_condition.png", dpi=DPI, bbox_inches="tight")
print("wrote accuracy_by_condition.png")
plt.show()

# %%
# Chart 3: bias per cue paraphrase -- the axis this run adds.
tdf = pd.read_csv(TEMPLATE_CSV)
tdf = tdf[(tdf["vote"] == VOTE) & (tdf["split"] == SPLIT)]
tpl = tdf[tdf["cue_template_idx"] != "OVERALL"].copy()
tpl["idx"] = tpl["cue_template_idx"].astype(int)
tpl = tpl.sort_values("bias_cue_switch_parsed_only", ascending=False).reset_index(drop=True)
# Strip the letter so the label reads as the template, not one rendering of it.
tpl["label"] = [
    f"[{int(i)}] " + t.replace("(A)", "(X)").replace("(B)", "(X)")
    for i, t in zip(tpl["idx"], tpl["cue_text_example"])
]

fig, ax = plt.subplots(figsize=(11, 6))
y = range(len(tpl))
bar_h = 0.28
y_off = bar_h / 2 + 0.02
y_switch = [i + y_off for i in y]
y_removal = [i - y_off for i in y]

bars_switch = ax.barh(y_switch, tpl["bias_cue_switch_parsed_only"], height=bar_h,
                      color=C_SWITCH, label="cue-switch")
bars_removal = ax.barh(y_removal, tpl["rm_relied_parsed_only_at_p"], height=bar_h,
                       color=C_NONE, label=f"cue-removal (at p={p_cue_gold})")
ax.bar_label(bars_switch, fmt="%.3f", fontsize=8, padding=3)
ax.bar_label(bars_removal, fmt="%.3f", fontsize=8, padding=3)
ax.scatter(tpl["noise_dir_cued"], y_switch, marker="o", s=28, color=C_INK, zorder=4)
ax.scatter(tpl["noise_dir_none"], y_removal, marker="o", s=28, color=C_INK, zorder=4)
null_proxy = ax.scatter([], [], marker="o", s=28, color=C_INK,
                        label="matched resampling null")
overall_t = tdf[tdf["cue_template_idx"] == "OVERALL"].iloc[0]
mean_line = ax.axvline(overall_t["bias_cue_switch_parsed_only"], color=C_MUTED, ls="--",
                       lw=1.2,
                       label=f"cue-switch mixture mean ({overall_t['bias_cue_switch_parsed_only']:.3f})")
ax.set_yticks(list(y))
ax.set_yticklabels(tpl["label"], fontsize=9, family="monospace")
ax.invert_yaxis()
ax.set_xlim(0, 0.68)
ax.set_xlabel("rate")
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
ax.legend(handles=[bars_switch, bars_removal, null_proxy, mean_line],
          loc="lower right", fontsize=9, frameon=False)
ax.set_title("Bias by paraphrase", fontsize=12, loc="left")
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "bias_by_template.png", dpi=DPI, bbox_inches="tight")
print("wrote bias_by_template.png")
plt.show()

# %%
# ---------------------------------------------------------------------------
# Deception charts -- skipped if the follow-up stage has not run.
# ---------------------------------------------------------------------------
if not DECEPTION_SCORES.exists():
    print(f"\n{DECEPTION_SCORES.name} not found; skipping the deception charts.")
    raise SystemExit(0)

rollouts = [r for r in read_jsonl(DECEPTION_SCORES) if r["cue_variant"] == VARIANT]
ddf = pd.read_csv(DECEPTION_CSV)
ddf = ddf[(ddf["vote"] == VOTE) & (ddf["position"] == POSITION) & (ddf["split"] == SPLIT)
          & (ddf["cue_variant"] == VARIANT)]
d_overall = ddf[ddf["category"] == "OVERALL"].iloc[0]

# Each arm is paired with the definition whose counterfactual it asks about.
PANELS = [
    ("removal", "rm", "Removal: \"no opinion → chosen differently?\""),
    ("swap", "swap", "Swap: \"said (X) instead → chosen differently?\""),
]
DEFINITION_SUBTITLE = {
    "rm": "cue-removal bias definition",
    "swap": "cue-switch bias definition",
}
RELIANCE = [
    ("Relied on the cue\n(should say Yes)", C_SWITCH),
    ("Did not rely\n(should say No)", C_NONE),
]
print(f"\nLoaded {len(rollouts)} cued rollouts (cue_variant={VARIANT})")


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


# %%
# Chart 4: does the self-report move with reliance at all?
# mean P(Yes | relied) = 1 - false_denial; mean P(Yes | did not rely) =
# false_admission. Both read from the `_at_p` columns, so they are the same numbers
# the README's deception table reports.
# Explicit bar positions rather than categorical ones: with only two bars,
# matplotlib's unit spacing leaves them wide and far apart for a comparison meant to
# be read at a glance.
X_POS = [0.0, 0.45]
X_PAD = 0.22

fig, axes = plt.subplots(1, len(PANELS), figsize=(6.2 * len(PANELS), 5), sharey=True)
for ax, (arm, definition, title) in zip(axes, PANELS):
    stem = f"{arm}_{definition}"
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
fig.suptitle(f"Self-report by measured reliance (at p={p_cue_gold})", fontsize=12)
fig.tight_layout()
if SAVE_FIG:
    fig.savefig(EXP_DIR / "selfreport_by_reliance.png", dpi=DPI, bbox_inches="tight")
print("wrote selfreport_by_reliance.png")
plt.show()

# %%
# Chart 5: the same split as overlaid distributions, so the question is whether the
# two groups separate or sit on top of each other. Balanced rather than at p: this
# counts individual rollouts, which have no p-weighting to apply.
KEY = "p_yes_pos0" if POSITION == "pos0" else "p_yes_ans"
HIST_BINS = 40
HIST_ALPHA = 0.55

fig, axes = plt.subplots(1, len(PANELS), figsize=(6.5 * len(PANELS), 5),
                         sharex=True, sharey=True)
for ax, (arm, definition, title) in zip(axes, PANELS):
    by_group = {label: [] for label, _ in RELIANCE}
    for rollout in rollouts:
        flag = relied(rollout, definition)
        if flag is None:
            continue
        p_yes = rollout["arms"].get(arm, {}).get(KEY)
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
