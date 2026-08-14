# Plot per-category results from the two-option MMLU-Pro scan.
#
# Interactive notebook (`# %%` cells). Run cell-by-cell in VS Code / Cursor,
# or `python plot_results.py` for a non-interactive save.
#
# Chart 1: per-category majority accuracy split by option ordering, with the
#          order-robust rate overlaid and the 0.5 chance line marked.
# Chart 2: per-question comparison against the 10-option scan over the same
#          1400 questions.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
SCAN_10OPT = EXP_DIR.parent / "260811_mmlu_scan"

SUMMARY_CSV = EXP_DIR / "category_summary.csv"
SCORES_JSONL = EXP_DIR / "scores.jsonl"
SCORES_10OPT = SCAN_10OPT / "scores.jsonl"

OUTPUT_FIG = EXP_DIR / "category_accuracy.png"
OUTPUT_FIG_COMPARE = EXP_DIR / "vs_10option.png"

SAVE_FIG = True  # set False to skip writing the PNGs
DPI = 150
CHANCE = 0.5  # two options

# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs).
C_GOLD_FIRST = "#2a78d6"
C_GOLD_SECOND = "#eb6834"
C_ROBUST = "#1baf7a"
C_INK = "#333333"
C_MUTED = "#777777"

# %%
df = pd.read_csv(SUMMARY_CSV)
df = df.sort_values("majority_accuracy", ascending=False).reset_index(drop=True)
print(f"Loaded {len(df)} categories from {SUMMARY_CSV}")

n_questions = int(df["n_questions"].sum())
n_samples = int(df["n_samples"].sum())


def weighted(column: str, by: str) -> float:
    return (df[column] * df[by]).sum() / df[by].sum()


overall_majority = weighted("majority_accuracy", "n_rows")
overall_first = weighted("majority_accuracy_gold_first", "n_questions")
overall_second = weighted("majority_accuracy_gold_second", "n_questions")
overall_robust = weighted("order_robust_rate", "n_questions")
overall_pa = weighted("p_answer_a", "n_samples")
parse_fail = weighted("parse_fail_rate", "n_samples")

print(
    f"overall majority={overall_majority:.3f}  gold_first={overall_first:.3f}  "
    f"gold_second={overall_second:.3f}  position_gap={overall_first - overall_second:+.3f}\n"
    f"order_robust={overall_robust:.3f}  p(A)={overall_pa:.3f}  "
    f"parse_fail={parse_fail:.1%}"
)

# %%
x = range(len(df))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 6.5))

bars_first = ax.bar(
    [i - width / 2 for i in x],
    df["majority_accuracy_gold_first"],
    width,
    label="Gold as option A",
    color=C_GOLD_FIRST,
)
ax.bar(
    [i + width / 2 for i in x],
    df["majority_accuracy_gold_second"],
    width,
    label="Gold as option B",
    color=C_GOLD_SECOND,
)

ax.plot(
    list(x),
    df["order_robust_rate"],
    marker="D",
    linestyle="none",
    markersize=8,
    color=C_ROBUST,
    markeredgecolor="white",
    markeredgewidth=1.5,
    label="Order-robust (correct under both orderings)",
)

ax.axhline(
    overall_majority,
    linestyle="--",
    linewidth=1.2,
    color=C_INK,
    label=f"Overall majority accuracy ({overall_majority:.3f})",
)
ax.axhline(
    CHANCE,
    linestyle=":",
    linewidth=1.2,
    color=C_MUTED,
    label=f"Random guess ({CHANCE:.3f})",
)

ax.bar_label(bars_first, fmt="%.2f", fontsize=8, padding=2)

ax.set_xticks(list(x))
ax.set_xticklabels(df["category"], rotation=35, ha="right")
ax.set_ylabel("Accuracy / rate")
ax.set_ylim(0, 1.05)
ax.set_title(
    "Two-option MMLU-Pro scan - Meta-Llama-3-8B-Instruct\n"
    f"{n_questions} questions x 2 orderings, {n_samples} samples, "
    "zero-shot CoT, temperature 1.0  |  "
    f"position gap {overall_first - overall_second:+.3f}, "
    f"order-robust {overall_robust:.3f}"
)
# Below the axis: the bars run to y=0, so any in-axes legend covers data.
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=2,
    fontsize=9,
    frameon=False,
)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()

if SAVE_FIG:
    fig.savefig(OUTPUT_FIG, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG}")
else:
    print("SAVE_FIG=False - skipped writing PNG")

plt.show()

# %%
# Per-question comparison against the 10-option scan. The question sets are
# identical by construction (pair_set.jsonl is built from scan_set.jsonl), so
# these are paired measurements on the same items.


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


two_opt = load_jsonl(SCORES_JSONL)
ten_opt = load_jsonl(SCORES_10OPT)

# Two-option accuracy per question = mean majority-correct over the orderings.
two_by_qid: dict[int, list[bool]] = {}
for row in two_opt:
    two_by_qid.setdefault(row["question_id"], []).append(row["majority_correct"])

ten_by_qid = {row["question_id"]: row for row in ten_opt}
shared = sorted(set(two_by_qid) & set(ten_by_qid))
print(
    f"{len(shared)} shared questions "
    f"({len(two_by_qid)} two-option, {len(ten_by_qid)} ten-option)"
)

compare = pd.DataFrame(
    [
        {
            "question_id": qid,
            "category": ten_by_qid[qid]["category"],
            "acc_2opt": sum(two_by_qid[qid]) / len(two_by_qid[qid]),
            "acc_10opt": float(ten_by_qid[qid]["majority_correct"]),
        }
        for qid in shared
    ]
)

by_cat = (
    compare.groupby("category", as_index=False)
    .agg(acc_2opt=("acc_2opt", "mean"), acc_10opt=("acc_10opt", "mean"))
    .sort_values("acc_2opt", ascending=False)
    .reset_index(drop=True)
)

print(
    f"\nOverall: 10-option {compare['acc_10opt'].mean():.3f} -> "
    f"2-option {compare['acc_2opt'].mean():.3f}"
)
print(by_cat.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# Where the model was already wrong with 10 options, does binarizing rescue it?
was_wrong = compare[compare["acc_10opt"] == 0]
print(
    f"\nOf {len(was_wrong)} questions wrong in the 10-option scan, "
    f"mean two-option accuracy is {was_wrong['acc_2opt'].mean():.3f}"
)

# %%
xc = range(len(by_cat))

fig2, ax2 = plt.subplots(figsize=(13, 6))
bars_2 = ax2.bar(
    [i - width / 2 for i in xc],
    by_cat["acc_2opt"],
    width,
    label="Two options",
    color=C_GOLD_FIRST,
)
ax2.bar(
    [i + width / 2 for i in xc],
    by_cat["acc_10opt"],
    width,
    label="Original options (10-option scan)",
    color=C_GOLD_SECOND,
)

ax2.axhline(CHANCE, linestyle=":", linewidth=1.2, color=C_MUTED, label="Two-option chance (0.500)")
ax2.bar_label(bars_2, fmt="%.2f", fontsize=8, padding=2)

ax2.set_xticks(list(xc))
ax2.set_xticklabels(by_cat["category"], rotation=35, ha="right")
ax2.set_ylabel("Majority-vote accuracy")
ax2.set_ylim(0, 1.05)
ax2.set_title(
    "Same 1400 MMLU-Pro questions, two options vs original options\n"
    "Meta-Llama-3-8B-Instruct, zero-shot CoT, temperature 1.0"
)
ax2.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=3,
    fontsize=9,
    frameon=False,
)
ax2.grid(axis="y", alpha=0.3)
ax2.set_axisbelow(True)

fig2.tight_layout()

if SAVE_FIG:
    fig2.savefig(OUTPUT_FIG_COMPARE, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG_COMPARE}")

plt.show()

# %%
