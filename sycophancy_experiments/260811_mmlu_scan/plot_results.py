# Plot per-category results from the MMLU-Pro difficulty scan.
#
# Interactive notebook (`# %%` cells). Run cell-by-cell in VS Code / Cursor,
# or `python plot_results.py` for a non-interactive save.
#
# Draws majority-vote accuracy and mean single-sample accuracy as grouped bars,
# overlays the self-consistency rate, and marks the random-guess baseline.

# %%
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent

SUMMARY_CSV = EXP_DIR / "category_summary.csv"
SCAN_SET = EXP_DIR / "scan_set.jsonl"
COMPLETIONS_JSONL = EXP_DIR / "completions.jsonl"
SCORES_JSONL = EXP_DIR / "scores.jsonl"
OUTPUT_FIG = EXP_DIR / "category_accuracy.png"

SAVE_FIG = True  # set False to skip writing the PNG
DPI = 150

# %%
df = pd.read_csv(SUMMARY_CSV)
df = df.sort_values("majority_accuracy", ascending=False).reset_index(drop=True)
print(f"Loaded {len(df)} categories from {SUMMARY_CSV}")

n_questions = int(df["n_questions"].sum())
n_samples = int(df["n_samples"].sum())
overall_majority = (df["majority_accuracy"] * df["n_questions"]).sum() / n_questions
overall_sample = (df["sample_accuracy"] * df["n_samples"]).sum() / n_samples
overall_agree = (df["all_agree_rate"] * df["n_questions"]).sum() / n_questions

if SCAN_SET.exists():
    chances = [
        1.0 / len(json.loads(line)["options"])
        for line in SCAN_SET.open()
        if line.strip()
    ]
    chance = sum(chances) / len(chances)
else:
    chance = None

parse_fail = (df["parse_fail_rate"] * df["n_samples"]).sum() / n_samples
print(
    f"overall majority={overall_majority:.3f}  sample={overall_sample:.3f}  "
    f"agree={overall_agree:.3f}  chance={chance:.3f}  parse_fail={parse_fail:.1%}"
    if chance is not None
    else f"overall majority={overall_majority:.3f}  sample={overall_sample:.3f}  "
    f"agree={overall_agree:.3f}  parse_fail={parse_fail:.1%}"
)

# %%
x = range(len(df))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 6.5))

bars_majority = ax.bar(
    [i - width / 2 for i in x],
    df["majority_accuracy"],
    width,
    label="Majority-vote accuracy (3 samples)",
    color="#3b74b8",
)
ax.bar(
    [i + width / 2 for i in x],
    df["sample_accuracy"],
    width,
    label="Mean single-sample accuracy",
    color="#9dc3e6",
)

ax.plot(
    list(x),
    df["all_agree_rate"],
    marker="D",
    linestyle="none",
    markersize=7,
    color="#d1495b",
    label="Self-consistency (all 3 samples agree)",
)

ax.axhline(
    overall_majority,
    linestyle="--",
    linewidth=1.2,
    color="#333333",
    label=f"Overall majority accuracy ({overall_majority:.3f})",
)
if chance is not None:
    ax.axhline(
        chance,
        linestyle=":",
        linewidth=1.2,
        color="#777777",
        label=f"Random guess ({chance:.3f})",
    )

ax.bar_label(bars_majority, fmt="%.2f", fontsize=8, padding=2)

ax.set_xticks(list(x))
ax.set_xticklabels(df["category"], rotation=35, ha="right")
ax.set_ylabel("Accuracy / rate")
ax.set_ylim(0, max(1.0, df["majority_accuracy"].max() + 0.12))
ax.set_title(
    "MMLU-Pro per-category difficulty scan - Meta-Llama-3-8B-Instruct\n"
    f"{n_questions} questions ({n_questions // len(df)}/category), "
    f"{n_samples} samples, zero-shot CoT, temperature 1.0  |  "
    f"mean single-sample {overall_sample:.3f}, self-consistency {overall_agree:.3f}"
)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()

if SAVE_FIG:
    fig.savefig(OUTPUT_FIG, dpi=DPI)
    print(f"Saved chart to {OUTPUT_FIG}")
else:
    print("SAVE_FIG=False — skipped writing PNG")

plt.show()

# %%
# Browse random completions. Edit the knobs and re-run this cell.
CATEGORIES = [
    # "biology",
    # "psychology",
    # "economics",
    # "business",
    # "health",
    # "computer science",
    # "other",
    # "history",
    # "math",
    # "physics",
    # "philosophy",
    # "chemistry",
    # "law",
    # "engineering",
]  # empty list → all categories
N_PER_CATEGORY = 3
SAMPLE_SEED = 0


def _user_content(prompt: str) -> str:
    """Strip Llama-3 chat wrappers; fall back to the raw prompt."""
    start = "<|start_header_id|>user<|end_header_id|>\n\n"
    end = "<|eot_id|>"
    if start not in prompt:
        return prompt
    body = prompt.split(start, 1)[1]
    if end in body:
        body = body.split(end, 1)[0]
    return body.strip()


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


completions = _load_jsonl(COMPLETIONS_JSONL)
score_by_qid = {
    row["question_id"]: row for row in _load_jsonl(SCORES_JSONL)
} if SCORES_JSONL.exists() else {}

available = sorted({c["category"] for c in completions})
selected = available if not CATEGORIES else [c for c in CATEGORIES if c in available]
missing = set(CATEGORIES) - set(available) if CATEGORIES else set()
if missing:
    print(f"WARNING: unknown categories ignored: {sorted(missing)}")

rng = random.Random(SAMPLE_SEED)
print(
    f"Sampling {N_PER_CATEGORY}/category from {selected} "
    f"(seed={SAMPLE_SEED}, pool={len(completions)} completions)\n"
)

for category in selected:
    pool = [c for c in completions if c["category"] == category]
    take = min(N_PER_CATEGORY, len(pool))
    samples = rng.sample(pool, take)
    print("=" * 88)
    print(f"CATEGORY: {category}  ({take}/{len(pool)} shown)")
    print("=" * 88)
    for i, rec in enumerate(samples, 1):
        score = score_by_qid.get(rec["question_id"], {})
        preds = score.get("preds") or []
        stages = score.get("stages") or []
        correct = score.get("correct") or []
        si = rec["sample_idx"]
        pred = preds[si] if si < len(preds) else None
        stage = stages[si] if si < len(stages) else None
        is_correct = correct[si] if si < len(correct) else None

        print(f"\n--- [{i}/{take}] qid={rec['question_id']}  sample={si} ---")
        print(
            f"gold={rec['answer']}  pred={pred}  correct={is_correct}  "
            f"extract={stage}"
            + (f"  preds={preds}" if preds else "")
        )
        print("\n[prompt]")
        print(_user_content(rec["prompt"]))
        print("\n[completion]")
        print(rec["completion"])
        print()

# %%
# Pairwise disagreement: P(two independent samples yield different extracted answers).
# Per question: fraction of unordered sample pairs whose preds differ (None counts as
# its own answer). Per category / overall: mean over questions.


def pairwise_disagreement(preds: list) -> float:
    n = len(preds)
    if n < 2:
        return float("nan")
    n_pairs = n * (n - 1) // 2
    n_disagree = sum(
        preds[i] != preds[j] for i in range(n) for j in range(i + 1, n)
    )
    return n_disagree / n_pairs


with SCORES_JSONL.open() as f:
    scores = [json.loads(line) for line in f if line.strip()]

for row in scores:
    row["pairwise_disagreement"] = pairwise_disagreement(row["preds"])

disagree_df = (
    pd.DataFrame(scores)
    .groupby("category", as_index=False)
    .agg(
        pairwise_disagreement=("pairwise_disagreement", "mean"),
        all_agree_rate=("all_agree", "mean"),
    )
    .sort_values("pairwise_disagreement", ascending=False)
    .reset_index(drop=True)
)

overall_disagree = float(pd.DataFrame(scores)["pairwise_disagreement"].mean())
print(
    f"Pairwise disagreement rate (mean over questions)\n"
    f"Overall: {overall_disagree:.3f}\n"
)
print(disagree_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# %%
