"""Parse and score the two-option MMLU-Pro scan completions.

Answer extraction follows the TIGER-AI-Lab/MMLU-Pro chain, restricted to A/B:
  1. "answer is (X)"
  2. "answer: X"
  3. the last *marked* letter -- "(A)", "A." or "A:"

Stage 3 is deliberately tighter than the reference's bare last-standalone-letter
rule. With only two options a stray letter is right 50% of the time by luck
instead of ~11%, and the bare rule matches the English article "A" at the start
of a sentence, so it would inflate accuracy. Every accuracy is therefore also
reported "strict": stage-3 extractions treated as parse failures.

As in the 10-option scan, unparseable completions are NOT scored as a random
guess -- they get pred=None and count as incorrect.

Each question appears twice, once per option ordering. Scoring is per
(question_id, order); the summary additionally reports the gold_first vs
gold_second accuracy gap (the model's A/B positional prior) and the
order-robust rate (majority-correct under both orderings).

Usage:
    python score_completions.py
    python score_completions.py --completions dry_run_completions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

PROFESSIONAL_LAW_SRC = "ori_mmlu-professional_law"

PATTERN_ANSWER_IS = re.compile(r"answer is \(?([AB])\)?")
PATTERN_ANSWER_COLON = re.compile(r".*[aA]nswer:\s*([AB])")
PATTERN_MARKED_LETTER = re.compile(r"\(([AB])\)|\b([AB])[.:]")

FALLBACK_STAGE = "marked_letter"

SUMMARY_FIELDS = [
    "category",
    "n_questions",
    "n_rows",
    "n_samples",
    "sample_accuracy",
    "sample_accuracy_strict",
    "majority_accuracy",
    "majority_accuracy_gold_first",
    "majority_accuracy_gold_second",
    "position_gap",
    "p_answer_a",
    "order_robust_rate",
    "all_agree_rate",
    "parse_fail_rate",
    "marked_letter_rate",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_answer(text: str) -> tuple[str | None, str]:
    """Return (predicted letter, extraction stage) using the A/B regex chain."""
    match = PATTERN_ANSWER_IS.search(text)
    if match:
        return match.group(1), "answer_is"
    match = PATTERN_ANSWER_COLON.search(text)
    if match:
        return match.group(1), "answer_colon"
    matches = PATTERN_MARKED_LETTER.findall(text)
    if matches:
        # findall yields (paren_group, punct_group); exactly one is non-empty.
        last = matches[-1]
        return last[0] or last[1], FALLBACK_STAGE
    return None, "none"


def majority_vote(preds: list[str | None]) -> str | None:
    """Most common parsed prediction; ties broken by earliest occurrence."""
    parsed = [p for p in preds if p is not None]
    if not parsed:
        return None
    counts = Counter(parsed)
    best = max(counts.values())
    for pred in parsed:
        if counts[pred] == best:
            return pred
    return None


def score_row(records: list[dict]) -> dict:
    """Aggregate the samples for one (question_id, order) into a scored record."""
    records = sorted(records, key=lambda r: r["sample_idx"])
    first = records[0]
    gold = first["answer"]

    preds, stages, correct = [], [], []
    for rec in records:
        pred, stage = extract_answer(rec["completion"])
        preds.append(pred)
        stages.append(stage)
        correct.append(pred == gold)

    # Strict view: a fallback extraction is treated as no extraction at all.
    strict_preds = [None if s == FALLBACK_STAGE else p for p, s in zip(preds, stages)]
    strict_correct = [p == gold for p in strict_preds]

    n = len(records)
    n_parsed = sum(p is not None for p in preds)
    majority_pred = majority_vote(preds)

    return {
        "question_id": first["question_id"],
        "order": first["order"],
        "category": first["category"],
        "src": first["src"],
        "answer": gold,
        "n_samples": n,
        "preds": preds,
        "stages": stages,
        "correct": correct,
        "strict_correct": strict_correct,
        "n_parsed": n_parsed,
        "sample_accuracy": sum(correct) / n,
        "majority_pred": majority_pred,
        "majority_correct": majority_pred == gold,
        # Consistency: every sample parsed and landed on the same letter.
        "all_agree": n_parsed == n and len(set(preds)) == 1,
    }


def metrics(rows: list[dict]) -> dict:
    """Metrics over a set of scored (question_id, order) rows."""
    n_rows = len(rows)
    n_samples = sum(r["n_samples"] for r in rows)
    n_parsed = sum(r["n_parsed"] for r in rows)
    n_fallback = sum(s == FALLBACK_STAGE for r in rows for s in r["stages"])

    by_order: dict[str, list[dict]] = defaultdict(list)
    by_question: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_order[row["order"]].append(row)
        by_question[row["question_id"]].append(row)

    def majority_acc(subset: list[dict]) -> float:
        return sum(r["majority_correct"] for r in subset) / len(subset) if subset else float("nan")

    acc_first = majority_acc(by_order.get("gold_first", []))
    acc_second = majority_acc(by_order.get("gold_second", []))

    # Order-robust: majority-correct under both orderings. Only defined for
    # questions where both were generated.
    paired = [rs for rs in by_question.values() if len(rs) == 2]
    order_robust = (
        sum(all(r["majority_correct"] for r in rs) for rs in paired) / len(paired)
        if paired
        else float("nan")
    )

    all_preds = [p for r in rows for p in r["preds"] if p is not None]
    p_answer_a = all_preds.count("A") / len(all_preds) if all_preds else float("nan")

    return {
        "n_questions": len(by_question),
        "n_rows": n_rows,
        "n_samples": n_samples,
        "sample_accuracy": sum(sum(r["correct"]) for r in rows) / n_samples,
        "sample_accuracy_strict": sum(sum(r["strict_correct"]) for r in rows) / n_samples,
        "majority_accuracy": majority_acc(rows),
        "majority_accuracy_gold_first": acc_first,
        "majority_accuracy_gold_second": acc_second,
        # Positive => the model favours option A regardless of where gold sits.
        "position_gap": acc_first - acc_second,
        "p_answer_a": p_answer_a,
        "order_robust_rate": order_robust,
        "all_agree_rate": sum(r["all_agree"] for r in rows) / n_rows,
        "parse_fail_rate": (n_samples - n_parsed) / n_samples,
        "marked_letter_rate": n_fallback / n_samples,
    }


def summarize(scored: list[dict]) -> list[dict]:
    """Per-category metrics, sorted by category name."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_category[row["category"]].append(row)
    return [
        {"category": category, **metrics(by_category[category])}
        for category in sorted(by_category)
    ]


def write_summary_csv(path: Path, summary: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row[k] for k in SUMMARY_FIELDS})
    print(f"  Wrote {len(summary)} rows to {path}")


def print_table(title: str, summary: list[dict], overall: dict) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'category':<18} {'maj_acc':>8} {'strict_s':>8} {'A_first':>8} "
        f"{'B_first':>8} {'robust':>8} {'p(A)':>7} {'agree':>7} {'fail':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(summary, key=lambda r: r["majority_accuracy"], reverse=True):
        print(
            f"{row['category']:<18} {row['majority_accuracy']:>8.3f} "
            f"{row['sample_accuracy_strict']:>8.3f} "
            f"{row['majority_accuracy_gold_first']:>8.3f} "
            f"{row['majority_accuracy_gold_second']:>8.3f} "
            f"{row['order_robust_rate']:>8.3f} {row['p_answer_a']:>7.3f} "
            f"{row['all_agree_rate']:>7.3f} {row['parse_fail_rate']:>7.3f}"
        )
    print("-" * len(header))
    print(
        f"{'OVERALL':<18} {overall['majority_accuracy']:>8.3f} "
        f"{overall['sample_accuracy_strict']:>8.3f} "
        f"{overall['majority_accuracy_gold_first']:>8.3f} "
        f"{overall['majority_accuracy_gold_second']:>8.3f} "
        f"{overall['order_robust_rate']:>8.3f} {overall['p_answer_a']:>7.3f} "
        f"{overall['all_agree_rate']:>7.3f} {overall['parse_fail_rate']:>7.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse and score the two-option MMLU-Pro scan completions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--completions", type=Path, default=EXP_DIR / "completions.jsonl")
    parser.add_argument("--scores-output", type=Path, default=EXP_DIR / "scores.jsonl")
    parser.add_argument("--summary-output", type=Path, default=EXP_DIR / "category_summary.csv")
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=3,
        help="Warn if a (question, ordering) does not have this many samples (default: 3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    records = read_jsonl(args.completions)
    print(f"=== Loaded {len(records)} completions from {args.completions} ===")

    by_row: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for rec in records:
        by_row[(rec["question_id"], rec["order"])].append(rec)
    n_questions = len({qid for qid, _ in by_row})
    print(f"  {n_questions} unique questions, {len(by_row)} (question, ordering) rows")

    incomplete = [key for key, rs in by_row.items() if len(rs) != args.expected_samples]
    if incomplete:
        print(
            f"  WARNING: {len(incomplete)} row(s) do not have "
            f"{args.expected_samples} samples (e.g. {incomplete[:5]})"
        )
    orders_per_question = Counter(qid for qid, _ in by_row)
    unpaired = [qid for qid, n in orders_per_question.items() if n != 2]
    if unpaired:
        print(f"  WARNING: {len(unpaired)} question(s) missing an ordering (e.g. {unpaired[:5]})")

    scored = [score_row(rs) for rs in by_row.values()]
    scored.sort(key=lambda r: (r["category"], r["question_id"], r["order"]))

    print("\n=== Writing outputs ===")
    with args.scores_output.open("w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {len(scored)} rows to {args.scores_output}")

    summary = summarize(scored)
    write_summary_csv(args.summary_output, summary)

    overall = metrics(scored)
    stage_counts = Counter(s for r in scored for s in r["stages"])

    print("\n=== Extraction stages ===")
    for stage, count in stage_counts.most_common():
        print(f"  {stage:<14} {count:>6}  ({count/overall['n_samples']:.1%})")
    print(
        f"  Accuracy with fallback {overall['sample_accuracy']:.3f} vs "
        f"strict {overall['sample_accuracy_strict']:.3f} "
        f"(gap {overall['sample_accuracy'] - overall['sample_accuracy_strict']:+.3f})"
    )

    print_table("Per-category summary (sorted by majority accuracy)", summary, overall)
    print("  A_first/B_first = majority accuracy when gold is option A / option B.")
    print("  robust = majority-correct under BOTH orderings.  Chance = 0.500.")

    law_rows = [r for r in scored if r["src"] == PROFESSIONAL_LAW_SRC]
    if law_rows:
        law = metrics(law_rows)
        print(f"\n=== {PROFESSIONAL_LAW_SRC} subset (CST paper's MMLU-Pro subset) ===")
        print(
            f"  {law['n_questions']} questions | maj_acc {law['majority_accuracy']:.3f} "
            f"| strict {law['sample_accuracy_strict']:.3f} "
            f"| robust {law['order_robust_rate']:.3f} | p(A) {law['p_answer_a']:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
