"""Parse and score MMLU-Pro scan completions.

Answer extraction follows the TIGER-AI-Lab/MMLU-Pro three-stage regex chain:
  1. "answer is (X)"
  2. "answer: X"
  3. the last standalone A-J token in the text

Unlike the reference implementation, unparseable completions are NOT scored as
a random guess -- they get pred=None and count as incorrect, and the parse
failure rate is reported per category instead.

Emits a per-question JSONL and a per-category summary CSV.

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

PATTERN_ANSWER_IS = re.compile(r"answer is \(?([A-J])\)?")
PATTERN_ANSWER_COLON = re.compile(r".*[aA]nswer:\s*([A-J])")
PATTERN_LAST_LETTER = re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", re.DOTALL)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_answer(text: str) -> tuple[str | None, str]:
    """Return (predicted letter, extraction stage) using the reference regex chain."""
    match = PATTERN_ANSWER_IS.search(text)
    if match:
        return match.group(1), "answer_is"
    match = PATTERN_ANSWER_COLON.search(text)
    if match:
        return match.group(1), "answer_colon"
    match = PATTERN_LAST_LETTER.search(text)
    if match:
        return match.group(0), "last_letter"
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


def score_question(records: list[dict]) -> dict:
    """Aggregate the samples for one question into a scored record."""
    records = sorted(records, key=lambda r: r["sample_idx"])
    gold = records[0]["answer"]

    preds, stages, correct = [], [], []
    for rec in records:
        pred, stage = extract_answer(rec["completion"])
        preds.append(pred)
        stages.append(stage)
        correct.append(pred == gold)

    n = len(records)
    n_parsed = sum(p is not None for p in preds)
    majority_pred = majority_vote(preds)
    # Consistency: every sample parsed and landed on the same letter.
    all_agree = n_parsed == n and len(set(preds)) == 1

    return {
        "question_id": records[0]["question_id"],
        "category": records[0]["category"],
        "answer": gold,
        "n_samples": n,
        "preds": preds,
        "stages": stages,
        "correct": correct,
        "n_parsed": n_parsed,
        "sample_accuracy": sum(correct) / n,
        "majority_pred": majority_pred,
        "majority_correct": majority_pred == gold,
        "all_agree": all_agree,
    }


def summarize(scored: list[dict]) -> list[dict]:
    """Per-category metrics, sorted by category name."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_category[row["category"]].append(row)

    summary = []
    for category in sorted(by_category):
        rows = by_category[category]
        n_questions = len(rows)
        n_samples = sum(r["n_samples"] for r in rows)
        n_parsed = sum(r["n_parsed"] for r in rows)
        n_stage3 = sum(s == "last_letter" for r in rows for s in r["stages"])
        summary.append(
            {
                "category": category,
                "n_questions": n_questions,
                "n_samples": n_samples,
                "sample_accuracy": sum(sum(r["correct"]) for r in rows) / n_samples,
                "majority_accuracy": sum(r["majority_correct"] for r in rows) / n_questions,
                "all_agree_rate": sum(r["all_agree"] for r in rows) / n_questions,
                "parse_fail_rate": (n_samples - n_parsed) / n_samples,
                "last_letter_rate": n_stage3 / n_samples,
            }
        )
    return summary


def write_summary_csv(path: Path, summary: list[dict]) -> None:
    fieldnames = [
        "category",
        "n_questions",
        "n_samples",
        "sample_accuracy",
        "majority_accuracy",
        "all_agree_rate",
        "parse_fail_rate",
        "last_letter_rate",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"  Wrote {len(summary)} rows to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse and score MMLU-Pro scan completions",
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
        help="Warn if a question does not have this many samples (default: 3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    records = read_jsonl(args.completions)
    print(f"=== Loaded {len(records)} completions from {args.completions} ===")

    by_question: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_question[rec["question_id"]].append(rec)
    print(f"  {len(by_question)} unique questions")

    incomplete = [q for q, rs in by_question.items() if len(rs) != args.expected_samples]
    if incomplete:
        print(
            f"  WARNING: {len(incomplete)} question(s) do not have "
            f"{args.expected_samples} samples (e.g. {incomplete[:5]})"
        )

    scored = [score_question(rs) for rs in by_question.values()]
    scored.sort(key=lambda r: (r["category"], r["question_id"]))

    print("\n=== Writing outputs ===")
    with args.scores_output.open("w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {len(scored)} rows to {args.scores_output}")

    summary = summarize(scored)
    write_summary_csv(args.summary_output, summary)

    total_samples = sum(s["n_samples"] for s in summary)
    total_questions = sum(s["n_questions"] for s in summary)
    overall = {
        "sample_accuracy": sum(sum(r["correct"]) for r in scored) / total_samples,
        "majority_accuracy": sum(r["majority_correct"] for r in scored) / total_questions,
        "all_agree_rate": sum(r["all_agree"] for r in scored) / total_questions,
        "parse_fail_rate": sum(r["n_samples"] - r["n_parsed"] for r in scored) / total_samples,
    }
    stage_counts = Counter(s for r in scored for s in r["stages"])

    print("\n=== Extraction stages ===")
    for stage, count in stage_counts.most_common():
        print(f"  {stage:<14} {count:>6}  ({count/total_samples:.1%})")

    print("\n=== Per-category summary (sorted by majority accuracy) ===")
    header = f"{'category':<18} {'maj_acc':>8} {'smp_acc':>8} {'agree':>8} {'parse_fail':>11}"
    print(header)
    print("-" * len(header))
    for row in sorted(summary, key=lambda r: r["majority_accuracy"], reverse=True):
        print(
            f"{row['category']:<18} {row['majority_accuracy']:>8.3f} "
            f"{row['sample_accuracy']:>8.3f} {row['all_agree_rate']:>8.3f} "
            f"{row['parse_fail_rate']:>11.3f}"
        )
    print("-" * len(header))
    print(
        f"{'OVERALL':<18} {overall['majority_accuracy']:>8.3f} "
        f"{overall['sample_accuracy']:>8.3f} {overall['all_agree_rate']:>8.3f} "
        f"{overall['parse_fail_rate']:>11.3f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
