"""Sample the MMLU-Pro per-category difficulty scan set.

Shuffles each of the 14 MMLU-Pro categories with a fixed seed and takes the
first N questions. Questions are used verbatim -- no modifications.

Follows TIGER-AI-Lab/MMLU-Pro `evaluate_from_local.py` preprocessing: drop the
"N/A" padding options so each question exposes only its real choices.

Usage:
    python build_scan_set.py
    python build_scan_set.py --seed 42 --per-category 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

EXP_DIR = Path(__file__).resolve().parent

DATASET_NAME = "TIGER-Lab/MMLU-Pro"
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]


def preprocess(split) -> list[dict]:
    """Drop 'N/A' padding options, as in the reference implementation."""
    rows = []
    for example in split:
        example = dict(example)
        example["options"] = [opt for opt in example["options"] if opt != "N/A"]
        rows.append(example)
    return rows


def check_answer_alignment(rows: list[dict], label: str) -> None:
    """Verify answer_index still points at the answer letter after the N/A drop."""
    for row in rows:
        n_options = len(row["options"])
        assert row["answer_index"] < n_options, (
            f"{label} q{row['question_id']}: answer_index {row['answer_index']} "
            f">= {n_options} options after dropping N/A"
        )
        expected = CHOICES[row["answer_index"]]
        assert row["answer"] == expected, (
            f"{label} q{row['question_id']}: answer {row['answer']!r} does not match "
            f"answer_index {row['answer_index']} (expected {expected!r})"
        )
    print(f"  CHECK: {label} answer/answer_index alignment holds for {len(rows)} rows")


def sample_per_category(rows: list[dict], per_category: int, seed: int) -> list[dict]:
    """Shuffle each category with a fixed seed and take the first N questions."""
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    sampled: list[dict] = []
    for category in sorted(by_category):
        pool = by_category[category]
        # Sort by question_id first so the shuffle does not depend on hub row order.
        pool = sorted(pool, key=lambda r: r["question_id"])
        random.Random(seed).shuffle(pool)

        if len(pool) < per_category:
            print(
                f"  WARNING: category {category!r} has only {len(pool)} questions, "
                f"requested {per_category}"
            )
        take = pool[:per_category]
        sampled.extend(take)
        print(f"  {category:<18} pool={len(pool):>5}  sampled={len(take):>4}")

    return sampled


def write_jsonl(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({k: row[k] for k in fields}) + "\n")
    line_count = sum(1 for _ in path.open())
    assert line_count == len(rows), f"{path} has {line_count} lines, expected {len(rows)}"
    print(f"  Wrote {line_count} rows to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample the MMLU-Pro per-category difficulty scan set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument(
        "--per-category",
        type=int,
        default=100,
        help="Questions to sample per category (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXP_DIR / "scan_set.jsonl",
        help="Output path for the sampled scan set",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"=== Loading {DATASET_NAME} ===")
    ds = load_dataset(DATASET_NAME)
    print(f"  Raw splits: { {k: len(v) for k, v in ds.items()} }")

    test_rows = preprocess(ds["test"])
    check_answer_alignment(test_rows, "test")

    print(f"\n=== Sampling {args.per_category}/category (seed={args.seed}) ===")
    sampled = sample_per_category(test_rows, args.per_category, args.seed)

    categories = sorted({r["category"] for r in test_rows})
    counts = Counter(r["category"] for r in sampled)
    assert len(counts) == len(categories), (
        f"Sampled {len(counts)} categories, expected {len(categories)}"
    )
    qids = [r["question_id"] for r in sampled]
    assert len(set(qids)) == len(qids), "Duplicate question_id in sampled scan set"
    print(f"  CHECK: {len(categories)} categories, {len(sampled)} unique questions")

    print("\n=== Writing outputs ===")
    write_jsonl(
        args.output,
        sampled,
        ["question_id", "category", "question", "options", "answer", "answer_index", "src"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
