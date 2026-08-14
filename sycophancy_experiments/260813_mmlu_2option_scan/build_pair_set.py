"""Build the two-option (binary) MMLU-Pro pair set.

Reads the question set from the 260811 10-option scan verbatim -- same 1400
questions, same question_ids -- so the two scans are comparable question by
question. For each question, one distractor is drawn uniformly from its wrong
options with a fixed per-question seed, and the pair is emitted twice: once with
the gold answer as option A and once as option B.

Following "Counterfactual Simulation Training for Chain-of-Thought Faithfulness"
(arXiv 2602.20710), which filters every benchmark "to two-way multiple choice,
where one option is correct and one is a distractor". The paper does not specify
how the distractor is chosen; uniform random is the neutral default. Emitting
both orderings makes the model's A/B positional prior measurable rather than
baked into the accuracy numbers.

Usage:
    python build_pair_set.py
    python build_pair_set.py --distractor-seed 4242
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SCAN_SET = EXP_DIR.parent / "260811_mmlu_scan" / "scan_set.jsonl"

DISTRACTOR_SEED = 4242
ORDERS = ["gold_first", "gold_second"]

FIELDS = [
    "question_id",
    "order",
    "category",
    "src",
    "question",
    "options",
    "answer",
    "answer_index",
    "gold_option",
    "distractor_option",
    "distractor_orig_index",
    "n_orig_options",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def make_pair(row: dict, seed: int) -> list[dict]:
    """Draw one distractor and emit the question in both option orderings."""
    options = row["options"]
    gold_index = row["answer_index"]
    gold = options[gold_index]

    wrong_indices = [i for i in range(len(options)) if i != gold_index]
    assert wrong_indices, f"q{row['question_id']}: no wrong options to draw from"
    # Per-question seed so adding or reordering questions cannot change the
    # distractor drawn for any other question.
    distractor_index = random.Random(seed + row["question_id"]).choice(wrong_indices)
    distractor = options[distractor_index]

    pair_rows = []
    for order in ORDERS:
        pair_options = [gold, distractor] if order == "gold_first" else [distractor, gold]
        answer_index = 0 if order == "gold_first" else 1
        pair_rows.append(
            {
                "question_id": row["question_id"],
                "order": order,
                "category": row["category"],
                "src": row["src"],
                "question": row["question"],
                "options": pair_options,
                "answer": "AB"[answer_index],
                "answer_index": answer_index,
                "gold_option": gold,
                "distractor_option": distractor,
                "distractor_orig_index": distractor_index,
                "n_orig_options": len(options),
            }
        )
    return pair_rows


def check_pairs(pair_rows: list[dict], n_questions: int) -> None:
    assert len(pair_rows) == 2 * n_questions, (
        f"Expected {2 * n_questions} rows, got {len(pair_rows)}"
    )

    by_question: dict[int, list[dict]] = {}
    for row in pair_rows:
        by_question.setdefault(row["question_id"], []).append(row)
    assert len(by_question) == n_questions, "Lost questions while pairing"

    for qid, rows in by_question.items():
        assert len(rows) == 2, f"q{qid}: {len(rows)} rows, expected 2"
        assert {r["order"] for r in rows} == set(ORDERS), f"q{qid}: orderings not distinct"
        golds = {r["gold_option"] for r in rows}
        distractors = {r["distractor_option"] for r in rows}
        assert len(golds) == 1 and len(distractors) == 1, f"q{qid}: pair content differs by order"
        for row in rows:
            assert len(row["options"]) == 2, f"q{qid}: {len(row['options'])} options"
            assert row["gold_option"] != row["distractor_option"], f"q{qid}: distractor is gold"
            assert row["options"][row["answer_index"]] == row["gold_option"], (
                f"q{qid} ({row['order']}): answer_index does not point at the gold option"
            )
            assert row["answer"] == "AB"[row["answer_index"]], f"q{qid}: answer letter mismatch"
        # The same question under both orderings must place gold on opposite sides.
        assert {r["answer"] for r in rows} == {"A", "B"}, f"q{qid}: gold on the same side twice"

    print(f"  CHECK: {n_questions} questions x 2 orderings, gold/distractor consistent")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the two-option MMLU-Pro pair set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scan-set",
        type=Path,
        default=SCAN_SET,
        help="10-option scan set to reuse the questions from",
    )
    parser.add_argument("--output", type=Path, default=EXP_DIR / "pair_set.jsonl")
    parser.add_argument("--distractor-seed", type=int, default=DISTRACTOR_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"=== Loading questions from {args.scan_set} ===")
    rows = read_jsonl(args.scan_set)
    qids = [r["question_id"] for r in rows]
    assert len(set(qids)) == len(qids), "Duplicate question_id in the scan set"
    print(f"  {len(rows)} questions, {len({r['category'] for r in rows})} categories")

    n_options = Counter(len(r["options"]) for r in rows)
    print(f"  Original option counts: {dict(sorted(n_options.items()))}")

    print(f"\n=== Pairing (distractor seed={args.distractor_seed}) ===")
    pair_rows = [pr for row in rows for pr in make_pair(row, args.distractor_seed)]
    check_pairs(pair_rows, len(rows))

    # A uniform draw over wrong options should not favour any original position.
    drawn = Counter(r["distractor_orig_index"] for r in pair_rows if r["order"] == "gold_first")
    print(f"  Distractor original-index spread: {dict(sorted(drawn.items()))}")

    by_category = Counter(r["category"] for r in pair_rows)
    for category in sorted(by_category):
        print(f"  {category:<18} rows={by_category[category]:>5}")

    print("\n=== Writing output ===")
    pair_rows.sort(key=lambda r: (r["category"], r["question_id"], r["order"]))
    with args.output.open("w") as f:
        for row in pair_rows:
            f.write(json.dumps({k: row[k] for k in FIELDS}) + "\n")
    line_count = sum(1 for _ in args.output.open())
    assert line_count == len(pair_rows), f"{args.output} has {line_count} lines"
    print(f"  Wrote {line_count} rows to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
