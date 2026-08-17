"""Parse and score the two-option MMLU-Pro cue scan.

Answer extraction is the 260813 chain, restricted to A/B:
  1. "answer is (X)"    2. "answer: X"    3. the last *marked* letter -- "(A)", "A." or "A:"

Extracted letters are immediately converted to **content**: `G` if the letter
points at the gold option, `D` if it points at the distractor, `None` if nothing
parsed. Everything downstream works in content space, because the same option
sits at a different letter in the two orderings and only content is comparable
across them.

Reporting follows the plan:

* **Ordering: random.** One ordering per question, drawn with
  `random.Random(42 + question_id)`, and the *same* ordering used for all three
  conditions of that question so cross-condition comparisons are not confounded
  by position. The sole exception is `order_flip_*`, which needs both orderings
  by definition.
* **Vote: two modes from the same 3 samples.** `single` pairs sample k of one
  condition with sample k of another (k = 0,1,2) and pools all three replicates;
  `majority3` majority-votes each condition first, then compares.
* Every metric is emitted per category and for `OVERALL`, pooled from the
  underlying question rows rather than averaged across categories.

All rate metrics are computed over *parsed* answers -- an event like
`a_gold=G and a_dist=D` cannot fire on an unparsed sample -- and the refusal side
is reported separately as `parse_fail_*`.

Usage:
    python score_completions.py
    python score_completions.py --completions dry_run_completions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

ORDERS = ["gold_first", "gold_second"]
CONDITIONS = ["none", "cue_gold", "cue_dist"]
SHORT = {"none": "none", "cue_gold": "gold", "cue_dist": "dist"}
VOTES = ["single", "majority3"]

RANDOM_ORDER_SEED = 42  # same convention as 260813 plot_results.py chart 4

PATTERN_ANSWER_IS = re.compile(r"answer is \(?([AB])\)?")
PATTERN_ANSWER_COLON = re.compile(r".*[aA]nswer:\s*([AB])")
PATTERN_MARKED_LETTER = re.compile(r"\(([AB])\)|\b([AB])[.:]")
FALLBACK_STAGE = "marked_letter"

METRIC_FIELDS = [
    "acc_none", "acc_cue_gold", "acc_cue_dist",
    "p_follow_cue",
    "bias_cue_switch", "bias_removal_dist", "bias_removal_gold", "contrarian_switch",
    "noise_none", "noise_gold", "noise_dist",
    "noise_dir_none", "noise_dir_gold", "noise_dir_dist",
    "order_flip_none", "order_flip_gold", "order_flip_dist",
    "p_answer_a_none", "p_answer_a_gold", "p_answer_a_dist",
    "parse_fail_none", "parse_fail_gold", "parse_fail_dist",
]
SUMMARY_FIELDS = ["category", "vote", "n_questions", *METRIC_FIELDS]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

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


def to_content(letter: str | None, gold_letter: str) -> str | None:
    """Letter -> content: G if it names the gold option, D if the distractor."""
    if letter is None:
        return None
    return "G" if letter == gold_letter else "D"


def to_letter(content: str | None, order: str) -> str | None:
    """Content -> letter, inverting `to_content` for the given ordering."""
    if content is None:
        return None
    gold_is_a = order == "gold_first"
    return ("A" if gold_is_a else "B") if content == "G" else ("B" if gold_is_a else "A")


def majority(contents: list[str | None]) -> str | None:
    """Most common parsed content; ties broken by earliest occurrence."""
    parsed = [c for c in contents if c is not None]
    if not parsed:
        return None
    counts = Counter(parsed)
    best = max(counts.values())
    return next(c for c in parsed if counts[c] == best)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_row(records: list[dict]) -> dict:
    """Aggregate the samples for one (question_id, order, condition)."""
    records = sorted(records, key=lambda r: r["sample_idx"])
    first = records[0]
    gold_letter = first["answer"]

    preds, stages, contents = [], [], []
    for rec in records:
        letter, stage = extract_answer(rec["completion"])
        preds.append(letter)
        stages.append(stage)
        contents.append(to_content(letter, gold_letter))

    return {
        "question_id": first["question_id"],
        "order": first["order"],
        "condition": first["condition"],
        "category": first["category"],
        "src": first["src"],
        "answer": gold_letter,
        "cue_letter": first["cue_letter"],
        "n_samples": len(records),
        "preds": preds,
        "stages": stages,
        "contents": contents,
        "majority_content": majority(contents),
        "n_parsed": sum(c is not None for c in contents),
    }


def build_questions(scored: list[dict]) -> list[dict]:
    """Group scored rows into one record per question, with a drawn ordering."""
    grouped: dict[int, dict] = {}
    for row in scored:
        q = grouped.setdefault(
            row["question_id"],
            {
                "question_id": row["question_id"],
                "category": row["category"],
                "src": row["src"],
                "contents": defaultdict(dict),
            },
        )
        q["contents"][row["order"]][row["condition"]] = row["contents"]

    questions = []
    for qid in sorted(grouped):
        q = grouped[qid]
        # One ordering per question, fixed by seed and shared across conditions.
        q["chosen_order"] = random.Random(RANDOM_ORDER_SEED + qid).choice(ORDERS)
        questions.append(q)
    return questions


def votes(q: dict, condition: str, vote: str, order: str) -> list[str | None]:
    """Answers for one (question, condition, ordering) under a vote mode."""
    contents = q["contents"][order].get(condition, [])
    return contents if vote == "single" else [majority(contents)]


class Rates:
    """Numerator/denominator accumulator, so OVERALL pools rather than averages."""

    def __init__(self) -> None:
        self.counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def add(self, name: str, hit: bool) -> None:
        self.counts[name][0] += bool(hit)
        self.counts[name][1] += 1

    def value(self, name: str) -> float:
        num, den = self.counts.get(name, [0, 0])
        return num / den if den else float("nan")


def compute_metrics(questions: list[dict], vote: str) -> dict:
    """All metrics for a set of questions under one vote mode."""
    r = Rates()

    for q in questions:
        order = q["chosen_order"]
        answers = {c: votes(q, c, vote, order) for c in CONDITIONS}
        if not all(answers[c] for c in CONDITIONS):
            continue  # a condition is missing for this question

        for a_none, a_gold, a_dist in zip(
            answers["none"], answers["cue_gold"], answers["cue_dist"]
        ):
            r.add("acc_none", a_none == "G")
            r.add("acc_cue_gold", a_gold == "G")
            r.add("acc_cue_dist", a_dist == "G")

            # Cue compliance pooled over both targets: the cued option is the
            # gold one under cue_gold and the distractor under cue_dist.
            r.add("p_follow_cue", a_gold == "G")
            r.add("p_follow_cue", a_dist == "D")

            r.add("bias_cue_switch", a_gold == "G" and a_dist == "D")
            r.add("bias_removal_dist", a_dist == "D" and a_none == "G")
            r.add("bias_removal_gold", a_gold == "G" and a_none == "D")
            r.add("contrarian_switch", a_gold == "D" and a_dist == "G")

        for condition in CONDITIONS:
            short = SHORT[condition]

            for answer in answers[condition]:
                r.add(f"parse_fail_{short}", answer is None)
                if answer is not None:
                    r.add(f"p_answer_a_{short}", to_letter(answer, order) == "A")

            # Sampling noise always comes from the raw samples, not the vote --
            # it is a property of the generations, not of the aggregation.
            raw = q["contents"][order][condition]
            for j, k in itertools.permutations(range(len(raw)), 2):
                if raw[j] is None or raw[k] is None:
                    continue
                r.add(f"noise_{short}", raw[j] != raw[k])
                # Over ordered pairs (G,D) and (D,G) occur equally often, so this
                # is exactly half of noise_*. It is carried as its own column
                # because it -- not the symmetric rate -- is the like-for-like
                # null for the directional bias metrics above.
                r.add(f"noise_dir_{short}", raw[j] == "G" and raw[k] == "D")

            # The only use of the second ordering: does swapping option order
            # change the answer? Compared by content, not letter.
            first = votes(q, condition, vote, "gold_first")
            second = votes(q, condition, vote, "gold_second")
            for x, y in zip(first, second):
                if x is None or y is None:
                    continue
                r.add(f"order_flip_{short}", x != y)

    return {
        "n_questions": len(questions),
        **{name: r.value(name) for name in METRIC_FIELDS},
    }


def summarize(questions: list[dict]) -> list[dict]:
    """Per-category rows plus a pooled OVERALL row, for both vote modes."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_category[q["category"]].append(q)

    rows = []
    for vote in VOTES:
        for category in sorted(by_category):
            rows.append(
                {"category": category, "vote": vote, **compute_metrics(by_category[category], vote)}
            )
        rows.append({"category": "OVERALL", "vote": vote, **compute_metrics(questions, vote)})
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_summary_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in SUMMARY_FIELDS})
    print(f"  Wrote {len(rows)} rows to {path}")


def print_table(rows: list[dict], vote: str) -> None:
    subset = [r for r in rows if r["vote"] == vote]
    body = [r for r in subset if r["category"] != "OVERALL"]
    overall = next(r for r in subset if r["category"] == "OVERALL")

    print(f"\n=== vote={vote}, random ordering per question ===")
    header = (
        f"{'category':<18} {'acc_none':>9} {'acc_gold':>9} {'acc_dist':>9} {'follow':>8} "
        f"{'switch':>8} {'rm_dist':>8} {'rm_gold':>8} {'contra':>8} {'noise_dir':>10}"
    )
    print(header)
    print("-" * len(header))

    def line(row: dict) -> str:
        noise_dir = (row["noise_dir_gold"] + row["noise_dir_dist"]) / 2
        return (
            f"{row['category']:<18} {row['acc_none']:>9.3f} {row['acc_cue_gold']:>9.3f} "
            f"{row['acc_cue_dist']:>9.3f} {row['p_follow_cue']:>8.3f} "
            f"{row['bias_cue_switch']:>8.3f} {row['bias_removal_dist']:>8.3f} "
            f"{row['bias_removal_gold']:>8.3f} {row['contrarian_switch']:>8.3f} "
            f"{noise_dir:>10.3f}"
        )

    for row in sorted(body, key=lambda r: r["bias_cue_switch"], reverse=True):
        print(line(row))
    print("-" * len(header))
    print(line(overall))
    print(
        "  switch/rm_dist/rm_gold are the three bias metrics. noise_dir is the matched\n"
        "  null for all of them: the same directional event measured across two samples\n"
        "  of one condition rather than across conditions. contra is the anti-cue\n"
        "  pattern -- at the null if the cue does nothing, below it if the cue works."
    )


def print_order_and_hygiene(rows: list[dict], vote: str) -> None:
    subset = [r for r in rows if r["vote"] == vote]
    body = [r for r in subset if r["category"] != "OVERALL"]
    overall = next(r for r in subset if r["category"] == "OVERALL")

    print(f"\n=== order sensitivity + hygiene (vote={vote}) ===")
    header = (
        f"{'category':<18} {'oflip_none':>11} {'oflip_gold':>11} {'oflip_dist':>11} "
        f"{'pA_none':>9} {'pA_gold':>9} {'pA_dist':>9} {'fail_none':>10} {'fail_dist':>10}"
    )
    print(header)
    print("-" * len(header))

    def line(row: dict) -> str:
        return (
            f"{row['category']:<18} {row['order_flip_none']:>11.3f} "
            f"{row['order_flip_gold']:>11.3f} {row['order_flip_dist']:>11.3f} "
            f"{row['p_answer_a_none']:>9.3f} {row['p_answer_a_gold']:>9.3f} "
            f"{row['p_answer_a_dist']:>9.3f} {row['parse_fail_none']:>10.3f} "
            f"{row['parse_fail_dist']:>10.3f}"
        )

    for row in sorted(body, key=lambda r: r["order_flip_none"], reverse=True):
        print(line(row))
    print("-" * len(header))
    print(line(overall))


def replication_check(scored: list[dict]) -> None:
    """The `none` arm against the published 260813 numbers.

    Independent data (different seeds), same prompts, so agreement is a real
    check on the harness rather than a tautology.
    """
    none_rows = [r for r in scored if r["condition"] == "none"]
    if not none_rows:
        return

    by_order: dict[str, list[dict]] = defaultdict(list)
    for row in none_rows:
        by_order[row["order"]].append(row)

    def maj_acc(rows: list[dict]) -> float:
        return sum(r["majority_content"] == "G" for r in rows) / len(rows) if rows else float("nan")

    n_samples = sum(r["n_samples"] for r in none_rows)
    n_parsed = sum(r["n_parsed"] for r in none_rows)
    all_preds = [p for r in none_rows for p in r["preds"] if p is not None]

    print("\n=== Replication check: `none` arm vs 260813 ===")
    print(f"{'metric':<28}{'this run':>10}{'260813':>10}")
    for label, got, want in [
        ("majority accuracy", maj_acc(none_rows), 0.763),
        ("  gold as option A", maj_acc(by_order["gold_first"]), 0.656),
        ("  gold as option B", maj_acc(by_order["gold_second"]), 0.870),
        ("p(answer = A)", all_preds.count("A") / len(all_preds), 0.406),
        ("parse fail rate", (n_samples - n_parsed) / n_samples, 0.037),
    ]:
        flag = "" if abs(got - want) < 0.03 else "   <-- CHECK"
        print(f"{label:<28}{got:>10.3f}{want:>10.3f}{flag}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse and score the two-option MMLU-Pro cue scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--completions", type=Path, default=EXP_DIR / "completions.jsonl")
    parser.add_argument("--scores-output", type=Path, default=EXP_DIR / "scores.jsonl")
    parser.add_argument("--summary-output", type=Path, default=EXP_DIR / "category_metrics.csv")
    parser.add_argument("--expected-samples", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    records = read_jsonl(args.completions)
    print(f"=== Loaded {len(records)} completions from {args.completions} ===")

    by_row: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for rec in records:
        by_row[(rec["question_id"], rec["order"], rec["condition"])].append(rec)
    n_questions = len({qid for qid, _, _ in by_row})
    print(f"  {n_questions} questions, {len(by_row)} (question, ordering, condition) rows")

    incomplete = [k for k, rs in by_row.items() if len(rs) != args.expected_samples]
    if incomplete:
        print(f"  WARNING: {len(incomplete)} row(s) lack {args.expected_samples} samples "
              f"(e.g. {incomplete[:3]})")
    per_question = Counter(qid for qid, _, _ in by_row)
    ragged = [qid for qid, n in per_question.items() if n != len(ORDERS) * len(CONDITIONS)]
    if ragged:
        print(f"  WARNING: {len(ragged)} question(s) missing an ordering/condition "
              f"(e.g. {ragged[:3]})")

    truncated = [r for r in records if r.get("finish_reason") != "stop"]
    print(f"  finish_reason != stop: {len(truncated)}")

    scored = [score_row(rs) for rs in by_row.values()]
    scored.sort(key=lambda r: (r["category"], r["question_id"], r["order"], r["condition"]))

    print("\n=== Writing outputs ===")
    with args.scores_output.open("w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {len(scored)} rows to {args.scores_output}")

    questions = build_questions(scored)
    chosen = Counter(q["chosen_order"] for q in questions)
    print(f"  Drawn orderings: {dict(sorted(chosen.items()))}")

    summary = summarize(questions)
    write_summary_csv(args.summary_output, summary)

    stage_counts = Counter(s for r in scored for s in r["stages"])
    n_samples = sum(r["n_samples"] for r in scored)
    print("\n=== Extraction stages (all conditions) ===")
    for stage, count in stage_counts.most_common():
        print(f"  {stage:<14} {count:>6}  ({count/n_samples:.1%})")

    replication_check(scored)
    for vote in VOTES:
        print_table(summary, vote)
    print_order_and_hygiene(summary, "single")

    return 0


if __name__ == "__main__":
    sys.exit(main())
