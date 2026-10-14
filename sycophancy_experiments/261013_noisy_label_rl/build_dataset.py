"""The RL question file: the confident slice with a noisy training label per question.

    python build_dataset.py     # -> dataset/labels.jsonl, dataset/summary.json

One row per slice question (1217 train / 489 eval). Each row is 260819's question
row with these fields added or replaced:

  train_label             the gold letter, or the other letter if the question is flipped
  label_flipped           True on exactly N_FLIPPED[split] questions per split (seeded)
  base_confident_answer   the base model's 6-of-8 no-cue answer (letter), from 261010
  base_confident_correct  base_confident_answer == answer
  p_cue_label             CUE_LABEL_PROB (replaces 260819's p_cue_gold)
  assigned_cue_letter     the in-loop eval cue: train_label iff cue_u < CUE_LABEL_PROB, else the other letter
  assigned_condition      cue_gold / cue_dist for that letter (picks user_content_cue_gold / _cue_dist)
"""

from __future__ import annotations

import json
import random
from collections import Counter

import noisy_spec as S
from domain_imports import to_letter


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def other_letter(letter: str) -> str:
    assert letter in ("A", "B"), letter
    return "B" if letter == "A" else "A"


def slice_answers() -> dict[int, tuple[str, str]]:
    """{question_id: (split, 6-of-8 no-cue answer as content G/D)} from 261010's slice labels."""
    answers: dict[int, tuple[str, str]] = {}
    for row in read_jsonl(S.SLICE_LABELS):
        entry = (row["split"], row["a_none"])
        assert answers.setdefault(row["question_id"], entry) == entry, row["question_id"]
    return answers


def main() -> None:
    assert not S.DATASET_PATH.exists(), f"{S.DATASET_PATH} exists"
    questions = read_jsonl(S.DOMAIN_QUESTIONS)
    assert len(questions) == S.N_DOMAIN_QUESTIONS
    answers = slice_answers()
    slice_questions = [q for q in questions if q["question_id"] in answers]
    assert len(slice_questions) == len(answers), "a slice question is missing from 260819"

    flipped: set[int] = set()
    for split, n_questions in S.N_SLICE_QUESTIONS.items():
        ids = sorted(q["question_id"] for q in slice_questions if q["split"] == split)
        assert len(ids) == n_questions, (split, len(ids))
        assert all(answers[i][0] == split for i in ids), f"split disagrees with 261010 in {split}"
        rng = random.Random(S.FLIP_SEED + S.SPLIT_SEED_OFFSET[split])
        flipped |= set(rng.sample(ids, S.N_FLIPPED[split]))

    rows = []
    for q in slice_questions:
        gold = q["answer"]
        assert to_letter("G", q["order"]) == gold, q["question_id"]
        flip = q["question_id"] in flipped
        train_label = other_letter(gold) if flip else gold
        confident_letter = to_letter(answers[q["question_id"]][1], q["order"])
        cue_letter = train_label if q["cue_u"] < S.CUE_LABEL_PROB else other_letter(train_label)
        row = {k: v for k, v in q.items() if k not in ("p_cue_gold", "assigned_condition", "assigned_cue_letter")}
        row.update(
            train_label=train_label,
            label_flipped=flip,
            base_confident_answer=confident_letter,
            base_confident_correct=confident_letter == gold,
            p_cue_label=S.CUE_LABEL_PROB,
            assigned_cue_letter=cue_letter,
            assigned_condition="cue_gold" if cue_letter == gold else "cue_dist",
        )
        rows.append(row)

    summary: dict = {"flip_rate": S.FLIP_RATE, "flip_seed": S.FLIP_SEED, "cue_label_prob": S.CUE_LABEL_PROB}
    for split in S.N_SLICE_QUESTIONS:
        split_rows = [r for r in rows if r["split"] == split]
        summary[split] = {
            "questions": len(split_rows),
            "flipped": sum(r["label_flipped"] for r in split_rows),
            "base_confident_correct": sum(r["base_confident_correct"] for r in split_rows),
            "flipped_by_category": dict(sorted(Counter(r["category"] for r in split_rows if r["label_flipped"]).items())),
            "questions_by_category": dict(sorted(Counter(r["category"] for r in split_rows).items())),
            "flipped_x_base_confident_correct": {
                f"flipped={f},base_correct={c}": n
                for (f, c), n in sorted(Counter((r["label_flipped"], r["base_confident_correct"]) for r in split_rows).items())
            },
            "eval_cue_at_label": sum(r["assigned_cue_letter"] == r["train_label"] for r in split_rows),
            "eval_cue_at_gold": sum(r["assigned_condition"] == "cue_gold" for r in split_rows),
        }
        assert summary[split]["flipped"] == S.N_FLIPPED[split]

    S.DATASET_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    S.DATASET_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
