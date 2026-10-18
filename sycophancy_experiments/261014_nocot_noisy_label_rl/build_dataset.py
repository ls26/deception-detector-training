"""The RL question file: the no-cue-confident questions in the no-CoT format, with a noisy training label.

    python build_dataset.py     # -> dataset/labels.jsonl, dataset/summary.json

One row per question whose no-cue top letter in 261007's saved no-CoT generation has probability
>= CONFIDENT_MIN (1304 train / 547 eval). Each row is 260819's question row with these fields
added or replaced:

  user_content_none / _cue_gold / _cue_dist   261007's no-CoT user messages (prompts.nocot_user_content:
                                              the CoT sentence swapped, asserted against 260819's renders)
  train_label             the gold letter, or the other letter if the question is flipped
  label_flipped           True on exactly N_FLIPPED[split] questions per split (seeded)
  base_confident_answer   the base model's no-cue top letter (261007's saved generation)
  base_confident_correct  base_confident_answer == answer
  base_p_top              the renormalized probability of that letter
  p_cue_label             CUE_LABEL_PROB (replaces 260819's p_cue_gold)
  assigned_cue_letter     the in-loop eval cue: train_label iff cue_u < CUE_LABEL_PROB, else the other letter
  assigned_condition      cue_gold / cue_dist for that letter (picks user_content_cue_gold / _cue_dist)

Imports 261007's prompts module, which imports 261007's `spec` and `domain_imports` by bare name, so
261007's folder goes first on sys.path and this script does not import this folder's domain_imports.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter

import nocot_spec as S

sys.path.insert(0, str(S.NOCOT_DIR))
import prompts as nocot_prompts  # noqa: E402  (261007's)

for module_name in ("prompts", "spec", "domain_imports"):
    module_file = sys.modules[module_name].__file__
    assert module_file == str(S.NOCOT_DIR / f"{module_name}.py"), f"{module_name} resolved to {module_file}"
nocot_spec_261007 = sys.modules["spec"]
for name in ("COT_SENTENCE", "NOCOT_SENTENCE", "NOCOT_PREFILL", "NOCOT_CLOSING", "PREFILL_LAST_TOKEN_ID", "LETTER_TOKEN_IDS"):
    assert getattr(S, name) == getattr(nocot_spec_261007, name), f"{name} differs from 261007's spec"
to_letter = sys.modules["domain_imports"].to_letter  # 260819's, through 261007's import hub


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def other_letter(letter: str) -> str:
    assert letter in ("A", "B"), letter
    return "B" if letter == "A" else "A"


def nocue_readings() -> dict[int, dict]:
    """{question_id: 261007's saved no-cue letter reading} (one per prompt; the 3 sample rows carry the same)."""
    readings: dict[int, dict] = {}
    for row in read_jsonl(S.NOCOT_TURN1):
        if row["condition"] != "none":
            continue
        reading = {k: row[k] for k in ("split", "lp_a", "lp_b", "p_a", "mass_ab", "top_letter")}
        assert readings.setdefault(row["question_id"], reading) == reading, row["question_id"]
    assert len(readings) == S.N_DOMAIN_QUESTIONS
    return readings


def is_confident(p_a: float | None) -> bool:
    """261007's label.py is_confident: the top letter's renormalized probability is >= CONFIDENT_MIN."""
    return p_a is not None and max(p_a, 1 - p_a) >= S.CONFIDENT_MIN


def main() -> None:
    assert not S.DATASET_PATH.exists(), f"{S.DATASET_PATH} exists"
    questions = read_jsonl(S.DOMAIN_QUESTIONS)
    assert len(questions) == S.N_DOMAIN_QUESTIONS
    readings = nocue_readings()
    confident = [q for q in questions if is_confident(readings[q["question_id"]]["p_a"])]

    flipped: set[int] = set()
    for split, n_questions in S.N_QUESTIONS.items():
        ids = sorted(q["question_id"] for q in confident if q["split"] == split)
        assert len(ids) == n_questions, (split, len(ids))
        assert all(readings[i]["split"] == split for i in ids), f"split disagrees with 261007 in {split}"
        rng = random.Random(S.FLIP_SEED + S.SPLIT_SEED_OFFSET[split])
        flipped |= set(rng.sample(ids, S.N_FLIPPED[split]))

    rows = []
    for q in confident:
        gold = q["answer"]
        assert to_letter("G", q["order"]) == gold, q["question_id"]
        reading = readings[q["question_id"]]
        assert reading["top_letter"] in ("A", "B"), q["question_id"]
        flip = q["question_id"] in flipped
        train_label = other_letter(gold) if flip else gold
        cue_letter = train_label if q["cue_u"] < S.CUE_LABEL_PROB else other_letter(train_label)
        row = {k: v for k, v in q.items() if k not in ("p_cue_gold", "assigned_condition", "assigned_cue_letter")}
        row.update({f"user_content_{c}": nocot_prompts.nocot_user_content(q, c) for c in S.EVAL_CONDITIONS})
        row.update(
            train_label=train_label,
            label_flipped=flip,
            base_confident_answer=reading["top_letter"],
            base_confident_correct=reading["top_letter"] == gold,
            base_p_top=max(reading["p_a"], 1 - reading["p_a"]),
            p_cue_label=S.CUE_LABEL_PROB,
            assigned_cue_letter=cue_letter,
            assigned_condition="cue_gold" if cue_letter == gold else "cue_dist",
        )
        rows.append(row)

    summary: dict = {"confident_min": S.CONFIDENT_MIN, "flip_rate": S.FLIP_RATE, "flip_seed": S.FLIP_SEED, "cue_label_prob": S.CUE_LABEL_PROB}
    for split in S.N_QUESTIONS:
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

    S.DATASET_PATH.parent.mkdir(exist_ok=True)
    S.DATASET_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    S.DATASET_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
