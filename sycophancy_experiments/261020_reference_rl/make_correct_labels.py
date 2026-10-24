"""The correct-label question files: each setup's question file with the training label set to the correct answer.

    python make_correct_labels.py     # -> dataset/cot/labels.jsonl, dataset/nocot/labels.jsonl

For each setup (261013's CoT file, 261014's no-CoT file) every row is copied with train_label = answer and
label_flipped = False. Every other field is kept, including the baked in-loop eval cue (assigned_cue_letter,
assigned_condition), so the trainer's in-loop eval serves the same prompts as in the noisy-label runs.
Asserts: the same rows in the same order, only those two fields differ, and the changed rows are exactly the
flipped questions (CoT 243 train + 98 eval, no-CoT 261 + 109).
"""

from __future__ import annotations

import json
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SETUP_FILES = {
    "cot": Path("/workspace/sycophancy_experiments/261013_noisy_label_rl/dataset/labels.jsonl"),
    "nocot": Path("/workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/dataset/labels.jsonl"),
}
N_FLIPPED = {"cot": {"train": 243, "eval": 98}, "nocot": {"train": 261, "eval": 109}}
CHANGED_FIELDS = {"train_label", "label_flipped"}


def main() -> None:
    for setup, source in SETUP_FILES.items():
        out = EXP_DIR / "dataset" / setup / "labels.jsonl"
        assert not out.exists(), f"{out} exists"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        correct_rows = [{**row, "train_label": row["answer"], "label_flipped": False} for row in rows]
        changed = {"train": 0, "eval": 0}
        for row, correct_row in zip(rows, correct_rows, strict=True):
            differing = {key for key in row if row[key] != correct_row[key]}
            assert differing <= CHANGED_FIELDS, (row["question_id"], differing)
            assert bool(differing) == row["label_flipped"], row["question_id"]
            changed[row["split"]] += bool(differing)
        assert changed == N_FLIPPED[setup], (setup, changed)
        out.parent.mkdir(parents=True)
        out.write_text("".join(json.dumps(row) + "\n" for row in correct_rows))
        print(f"{setup}: {len(correct_rows)} rows, {changed} changed -> {out}")


if __name__ == "__main__":
    main()
