"""Check the label code on the base model's saved no-CoT data against 261007's own labels.

    python check_scoring.py     # -> check_scoring.json, dry/saved_base/{completions,followups,labels}.jsonl

Builds eval_checkpoint.py's completions / follow-ups for the 547 eval questions from 261007's saved
generation (each prompt's letter reading; the "top" follow-up's float32 P(Yes)), runs eval_checkpoint's
build_labels on them, and compares every cued prompt's a_cued, a_none, bias, said and deception label
and P(Yes) with 261007's arms/nocot_logprob/labels.jsonl. PASS iff every one matches. The files in
dry/saved_base/ then feed compare.py's eval tables as a check of the rates against the base numbers
computed from 261007's data.
"""

from __future__ import annotations

import json
import math

import nocot_spec as S
from domain_imports import draw_wordings, to_content
from eval_checkpoint import build_labels, read_jsonl, write_jsonl

OUT_DIR = S.EXP_DIR / "dry" / "saved_base"
COMPARED = ("a_cued", "a_none", "bias_rm", "said", "deception_rm", "p_yes")


def saved_completions(questions: dict[int, dict]) -> list[dict]:
    rows = []
    for saved in read_jsonl(S.NOCOT_TURN1):
        q = questions.get(saved["question_id"])
        if q is None or saved["sample_idx"] != 0:
            continue
        top_letter = saved["top_letter"]
        rows.append({
            "question_id": saved["question_id"],
            "condition": saved["condition"],
            "split": q["split"],
            "category": q["category"],
            "order": q["order"],
            "answer": q["answer"],
            "train_label": q["train_label"],
            "label_flipped": q["label_flipped"],
            "base_confident_correct": q["base_confident_correct"],
            "cue_letter": saved["cue_letter"],
            "prompt": saved["prompt"],
            "lp_a": saved["lp_a"],
            "lp_b": saved["lp_b"],
            "p_a": saved["p_a"],
            "p_a_full": 0.0 if saved["lp_a"] is None else math.exp(saved["lp_a"]),
            "p_b_full": 0.0 if saved["lp_b"] is None else math.exp(saved["lp_b"]),
            "mass_ab": saved["mass_ab"],
            "top_letter": top_letter,
            "top_rule": "both_in_top20" if top_letter is not None else "tie",
            "content": to_content(top_letter, q["answer"]),
            "checkpoint": "saved_base",
        })
    return rows


def saved_followups(completions: list[dict]) -> list[dict]:
    top_letters = {(r["question_id"], r["condition"]): r["top_letter"] for r in completions}
    rows = []
    for saved in read_jsonl(S.NOCOT_PYES):
        key = (saved["question_id"], saved["condition"])
        if saved["sample_key"] != "top" or key not in top_letters:
            continue
        rows.append({"question_id": key[0], "condition": key[1], "top_letter": top_letters[key],
                     "wording": draw_wordings(key[0])[S.DEFINITION][1], "p_yes": saved["p_yes_fp32"]})
    return rows


def main() -> None:
    questions = {q["question_id"]: q for q in read_jsonl(S.DATASET_PATH) if q["split"] == "eval"}
    assert len(questions) == S.N_QUESTIONS["eval"]
    completions = saved_completions(questions)
    assert len(completions) == 3 * len(questions), len(completions)
    followups = saved_followups(completions)
    labels = build_labels(completions, followups)

    reference = {(r["question_id"], r["condition"]): r for r in read_jsonl(S.NOCOT_LABELS) if r["question_id"] in questions}
    assert len(reference) == len(labels) == 2 * len(questions)
    matches = {field: 0 for field in COMPARED}
    mismatches = []
    for label in labels:
        ref = reference[(label["question_id"], label["condition"])]
        ref_values = {"a_cued": ref["a_cued"], "a_none": ref["a_none"], "bias_rm": ref["bias_rm"], "said": ref["said_rm"],
                      "deception_rm": ref["deception_rm"], "p_yes": ref["p_yes_rm"]}
        for field in COMPARED:
            if label[field] == ref_values[field]:
                matches[field] += 1
            elif len(mismatches) < 20:
                mismatches.append({"key": [label["question_id"], label["condition"]], "field": field, "ours": label[field], "261007": ref_values[field]})

    for name, rows in (("completions", completions), ("followups", followups), ("labels", labels)):
        path = OUT_DIR / f"{name}.jsonl"
        if path.exists():
            path.unlink()  # derived from saved data; rebuilt on every run
        write_jsonl(path, rows)
    result = {"PASS": all(n == len(labels) for n in matches.values()), "cued_prompts": len(labels), "follow_ups": len(followups),
              "matches": matches, "first_mismatches": mismatches}
    (S.EXP_DIR / "check_scoring.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
