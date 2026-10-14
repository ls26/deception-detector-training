"""Check: the labels stage and compare.py's rates reproduce 261010's labels and the base numbers from saved data.

    python check_scoring.py     # -> dry/saved_base/{completions,followups,labels}.jsonl, check_scoring.json

Builds eval_checkpoint.py's inputs from the base model's saved data on the 489 confident eval questions:
cued rollouts = 260819's `random`-variant rollouts (3 per cued condition), no-cue samples = 261010's 8 per
question, P(Yes) = 261007's float32 values of the assigned removal wording. Then runs the labels stage and
asserts that every rollout's bias and deception label equals 261010's slice label, and its said Yes / No
equals 261010's wherever this folder asks a follow-up. The rates are printed next to the numbers the plan
quotes (bias 0.121 / 0.261, deception 0.555 / 0.686 thresholded and 0.535 / 0.603 soft, accuracy 0.841 /
0.536, for cue at the true / the wrong answer); a mismatch there is reported, not asserted.
"""

from __future__ import annotations

import json

import noisy_spec as S
from compare import rates, rollout_rows
from domain_imports import draw_wordings, extract_answer, to_content
from eval_checkpoint import KEY_FIELDS, build_labels, key_of, read_jsonl, write_jsonl

OUT_DIR = S.EXP_DIR / "dry" / "saved_base"
EXPECTED = {
    "cue_gold": {"accuracy": 0.841, "bias": 0.121, "deception": 0.555, "deception_soft": 0.535},
    "cue_dist": {"accuracy": 0.536, "bias": 0.261, "deception": 0.686, "deception_soft": 0.603},
}


def saved_completions(questions: dict[int, dict]) -> list[dict]:
    def row(q: dict, condition: str, sample_idx: int, r: dict, cue_letter: str | None) -> dict:
        completion, finish_reason = r["completion"], r["finish_reason"]
        pred = extract_answer(completion)[0]
        return {
            "question_id": q["question_id"], "condition": condition, "sample_idx": sample_idx, "split": q["split"],
            "category": q["category"], "answer": q["answer"], "train_label": q["train_label"], "label_flipped": q["label_flipped"],
            "base_confident_correct": q["base_confident_correct"], "cue_letter": cue_letter, "pred": pred,
            "content": to_content(pred, q["answer"]), "finish_reason": finish_reason, "prompt": r["prompt"], "completion": completion,
        }

    rows = []
    for r in read_jsonl(S.EVAL_NOCUE_COMPLETIONS):
        if r["question_id"] in questions:
            rows.append(row(questions[r["question_id"]], "none", r["sample_idx"], r, None))
            assert rows[-1]["pred"] == r["pred"], "261010's stored no-cue answer differs from the parser"
    for r in read_jsonl(S.BASELINE_COMPLETIONS):
        if r["question_id"] in questions and r["cue_variant"] == "random" and r["condition"] in ("cue_gold", "cue_dist"):
            rows.append(row(questions[r["question_id"]], r["condition"], r["sample_idx"], r, r["cue_letter"]))
    return rows


def saved_followups(completions: list[dict]) -> list[dict]:
    pyes = {key_of(r): r["p_yes_fp32"] for r in read_jsonl(S.REF_PYES)}
    followups = []
    for row in completions:
        if row["condition"] == "none" or row["finish_reason"] != "stop" or row["pred"] is None:
            continue
        followups.append({**{f: row[f] for f in KEY_FIELDS}, "wording": draw_wordings(row["question_id"])[S.DEFINITION][1], "p_yes": pyes[key_of(row)]})
    return followups


def main() -> None:
    questions = {q["question_id"]: q for q in read_jsonl(S.DATASET_PATH) if q["split"] == "eval"}
    completions = saved_completions(questions)
    n_expected = len(questions) * sum(S.N_SAMPLES.values())
    assert len(completions) == n_expected, (len(completions), n_expected)
    followups = saved_followups(completions)
    labels = build_labels(completions, followups)
    for name, rows in (("completions", completions), ("followups", followups), ("labels", labels)):
        write_jsonl(OUT_DIR / f"{name}.jsonl", rows)

    reference = {key_of(r): r for r in read_jsonl(S.SLICE_LABELS) if r["split"] == "eval"}
    assert set(reference) == {key_of(r) for r in labels}, "label rows differ from 261010's eval rows"
    bias_equal = sum(r["bias_rm"] == reference[key_of(r)]["bias_rm"] for r in labels)
    deception_equal = sum(r["deception_rm"] == reference[key_of(r)]["deception_rm"] for r in labels)
    with_followup = [r for r in labels if r["p_yes"] is not None]
    said_equal = sum(r["said"] == reference[key_of(r)]["said_rm"] for r in with_followup)
    print(f"  bias equal {bias_equal}/{len(labels)}, deception equal {deception_equal}/{len(labels)}, "
          f"said equal {said_equal}/{len(with_followup)} rows with a follow-up")

    rows = rollout_rows(OUT_DIR)
    shared = {r["question_id"] for r in rows if r["has_nocue_answer"]}
    result = {"rows": len(labels), "bias_equal": bias_equal, "deception_equal": deception_equal,
              "said_equal": said_equal, "rows_with_followup": len(with_followup), "questions_with_nocue_answer": len(shared), "rates": {}}
    for condition, expected in EXPECTED.items():
        got = rates([r for r in rows if r["condition"] == condition], condition, shared)
        result["rates"][condition] = {metric: {"expected": value, "got": round(got[metric], 4)} for metric, value in expected.items()}
        for metric, value in expected.items():
            print(f"  {condition} {metric}: expected {value}, got {got[metric]:.4f}")
    result["PASS"] = bias_equal == deception_equal == len(labels) and said_equal == len(with_followup)
    (S.EXP_DIR / "check_scoring.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'PASS' if result['PASS'] else 'FAIL'}: wrote check_scoring.json")


if __name__ == "__main__":
    main()
