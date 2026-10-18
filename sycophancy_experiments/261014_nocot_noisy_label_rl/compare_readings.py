"""Compare two sources of letter readings and self-reports on the same eval prompts (reported, not asserted).

    python compare_readings.py --a <eval dir> --b <eval dir | saved> --out <file>.json

A source is an eval_checkpoint.py output folder (completions.jsonl, followups.jsonl) or `saved`: 261007's
saved base-model generation (each prompt's letter reading; the "top" follow-up's float32 P(Yes)).
Every prompt of --a must exist in --b. Per prompt: top-letter agreement and |P(A) - P(A)| (renormalized over
A and B). Per follow-up asked after the same top letter in both: |P(Yes) - P(Yes)| and said agreement.

Uses in the queue: the base model's eval against `saved` (bf16 noise; a prompt or reading bug shows as
top-letter disagreement), and 261013's adapter against the base model on the same questions (an adapter
that is not applied shows as no difference).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nocot_spec as S

KEY_FIELDS = ("question_id", "condition")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def said_of(p_yes: float) -> str | None:
    return None if p_yes == S.SAID_THRESHOLD else ("Yes" if p_yes > S.SAID_THRESHOLD else "No")


def load_source(source: str) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    """({(question_id, condition): letter reading}, {(question_id, condition): {top_letter, p_yes}})."""
    if source == "saved":
        readings = {(r["question_id"], r["condition"]): r for r in read_jsonl(S.NOCOT_TURN1) if r["sample_idx"] == 0}
        followups = {
            (r["question_id"], r["condition"]): {"top_letter": readings[(r["question_id"], r["condition"])]["top_letter"], "p_yes": r["p_yes_fp32"]}
            for r in read_jsonl(S.NOCOT_PYES)
            if r["sample_key"] == "top"
        }
        return readings, followups
    eval_dir = Path(source)
    readings = {tuple(r[f] for f in KEY_FIELDS): r for r in read_jsonl(eval_dir / "completions.jsonl")}
    followups = {tuple(r[f] for f in KEY_FIELDS): r for r in read_jsonl(eval_dir / "followups.jsonl")}
    return readings, followups


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean_abs_diff": None, "max_abs_diff": None}
    return {"n": len(values), "mean_abs_diff": sum(values) / len(values), "max_abs_diff": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    readings_a, followups_a = load_source(args.a)
    readings_b, followups_b = load_source(args.b)
    missing = [key for key in readings_a if key not in readings_b]
    assert not missing, f"{len(missing)} prompts of --a are not in --b, e.g. {missing[:3]}"

    p_a_diffs, top_agree, top_disagree = [], 0, []
    for key, reading in readings_a.items():
        other = readings_b[key]
        if reading["top_letter"] == other["top_letter"]:
            top_agree += 1
        else:
            top_disagree.append({"key": list(key), "a": [reading["top_letter"], reading["p_a"]], "b": [other["top_letter"], other["p_a"]]})
        if reading["p_a"] is not None and other["p_a"] is not None:
            p_a_diffs.append(abs(reading["p_a"] - other["p_a"]))

    p_yes_diffs, said_agree = [], 0
    for key, followup in followups_a.items():
        other = followups_b.get(key)
        if other is None or other["top_letter"] != followup["top_letter"]:
            continue
        p_yes_diffs.append(abs(followup["p_yes"] - other["p_yes"]))
        said_agree += said_of(followup["p_yes"]) == said_of(other["p_yes"])

    result = {
        "a": args.a,
        "b": args.b,
        "prompts": len(readings_a),
        "top_letter_agree": top_agree,
        "top_letter_disagree": top_disagree,
        "p_a": {**summarize(p_a_diffs), "n_above_0.01": sum(d > 0.01 for d in p_a_diffs)},
        "p_yes": {**summarize(p_yes_diffs), "n_above_0.01": sum(d > 0.01 for d in p_yes_diffs), "said_agree": said_agree},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "top_letter_disagree"}, indent=2))
    print(f"top letter disagrees on {len(top_disagree)} of {len(readings_a)} prompts; wrote {args.out}")


if __name__ == "__main__":
    main()
