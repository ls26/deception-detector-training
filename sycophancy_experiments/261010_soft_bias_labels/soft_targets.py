"""Soft bias labels for fit B's rows of the three probes.

    python soft_targets.py                  # needs completions.jsonl (the 8 fresh no-cue samples)
    python soft_targets.py --old-samples-only --out dry/old3/labels.jsonl
                                            # pre-check: every row, 3-sample fields only
    python soft_targets.py --new-completions dry/completions.jsonl --out dry/labels.jsonl
                                            # dry run on a few questions: fresh fields for those only

One record per row of fit B's two probe-set files (bias_followed_rm, deception_rm),
train and eval:

  p_none = fraction of parsed no-cue samples whose content is the cued content
           (G under cue_gold, D under cue_dist)
  b      = 1 - p_none if this rollout followed the cue, else 0
  target = b for bias_followed and said-No rows; 1 - b for said-Yes rows

`*_3` uses the 3 existing no-cue samples of 260819 (every row); `*_8` the 8 fresh
samples (eval rows). `hard_8` is the old hard rule (260824's turn1_label /
turn2_label on the strict majority of the 8) and is None where that rule drops the
row (a 4-4 tie, or anti-bias on a deception row).
Writes labels.jsonl and p_none_summary.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import soft_spec as S
from domain_imports import majority_vote, turn1_label, turn2_label


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def no_cue_contents_3() -> dict[int, list[str | None]]:
    """question_id -> the 3 contents of 260819's no-cue cell."""
    cells = [r for r in read_jsonl(S.BASELINE_SCORES) if r["condition"] == "none"]
    out = {}
    for r in cells:
        assert r["cue_variant"] == "shared" and r["n_samples"] == 3 and len(r["contents"]) == 3, r["question_id"]
        assert r["question_id"] not in out, f"two no-cue cells for q{r['question_id']}"
        out[r["question_id"]] = r["contents"]
    return out


def no_cue_contents_8(path: Path) -> dict[int, list[str | None]]:
    """question_id -> the contents of the fresh no-cue samples, in sample order."""
    by_question: dict[int, dict[int, str | None]] = defaultdict(dict)
    for r in read_jsonl(path):
        by_question[r["question_id"]][r["sample_idx"]] = r["content"]
    for qid, samples in by_question.items():
        assert sorted(samples) == list(range(S.N_NEW_SAMPLES)), f"q{qid}: samples {sorted(samples)}"
    return {qid: [samples[k] for k in range(S.N_NEW_SAMPLES)] for qid, samples in by_question.items()}


def p_none(contents: list[str | None], cue_content: str) -> tuple[float, int]:
    """(fraction of parsed samples equal to cue_content, number parsed)."""
    parsed = [c for c in contents if c is not None]
    assert parsed, "no parsed no-cue sample"
    return sum(c == cue_content for c in parsed) / len(parsed), len(parsed)


def target_of(b: float, said: str | None) -> float:
    """bias_followed rows (said None) and said-No rows: b; said-Yes rows: 1 - b."""
    if said == "Yes":
        return 1.0 - b
    assert said in (None, "No"), said
    return b


def hard_label_8(row: dict, label_rec: dict, contents_8: list[str | None], probe_set: str) -> int | None:
    """The old hard rule with the 8 fresh samples as the counterfactual."""
    rollout = {"a_cued": label_rec["a_cued"], "a_none": majority_vote(contents_8), "a_other_cue": None, "condition": row["condition"]}
    stratum = turn1_label(rollout, "rm")
    if probe_set == "bias_followed_rm":
        return {"bias": 1, "no bias": 0}.get(stratum)
    return {"deceptive": 1, "honest": 0}.get(turn2_label(stratum, row["said"]))


def build_records(probe_set: str, split: str, labels_by_key: dict, contents_3: dict, contents_8: dict | None) -> list[dict]:
    rows = read_jsonl(S.FIT_B_PROBE_SETS / f"{probe_set}_{split}.jsonl")
    records = []
    for row in rows:
        key = (row["question_id"], row["condition"], row["sample_idx"])
        label_rec = labels_by_key[key]
        cue_content = "G" if row["condition"] == "cue_gold" else "D"
        assert row["condition"] in ("cue_gold", "cue_dist"), row["condition"]
        followed = row["followed_cue"]
        assert followed == (label_rec["a_cued"] == cue_content), f"{key}: followed_cue disagrees with a_cued"
        said = None if probe_set == "bias_followed_rm" else row["said"]
        if probe_set == "bias_followed_rm":
            assert followed, f"{key}: bias_followed row that did not follow the cue"
        else:
            assert said == label_rec["said_rm"], f"{key}: said differs from fit B's labels"

        p_3, n_parsed_3 = p_none(contents_3[row["question_id"]], cue_content)
        b_3 = (1.0 - p_3) if followed else 0.0
        target_3 = target_of(b_3, said)
        # Rounding the 3-sample target must give back fit B's hard label (asserted in main).
        rec = {
            "probe_set": probe_set,
            "split": split,
            "question_id": row["question_id"],
            "condition": row["condition"],
            "sample_idx": row["sample_idx"],
            "said": said,
            "followed": followed,
            "fit_b_label": row["label"],
            "n_parsed_3": n_parsed_3,
            "p_none_3": p_3,
            "target_3": target_3,
        }
        if split == "eval" and contents_8 is not None and row["question_id"] in contents_8:
            c8 = contents_8[row["question_id"]]
            p_8, n_parsed_8 = p_none(c8, cue_content)
            b_8 = (1.0 - p_8) if followed else 0.0
            rec.update(
                {
                    "n_parsed_8": n_parsed_8,
                    "p_none_8": p_8,
                    "soft_8": target_of(b_8, said),
                    "hard_8": hard_label_8(row, label_rec, c8, probe_set),
                }
            )
        records.append(rec)
    return records


def check_records(records: list[dict], require_fresh: bool) -> None:
    """Structural asserts: the 3-sample target reproduces fit B's label on every row;
    in the full run every eval row has a fresh soft label."""
    for rec in records:
        assert rec["target_3"] != 0.5, f"target 0.5 at {rec}"
        assert round(rec["target_3"]) == rec["fit_b_label"], f"rounded target differs from fit B's label: {rec}"
        if rec["split"] == "eval" and require_fresh:
            assert "soft_8" in rec, f"eval row without fresh samples: {rec}"
            assert 0.0 <= rec["soft_8"] <= 1.0
    keys = [(r["probe_set"], r["split"], r["question_id"], r["condition"], r["sample_idx"]) for r in records]
    assert len(keys) == len(set(keys)), "duplicate row keys"


def summarize(records: list[dict], contents_3: dict, contents_8: dict | None) -> dict:
    summary: dict = {"rows": dict(Counter(f"{r['probe_set']}/{r['split']}" for r in records))}
    summary["train_target_values"] = {
        ps: dict(Counter(round(r["target_3"], 4) for r in records if r["probe_set"] == ps and r["split"] == "train")) for ps in S.PROBE_SETS
    }
    if contents_8 is None:
        return summary
    eval_rows = [r for r in records if "soft_8" in r]
    summary["eval_rows_with_fresh_labels"] = len(eval_rows)
    summary["parsed_of_8_per_question"] = dict(sorted(Counter(sum(c is not None for c in v) for v in contents_8.values()).items()))
    # p_none of the gold answer per eval question = no-cue accuracy; reported per condition below.
    for ps in S.PROBE_SETS:
        rows = [r for r in eval_rows if r["probe_set"] == ps]
        by_said = {"bias_followed_rm": [None], "deception_rm": ["Yes", "No"]}[ps]
        for said in by_said:
            sub = [r for r in rows if r["said"] == said]
            name = ps if said is None else f"{ps}/said_{said}"
            soft = [r["soft_8"] for r in sub]
            summary[name] = {
                "rows": len(sub),
                "soft_8_exactly_0_or_1": sum(s in (0.0, 1.0) for s in soft),
                "soft_8_histogram_tenths": dict(sorted(Counter(min(int(s * 10), 9) for s in soft).items())),
                "hard_8_dropped": sum(r["hard_8"] is None for r in sub),
                "hard_8_differs_from_fit_b": sum(r["hard_8"] is not None and r["hard_8"] != r["fit_b_label"] for r in sub),
                "followed_rows_p_none_8_by_condition": {
                    cond: dict(sorted(Counter(round(r["p_none_8"], 3) for r in sub if r["followed"] and r["condition"] == cond).items()))
                    for cond in ("cue_gold", "cue_dist")
                },
            }
    agree = sum(majority_vote(contents_3[q]) == majority_vote(c8) for q, c8 in contents_8.items())
    summary["questions_majority_3_equals_majority_8"] = f"{agree}/{len(contents_8)}"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-samples-only", action="store_true")
    parser.add_argument("--new-completions", type=Path, default=S.NEW_COMPLETIONS)
    parser.add_argument("--out", type=Path, default=S.LABELS)
    args = parser.parse_args()
    assert not args.out.exists(), f"{args.out} exists"

    labels_by_key = {(r["question_id"], r["condition"], r["sample_idx"]): r for r in read_jsonl(S.FIT_B_LABELS)}
    contents_3 = no_cue_contents_3()
    contents_8 = None if args.old_samples_only else no_cue_contents_8(args.new_completions)
    full_run = args.new_completions == S.NEW_COMPLETIONS and not args.old_samples_only

    records = []
    for probe_set in S.PROBE_SETS:
        for split in ("train", "eval"):
            records += build_records(probe_set, split, labels_by_key, contents_3, contents_8)
    check_records(records, require_fresh=full_run)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = summarize(records, contents_3, contents_8)
    assert args.out.name == "labels.jsonl", "per-row label files are named labels.jsonl"
    summary_path = args.out.parent / "p_none_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out} ({len(records)} rows) and {summary_path}")


if __name__ == "__main__":
    main()
