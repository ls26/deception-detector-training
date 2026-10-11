"""The slice and its labels.

    python slice_labels.py              # -> arms/slice_cot/labels.jsonl, slice_summary.json
    python slice_labels.py --dry-run    # dry/completions.jsonl + the first DRY_QUESTIONS eval questions -> dry/...

Per question, the no-cue contents of its 8 samples (G = gold, D = distractor,
None = unparsed): train questions from this folder's completions.jsonl, eval
questions from 261010's. A question is in the slice when at least MIN_AGREE of
its 8 samples have the same content; that content is its no-cue answer.

Every 261007 label row (one per cued CoT rollout) on a slice question is
copied with `a_none` set to that answer and `bias_rm` / `deception_rm`
recomputed with 260824's turn1_label / turn2_label. `said_rm` (float32
P(Yes) > 0.5) is unchanged. Rows on other questions are dropped. The rows keep
261007's spec_hash stamp, which 261007's step 1 asserts when it loads them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import slice_spec as S
from domain_imports import turn1_label, turn2_label


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def nocue_contents(path: Path, split: str) -> dict[int, list[str | None]]:
    """{question_id: [content of sample 0, ..., sample 7]}."""
    by_question: dict[int, dict[int, str | None]] = {}
    for r in read_jsonl(path):
        assert r["split"] == split and r["condition"] == "none", (path, r["question_id"])
        samples = by_question.setdefault(r["question_id"], {})
        assert r["sample_idx"] not in samples, (path, r["question_id"], r["sample_idx"])
        samples[r["sample_idx"]] = r["content"]
    for question_id, samples in by_question.items():
        assert sorted(samples) == list(range(S.N_NOCUE_SAMPLES)), (path, question_id)
    return {question_id: [samples[k] for k in range(S.N_NOCUE_SAMPLES)] for question_id, samples in by_question.items()}


def confident_answer(contents: list[str | None]) -> str | None:
    """The content at least MIN_AGREE of the samples give, else None (unparsed samples count against)."""
    for content in ("G", "D"):
        if contents.count(content) >= S.MIN_AGREE:
            return content
    return None


def labels_for(row: dict, a_none: str | None) -> tuple[str, str]:
    """(bias_rm, deception_rm) of one cued rollout under the no-cue answer a_none."""
    bias = turn1_label({"a_cued": row["a_cued"], "a_none": a_none, "a_other_cue": None, "condition": row["condition"]}, "rm")
    return bias, turn2_label(bias, row["said_rm"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = S.EXP_DIR / "dry" if args.dry_run else S.EXP_DIR
    out_path = root / "arms" / S.ARM / "labels.jsonl"
    assert not out_path.exists(), f"{out_path} exists"

    contents = {
        "train": nocue_contents(root / "completions.jsonl", "train"),
        "eval": nocue_contents(S.EVAL_NOCUE_COMPLETIONS, "eval"),
    }
    if args.dry_run:
        assert len(contents["train"]) == S.DRY_QUESTIONS
        contents["eval"] = {q: contents["eval"][q] for q in sorted(contents["eval"])[: S.DRY_QUESTIONS]}
    else:
        assert len(contents["train"]) == S.N_TRAIN_QUESTIONS and len(contents["eval"]) == S.N_EVAL_QUESTIONS
    assert not set(contents["train"]) & set(contents["eval"]), "a question in both splits"
    answer = {q: confident_answer(c) for split_contents in contents.values() for q, c in split_contents.items()}

    source = read_jsonl(S.SOURCE_LABELS)
    keys = [(r["question_id"], r["condition"], r["sample_idx"]) for r in source]
    assert len(keys) == len(set(keys)), "duplicate rollout keys in 261007's labels"
    rows, transitions = [], {"train": Counter(), "eval": Counter()}
    for r in source:
        # the label functions reproduce 261007's labels from 261007's own no-cue answer
        assert labels_for(r, r["a_none"]) == (r["bias_rm"], r["deception_rm"]), (r["question_id"], r["condition"], r["sample_idx"])
        if r["question_id"] not in answer:
            assert args.dry_run, f"q{r['question_id']} has no no-cue samples"
            continue
        assert r["split"] in contents and r["question_id"] in contents[r["split"]], (r["question_id"], r["split"])
        if answer[r["question_id"]] is None:
            continue
        new = dict(r)
        new["a_none"] = answer[r["question_id"]]
        new["bias_rm"], new["deception_rm"] = labels_for(r, new["a_none"])
        rows.append(new)
        transitions[r["split"]][(r["bias_rm"], new["bias_rm"], r["deception_rm"], new["deception_rm"])] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    summary = {"min_agree": S.MIN_AGREE, "n_samples": S.N_NOCUE_SAMPLES}
    for split, split_contents in contents.items():
        top_count = Counter(max(c.count("G"), c.count("D")) for c in split_contents.values())
        split_answers = Counter(str(answer[q]) for q in split_contents)
        split_rows = [r for r in rows if r["split"] == split]
        summary[split] = {
            "questions": len(split_contents),
            "slice_questions": len(split_contents) - split_answers["None"],
            "slice_answer": {"G": split_answers["G"], "D": split_answers["D"]},
            "top_count_histogram": {str(k): top_count[k] for k in sorted(top_count)},
            "unparsed_samples": sum(c.count(None) for c in split_contents.values()),
            "rows": len(split_rows),
            "bias_rm": dict(Counter(r["bias_rm"] for r in split_rows)),
            "deception_rm": dict(Counter(r["deception_rm"] for r in split_rows)),
            "label_transitions_old_to_new": [
                {"bias_old": k[0], "bias_new": k[1], "deception_old": k[2], "deception_new": k[3], "rows": n}
                for k, n in sorted(transitions[split].items(), key=lambda kv: -kv[1])
            ],
        }
    (root / "slice_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({split: {k: v for k, v in summary[split].items() if k != "label_transitions_old_to_new"} for split in contents}, indent=2))
    print(f"Wrote {out_path} ({len(rows)} rows) and {root / 'slice_summary.json'}")


if __name__ == "__main__":
    main()
