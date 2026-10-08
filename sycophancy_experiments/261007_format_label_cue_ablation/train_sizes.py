"""The matched train size per probe kind across arms (spec's size rule).

For every arm's labels.jsonl, count the rows of each (class, condition) cell
of a probe kind in step 1's TRAIN pool (`step1_build_probe_sets.train_pool_per_cell`:
one row per (question_id, condition) prompt, picked with POOL_SEED after the
class filter), i.e. the most step 1 can draw from that cell. A prompt whose
rows fall in both classes counts in one class only. anti-bias and unparsed
rows are never eligible.

  full_n_per_class (arm)  = 2 * min over its (class, condition) cells
  matched_n_per_class     = min over arms of full_n_per_class
  rule: matched >= MATCHED_MIN_PER_CLASS      -> "matched"
        matched <  INSUFFICIENT_PER_CLASS     -> "insufficient" (everywhere)
        otherwise                              -> "unmatched" (each arm at its own full size)

`deception` is the union of the deception_yes and deception_no draws, so it
has no entry. Writes arms/sizes.json (SCHEMA.md) and arms/train_sizes.md.

Usage:
    python train_sizes.py [--arms-dir arms] [--arms ref_cot nocot_sampled ...]
"""

from __future__ import annotations

import argparse
import json

import spec
import step1_build_probe_sets as s1
from label_common import read_jsonl
from probe_config import PipelineConfig

SIZED_KINDS: tuple[str, ...] = ("bias", "bias_followed", "deception_yes", "deception_no")


def rule_for(matched_n_per_class: int) -> str:
    if matched_n_per_class >= spec.MATCHED_MIN_PER_CLASS:
        return "matched"
    if matched_n_per_class < spec.INSUFFICIENT_PER_CLASS:
        return "insufficient"
    return "unmatched"


def what_step1_uses(kind: str, entry: dict) -> str:
    rule, matched = entry["rule"], entry["matched_n_per_class"]
    if rule == "matched":
        return f"`{kind}`: matched; step 1 draws {matched} per class in every arm ({matched // 2} per condition)."
    if rule == "unmatched":
        sizes = ", ".join(f"{arm} {per_arm['full_n_per_class']}" for arm, per_arm in entry["per_arm"].items())
        return (f"`{kind}`: unmatched (matched size {matched} < {spec.MATCHED_MIN_PER_CLASS}); step 1 draws each arm's "
                f"full size ({sizes}) and the probe is marked unmatched.")
    return (f"`{kind}`: insufficient data (matched size {matched} < {spec.INSUFFICIENT_PER_CLASS}); "
            f"step 1 skips this probe in every arm.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms-dir", default="arms")
    parser.add_argument("--arms", nargs="+", default=list(spec.ARMS), choices=spec.ARMS)
    args = parser.parse_args()
    arms_dir = spec.EXP_DIR / args.arms_dir

    rows_by_arm = {}
    for arm in args.arms:
        rows = read_jsonl(arms_dir / arm / "labels.jsonl")
        for row in rows:
            assert row["spec_hash"] == spec.spec_hash(), (arm, row["spec_hash"])
        rows_by_arm[arm] = rows
        print(f"  {arm}: {len(rows)} rows, {sum(r['split'] == 'train' for r in rows)} train")

    per_probe = {}
    for kind in SIZED_KINDS:
        per_arm = {}
        for arm, rows in rows_by_arm.items():
            cells = s1.train_pool_per_cell(rows, kind, spec.DEFINITION, PipelineConfig(arm=arm, arms_dir=arms_dir))
            smallest_cell = min(n for by_condition in cells.values() for n in by_condition.values())
            per_arm[arm] = {"distinct_per_cell": cells, "full_n_per_class": 2 * smallest_cell}
        matched = min(entry["full_n_per_class"] for entry in per_arm.values())
        per_probe[kind] = {"matched_n_per_class": matched, "rule": rule_for(matched), "per_arm": per_arm}

    (arms_dir / "sizes.json").write_text(json.dumps({"spec_hash": spec.spec_hash(), "per_probe": per_probe}, indent=1))

    md = ["# Train sizes", "",
          f"spec {spec.spec_hash()}; arms {', '.join(args.arms)}. Cells count the rows of step 1's train pool "
          f"(one row per (question_id, condition) prompt after the class filter, POOL_SEED {spec.POOL_SEED}), i.e. the most "
          f"step 1 can draw per cell. full_n_per_class = 2 x the arm's smallest cell; "
          f"matched = min over arms. Rule: >= {spec.MATCHED_MIN_PER_CLASS} matched, < {spec.INSUFFICIENT_PER_CLASS} "
          "insufficient, otherwise unmatched.", ""]
    for kind, entry in per_probe.items():
        md += [f"## {kind}", "", "| arm | class | cue_gold | cue_dist | full_n_per_class |", "|---|---|---|---|---|"]
        for arm, per_arm in entry["per_arm"].items():
            for class_value, by_condition in per_arm["distinct_per_cell"].items():
                md.append(f"| {arm} | {class_value} | {by_condition['cue_gold']} | {by_condition['cue_dist']} | "
                          f"{per_arm['full_n_per_class']} |")
        md += ["", f"matched_n_per_class = {entry['matched_n_per_class']}, rule = **{entry['rule']}**.", "",
               what_step1_uses(kind, entry), ""]
        print(f"  {kind}: matched {entry['matched_n_per_class']} -> {entry['rule']}")
    md += ["`deception` (unstratified) is the union of the deception_yes and deception_no draws and takes no size of its own.", ""]
    (arms_dir / "train_sizes.md").write_text("\n".join(md))
    print(f"  wrote {arms_dir / 'sizes.json'} and train_sizes.md")


if __name__ == "__main__":
    main()
