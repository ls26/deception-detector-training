"""Recompute the numbers behind comparison.md from the files on disk.

    python audit.py

Checks, each printed as PASS / FAIL with its numbers, then one assert over all:
  1. every `auc_combined` in runs/<arm>/probe_metrics.json equals the AUC
     recomputed from runs/<arm>/eval_scores.json (within 1e-6);
  2. the train-pool row counts per probe kind recounted from
     arms/<arm>/labels.jsonl equal arms/sizes.json's `distinct_per_cell`, and
     `full_n_per_class` is twice the smallest cell;
  3. every probe_sets eval file has one row per (question, condition) prompt,
     its prompts are exactly the eligible eval prompts of labels.jsonl, and
     every row's label is the labels.jsonl row's.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import spec  # noqa: E402
import step1_build_probe_sets as s1  # noqa: E402
from probe_config import PipelineConfig  # noqa: E402

AUC_TOL = 1e-6


def check(results: list[tuple[bool, str]], ok: bool, message: str) -> None:
    results.append((ok, message))
    print(f"  {'PASS' if ok else 'FAIL'}  {message}")


def audit_aucs(runs_dir: Path, arm: str, results: list) -> None:
    run_dir = runs_dir / arm
    if not (run_dir / "probe_metrics.json").exists():
        print(f"  (no final fit for {arm})")
        return
    metrics = json.loads((run_dir / "probe_metrics.json").read_text())
    scores = json.loads((run_dir / "eval_scores.json").read_text())
    check(results, metrics["spec_hash"] == spec.spec_hash() and scores["spec_hash"] == spec.spec_hash(), f"{arm}: run files stamped with the current spec")
    for p in metrics["probes"]:
        s = scores[p["probe"]]
        recomputed = float(roc_auc_score(np.array(s["label"]), np.array(s["score"])))
        check(results, abs(recomputed - p["auc_combined"]) < AUC_TOL, f"{arm}/{p['probe']}: auc_combined {p['auc_combined']:.6f} vs recomputed {recomputed:.6f} on {len(s['label'])} eval rows")
        check(results, p["combined_score"] == spec.COMBINED_SCORE and abs(p["auc_mean_sigmoid"] - p["auc_combined"]) < AUC_TOL, f"{arm}/{p['probe']}: headline is the {spec.COMBINED_SCORE} AUC")


def audit_sizes(arms_dir: Path, arm: str, labels: list[dict], sizes: dict, results: list) -> None:
    # the unstratified deception probe has no size entry (it is the union of the two strata)
    for kind in [kind for kind in spec.PROBE_KINDS if kind != "deception"]:
        entry = sizes["per_probe"][kind]
        if arm not in entry["per_arm"]:
            check(results, False, f"{arm}/{kind}: no per_arm entry in sizes.json")
            continue
        recorded = entry["per_arm"][arm]
        counted = s1.train_pool_per_cell(labels, kind, spec.DEFINITION, PipelineConfig(arm=arm, arms_dir=arms_dir))
        check(results, counted == recorded["distinct_per_cell"], f"{arm}/{kind}: train pool rows per cell {counted} vs sizes.json {recorded['distinct_per_cell']}")
        smallest_cell = min(n for cells in counted.values() for n in cells.values())
        check(results, recorded["full_n_per_class"] == 2 * smallest_cell, f"{arm}/{kind}: full_n_per_class {recorded['full_n_per_class']} vs 2 x smallest cell {2 * smallest_cell} (rule {entry['rule']})")


def audit_eval_sets(arms_dir: Path, arm: str, labels: list[dict], results: list) -> None:
    probe_sets_dir = arms_dir / arm / "probe_sets"
    labels_by_key = {(r["question_id"], r["condition"], r["sample_idx"]): r for r in labels}
    for family in ("bias", "deception", "bias_followed"):
        path = probe_sets_dir / f"{family}_{spec.DEFINITION}_eval.jsonl"
        if not path.exists():
            print(f"  (no {path.name} for {arm})")
            continue
        rows = s1.load_jsonl(path)
        prompt_counts = Counter((r["question_id"], r["condition"]) for r in rows)
        check(results, max(prompt_counts.values()) == 1, f"{arm}/{path.name}: {len(rows)} rows over {len(prompt_counts)} prompts (max rows per prompt {max(prompt_counts.values())})")
        eligible_prompts = {(r["question_id"], r["condition"]) for r in labels if r["split"] == "eval" and s1.is_eligible(r, family, spec.DEFINITION)}
        check(results, set(prompt_counts) == eligible_prompts, f"{arm}/{path.name}: prompts equal the eligible eval prompts of labels.jsonl ({len(eligible_prompts)})")
        label_of = s1.label_fn(family, spec.DEFINITION)
        n_label_mismatch = 0
        for r in rows:
            source = labels_by_key[(r["question_id"], r["condition"], r["sample_idx"])]
            expected = 1 if s1.is_positive(family, label_of(source)) else 0
            n_label_mismatch += int(expected != r["label"] or source["split"] != "eval")
        check(results, n_label_mismatch == 0, f"{arm}/{path.name}: {n_label_mismatch} rows whose label or split differs from labels.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms-dir", type=Path, default=spec.ARMS_DIR)
    args = parser.parse_args()
    arms_dir = args.arms_dir.resolve()
    runs_dir = arms_dir.parent / "runs"
    sizes = json.loads((arms_dir / "sizes.json").read_text())
    results: list[tuple[bool, str]] = []
    check(results, sizes["spec_hash"] == spec.spec_hash(), "sizes.json stamped with the current spec")

    for arm in spec.ARMS:
        labels_path = arms_dir / arm / "labels.jsonl"
        if not labels_path.exists():
            print(f"\n=== {arm}: no labels.jsonl, skipped ===")
            continue
        print(f"\n=== {arm} ===")
        labels = s1.load_jsonl(labels_path)
        audit_aucs(runs_dir, arm, results)
        audit_sizes(arms_dir, arm, labels, sizes, results)
        audit_eval_sets(arms_dir, arm, labels, results)

    n_fail = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    assert n_fail == 0, f"{n_fail} audit checks failed"
    print("AUDIT PASS")


if __name__ == "__main__":
    main()
