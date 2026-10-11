"""Probe sets and activations for the slice: 261007's step 1 and step 2, unchanged.

    python run_pipeline.py              # -> arms/train_sizes.json, arms/slice_cot/{probe_sets,activations}/
    python run_pipeline.py --dry-run    # the same under dry/

Reads arms/slice_cot/labels.jsonl (slice_labels.py). Train size per probe kind
is the baseline probe's rule (261007 ref_decomposition.part_full3_oldrecipe):
every rollout of a prompt can be a row, and each class takes 2 x the smallest
(class, cue condition) cell of the train pool, split evenly over the two cue
conditions (step 1's draw, seed 53). Eval: one rollout per prompt (seed 53),
every eligible prompt. Loads the model on the GPU for extraction.
"""

from __future__ import annotations

import argparse
import json

import slice_spec as S
from slice_pipeline import POSITION_OF_FAMILY, bootstrap_hf_env, s1, s2, s3, slice_config, step1_gates, train_sizes_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = S.EXP_DIR / "dry" if args.dry_run else S.EXP_DIR
    bootstrap_hf_env()

    counting_cfg = slice_config(root, n_train_per_class={})
    assert not counting_cfg.probe_sets_dir.exists() and not counting_cfg.act_dir.exists(), "probe sets or activations exist"
    labels = s1.load_labels(counting_cfg)
    cells, n_train_per_class = {}, {}
    for kind in S.PROBE_KINDS:
        cells[kind] = s1.train_pool_per_cell(labels, kind, S.DEFINITION, counting_cfg)
        n_train_per_class[kind] = 2 * min(n for by_condition in cells[kind].values() for n in by_condition.values())
        print(f"{kind}: train pool cells {cells[kind]} -> {n_train_per_class[kind]} per class")
    train_sizes_path(root).write_text(json.dumps({"n_train_per_class": n_train_per_class, "train_pool_cells": cells}, indent=2))

    cfg = slice_config(root)
    probe_sets = s1.build_all(cfg, labels)
    # step 1's gates, minus "one train row per prompt": every rollout of a prompt can be a train row here
    failures = [f for f in step1_gates(cfg, probe_sets) if not ("/train:" in f and "rows over" in f)]
    assert not failures, failures
    for ps in probe_sets:
        for split in cfg.splits:
            assert all(r["split"] == split for r in ps.rows[split]), f"{ps.name}/{split}: a row from the other split"
        train_questions = {r["question_id"] for r in ps.rows["train"]}
        eval_questions = {r["question_id"] for r in ps.rows["eval"]}
        assert not train_questions & eval_questions, f"{ps.name}: a question in train and eval"

    model, tokenizer = s2.load_model_and_tokenizer(cfg)
    extraction = s2.extract_all(cfg, model, tokenizer, {})
    del model

    # every activation file holds its probe-set rows, in file order, with the same labels
    for family, definition in cfg.dataset_keys():
        for split in cfg.splits:
            rows = s1.load_jsonl(s2.probe_set_path(cfg, family, definition, split))
            d = s3.load_activation_file(s3.activation_path(cfg, cfg.dataset_name(family, definition), POSITION_OF_FAMILY[family], split))
            assert d["row_keys"] == [(r["question_id"], r["condition"], r["sample_idx"], r["category"]) for r in rows], (family, split)
            assert d["labels"].tolist() == [r["label"] for r in rows], (family, split)
    (root / "arms" / "extraction_summary.json").write_text(json.dumps(extraction, indent=2))
    print(f"\nWrote {cfg.probe_sets_dir}, {cfg.act_dir}")


if __name__ == "__main__":
    main()
