"""Final fit of the three slice probes, and the baseline probe scored on the same eval rows.

    python final_fit.py              # -> runs/slice_cot/, runs/baseline_on_slice/
    python final_fit.py --dry-run    # the same under dry/

1. Rescoring check: the baseline probe's saved weights and band scored on its
   own eval activations, against its saved eval scores (row lists asserted
   equal; the largest score difference and both AUCs reported).
2. Slice probes: 261007's train_eval_all with each probe's CV choice
   (cv/slice_cot/<probe>/selection.json), seed 0, on the full slice train set,
   scored on the slice eval rows -> runs/slice_cot/{config.json,
   probe_metrics.json, eval_scores.json, probe_weights/}.
3. The baseline probe's saved weights and band scored on the slice eval
   activations, whose labels are the slice labels ->
   runs/baseline_on_slice/eval_scores.json.
This is the first read of the slice eval activations.
"""

from __future__ import annotations

import argparse
import ast
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import slice_spec as S
from slice_pipeline import apply_overrides, bootstrap_hf_env, config_to_json, s3, slice_config, tokenizer_for
from obfuscation_atlas.detectors.train import load_model


def load_baseline_probe(probe_name: str) -> tuple[dict, list[int]]:
    """The baseline probe's {layer: probe} and the layers its saved eval scores averaged over."""
    probe_dir = S.BASELINE_PROBE_RUN_DIR / "probe_weights" / probe_name
    layers = ast.literal_eval((probe_dir / "filtered_layers.txt").read_text())
    return load_model(str(probe_dir)), layers


def score_activation_file(probe_dict: dict, layers: list[int], act_file: dict, said: str | None, cfg, tokenizer) -> dict:
    """Mean-sigmoid score of every row of one activation file (said-masked), with its row records."""
    ds = s3.build_activation_dataset(tokenizer, act_file, cfg, said=said)
    scores = s3.combined_score(s3.layer_logits(probe_dict, ds, layers), cfg.combined_score)
    question_ids, conditions = s3.eval_row_records(act_file, s3.said_mask(act_file, said))
    return {"question_id": question_ids, "condition": conditions, "label": ds.labels.numpy().tolist(), "score": scores.tolist()}


def row_records(scores: dict) -> list[tuple]:
    return list(zip(scores["question_id"], scores["condition"], scores["label"]))


def rescoring_check(cfg, tokenizer) -> dict:
    saved = json.loads((S.BASELINE_PROBE_RUN_DIR / "eval_scores.json").read_text())
    out = {}
    for probe_name in S.PROBES:
        probe = cfg.spec(probe_name)
        probe_dict, layers = load_baseline_probe(probe_name)
        act_file = s3.load_activation_file(S.BASELINE_PROBE_ACT_DIR / f"{probe.dataset}_{probe.position}_eval.pt")
        rescored = score_activation_file(probe_dict, layers, act_file, probe.said, cfg, tokenizer)
        assert row_records(rescored) == row_records(saved[probe_name]), f"{probe_name}: rescored rows differ from the saved eval rows"
        out[probe_name] = {
            "n_rows": len(rescored["score"]),
            "layers": layers,
            "max_abs_score_diff": float(np.max(np.abs(np.array(rescored["score"]) - np.array(saved[probe_name]["score"])))),
            "auc_saved": float(roc_auc_score(saved[probe_name]["label"], saved[probe_name]["score"])),
            "auc_rescored": float(roc_auc_score(rescored["label"], rescored["score"])),
        }
        print(f"  rescoring check {probe_name}: {out[probe_name]}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = S.EXP_DIR / "dry" if args.dry_run else S.EXP_DIR
    bootstrap_hf_env()

    cfg = slice_config(root)
    tokenizer = tokenizer_for(cfg)
    baseline_dir = root / "runs" / "baseline_on_slice"
    assert not cfg.run_dir.exists() and not baseline_dir.exists(), "a final fit exists"

    print("=== 1. rescoring check ===")
    check = rescoring_check(cfg, tokenizer)

    print("\n=== 2. slice probes ===")
    overrides = {}
    for probe_name in S.PROBES:
        chosen = json.loads((cfg.cv_dir / probe_name / "selection.json").read_text())["chosen"]
        overrides[probe_name] = {
            "train_cfg": {"learning_rate": chosen["learning_rate"], "max_steps": chosen["max_steps"]},
            "layer_selection": "band",
            "layer_band": chosen["layer_band"],
        }
        # balanced by construction; asserted because the trainer would silently truncate an imbalanced set
        probe = cfg.spec(probe_name)
        train_file = s3.load_activation_file(s3.activation_path(cfg, probe.dataset, probe.position, "train"))
        train_labels = train_file["labels"][s3.said_mask(train_file, probe.said)]
        assert int((train_labels == 1).sum()) == int((train_labels == 0).sum()), f"{probe_name}: train set not balanced"
        del train_file
    cfg = apply_overrides(cfg, {"per_probe_overrides": overrides})
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "config.json").write_text(json.dumps(config_to_json(cfg), indent=2))
    s3.train_eval_all(cfg, tokenizer)
    slice_scores = json.loads(cfg.eval_scores_path.read_text())

    print("\n=== 3. baseline probe on the slice eval rows ===")
    baseline_scores = {"rescoring_check": check}
    for probe_name in S.PROBES:
        probe = cfg.spec(probe_name)
        probe_dict, layers = load_baseline_probe(probe_name)
        act_file = s3.load_activation_file(s3.activation_path(cfg, probe.dataset, probe.position, "eval"))
        baseline_scores[probe_name] = score_activation_file(probe_dict, layers, act_file, probe.said, cfg, tokenizer)
        baseline_scores[probe_name]["layers"] = layers
        assert row_records(baseline_scores[probe_name]) == row_records(slice_scores[probe_name]), f"{probe_name}: the two probes scored different rows"
        print(f"  {probe_name}: AUC {roc_auc_score(baseline_scores[probe_name]['label'], baseline_scores[probe_name]['score']):.4f} on {len(layers)} layers {layers}")
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "eval_scores.json").write_text(json.dumps(baseline_scores))
    print(f"\nWrote {cfg.run_dir} and {baseline_dir}")


if __name__ == "__main__":
    main()
