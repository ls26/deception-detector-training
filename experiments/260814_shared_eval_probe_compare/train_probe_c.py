#!/usr/bin/env python3
"""Train probe C: the round-6 training set, but on BASE-model activations.

Probe C is the initial model's counterpart to the round-6 probe (A): identical
training text, different model. Rows come verbatim from each lineage's existing
round-6 `probe_train_dataset.json` -- never rebuilt -- so A and C are guaranteed to
have seen byte-identical training data and differ only in whose activations they read.

Writes to `round6/probe_weights_base_model/`; the round-6 artifacts are never touched.
The base model is loaded once and reused across all lineages.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/workspace")))

from pipeline.build_dataset import rows_to_dataset
from pipeline.grpo_launcher import check_disk
from pipeline.paths import WORKSPACE
from pipeline.probe_training import (
    MODEL_NAME,
    bootstrap_hf_env,
    load_eval_task_dataset,
    train_probe_and_eval_on_dataset,
)

from common import (
    DATA_SEED,
    PROBE_C,
    label_for,
    normalize_variant,
    probe_c_activations,
    probe_c_metrics,
    probe_c_weights,
    resolve_targets,
    round6_manifest,
)

# nofu: probe at the completion's end-of-turn token, matching every round-2+ probe in
# these lineages (trial configs set nofu: true; see EXPERIMENT_LOG.md #21).
FOLLOW_UP: list = []
MIN_ON_DATASET_AUC = 0.99


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    print(f"OK: loaded base model {MODEL_NAME}")
    return model, tokenizer


def _clear_existing(trial: int, variant: str) -> None:
    for path in (probe_c_weights(trial, variant), probe_c_activations(trial, variant)):
        if path.exists():
            shutil.rmtree(path)
            print(f"OK: deleted {path} (--force)")


def train_lineage(model, tokenizer, eval_task_dataset, trial: int, variant: str) -> dict:
    label = label_for(trial, variant)
    print(f"\n{'=' * 60}\nProbe C ({PROBE_C}): {label}\n{'=' * 60}")

    manifest_path = round6_manifest(trial, variant)
    if not manifest_path.exists():
        raise FileNotFoundError(f"{label}: missing round-6 manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    n_per_class = manifest["n_train_per_class"]
    if not (len(manifest["deceptive"]) == len(manifest["honest"]) == n_per_class):
        raise ValueError(
            f"{label}: manifest class counts mismatch "
            f"(deceptive={len(manifest['deceptive'])}, honest={len(manifest['honest'])}, "
            f"n_train_per_class={n_per_class})"
        )
    print(f"OK: reusing {manifest_path} ({n_per_class}/class, strategy={manifest.get('strategy')})")

    result = train_probe_and_eval_on_dataset(
        model,
        tokenizer,
        train_pos=rows_to_dataset(manifest["deceptive"]),
        train_neg=rows_to_dataset(manifest["honest"]),
        eval_task_dataset=eval_task_dataset,
        weights_dir=probe_c_weights(trial, variant),
        activations_dir=probe_c_activations(trial, variant),
        follow_up=FOLLOW_UP,
    )
    summary = result["summary"]
    if summary["auc_filtered"] < MIN_ON_DATASET_AUC:
        print(
            f"WARN {label}: probe C on-dataset AUC={summary['auc_filtered']:.4f} "
            f"< {MIN_ON_DATASET_AUC} -- the round-6 training set may not be linearly "
            "separable from base-model activations; treat the comparison with care"
        )

    metrics = {
        "probe": PROBE_C,
        "trial": trial,
        "variant": variant or "original",
        "activations_model": "base",
        "model_name": MODEL_NAME,
        "nofu": True,
        "data_seed": DATA_SEED,
        "train_manifest": str(manifest_path),
        "n_train_per_class": n_per_class,
        "filtered_layers": result["filtered_layers"],
        "on_dataset_auc": summary["auc_filtered"],
        "on_dataset_auc_all_layers": summary["auc_all_layers"],
        "on_dataset_accuracy": summary["accuracy"],
        "on_dataset_f1": summary["f1"],
        "on_dataset_mean_scores": summary["mean_scores_layer_averaged"],
        "weights_dir": str(probe_c_weights(trial, variant)),
        "activations_dir": str(probe_c_activations(trial, variant)),
    }
    metrics_path = probe_c_metrics(trial, variant)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"OK {label}: on_dataset_auc={summary['auc_filtered']:.4f}, wrote {metrics_path.name}")

    del result
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train probe C (round-6 data, base activations)")
    parser.add_argument("--trial", type=int, action="append", dest="trials")
    parser.add_argument("--variant", type=str, action="append", dest="variants")
    parser.add_argument("--force", action="store_true", help="Retrain even if weights exist")
    args = parser.parse_args()

    bootstrap_hf_env()
    check_disk(min_free_gb=30, path=WORKSPACE)

    variants = [normalize_variant(v) for v in args.variants] if args.variants else None
    targets = resolve_targets(trials=args.trials, variants=variants)
    print(f"Targets ({len(targets)}): {targets}")

    if args.force:
        for trial, variant in targets:
            _clear_existing(trial, variant)

    pending = [
        (t, v) for t, v in targets if not (probe_c_weights(t, v) / "layer_0" / "model.pt").exists()
    ]
    for trial, variant in targets:
        if (trial, variant) not in pending:
            print(f"SKIP {label_for(trial, variant)}: probe C weights already exist (--force to retrain)")
    if not pending:
        print("Nothing to train.")
        return

    model, tokenizer = load_base_model()
    # Same 400-prompt on-dataset test split for every lineage; load once.
    eval_task_dataset = load_eval_task_dataset(tokenizer, seed=DATA_SEED)

    results = [train_lineage(model, tokenizer, eval_task_dataset, t, v) for t, v in pending]

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\nProbe C summary\n{'=' * 60}")
    for row in results:
        print(
            f"  trial{row['trial']} {row['variant']:28s} "
            f"on_dataset_auc={row['on_dataset_auc']:.4f}  "
            f"layers={len(row['filtered_layers'])}"
        )


if __name__ == "__main__":
    main()
