#!/usr/bin/env python3
"""Train mixed round-1 nofu probe (300/class) and eval on-dataset + base/det0 on-policy."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORKSPACE = Path(__file__).resolve().parents[2]  # archive/ is one level deeper than scratch/ was
EXPERIMENTS = WORKSPACE / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from build_dataset import (  # noqa: E402
    MAX_SEQUENCE_LENGTH,
    N_FIXED_PER_CLASS,
    N_ROUND1_MIXED_PER_CLASS,
    ON_POLICY_ROUND1_MIXED,
    SEED,
    build_train_dataset_round1_mixed,
)
from generation import bootstrap_hf_env  # noqa: E402
from probe_training import (  # noqa: E402
    eval_probe_on_policy,
    load_eval_task_dataset,
    smoke_forward,
    train_probe_and_eval_on_dataset,
    verify_cache_metadata,
)

OUT = EXPERIMENTS / "archive" / "mixed_round1_nofu"
ARTIFACTS_BASE = EXPERIMENTS / "artifacts" / "base"
ARTIFACTS_DET0 = EXPERIMENTS / "artifacts" / "det0"
BASE_EVAL = ARTIFACTS_BASE / "base_eval_generations.jsonl"
BASE_TRAIN = ARTIFACTS_BASE / "base_train_generations.jsonl"
DET0_EVAL = ARTIFACTS_DET0 / "det0_eval_generations.jsonl"
DET0_TRAIN = ARTIFACTS_DET0 / "det0_train_generations.jsonl"

MANIFEST_PATH = OUT / "probe_train_dataset.json"
PROBE_WEIGHTS = OUT / "probe_weights"
ON_DATASET_ACTIVATIONS = OUT / "on_dataset_activations"
BASE_ACTIVATIONS = OUT / "base_activations"
DET0_ACTIVATIONS = OUT / "det0_activations"
METRICS_PATH = OUT / "probe_metrics.json"

EVAL_SAMPLES = 400
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TRAIN_SAMPLES = N_ROUND1_MIXED_PER_CLASS * 2
TEST_SAMPLES = EVAL_SAMPLES * 2

REQUIRED_METRIC_KEYS = (
    "on_dataset_auc",
    "on_policy_base_auc",
    "on_policy_base_split_auc",
    "on_policy_det0_auc",
    "on_policy_det0_split_auc",
    "deceptive_sources",
    "honest_sources",
    "filtered_layers",
)


def check_jsonl_rows(path: Path, expected: int | None, *, min_rows: int = 0) -> dict:
    assert path.exists(), f"Missing {path}"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if expected is not None:
        assert len(rows) == expected, f"{path.name}: expected {expected} rows, got {len(rows)}"
    else:
        assert len(rows) >= min_rows, f"{path.name}: expected >= {min_rows} rows, got {len(rows)}"
    return {"rows": rows, "n": len(rows)}


def preflight_artifacts() -> None:
    for path in (BASE_EVAL, BASE_TRAIN, DET0_EVAL, DET0_TRAIN):
        assert path.exists(), f"Missing artifact: {path}"
    check_jsonl_rows(BASE_EVAL, EVAL_SAMPLES)
    check_jsonl_rows(DET0_EVAL, EVAL_SAMPLES)
    check_jsonl_rows(BASE_TRAIN, None, min_rows=1)
    check_jsonl_rows(DET0_TRAIN, None, min_rows=1)
    print("OK: preflight passed (4 artifact JSONLs present)")


def maybe_clear_outputs(*, force: bool) -> None:
    if not force:
        return
    for path in (
        PROBE_WEIGHTS,
        ON_DATASET_ACTIVATIONS,
        BASE_ACTIVATIONS,
        DET0_ACTIVATIONS,
        METRICS_PATH,
        MANIFEST_PATH,
    ):
        if path.is_dir():
            shutil.rmtree(path)
            print(f"OK: deleted {path} (--force)")
        elif path.is_file():
            path.unlink()
            print(f"OK: deleted {path} (--force)")


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    assert next(model.parameters()).is_cuda, "Expected model on CUDA"
    print(f"OK: loaded base model {BASE_MODEL}")
    return model, tokenizer


def verify_manifest(manifest: dict, tokenizer) -> None:
    assert manifest["deceptive_sources"] == {
        "on_dataset": N_FIXED_PER_CLASS,
        "on_policy": ON_POLICY_ROUND1_MIXED,
    }
    assert manifest["honest_sources"] == {
        "on_dataset": N_FIXED_PER_CLASS,
        "on_policy": ON_POLICY_ROUND1_MIXED,
    }
    assert len(manifest["deceptive"]) == len(manifest["honest"]) == N_ROUND1_MIXED_PER_CLASS
    for row in manifest["deceptive"] + manifest["honest"]:
        text = row["prompt"] + row["completion"]
        assert len(tokenizer.encode(text, add_special_tokens=False)) <= MAX_SEQUENCE_LENGTH
    print(f"OK: manifest verified ({N_ROUND1_MIXED_PER_CLASS} per class)")


def verify_on_policy_cache(activations_root: Path) -> None:
    cache = activations_root / "on_policy"
    assert cache.exists(), f"Missing on-policy cache: {cache}"
    meta = cache / "rank_0" / "combined_examples" / "metadata.json"
    assert meta.exists(), f"Missing on-policy metadata: {meta}"
    print(f"OK: on-policy cache exists at {cache}")


def build_metrics(
    *,
    train_result: dict,
    base_on_policy: dict,
    det0_on_policy: dict,
    manifest: dict,
) -> dict:
    on_dataset = train_result["summary"]
    base_summary = base_on_policy["summary"]
    base_split = base_on_policy["split_summary"]
    det0_summary = det0_on_policy["summary"]
    det0_split = det0_on_policy["split_summary"]
    filtered_layers = train_result["filtered_layers"]

    return {
        "run_id": "mixed_round1_nofu",
        "filtered_layers": filtered_layers,
        "on_dataset_auc": on_dataset["auc_filtered"],
        "on_dataset_accuracy": on_dataset["accuracy"],
        "on_dataset_f1": on_dataset["f1"],
        "on_dataset_mean_scores": on_dataset["mean_scores_layer_averaged"],
        "on_policy_base_auc": base_summary["auc_filtered"],
        "on_policy_base_accuracy": base_summary["accuracy"],
        "on_policy_base_f1": base_summary["f1"],
        "on_policy_base_mean_scores": base_summary["mean_scores_layer_averaged"],
        "on_policy_base_split_auc": base_split["auc_filtered"],
        "on_policy_base_split_accuracy": base_split["accuracy"],
        "on_policy_base_split_f1": base_split["f1"],
        "on_policy_base_split_stats": base_split["stats"],
        "on_policy_base_split_mean_scores": base_split["mean_scores_layer_averaged"],
        "on_policy_det0_auc": det0_summary["auc_filtered"],
        "on_policy_det0_accuracy": det0_summary["accuracy"],
        "on_policy_det0_f1": det0_summary["f1"],
        "on_policy_det0_mean_scores": det0_summary["mean_scores_layer_averaged"],
        "on_policy_det0_split_auc": det0_split["auc_filtered"],
        "on_policy_det0_split_accuracy": det0_split["accuracy"],
        "on_policy_det0_split_f1": det0_split["f1"],
        "on_policy_det0_split_stats": det0_split["stats"],
        "on_policy_det0_split_mean_scores": det0_split["mean_scores_layer_averaged"],
        "deceptive_sources": manifest["deceptive_sources"],
        "honest_sources": manifest["honest_sources"],
        "train_manifest": str(MANIFEST_PATH),
        "weights_dir": str(PROBE_WEIGHTS),
        "on_dataset_activations_dir": str(ON_DATASET_ACTIVATIONS),
        "base_activations_dir": str(BASE_ACTIVATIONS),
        "det0_activations_dir": str(DET0_ACTIVATIONS),
    }


def verify_metrics(metrics: dict) -> None:
    for key in REQUIRED_METRIC_KEYS:
        assert key in metrics, f"probe_metrics.json missing {key!r}"
    assert metrics["on_dataset_auc"] > 0.9, f"on_dataset_auc={metrics['on_dataset_auc']:.4f} too low"
    for path_key in (
        "train_manifest",
        "weights_dir",
        "on_dataset_activations_dir",
        "base_activations_dir",
        "det0_activations_dir",
    ):
        assert str(OUT) in metrics[path_key], f"{path_key} not under {OUT}: {metrics[path_key]}"
    print(f"OK: probe metrics verified at {METRICS_PATH}")


def run(*, force: bool) -> None:
    bootstrap_hf_env()
    OUT.mkdir(parents=True, exist_ok=True)
    preflight_artifacts()
    maybe_clear_outputs(force=force)

    model, tokenizer = load_base_model()
    smoke_forward(model, tokenizer)

    train_pos, train_neg, manifest = build_train_dataset_round1_mixed(
        tokenizer,
        manifest_path=MANIFEST_PATH,
        base_train_jsonl=BASE_TRAIN,
        det0_train_jsonl=DET0_TRAIN,
        seed=SEED,
    )
    verify_manifest(manifest, tokenizer)
    assert len(train_pos) == len(train_neg) == N_ROUND1_MIXED_PER_CLASS

    eval_task_dataset = load_eval_task_dataset(tokenizer, seed=SEED)
    train_result = train_probe_and_eval_on_dataset(
        model,
        tokenizer,
        train_pos=train_pos,
        train_neg=train_neg,
        eval_task_dataset=eval_task_dataset,
        weights_dir=PROBE_WEIGHTS,
        activations_dir=ON_DATASET_ACTIVATIONS,
        follow_up=[],
    )
    assert (PROBE_WEIGHTS / "layer_0" / "model.pt").exists(), f"Missing {PROBE_WEIGHTS}/layer_0/model.pt"
    verify_cache_metadata(ON_DATASET_ACTIVATIONS, expected_samples=TRAIN_SAMPLES, split="train")
    verify_cache_metadata(ON_DATASET_ACTIVATIONS, expected_samples=TEST_SAMPLES, split="test")
    assert train_result["summary"]["auc_filtered"] > 0.9
    print(f"OK: on-dataset train AUC {train_result['summary']['auc_filtered']:.4f}")

    base_on_policy = eval_probe_on_policy(
        model,
        tokenizer,
        probe_dict=train_result["probe_dict"],
        filtered_layers=train_result["filtered_layers"],
        eval_task_dataset=eval_task_dataset,
        activations_dir=BASE_ACTIVATIONS,
        val_jsonl=BASE_EVAL,
        follow_up=[],
        deceptive_definition="pass_only_shown",
    )
    verify_on_policy_cache(BASE_ACTIVATIONS)
    print(f"OK: base on-policy AUC {base_on_policy['summary']['auc_filtered']:.4f}")
    print(f"    base split stats: {base_on_policy['split_summary']['stats']}")

    det0_on_policy = eval_probe_on_policy(
        model,
        tokenizer,
        probe_dict=train_result["probe_dict"],
        filtered_layers=train_result["filtered_layers"],
        eval_task_dataset=eval_task_dataset,
        activations_dir=DET0_ACTIVATIONS,
        val_jsonl=DET0_EVAL,
        follow_up=[],
        deceptive_definition="pass_only_shown",
    )
    verify_on_policy_cache(DET0_ACTIVATIONS)
    print(f"OK: det0 on-policy AUC {det0_on_policy['summary']['auc_filtered']:.4f}")
    print(f"    det0 split stats: {det0_on_policy['split_summary']['stats']}")

    metrics = build_metrics(
        train_result=train_result,
        base_on_policy=base_on_policy,
        det0_on_policy=det0_on_policy,
        manifest=manifest,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    verify_metrics(metrics)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\n=== Mixed round1 nofu probe complete ===")
    print(f"  on_dataset_auc:          {metrics['on_dataset_auc']:.4f}")
    print(f"  on_policy_base_auc:        {metrics['on_policy_base_auc']:.4f}")
    print(f"  on_policy_base_split_auc:  {metrics['on_policy_base_split_auc']:.4f}")
    print(f"  on_policy_det0_auc:        {metrics['on_policy_det0_auc']:.4f}")
    print(f"  on_policy_det0_split_auc:  {metrics['on_policy_det0_split_auc']:.4f}")
    print(f"  saved to:                  {METRICS_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mixed round-1 nofu probe train/eval")
    parser.add_argument("--force", action="store_true", help="Clear outputs under mixed_round1_nofu before run")
    args = parser.parse_args()
    run(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
