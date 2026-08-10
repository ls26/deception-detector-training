#!/usr/bin/env python3
"""Supplemented on-policy split probe eval for multi-round trial lineages."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path("/workspace")))

from obfuscation_atlas.detectors.evaluate import eval_multiple_detectors
from obfuscation_atlas.detectors.probe_archs import SequenceAggregator
from obfuscation_atlas.detectors.train import load_model
from obfuscation_atlas.tasks.code_generation import deceptive_reward_key
from obfuscation_atlas.training.obfuscation_training import (
    HONEST_REWARD_KEY,
    run_on_policy_split_eval,
)
from pipeline.paths import WORKSPACE
from pipeline.probe_training import (
    EVAL_SEQUENCE_AGGREGATOR,
    MAX_SEQUENCE_LENGTH,
    MODEL_BATCH_SIZE,
    MODEL_NAME,
    _assert_probed_token,
    _load_filtered_layers,
    _summarize_split_metrics,
    bootstrap_hf_env,
    detector_key_for,
    load_final_policy_model,
)
from pipeline.round_config import resolve_checkpoint_final

from experiments.supplemented_probe_eval.common import (
    DATA_SEED,
    DECEPTIVE_DEFINITION,
    LINEAGES,
    build_supplemented_rows,
    load_jsonl,
    load_trial_state,
    results_path,
    rounds_dir,
    source_eval_rows,
)

FOLLOW_UP: list = []
CACHE_ROOT = WORKSPACE / "experiments/supplemented_probe_eval/activations"
SPLIT_FPR = 0.01


def load_base_model():
    bootstrap_hf_env()
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
    return model, tokenizer


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def probe_weights_dir(trial: int, variant: str, round_num: int, *, frozen: bool) -> Path:
    rd = rounds_dir(trial, variant)
    probe_round = 1 if frozen else round_num
    return rd / f"round{probe_round}" / "probe_weights"


def load_probe(trial: int, variant: str, round_num: int, *, frozen: bool):
    weights_dir = probe_weights_dir(trial, variant, round_num, frozen=frozen)
    if not weights_dir.exists():
        raise FileNotFoundError(f"Missing probe weights: {weights_dir}")
    probe_dict = load_model(str(weights_dir))
    filtered_layers = _load_filtered_layers(weights_dir)
    return probe_dict, filtered_layers


def load_model_k(trial: int, variant: str, model_k: int, state: dict):
    if model_k == 0:
        return load_base_model()
    run_id = state["round_grpo_ids"][str(model_k)]
    checkpoint_final = resolve_checkpoint_final(WORKSPACE, run_id, trial=trial)
    adapter_path = checkpoint_final / "lora"
    if not adapter_path.is_dir():
        adapter_path = checkpoint_final
    return load_final_policy_model(checkpoint_final=checkpoint_final, adapter_path=adapter_path)


def score_supplemented_rows(
    model,
    tokenizer,
    *,
    probe_dict,
    filtered_layers: list[int],
    rows: list[dict],
    cache_dir: Path,
) -> dict:
    prompts = [r["prompt"] for r in rows]
    completions = [r["completion"] for r in rows]
    val_gens = Dataset.from_dict({"prompt": prompts, "completion": completions})
    val_neg = Dataset.from_dict({"prompt": prompts, "completion": completions})

    detector_key = detector_key_for("last-token-generation")
    cache_dir.mkdir(parents=True, exist_ok=True)
    on_policy_results = eval_multiple_detectors(
        detectors=[probe_dict],
        detector_types=["linear-probe"],
        test_dataset=(val_gens, val_neg),
        obfuscate_over="last-token-generation",
        model=model,
        tokenizer=tokenizer,
        test_completion_columns=("completion", "completion"),
        test_congruent_follow_up_prompts=FOLLOW_UP,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        model_batch_size=MODEL_BATCH_SIZE,
        eval_sequence_aggregator=EVAL_SEQUENCE_AGGREGATOR,
        test_cache_activations_save_path=str(cache_dir),
        accelerator=None,
    )
    _assert_probed_token(cache_dir, tokenizer=tokenizer, follow_up=FOLLOW_UP, label=str(cache_dir.name))

    deceptive_key = deceptive_reward_key(DECEPTIVE_DEFINITION)
    reward_metrics = {
        deceptive_key: [bool(r["deceptive"]) for r in rows],
        HONEST_REWARD_KEY: [bool(r["honest"]) for r in rows],
    }
    split_results = run_on_policy_split_eval(
        eval_results=on_policy_results,
        reward_metrics=reward_metrics,
        save_path=cache_dir,
        plot_key="supplemented.on_policy",
        filtered_layers=filtered_layers,
        false_positive_rate=SPLIT_FPR,
        deceptive_key=deceptive_key,
        save_json=False,
    )
    split_summary = _summarize_split_metrics(split_results, detector_key, filtered_layers)
    return {
        "deceptive": split_summary["mean_scores_layer_averaged"]["deceptive"],
        "honest": split_summary["mean_scores_layer_averaged"]["honest"],
        "auc": split_summary["auc_filtered"],
        "accuracy": split_summary["accuracy"],
        "f1": split_summary["f1"],
        "n_deceptive": int(sum(r["deceptive"] for r in rows)),
        "n_honest": int(sum(r["honest"] for r in rows)),
        "split_stats": split_summary.get("stats", {}),
    }


def native_class_counts(trial: int, variant: str, model_k: int) -> dict[str, int]:
    rows = source_eval_rows(trial, variant, "base" if model_k == 0 else f"round{model_k}")
    return {
        "deceptive": sum(r["deceptive"] for r in rows),
        "honest": sum(r["honest"] for r in rows),
    }


def validate_against_probe_metrics(trial: int, variant: str, round_num: int, start_metrics: dict) -> None:
    probe_metrics_path = rounds_dir(trial, variant) / f"round{round_num}" / "probe_metrics.json"
    if not probe_metrics_path.exists():
        print(f"  validate round{round_num}: no probe_metrics.json — skip")
        return
    ref = json.loads(probe_metrics_path.read_text())
    ref_scores = ref.get("on_policy_split_mean_scores", {})
    ref_auc = ref.get("on_policy_split_auc")
    tol = 0.02
    for key in ("deceptive", "honest"):
        ref_val = ref_scores.get(key)
        got = start_metrics.get(key)
        if ref_val is None or got is None or (isinstance(got, float) and np.isnan(got)):
            continue
        if abs(ref_val - got) > tol:
            print(
                f"  WARN validate round{round_num} start {key}: ref={ref_val:.4f} got={got:.4f} (>|{tol}|)"
            )
        else:
            print(f"  OK validate round{round_num} start {key}: ref={ref_val:.4f} got={got:.4f}")
    if ref_auc is not None and start_metrics.get("auc") is not None:
        if abs(ref_auc - start_metrics["auc"]) > tol:
            print(
                f"  WARN validate round{round_num} start auc: ref={ref_auc:.4f} got={start_metrics['auc']:.4f}"
            )
        else:
            print(f"  OK validate round{round_num} start auc: ref={ref_auc:.4f} got={start_metrics['auc']:.4f}")


def run_lineage(trial: int, variant: str, *, validate: bool = True) -> dict:
    frozen = variant == "frozen"
    state = load_trial_state(trial, variant)
    lineage_cache = CACHE_ROOT / f"trial{trial}" / (variant or "original")
    lineage_cache.mkdir(parents=True, exist_ok=True)

    round_results: dict[str, dict] = {str(r): {} for r in range(1, 6)}
    model_meta: dict[str, dict] = {}

    for model_k in range(0, 6):
        print(f"\n{'=' * 60}")
        print(f"trial{trial} {variant or 'original'}: model_k={model_k}")
        rows, sup_meta = build_supplemented_rows(trial=trial, variant=variant, model_k=model_k)
        model_meta[str(model_k)] = sup_meta
        native = native_class_counts(trial, variant, model_k)
        print(f"  native counts: deceptive={native['deceptive']} honest={native['honest']}")
        print(f"  supplemented: deceptive={sup_meta['n_deceptive']} honest={sup_meta['n_honest']}")
        print(f"  provenance: dec={sup_meta['deceptive_provenance']} hon={sup_meta['honest_provenance']}")

        model, tokenizer = load_model_k(trial, variant, model_k, state)
        cache_dir = lineage_cache / f"model_{model_k}"

        if model_k >= 1:
            probe_dict, filtered_layers = load_probe(trial, variant, model_k, frozen=frozen)
            end_metrics = score_supplemented_rows(
                model,
                tokenizer,
                probe_dict=probe_dict,
                filtered_layers=filtered_layers,
                rows=rows,
                cache_dir=cache_dir / "probe_end",
            )
            end_metrics["supplement"] = sup_meta
            round_results[str(model_k)]["end"] = end_metrics
            print(
                f"  END round{model_k}: dec={end_metrics['deceptive']:.4f} "
                f"hon={end_metrics['honest']:.4f} auc={end_metrics['auc']:.4f}"
            )

        if model_k < 5:
            probe_dict, filtered_layers = load_probe(trial, variant, model_k + 1, frozen=frozen)
            start_metrics = score_supplemented_rows(
                model,
                tokenizer,
                probe_dict=probe_dict,
                filtered_layers=filtered_layers,
                rows=rows,
                cache_dir=cache_dir / "probe_start",
            )
            start_metrics["supplement"] = sup_meta
            round_results[str(model_k + 1)]["start"] = start_metrics
            print(
                f"  START round{model_k + 1}: dec={start_metrics['deceptive']:.4f} "
                f"hon={start_metrics['honest']:.4f} auc={start_metrics['auc']:.4f}"
            )
            if (
                validate
                and native["deceptive"] >= 20
                and native["honest"] >= 20
                and sup_meta["deceptive_provenance"].get(f"round{model_k}" if model_k else "base", 0)
                == sup_meta["n_deceptive"]
                and sup_meta["honest_provenance"].get(f"round{model_k}" if model_k else "base", 0)
                == sup_meta["n_honest"]
            ):
                validate_against_probe_metrics(trial, variant, model_k + 1, start_metrics)

        unload_model(model)
        del tokenizer

    output = {
        "trial": trial,
        "variant": variant or "original",
        "min_class_count": 20,
        "deceptive_definition": DECEPTIVE_DEFINITION,
        "data_seed": DATA_SEED,
        "model_meta": model_meta,
        "rounds": round_results,
    }
    out_path = results_path(trial, variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, action="append", dest="trials")
    parser.add_argument("--variant", type=str, action="append", dest="variants")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    bootstrap_hf_env()
    targets = LINEAGES
    if args.trials or args.variants:
        trial_set = set(args.trials or [t for t, _ in LINEAGES])
        variant_set = set(args.variants if args.variants is not None else [v for _, v in LINEAGES])
        targets = [(t, v) for t, v in LINEAGES if t in trial_set and v in variant_set]

    for trial, variant in targets:
        run_lineage(trial, variant, validate=not args.no_validate)


if __name__ == "__main__":
    main()
