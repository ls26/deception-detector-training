#!/usr/bin/env python3
"""Round-6 probe monitorability eval: train a fresh probe on the round-5 model, no GRPO."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/workspace")))

from obfuscation_atlas.detectors.train import load_model
from pipeline.grpo_launcher import check_disk, check_vllm
from pipeline.paths import WORKSPACE
from pipeline.probe_training import _load_filtered_layers, bootstrap_hf_env, load_final_policy_model
from pipeline.round_config import RoundConfig, resolve_checkpoint_final
from pipeline.run_trial import load_trial_config, preflight, run_single_round

from common import (
    EXPECTED_LAST_ROUND,
    MODEL_K,
    PROBE_ROUND,
    resolve_targets,
    results_path,
)
from experiments.supplemented_probe_eval.common import (
    DATA_SEED,
    DECEPTIVE_DEFINITION,
    build_supplemented_rows,
    load_trial_state,
)
from experiments.supplemented_probe_eval.run_eval import score_supplemented_rows


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_round5_model(trial: int, variant: str, state: dict):
    run_id = state["round_grpo_ids"][str(MODEL_K)]
    checkpoint_final = resolve_checkpoint_final(WORKSPACE, run_id, trial=trial)
    adapter_path = checkpoint_final / "lora"
    if not adapter_path.is_dir():
        adapter_path = checkpoint_final
    return load_final_policy_model(checkpoint_final=checkpoint_final, adapter_path=adapter_path)


def load_round6_probe(cfg: RoundConfig):
    weights_dir = cfg.probe_weights
    if not weights_dir.exists():
        raise FileNotFoundError(f"Missing round-{PROBE_ROUND} probe weights: {weights_dir}")
    probe_dict = load_model(str(weights_dir))
    filtered_layers = _load_filtered_layers(weights_dir)
    return probe_dict, filtered_layers


def build_results_payload(
    *,
    trial: int,
    variant: str,
    state: dict,
    probe_metrics: dict,
    supplemented: dict,
    supplement_meta: dict,
) -> dict:
    return {
        "trial": trial,
        "variant": variant or "original",
        "probe_round": PROBE_ROUND,
        "model_k": MODEL_K,
        "round5_run_id": state["previous_run_id"],
        "deceptive_definition": DECEPTIVE_DEFINITION,
        "data_seed": DATA_SEED,
        "on_dataset": {
            "auc": probe_metrics["on_dataset_auc"],
            "accuracy": probe_metrics["on_dataset_accuracy"],
            "f1": probe_metrics["on_dataset_f1"],
            "mean_scores": probe_metrics["on_dataset_mean_scores"],
        },
        "on_policy_split": {
            "auc": probe_metrics["on_policy_split_auc"],
            "accuracy": probe_metrics["on_policy_split_accuracy"],
            "f1": probe_metrics["on_policy_split_f1"],
            "mean_scores": probe_metrics["on_policy_split_mean_scores"],
            "stats": probe_metrics.get("on_policy_split_stats", {}),
        },
        "supplemented_on_policy": supplemented,
        "supplement_meta": supplement_meta,
        "probe_metrics_path": str(
            WORKSPACE
            / f"experiments/trials/trial{trial}"
            / (f"rounds_{variant}" if variant else "rounds")
            / f"round{PROBE_ROUND}"
            / "probe_metrics.json"
        ),
    }


def _cleanup_partial_round6(cfg: RoundConfig) -> None:
    """Remove incomplete probe artifacts if a prior run crashed mid-round."""
    if cfg.probe_metrics.exists():
        return
    for path in (cfg.probe_weights, cfg.probe_activations):
        if path.exists():
            import shutil

            shutil.rmtree(path)
            print(f"OK: removed partial {path} from prior failed run")


def run_lineage(trial: int, variant: str, *, force: bool = False) -> dict:
    label = f"trial{trial} {variant or 'original'}"
    print(f"\n{'=' * 60}")
    print(f"Round-6 probe eval: {label}")
    print(f"{'=' * 60}")

    state = load_trial_state(trial, variant)
    if state["last_completed_round"] != EXPECTED_LAST_ROUND:
        raise ValueError(
            f"{label}: expected last_completed_round={EXPECTED_LAST_ROUND}, "
            f"got {state['last_completed_round']}"
        )

    cfg = RoundConfig(
        trial=trial,
        round=PROBE_ROUND,
        previous_run_id=state["previous_run_id"],
        variant=variant,
    )
    trial_cfg = load_trial_config(trial, variant)
    if trial_cfg.get("frozen_probe_weights_dir"):
        raise ValueError(f"{label}: frozen variant is excluded from round-6 probe eval")

    preflight(cfg, state, trial_cfg)

    out_path = results_path(trial, variant)
    if out_path.exists() and not force and cfg.probe_metrics.exists():
        print(f"SKIP: {out_path.name} and round{PROBE_ROUND} probe_metrics already exist (--force to rerun)")
        return json.loads(out_path.read_text())

    _cleanup_partial_round6(cfg)
    run_single_round(cfg, dry_run=True, force=force)

    if not cfg.probe_metrics.exists():
        raise FileNotFoundError(f"Missing probe metrics after round-{PROBE_ROUND} pipeline: {cfg.probe_metrics}")

    probe_metrics = json.loads(cfg.probe_metrics.read_text())
    print(
        f"OK: round-{PROBE_ROUND} probe "
        f"on_dataset_auc={probe_metrics['on_dataset_auc']:.4f} "
        f"on_policy_split_auc={probe_metrics['on_policy_split_auc']:.4f}"
    )

    rows, supplement_meta = build_supplemented_rows(trial=trial, variant=variant, model_k=MODEL_K)
    print(
        f"  supplemented rows: deceptive={supplement_meta['n_deceptive']} "
        f"honest={supplement_meta['n_honest']}"
    )

    model, tokenizer = load_round5_model(trial, variant, state)
    probe_dict, filtered_layers = load_round6_probe(cfg)
    supplemented = score_supplemented_rows(
        model,
        tokenizer,
        probe_dict=probe_dict,
        filtered_layers=filtered_layers,
        rows=rows,
    )
    supplemented["supplement"] = supplement_meta
    print(
        f"  supplemented: dec={supplemented['deceptive']:.4f} "
        f"hon={supplemented['honest']:.4f} auc={supplemented['auc']:.4f}"
    )
    unload_model(model)
    del tokenizer

    payload = build_results_payload(
        trial=trial,
        variant=variant,
        state=state,
        probe_metrics=probe_metrics,
        supplemented=supplemented,
        supplement_meta=supplement_meta,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_path}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _normalize_variant(variant: str) -> str:
    return "" if variant == "original" else variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-6 probe monitorability eval (no GRPO)")
    parser.add_argument("--trial", type=int, action="append", dest="trials")
    parser.add_argument("--variant", type=str, action="append", dest="variants")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    args = parser.parse_args()

    bootstrap_hf_env()
    check_vllm()
    check_disk(min_free_gb=50, path=WORKSPACE)

    variants = [_normalize_variant(v) for v in args.variants] if args.variants else None
    targets = resolve_targets(trials=args.trials, variants=variants)
    print(f"Targets ({len(targets)}): {targets}")

    results: list[dict] = []
    for trial, variant in targets:
        results.append(run_lineage(trial, variant, force=args.force))

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for row in results:
        t, v = row["trial"], row["variant"]
        print(
            f"  trial{t} {v:30s}  "
            f"on_ds={row['on_dataset']['auc']:.4f}  "
            f"split={row['on_policy_split']['auc']:.4f}  "
            f"supp={row['supplemented_on_policy']['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
