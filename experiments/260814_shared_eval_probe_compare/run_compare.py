#!/usr/bin/env python3
"""Score all three probes on each lineage's shared on-policy evaluation set.

Two passes, so each model is loaded once rather than once per probe:

  pass 1  base model    -> probe B (initial_probe1) and probe C (initial_round6_data)
  pass 2  round-5 model -> probe A (final_round6)

Scoring goes through `score_supplemented_rows()` unchanged, so the AUC / split-metric
machinery is identical to the round-6 probe eval. Per-lineage results are written
incrementally, so a crash in pass 2 doesn't lose pass 1.
"""

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
from pipeline.paths import WORKSPACE
from pipeline.probe_training import _load_filtered_layers, bootstrap_hf_env, load_final_policy_model
from pipeline.round_config import resolve_checkpoint_final

from experiments.supplemented_probe_eval.common import load_trial_state
from experiments.supplemented_probe_eval.run_eval import score_supplemented_rows

from common import (
    DATA_SEED,
    DECEPTIVE_DEFINITION,
    EXPECTED_LAST_ROUND,
    MODEL_K,
    PROBE_A,
    PROBE_B,
    PROBE_C,
    PROBE_ROUND,
    label_for,
    load_jsonl,
    normalize_variant,
    probe_a_weights,
    probe_b_weights,
    probe_c_weights,
    provenance_path,
    resolve_targets,
    results_path,
    shared_eval_path,
)
from train_probe_c import load_base_model

PROBE_SPECS = {
    PROBE_A: {"weights": probe_a_weights, "activations_model": "round5"},
    PROBE_B: {"weights": probe_b_weights, "activations_model": "base"},
    PROBE_C: {"weights": probe_c_weights, "activations_model": "base"},
}


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_probe(weights_dir: Path, label: str):
    if not (weights_dir / "layer_0" / "model.pt").exists():
        raise FileNotFoundError(f"{label}: missing probe weights at {weights_dir}")
    return load_model(str(weights_dir)), _load_filtered_layers(weights_dir)


def load_round5_model(trial: int, state: dict):
    run_id = state["round_grpo_ids"][str(MODEL_K)]
    checkpoint_final = resolve_checkpoint_final(WORKSPACE, run_id, trial=trial)
    adapter_path = checkpoint_final / "lora"
    if not adapter_path.is_dir():
        adapter_path = checkpoint_final
    return load_final_policy_model(checkpoint_final=checkpoint_final, adapter_path=adapter_path)


def load_rows(trial: int, variant: str) -> list[dict]:
    path = shared_eval_path(trial, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing shared eval set: {path} (run build_shared_eval.py first)")
    rows = load_jsonl(path)
    n_dec = sum(bool(r["deceptive"]) for r in rows)
    n_hon = sum(bool(r["honest"]) for r in rows)
    if n_dec + n_hon != len(rows):
        raise ValueError(f"{path}: {len(rows) - n_dec - n_hon} row(s) are neither deceptive nor honest")
    return rows


def result_skeleton(trial: int, variant: str, state: dict) -> dict:
    meta = json.loads(provenance_path(trial, variant).read_text())
    return {
        "trial": trial,
        "variant": variant or "original",
        "probe_round": PROBE_ROUND,
        "model_k": MODEL_K,
        "round5_run_id": state["round_grpo_ids"][str(MODEL_K)],
        "deceptive_definition": DECEPTIVE_DEFINITION,
        "data_seed": DATA_SEED,
        "shared_eval": meta,
        "probes": {},
    }


def load_or_init_result(trial: int, variant: str, state: dict, *, force: bool) -> dict:
    path = results_path(trial, variant)
    if path.exists() and not force:
        data = json.loads(path.read_text())
        data.setdefault("probes", {})
        return data
    return result_skeleton(trial, variant, state)


def save_result(trial: int, variant: str, data: dict) -> Path:
    path = results_path(trial, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def score_probe(model, tokenizer, *, probe_key: str, trial: int, variant: str, rows: list[dict]) -> dict:
    spec = PROBE_SPECS[probe_key]
    weights_dir = spec["weights"](trial, variant)
    probe_dict, filtered_layers = load_probe(weights_dir, f"{label_for(trial, variant)} {probe_key}")
    scores = score_supplemented_rows(
        model,
        tokenizer,
        probe_dict=probe_dict,
        filtered_layers=filtered_layers,
        rows=rows,
    )
    scores["probe"] = probe_key
    scores["weights_dir"] = str(weights_dir)
    scores["activations_model"] = spec["activations_model"]
    scores["filtered_layers"] = filtered_layers
    scores["n_filtered_layers"] = len(filtered_layers)
    print(
        f"  {probe_key:22s} auc={scores['auc']:.4f}  dec={scores['deceptive']:.4f}  "
        f"hon={scores['honest']:.4f}  layers={len(filtered_layers)}"
    )
    del probe_dict
    gc.collect()
    return scores


def prepare(targets: list[tuple[int, str]], *, force: bool) -> dict[tuple[int, str], dict]:
    prepared: dict[tuple[int, str], dict] = {}
    for trial, variant in targets:
        label = label_for(trial, variant)
        state = load_trial_state(trial, variant)
        if state["last_completed_round"] != EXPECTED_LAST_ROUND:
            raise ValueError(
                f"{label}: expected last_completed_round={EXPECTED_LAST_ROUND}, "
                f"got {state['last_completed_round']}"
            )
        prepared[(trial, variant)] = {
            "state": state,
            "rows": load_rows(trial, variant),
            "result": load_or_init_result(trial, variant, state, force=force),
        }
        meta = prepared[(trial, variant)]["result"]["shared_eval"]
        print(
            f"OK {label}: {len(prepared[(trial, variant)]['rows'])} shared eval rows "
            f"(dec={meta['n_deceptive']}, hon={meta['n_honest']})"
        )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Score 3 probes on the shared on-policy eval set")
    parser.add_argument("--trial", type=int, action="append", dest="trials")
    parser.add_argument("--variant", type=str, action="append", dest="variants")
    parser.add_argument("--force", action="store_true", help="Re-score probes already in the result file")
    args = parser.parse_args()

    bootstrap_hf_env()
    variants = [normalize_variant(v) for v in args.variants] if args.variants else None
    targets = resolve_targets(trials=args.trials, variants=variants)
    print(f"Targets ({len(targets)}): {targets}\n")

    prepared = prepare(targets, force=args.force)

    # Pass 1 -- base model, probes B and C.
    base_probes = [PROBE_B, PROBE_C]
    needs_base = [
        key for key in targets
        if any(p not in prepared[key]["result"]["probes"] for p in base_probes)
    ]
    if needs_base:
        print(f"\n{'=' * 60}\nPass 1: base model -- {PROBE_B}, {PROBE_C}\n{'=' * 60}")
        model, tokenizer = load_base_model()
        for trial, variant in needs_base:
            entry = prepared[(trial, variant)]
            print(f"\n{label_for(trial, variant)}")
            for probe_key in base_probes:
                if probe_key in entry["result"]["probes"]:
                    print(f"  {probe_key:22s} cached")
                    continue
                entry["result"]["probes"][probe_key] = score_probe(
                    model, tokenizer, probe_key=probe_key, trial=trial, variant=variant, rows=entry["rows"]
                )
                save_result(trial, variant, entry["result"])
        unload(model)
        del tokenizer
    else:
        print("Pass 1: nothing to score (all base-model probes cached)")

    # Pass 2 -- one round-5 model per lineage, probe A.
    needs_round5 = [key for key in targets if PROBE_A not in prepared[key]["result"]["probes"]]
    if needs_round5:
        print(f"\n{'=' * 60}\nPass 2: round-5 models -- {PROBE_A}\n{'=' * 60}")
        for trial, variant in needs_round5:
            entry = prepared[(trial, variant)]
            print(f"\n{label_for(trial, variant)}")
            model, tokenizer = load_round5_model(trial, entry["state"])
            entry["result"]["probes"][PROBE_A] = score_probe(
                model, tokenizer, probe_key=PROBE_A, trial=trial, variant=variant, rows=entry["rows"]
            )
            save_result(trial, variant, entry["result"])
            unload(model)
            del tokenizer
    else:
        print("Pass 2: nothing to score (all round-5 probes cached)")

    print(f"\n{'=' * 60}\nDone\n{'=' * 60}")
    for trial, variant in targets:
        path = save_result(trial, variant, prepared[(trial, variant)]["result"])
        probes = prepared[(trial, variant)]["result"]["probes"]
        aucs = "  ".join(f"{k}={probes[k]['auc']:.4f}" for k in (PROBE_A, PROBE_B, PROBE_C) if k in probes)
        print(f"  {label_for(trial, variant):32s} {aucs}")
        print(f"      -> {path}")


if __name__ == "__main__":
    main()
