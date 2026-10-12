"""Train the final probe of one lineage: one more generate-and-train round on the last
completed round's policy, with no GRPO after it, then score that probe on the policy's
supplemented eval completions.

Usage:
    python -m probe_eval.train_final_probe --trial 1 --variant adaptive_score

Needs the vLLM server, like a training round. Writes rounds_<variant>/round<last+1>/ and
experiments/trials/trial<N>/results/final_probe_<variant>.json. Not applicable to the frozen
variant, which trains no probes after round 1.
"""

from __future__ import annotations

import argparse
import json
import shutil

from training.common import DATA_SEED, DECEPTIVE_DEFINITION, bootstrap_hf_env
from training.round_config import RoundConfig
from training.run_trial import load_trial_config, load_trial_state, preflight, prepare_round_probe

from probe_eval.common import (
    build_supplemented_rows,
    final_probe_results_path,
    load_policy_model,
    load_probe,
    score_rows,
    unload_model,
)


def remove_partial_round(cfg: RoundConfig) -> None:
    """A previous run that crashed between writing the probe weights and the probe metrics
    leaves weights the pipeline would silently reuse; remove them so the probe is retrained."""
    if cfg.probe_metrics.exists():
        return
    for path in (cfg.probe_weights, cfg.probe_activations):
        if path.is_symlink():
            raise RuntimeError(f"{path} is a symlink; refusing to remove it")
        if path.exists():
            shutil.rmtree(path)
            print(f"OK: removed partial {path} from a previous failed run")


def run_lineage(trial: int, variant: str, *, force: bool) -> dict:
    """Train and score the final probe; returns the written payload."""
    label = f"trial{trial} {variant}"
    print(f"\n{'=' * 60}\nFinal probe: {label}\n{'=' * 60}")
    state = load_trial_state(trial, variant)
    trial_cfg = load_trial_config(trial, variant)
    if "frozen_probe_weights_dir" in trial_cfg:
        raise ValueError(f"{label}: the frozen variant has no probe to retrain")
    last_round = state["last_completed_round"]
    probe_round = last_round + 1
    cfg = RoundConfig(trial=trial, round=probe_round, previous_run_id=state["previous_run_id"], variant=variant)
    preflight(cfg)

    out_path = final_probe_results_path(trial, variant)
    if out_path.exists() and not force and cfg.probe_metrics.exists():
        print(f"SKIP: {out_path.name} and round{probe_round} probe_metrics already exist (--force to rerun)")
        return json.loads(out_path.read_text())

    remove_partial_round(cfg)
    prepare_round_probe(cfg, trial_cfg, force=force)
    probe_metrics = json.loads(cfg.probe_metrics.read_text())
    print(
        f"OK: round-{probe_round} probe on_dataset_auc={probe_metrics['on_dataset_auc']:.4f} "
        f"on_policy_split_auc={probe_metrics['on_policy_split_auc']:.4f}"
    )

    rows, supplement_meta = build_supplemented_rows(trial=trial, variant=variant, policy_round=last_round)
    print(f"  supplemented rows: deceptive={supplement_meta['n_deceptive']} honest={supplement_meta['n_honest']}")
    model, tokenizer = load_policy_model(state["round_grpo_ids"][str(last_round)])
    probe_dict, filtered_layers = load_probe(cfg.probe_weights)
    supplemented = score_rows(model, tokenizer, probe_dict=probe_dict, filtered_layers=filtered_layers, rows=rows)
    supplemented["supplement"] = supplement_meta
    print(
        f"  supplemented: deceptive={supplemented['deceptive']:.4f} "
        f"honest={supplemented['honest']:.4f} auc={supplemented['auc']:.4f}"
    )
    unload_model(model)
    del tokenizer

    payload = {
        "trial": trial,
        "variant": variant,
        "probe_round": probe_round,
        "policy_round": last_round,
        "final_model_run_id": state["previous_run_id"],
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
            "stats": probe_metrics["on_policy_split_stats"],
        },
        "supplemented_on_policy": supplemented,
        "probe_metrics_path": str(cfg.probe_metrics),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--variant", type=str, required=True, help="Lineage name (e.g. adaptive_score)")
    parser.add_argument("--force", action="store_true", help="Retrain the probe and rescore even if outputs exist")
    args = parser.parse_args()

    bootstrap_hf_env()
    run_lineage(args.trial, args.variant, force=args.force)


if __name__ == "__main__":
    main()
