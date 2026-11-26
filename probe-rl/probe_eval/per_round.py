"""Per-round probe metrics for one lineage: each round's probe scored on the policy's own
eval completions at the start of the round (the policy the probe was trained on) and at the
end (the policy GRPO produced against it), with rare classes topped up from earlier
checkpoints.

Usage:
    python -m probe_eval.per_round --trial 1 --variant frozen

Writes experiments/trials/trial<N>/results/per_round_probe_eval_<variant>.json.
"""

from __future__ import annotations

import argparse
import json

from training.common import DATA_SEED, DECEPTIVE_DEFINITION, bootstrap_hf_env
from training.probe_training import load_base_model
from training.run_trial import load_trial_config, load_trial_state

from probe_eval.common import (
    MIN_CLASS_COUNT,
    build_supplemented_rows,
    ensure_last_model_eval,
    load_policy_model,
    load_probe,
    per_round_results_path,
    probe_round_dir,
    score_rows,
    source_eval_rows,
    unload_model,
)

# Tolerance when comparing a start-of-round score with the training pipeline's own number.
VALIDATION_TOLERANCE = 0.02


def load_round_probe(trial: int, variant: str, probe_round: int, *, frozen: bool):
    weights_round = 1 if frozen else probe_round
    return load_probe(probe_round_dir(trial, variant, weights_round) / "probe_weights")


def load_policy(trial: int, policy_round: int, state: dict):
    if policy_round == 0:
        return load_base_model()
    return load_policy_model(state["round_grpo_ids"][str(policy_round)])


def native_class_counts(trial: int, variant: str, policy_round: int) -> dict[str, int]:
    rows = source_eval_rows(trial, variant, "base" if policy_round == 0 else f"round{policy_round}")
    return {"deceptive": sum(row["deceptive"] for row in rows), "honest": sum(row["honest"] for row in rows)}


def validate_against_probe_metrics(trial: int, variant: str, probe_round: int, start_metrics: dict) -> None:
    """Compare the start-of-round numbers with probe_metrics.json, which the training
    pipeline computed from the same probe and completions (frozen rounds have none)."""
    probe_metrics_path = probe_round_dir(trial, variant, probe_round) / "probe_metrics.json"
    if not probe_metrics_path.exists():
        return
    reference = json.loads(probe_metrics_path.read_text())
    comparisons = {
        "deceptive": reference["on_policy_split_mean_scores"]["deceptive"],
        "honest": reference["on_policy_split_mean_scores"]["honest"],
        "auc": reference["on_policy_split_auc"],
    }
    for key, reference_value in comparisons.items():
        status = "WARN" if abs(reference_value - start_metrics[key]) > VALIDATION_TOLERANCE else "OK"
        print(f"  {status} validate round{probe_round} start {key}: ref={reference_value:.4f} got={start_metrics[key]:.4f}")


def run_lineage(trial: int, variant: str) -> dict:
    """Score every round's probe at its start and end; returns the written payload."""
    frozen = "frozen_probe_weights_dir" in load_trial_config(trial, variant)
    state = load_trial_state(trial, variant)
    num_rounds = state["last_completed_round"]
    ensure_last_model_eval(trial, variant, state)

    round_results: dict[str, dict] = {str(round_number): {} for round_number in range(1, num_rounds + 1)}
    policy_meta: dict[str, dict] = {}

    for policy_round in range(0, num_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"trial{trial} {variant}: policy round {policy_round}")
        rows, supplement_meta = build_supplemented_rows(trial=trial, variant=variant, policy_round=policy_round)
        policy_meta[str(policy_round)] = supplement_meta
        native = native_class_counts(trial, variant, policy_round)
        print(f"  native counts: deceptive={native['deceptive']} honest={native['honest']}")
        print(f"  supplemented: deceptive={supplement_meta['n_deceptive']} honest={supplement_meta['n_honest']}")
        print(
            f"  provenance: deceptive={supplement_meta['deceptive_provenance']} "
            f"honest={supplement_meta['honest_provenance']}"
        )

        model, tokenizer = load_policy(trial, policy_round, state)

        if policy_round >= 1:
            probe_dict, filtered_layers = load_round_probe(trial, variant, policy_round, frozen=frozen)
            end_metrics = score_rows(model, tokenizer, probe_dict=probe_dict, filtered_layers=filtered_layers, rows=rows)
            end_metrics["supplement"] = supplement_meta
            round_results[str(policy_round)]["end"] = end_metrics
            print(
                f"  END round{policy_round}: deceptive={end_metrics['deceptive']:.4f} "
                f"honest={end_metrics['honest']:.4f} auc={end_metrics['auc']:.4f}"
            )

        if policy_round < num_rounds:
            probe_dict, filtered_layers = load_round_probe(trial, variant, policy_round + 1, frozen=frozen)
            start_metrics = score_rows(model, tokenizer, probe_dict=probe_dict, filtered_layers=filtered_layers, rows=rows)
            start_metrics["supplement"] = supplement_meta
            round_results[str(policy_round + 1)]["start"] = start_metrics
            print(
                f"  START round{policy_round + 1}: deceptive={start_metrics['deceptive']:.4f} "
                f"honest={start_metrics['honest']:.4f} auc={start_metrics['auc']:.4f}"
            )
            # The training pipeline scored exactly these rows only when nothing was topped up.
            current_source = f"round{policy_round}" if policy_round else "base"
            not_supplemented = (
                supplement_meta["deceptive_provenance"].get(current_source, 0) == supplement_meta["n_deceptive"]
                and supplement_meta["honest_provenance"].get(current_source, 0) == supplement_meta["n_honest"]
            )
            if native["deceptive"] >= MIN_CLASS_COUNT and native["honest"] >= MIN_CLASS_COUNT and not_supplemented:
                validate_against_probe_metrics(trial, variant, policy_round + 1, start_metrics)

        unload_model(model)
        del tokenizer

    payload = {
        "trial": trial,
        "variant": variant,
        "min_class_count": MIN_CLASS_COUNT,
        "deceptive_definition": DECEPTIVE_DEFINITION,
        "data_seed": DATA_SEED,
        "policy_meta": policy_meta,
        "rounds": round_results,
    }
    out_path = per_round_results_path(trial, variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--variant", type=str, required=True, help="Lineage name (e.g. frozen, adaptive_score)")
    args = parser.parse_args()

    bootstrap_hf_env()
    run_lineage(args.trial, args.variant)


if __name__ == "__main__":
    main()
