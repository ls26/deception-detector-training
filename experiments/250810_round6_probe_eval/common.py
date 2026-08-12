"""Shared config for round-6 probe monitorability eval (no GRPO)."""

from __future__ import annotations

from pathlib import Path

from experiments.supplemented_probe_eval.common import trial_dir

# Same lineages as supplemented_probe_eval, excluding frozen (no on-policy train gens).
ROUND6_LINEAGES: list[tuple[int, str]] = [
    (4, "v2"),
    (4, "early_retraining_v2"),
    (4, "early_retraining_auc_v2"),
    (6, ""),
    (6, "early_retraining"),
    (6, "early_retraining_auc_v3"),
    (8, ""),
    (8, "early_retraining"),
    (8, "early_retraining_auc_v3"),
]

EXPECTED_LAST_ROUND = 5
PROBE_ROUND = 6
MODEL_K = 5


def results_path(trial: int, variant: str) -> Path:
    suffix = f"_{variant}" if variant else ""
    return trial_dir(trial) / "results" / f"round6_probe_eval{suffix}.json"


def resolve_targets(
    *,
    trials: list[int] | None,
    variants: list[str] | None,
) -> list[tuple[int, str]]:
    if trials and variants:
        return [(t, v) for t in trials for v in variants]
    if trials:
        trial_set = set(trials)
        return [(t, v) for t, v in ROUND6_LINEAGES if t in trial_set]
    if variants:
        variant_set = set(variants)
        return [(t, v) for t, v in ROUND6_LINEAGES if v in variant_set]
    return ROUND6_LINEAGES
