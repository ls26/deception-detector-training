"""Shared config for the shared-eval-set 3-probe monitorability comparison.

Compares monitorability of the INITIAL (base) model against the FINAL (round-5)
model of each completed lineage, by scoring three probes on one shared on-policy
evaluation set built from every intermediate policy in the lineage:

  A  "final_round6"        round-6 probe (round-6 train set, round-5 activations)
  B  "initial_probe1"      round-1 probe (probe-1 train set, base activations)
  C  "initial_round6_data" new probe    (round-6 train set, base activations)

A and B already exist on disk; only C is trained here. A probe is a function of
activations, so "same evaluation set" means the same *texts*: A reads them through
the round-5 model, B and C through the base model. That is what makes the
initial-vs-final comparison meaningful -- each probe reads its own model's
representations.
"""

from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

from experiments.supplemented_probe_eval.common import rounds_dir, trial_dir

# Same lineages as the round-6 probe eval (experiments/250810_round6_probe_eval),
# excluding frozen: every one has round-6 probe weights + a round-5 checkpoint.
LINEAGES: list[tuple[int, str]] = [
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

N_PER_CLASS = 100
BUILD_SEED = 53
DATA_SEED = 53
DECEPTIVE_DEFINITION = "pass_only_shown"

# Deceptive on-policy anchor is det0, honest is the base model -- the base policy is
# deceptive on 1/400 eval prompts, so a base-anchored deceptive pool is unusable.
# Mirrors how build_train_dataset_split_sources() picks its per-class anchors.
DECEPTIVE_ANCHOR = "det0"
HONEST_ANCHOR = "base"

PROBE_A = "final_round6"
PROBE_B = "initial_probe1"
PROBE_C = "initial_round6_data"
PROBE_KEYS = (PROBE_A, PROBE_B, PROBE_C)

EXPERIMENT_DIR = Path(__file__).resolve().parent
SHARED_EVAL_DIR = EXPERIMENT_DIR / "shared_eval_sets"


def _suffix(variant: str) -> str:
    return f"_{variant}" if variant else ""


def label_for(trial: int, variant: str) -> str:
    return f"trial{trial} {variant or 'original'}"


def shared_eval_path(trial: int, variant: str) -> Path:
    return SHARED_EVAL_DIR / f"shared_eval_set_trial{trial}{_suffix(variant)}.jsonl"


def provenance_path(trial: int, variant: str) -> Path:
    return SHARED_EVAL_DIR / f"provenance_trial{trial}{_suffix(variant)}.json"


def round6_dir(trial: int, variant: str) -> Path:
    return rounds_dir(trial, variant) / f"round{PROBE_ROUND}"


def probe_a_weights(trial: int, variant: str) -> Path:
    return round6_dir(trial, variant) / "probe_weights"


def probe_b_weights(trial: int, variant: str) -> Path:
    return rounds_dir(trial, variant) / "round1" / "probe_weights"


# Probe C never shares a directory with probe A: train_probe_and_eval_on_dataset()
# silently reuses whatever weights it finds at weights_dir/layer_0/model.pt, so a
# shared path would either no-op or clobber the round-6 probe.
def probe_c_weights(trial: int, variant: str) -> Path:
    return round6_dir(trial, variant) / "probe_weights_base_model"


def probe_c_activations(trial: int, variant: str) -> Path:
    return round6_dir(trial, variant) / "probe_activations_base_model"


def probe_c_metrics(trial: int, variant: str) -> Path:
    return round6_dir(trial, variant) / "probe_metrics_base_model.json"


def round6_manifest(trial: int, variant: str) -> Path:
    return round6_dir(trial, variant) / "probe_train_dataset.json"


def results_path(trial: int, variant: str) -> Path:
    return trial_dir(trial) / "results" / f"shared_eval_probe_compare{_suffix(variant)}.json"


def summary_path() -> Path:
    return EXPERIMENT_DIR / "summary.json"


def build_rng(trial: int, variant: str, name: str) -> random.Random:
    """Stable per-(lineage, name) RNG (zlib.crc32, not Python's salted str.__hash__)."""
    key = f"trial{trial}|{variant}|{name}".encode()
    return random.Random(BUILD_SEED + trial * 1000 + (zlib.crc32(key) & 0xFFFFFFFF))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def resolve_targets(
    *,
    trials: list[int] | None,
    variants: list[str] | None,
) -> list[tuple[int, str]]:
    if trials and variants:
        return [(t, v) for t in trials for v in variants]
    if trials:
        trial_set = set(trials)
        return [(t, v) for t, v in LINEAGES if t in trial_set]
    if variants:
        variant_set = set(variants)
        return [(t, v) for t, v in LINEAGES if v in variant_set]
    return LINEAGES


def normalize_variant(variant: str) -> str:
    return "" if variant == "original" else variant
