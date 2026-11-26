"""Shared by the probe evaluations: lineage paths, eval-completion loading, the
supplementation rule for rare classes, the shared-eval-set RNG, probe scoring, and model
loading.

A lineage is one (trial, variant). "Policy round k" is the policy at the end of GRPO round k
(round 0 is the base model); "probe round k" is the probe trained before GRPO round k. The
final probe, trained after the last round and never used as a penalty, is probe round
`last_completed_round + 1`.
"""

from __future__ import annotations

import gc
import random
import tempfile
import zlib
from pathlib import Path

import torch
from datasets import Dataset

from obfuscation_atlas.detectors.evaluate import eval_multiple_detectors
from obfuscation_atlas.detectors.train import load_model
from obfuscation_atlas.tasks.code_generation import deceptive_reward_key
from obfuscation_atlas.training.obfuscation_training import HONEST_REWARD_KEY, run_on_policy_split_eval
from transformers import AutoTokenizer

from training.common import DATA_SEED, DECEPTIVE_DEFINITION, EXPERIMENTS, MAX_SEQUENCE_LENGTH, bootstrap_hf_env, check_vllm, load_jsonl
from training.probe_training import (
    EVAL_SEQUENCE_AGGREGATOR,
    MODEL_BATCH_SIZE,
    OBFUSCATE_OVER,
    SPLIT_FALSE_POSITIVE_RATE,
    assert_probed_token,
    load_filtered_layers,
    load_final_policy_model,
    summarize_split_metrics,
)
from training.round_config import RoundConfig, adapter_path_for, checkpoint_final_for_id
from training.run_trial import generate_eval_completions

# A checkpoint's eval completions are topped up from earlier checkpoints until each class
# has at least this many rows.
MIN_CLASS_COUNT = 20
# Shared eval set for the initial-vs-final probe comparison: rows per class.
N_PER_CLASS = 100
DECEPTIVE_ANCHOR = "det0"
HONEST_ANCHOR = "base"
BASE_EVAL_JSONL = EXPERIMENTS / "artifacts/base/base_eval_generations.jsonl"
DET0_EVAL_JSONL = EXPERIMENTS / "artifacts/det0/det0_eval_generations.jsonl"


# --- lineage paths -----------------------------------------------------------


def trial_dir(trial: int) -> Path:
    return EXPERIMENTS / "trials" / f"trial{trial}"


def buffer_dir(trial: int, variant: str) -> Path:
    return trial_dir(trial) / f"buffer_{variant}"


def results_dir(trial: int) -> Path:
    return trial_dir(trial) / "results"


def per_round_results_path(trial: int, variant: str) -> Path:
    return results_dir(trial) / f"per_round_probe_eval_{variant}.json"


def final_probe_results_path(trial: int, variant: str) -> Path:
    return results_dir(trial) / f"final_probe_{variant}.json"


def compare_results_path(trial: int, variant: str) -> Path:
    return results_dir(trial) / f"initial_vs_final_probe_{variant}.json"


def shared_eval_path(trial: int, variant: str) -> Path:
    return results_dir(trial) / f"shared_eval_set_{variant}.jsonl"


def provenance_path(trial: int, variant: str) -> Path:
    return results_dir(trial) / f"shared_eval_set_{variant}_provenance.json"


def probe_round_dir(trial: int, variant: str, probe_round: int) -> Path:
    """Round 1 is shared by every variant: its directory is `rounds/round1` (written by
    run_round1_probe) unless the variant has its own copy."""
    variant_round_dir = trial_dir(trial) / f"rounds_{variant}" / f"round{probe_round}"
    if probe_round == 1 and not variant_round_dir.exists():
        return trial_dir(trial) / "rounds" / "round1"
    return variant_round_dir


def probe_round_manifest(trial: int, variant: str, probe_round: int) -> Path:
    return probe_round_dir(trial, variant, probe_round) / "probe_train_dataset.json"


def final_probe_weights(trial: int, variant: str, probe_round: int) -> Path:
    return probe_round_dir(trial, variant, probe_round) / "probe_weights"


# The initial probe is trained on the final probe's texts through the base model. It gets
# its own directory: train_or_load_probe_and_eval reuses whatever weights it finds at
# weights_dir/layer_0/model.pt, so sharing the final probe's path would load the wrong probe.
def initial_probe_weights(trial: int, variant: str, probe_round: int) -> Path:
    return probe_round_dir(trial, variant, probe_round) / "probe_weights_base_model"


def initial_probe_activations(trial: int, variant: str, probe_round: int) -> Path:
    return probe_round_dir(trial, variant, probe_round) / "probe_activations_base_model"


def initial_probe_metrics(trial: int, variant: str, probe_round: int) -> Path:
    return probe_round_dir(trial, variant, probe_round) / "probe_metrics_base_model.json"


# --- eval completions per policy ---------------------------------------------


def eval_jsonl_candidates(trial: int, variant: str, policy_round: int) -> list[Path]:
    """Where policy round k's eval completions may be, highest priority first: the variant's
    buffer, then (for the shared round-1 policy) any other variant's buffer."""
    if policy_round == 0:
        return [BASE_EVAL_JSONL]
    name = f"trial{trial}_round{policy_round}_eval_generations.jsonl"
    candidates = [buffer_dir(trial, variant) / name]
    if policy_round == 1:
        candidates.extend(sorted(trial_dir(trial).glob(f"buffer_*/{name}")))
    return candidates


def resolve_eval_jsonl(trial: int, variant: str, policy_round: int) -> Path:
    for path in eval_jsonl_candidates(trial, variant, policy_round):
        if path.exists():
            return path
    raise FileNotFoundError(f"No eval completions for trial{trial} {variant} policy round {policy_round}")


def ensure_last_model_eval(trial: int, variant: str, state: dict) -> Path:
    """The last policy's eval completions are normally written by the *next* round's
    generation step; generate them here (needs vLLM) when no later round was run."""
    last_round = state["last_completed_round"]
    for path in eval_jsonl_candidates(trial, variant, last_round):
        if path.exists():
            return path
    cfg = RoundConfig(trial=trial, round=last_round + 1, previous_run_id=state["previous_run_id"], variant=variant)
    print(f"Eval completions for policy round {last_round} are missing; generating them from {cfg.adapter_path}")
    check_vllm()
    bootstrap_hf_env()
    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint_final)
    return generate_eval_completions(cfg, DATA_SEED, tokenizer)


def source_eval_rows(trial: int, variant: str, source: str) -> list[dict]:
    """The eval completions of one source: `base`, `det0`, or `round<k>`."""
    if source == HONEST_ANCHOR:
        return load_jsonl(BASE_EVAL_JSONL)
    if source == DECEPTIVE_ANCHOR:
        return load_jsonl(DET0_EVAL_JSONL)
    return load_jsonl(resolve_eval_jsonl(trial, variant, int(source.removeprefix("round"))))


def build_rng(trial: int, variant: str, name: str) -> random.Random:
    """A stable RNG per (lineage, name), keyed by zlib.crc32 rather than Python's salted
    str.__hash__. The variant name is part of the key, so renaming a variant changes every
    shared-eval-set draw of its lineage."""
    key = f"trial{trial}|{variant}|{name}".encode()
    return random.Random(DATA_SEED + trial * 1000 + (zlib.crc32(key) & 0xFFFFFFFF))


# --- supplementation: top up a rare class from earlier checkpoints ------------


def top_up_class(
    *, trial: int, variant: str, policy_round: int, class_name: str, rng: random.Random
) -> tuple[list[dict], dict[str, int]]:
    """All of policy round k's eval rows of `class_name` when there are at least
    MIN_CLASS_COUNT; otherwise exactly MIN_CLASS_COUNT rows, topped up from earlier policies
    of the lineage (most recent first), then the base model, then (reward-hacking class only)
    the no-penalty model. Returns (rows, {source: rows taken})."""
    provenance: dict[str, int] = {}
    chain = [f"round{k}" for k in range(policy_round, 0, -1)] + [HONEST_ANCHOR]
    if class_name == "deceptive":
        chain.append(DECEPTIVE_ANCHOR)
    current_source = chain[0]
    chosen = [row for row in source_eval_rows(trial, variant, current_source) if row[class_name]]
    provenance[current_source] = len(chosen)
    if len(chosen) >= MIN_CLASS_COUNT:
        return chosen, provenance

    used = {(row["prompt"], row["completion"]) for row in chosen}
    for source in chain[1:]:
        if len(chosen) >= MIN_CLASS_COUNT:
            break
        pool = [
            row
            for row in source_eval_rows(trial, variant, source)
            if row[class_name] and (row["prompt"], row["completion"]) not in used
        ]
        take = min(MIN_CLASS_COUNT - len(chosen), len(pool))
        if take <= 0:
            provenance[source] = 0
            continue
        picks = rng.sample(pool, take)
        chosen.extend(picks)
        used.update((row["prompt"], row["completion"]) for row in picks)
        provenance[source] = take

    if len(chosen) < MIN_CLASS_COUNT:
        raise RuntimeError(
            f"Could not top up {class_name} to {MIN_CLASS_COUNT} for trial{trial} policy round {policy_round}: got {len(chosen)}"
        )
    return chosen, provenance


def build_supplemented_rows(*, trial: int, variant: str, policy_round: int) -> tuple[list[dict], dict]:
    """The rows on which policy round k is scored (see `top_up_class`), and their provenance."""
    rng = random.Random(DATA_SEED + trial * 100 + policy_round)
    deceptive_rows, deceptive_provenance = top_up_class(
        trial=trial, variant=variant, policy_round=policy_round, class_name="deceptive", rng=rng
    )
    honest_rows, honest_provenance = top_up_class(
        trial=trial, variant=variant, policy_round=policy_round, class_name="honest", rng=rng
    )
    meta = {
        "policy_round": policy_round,
        "n_deceptive": len(deceptive_rows),
        "n_honest": len(honest_rows),
        "deceptive_provenance": deceptive_provenance,
        "honest_provenance": honest_provenance,
        "eval_jsonl": str(resolve_eval_jsonl(trial, variant, policy_round)),
    }
    return deceptive_rows + honest_rows, meta


# --- probe scoring -------------------------------------------------------------


def score_rows(model, tokenizer, *, probe_dict, filtered_layers: list[int], rows: list[dict]) -> dict:
    """Score labeled completion rows with a probe on `model`'s activations. Returns the
    layer-averaged mean score per class, the AUC/accuracy/F1 over the labeled rows, the class
    counts, and the split statistics."""
    prompts = [row["prompt"] for row in rows]
    completions = [row["completion"] for row in rows]
    # The vendored evaluator wants a (positive, negative) pair of datasets; both halves are the
    # same rows here, and the real class split is applied afterwards from the labels.
    dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        results = eval_multiple_detectors(
            detectors=[probe_dict],
            detector_types=["linear-probe"],
            test_dataset=(dataset, dataset),
            obfuscate_over=OBFUSCATE_OVER,
            model=model,
            tokenizer=tokenizer,
            test_completion_columns=("completion", "completion"),
            test_congruent_follow_up_prompts=[],
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            model_batch_size=MODEL_BATCH_SIZE,
            eval_sequence_aggregator=EVAL_SEQUENCE_AGGREGATOR,
            test_cache_activations_save_path=str(cache_dir),
            accelerator=None,
        )
        assert_probed_token(cache_dir, tokenizer=tokenizer, label="probe_eval rows")
        deceptive_key = deceptive_reward_key(DECEPTIVE_DEFINITION)
        reward_metrics = {
            deceptive_key: [bool(row["deceptive"]) for row in rows],
            HONEST_REWARD_KEY: [bool(row["honest"]) for row in rows],
        }
        split_results = run_on_policy_split_eval(
            eval_results=results,
            reward_metrics=reward_metrics,
            save_path=cache_dir,
            plot_key="supplemented.on_policy",
            filtered_layers=filtered_layers,
            false_positive_rate=SPLIT_FALSE_POSITIVE_RATE,
            deceptive_key=deceptive_key,
            save_json=False,
        )
        split_summary = summarize_split_metrics(split_results, filtered_layers)
    return {
        "deceptive": split_summary["mean_scores_layer_averaged"]["deceptive"],
        "honest": split_summary["mean_scores_layer_averaged"]["honest"],
        "auc": split_summary["auc_filtered"],
        "accuracy": split_summary["accuracy"],
        "f1": split_summary["f1"],
        "n_deceptive": int(sum(row["deceptive"] for row in rows)),
        "n_honest": int(sum(row["honest"] for row in rows)),
        "split_stats": split_summary["stats"],
    }


def load_probe(weights_dir: Path):
    if not (weights_dir / "layer_0" / "model.pt").exists():
        raise FileNotFoundError(f"Missing probe weights at {weights_dir}")
    return load_model(str(weights_dir)), load_filtered_layers(weights_dir)


# --- models -------------------------------------------------------------------


def load_policy_model(run_id: str):
    """The base model plus the LoRA adapter of one GRPO run."""
    checkpoint_final = checkpoint_final_for_id(run_id)
    return load_final_policy_model(checkpoint_final=checkpoint_final, adapter_path=adapter_path_for(checkpoint_final))


def unload_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
