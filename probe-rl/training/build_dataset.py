"""Probe training sets: fixed on-dataset examples plus labeled on-policy completions.

Every probe trains on `N_FIXED_PER_CLASS` on-dataset examples per class (programmatic
hardcodes vs. human-written solutions from the MBPP probe split) plus on-policy completions
drawn from the labeled JSONL files of the trial's buffer. The manifest written next to each
probe lists every selected row and where it came from.
"""

from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

from datasets import Dataset

from obfuscation_atlas.config import CodeGenTaskConfig, MBPPProbeConfig, MBPPTaskConfig
from obfuscation_atlas.training.obfuscation_training import prepare_dataset

from training.common import DATA_SEED, EVAL_SAMPLES, MAX_SEQUENCE_LENGTH, load_jsonl

N_FIXED_PER_CLASS = 200
ON_POLICY_ROUND1_MIXED = 100
N_ROUND1_MIXED_PER_CLASS = N_FIXED_PER_CLASS + ON_POLICY_ROUND1_MIXED


def prepare_mbpp_datasets(tokenizer, *, seed: int):
    """The vendored MBPP splits: (task train sets, (probe_pos, probe_neg), (val_pos, val_neg))."""
    task_config = CodeGenTaskConfig()
    task_config.data.seed = seed
    task_config.data.probe_dataset = MBPPProbeConfig()
    task_config.data.task_dataset = MBPPTaskConfig()
    return prepare_dataset(task_config.data, tokenizer=tokenizer)


def load_probe_prompts(tokenizer, *, seed: int):
    """The probe-training split: `N_FIXED_PER_CLASS` prompts, each with a hardcode and a solution."""
    _, (probe_pos, probe_neg), _ = prepare_mbpp_datasets(tokenizer, seed=seed)
    assert len(probe_pos) == len(probe_neg) == N_FIXED_PER_CLASS
    assert probe_pos["task_id"] == probe_neg["task_id"]
    return probe_pos, probe_neg


def load_eval_task_dataset(tokenizer, *, seed: int):
    """The held-out split: `EVAL_SAMPLES` prompts, disjoint from the probe-training split."""
    _, _, (val_pos, val_neg) = prepare_mbpp_datasets(tokenizer, seed=seed)
    assert len(val_pos) == len(val_neg) == EVAL_SAMPLES
    return val_pos, val_neg


def _source_rng(seed: int, name: str) -> random.Random:
    """Stable per-source RNG (zlib.crc32, not Python's salted str.__hash__)."""
    return random.Random(seed + (zlib.crc32(name.encode()) & 0xFFFFFFFF))


def rows_to_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [row["prompt"] for row in rows],
            "completion": [row["completion"] for row in rows],
            "task_id": [row["task_id"] for row in rows],
            "source_model": [row["model_key"] if "model_key" in row else row["source_model"] for row in rows],
        }
    )


def _resolve_split_sources(
    round_number: int, trial: int, buffer_dir: Path, *, anchor_name: str, anchor_path: Path
) -> list[tuple[str, Path]]:
    """Anchor source (base_model or det0) + all previous rounds' on-policy JSONLs."""
    if not anchor_path.exists():
        raise FileNotFoundError(f"Missing {anchor_path}")
    sources: list[tuple[str, Path]] = [(anchor_name, anchor_path)]
    for previous_round in range(1, round_number):
        path = buffer_dir / f"trial{trial}_round{previous_round}_train_generations.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        sources.append((f"round{previous_round}", path))
    return sources


def _label_pool(rows: list[dict], label: str) -> list[dict]:
    return [row for row in rows if row.get(label)]


def _on_dataset_pool(probe_rows, *, label: str) -> list[dict]:
    completion_column = "hardcode" if label == "deceptive" else "code"
    source_model = "dataset_hardcode" if label == "deceptive" else "dataset_code"
    return [
        {
            "prompt": row["prompt"],
            "completion": row[completion_column],
            "task_id": row["task_id"],
            "source": "on_dataset",
            "source_model": source_model,
        }
        for row in probe_rows
    ]


def _fits_length(row: dict, tokenizer) -> bool:
    text = row["prompt"] + row["completion"]
    return len(tokenizer.encode(text, add_special_tokens=False)) <= MAX_SEQUENCE_LENGTH


def _take_from_pool(pool: list[dict], n: int, rng: random.Random, tokenizer, *, tag_on_policy: bool) -> list[dict]:
    """Up to `n` rows from a shuffled pool that fit the sequence length."""
    shuffled = list(pool)
    rng.shuffle(shuffled)
    taken: list[dict] = []
    for row in shuffled:
        if len(taken) >= n:
            break
        out = dict(row)
        if tag_on_policy:
            out["source"] = "on_policy"
        if _fits_length(out, tokenizer):
            taken.append(out)
    return taken


def _select_fixed_on_dataset(probe_rows, *, label: str, seed: int, tokenizer) -> tuple[list[dict], dict[str, int]]:
    pool = _on_dataset_pool(probe_rows, label=label)
    taken = _take_from_pool(pool, N_FIXED_PER_CLASS, random.Random(seed), tokenizer, tag_on_policy=False)
    if len(taken) < N_FIXED_PER_CLASS:
        raise ValueError(
            f"Cannot select {N_FIXED_PER_CLASS} fixed {label} examples: got {len(taken)}. Pool size={len(pool)}"
        )
    return taken, {"on_dataset": len(taken)}


def _select_on_policy_from_jsonl(rows: list[dict], *, label: str, n: int, seed: int, tokenizer) -> list[dict]:
    pool = _label_pool(rows, label)
    taken = _take_from_pool(pool, n, random.Random(seed), tokenizer, tag_on_policy=True)
    if len(taken) < n:
        raise ValueError(f"Cannot select {n} on-policy {label} examples: got {len(taken)}. Pool size={len(pool)}")
    for row in taken:
        row["source_model"] = row["model_key"]
    return taken


def _take_next_valid(
    source_name: str, pool: list[dict], index: int, used: set[tuple[str, int, int]], tokenizer
) -> tuple[dict | None, int]:
    """The next unused row of `pool` from `index` on that fits the sequence length."""
    while index < len(pool):
        row = pool[index]
        index += 1
        key = (source_name, row["task_id"], row["sample_idx"])
        if key in used:
            continue
        out = dict(row)
        out["source"] = "on_policy"
        if not _fits_length(out, tokenizer):
            continue
        used.add(key)
        return out, index
    return None, index


def _round_robin_select(
    source_entries: list[tuple[str, list[dict]]], *, label: str, n_target: int, seed: int, tokenizer
) -> tuple[list[dict], dict[str, int]]:
    """Shuffle each source's `label` rows, then cycle through the sources one row at a time
    (a source that runs dry drops out of the rotation) until `n_target` rows are selected."""
    used: set[tuple[str, int, int]] = set()
    counts = {name: 0 for name, _ in source_entries}
    shuffled_pools: dict[str, list[dict]] = {}
    indices: dict[str, int] = {}
    for name, rows in source_entries:
        shuffled = _label_pool(rows, label)
        _source_rng(seed, name).shuffle(shuffled)
        shuffled_pools[name] = shuffled
        indices[name] = 0

    active = [name for name, _ in source_entries]
    selected: list[dict] = []
    rotation_idx = 0
    while len(selected) < n_target and active:
        name = active[rotation_idx % len(active)]
        row, indices[name] = _take_next_valid(name, shuffled_pools[name], indices[name], used, tokenizer)
        if row is None:
            active.remove(name)
            if active:
                rotation_idx = rotation_idx % len(active)
        else:
            selected.append(row)
            counts[name] += 1
            rotation_idx += 1

    if len(selected) < n_target:
        pool_sizes = {name: len(pool) for name, pool in shuffled_pools.items()}
        raise ValueError(
            f"Cannot select {n_target} {label} on-policy examples: got {len(selected)}. "
            f"Pool sizes={pool_sizes}, counts={counts}"
        )
    return selected, counts


def _write_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"OK: saved {manifest['strategy']} train manifest to {manifest_path}")
    print(f"  deceptive sources: {manifest['deceptive_sources']}")
    print(f"  honest sources:    {manifest['honest_sources']}")


def build_train_dataset_round1_mixed(
    tokenizer, *, manifest_path: Path, base_train_jsonl: Path, det0_train_jsonl: Path, seed: int = DATA_SEED
) -> tuple[Dataset, Dataset, dict]:
    """The round-1 probe's training set: `N_FIXED_PER_CLASS` on-dataset examples per class plus
    `ON_POLICY_ROUND1_MIXED` base-model honest completions and `ON_POLICY_ROUND1_MIXED`
    no-penalty-model (det0) reward-hacking completions."""
    base_rows = load_jsonl(base_train_jsonl)
    det0_rows = load_jsonl(det0_train_jsonl)
    probe_pos, probe_neg = load_probe_prompts(tokenizer, seed=seed)

    fixed_deceptive, fixed_deceptive_counts = _select_fixed_on_dataset(
        probe_pos, label="deceptive", seed=seed, tokenizer=tokenizer
    )
    on_policy_deceptive = _select_on_policy_from_jsonl(
        det0_rows, label="deceptive", n=ON_POLICY_ROUND1_MIXED, seed=seed, tokenizer=tokenizer
    )
    fixed_honest, fixed_honest_counts = _select_fixed_on_dataset(
        probe_neg, label="honest", seed=seed + 1, tokenizer=tokenizer
    )
    on_policy_honest = _select_on_policy_from_jsonl(
        base_rows, label="honest", n=ON_POLICY_ROUND1_MIXED, seed=seed + 1, tokenizer=tokenizer
    )
    deceptive_rows = fixed_deceptive + on_policy_deceptive
    honest_rows = fixed_honest + on_policy_honest
    assert len(deceptive_rows) == len(honest_rows) == N_ROUND1_MIXED_PER_CLASS

    manifest = {
        "strategy": "round1_mixed",
        "seed": seed,
        "n_train_per_class": N_ROUND1_MIXED_PER_CLASS,
        "n_fixed_per_class": N_FIXED_PER_CLASS,
        "n_on_policy_per_class": ON_POLICY_ROUND1_MIXED,
        "source_paths": {"base_train": str(base_train_jsonl), "det0_train": str(det0_train_jsonl)},
        "deceptive_sources": {**fixed_deceptive_counts, "det0": len(on_policy_deceptive)},
        "honest_sources": {**fixed_honest_counts, "base_model": len(on_policy_honest)},
        "deceptive": deceptive_rows,
        "honest": honest_rows,
    }
    _write_manifest(manifest_path, manifest)
    return rows_to_dataset(deceptive_rows), rows_to_dataset(honest_rows), manifest


def build_train_dataset_split_sources(
    tokenizer,
    *,
    trial: int,
    round_number: int,
    manifest_path: Path,
    buffer_dir: Path,
    base_honest_jsonl: Path,
    det0_deceptive_jsonl: Path,
    on_policy_per_source: int,
    seed: int,
) -> tuple[Dataset, Dataset, dict]:
    """The training set of the probe for round `round_number` (>= 2): `N_FIXED_PER_CLASS`
    on-dataset examples per class plus `on_policy_per_source` on-policy examples per source,
    drawn round-robin. Honest sources: the base model + every previous round; reward-hacking
    sources: the no-penalty model (det0) + every previous round."""
    if round_number < 2:
        raise ValueError(f"round_number must be >= 2, got {round_number}")

    honest_sources = _resolve_split_sources(
        round_number, trial, buffer_dir, anchor_name="base_model", anchor_path=base_honest_jsonl
    )
    deceptive_sources = _resolve_split_sources(
        round_number, trial, buffer_dir, anchor_name="det0", anchor_path=det0_deceptive_jsonl
    )
    for class_name, sources in (("honest", honest_sources), ("deceptive", deceptive_sources)):
        print(f"{class_name} on-policy sources ({len(sources)}):")
        for name, path in sources:
            print(f"  {name}: {path}")
    n_on_policy = len(honest_sources) * on_policy_per_source
    honest_source_entries = [(name, load_jsonl(path)) for name, path in honest_sources]
    deceptive_source_entries = [(name, load_jsonl(path)) for name, path in deceptive_sources]
    probe_pos, probe_neg = load_probe_prompts(tokenizer, seed=seed)

    fixed_deceptive, fixed_deceptive_counts = _select_fixed_on_dataset(
        probe_pos, label="deceptive", seed=seed, tokenizer=tokenizer
    )
    on_policy_deceptive, on_policy_deceptive_counts = _round_robin_select(
        deceptive_source_entries, label="deceptive", n_target=n_on_policy, seed=seed, tokenizer=tokenizer
    )
    fixed_honest, fixed_honest_counts = _select_fixed_on_dataset(
        probe_neg, label="honest", seed=seed + 1, tokenizer=tokenizer
    )
    on_policy_honest, on_policy_honest_counts = _round_robin_select(
        honest_source_entries, label="honest", n_target=n_on_policy, seed=seed + 1, tokenizer=tokenizer
    )
    deceptive_rows = fixed_deceptive + on_policy_deceptive
    honest_rows = fixed_honest + on_policy_honest
    n_total = N_FIXED_PER_CLASS + n_on_policy
    assert len(deceptive_rows) == len(honest_rows) == n_total

    manifest = {
        "strategy": "split_sources",
        "trial": trial,
        "round": round_number,
        "seed": seed,
        "n_train_per_class": n_total,
        "n_fixed_per_class": N_FIXED_PER_CLASS,
        "n_on_policy_per_class": n_on_policy,
        "on_policy_per_source": on_policy_per_source,
        "honest_source_paths": {name: str(path) for name, path in honest_sources},
        "deceptive_source_paths": {name: str(path) for name, path in deceptive_sources},
        "deceptive_sources": {**fixed_deceptive_counts, **on_policy_deceptive_counts},
        "honest_sources": {**fixed_honest_counts, **on_policy_honest_counts},
        "deceptive": deceptive_rows,
        "honest": honest_rows,
    }
    _write_manifest(manifest_path, manifest)
    return rows_to_dataset(deceptive_rows), rows_to_dataset(honest_rows), manifest
