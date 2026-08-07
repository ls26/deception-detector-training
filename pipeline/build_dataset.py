# Build mixed on-policy + on-dataset probe training datasets.

# %%
import json
import random
import zlib
from pathlib import Path

from datasets import Dataset

from obfuscation_atlas.config import CodeGenTaskConfig, MBPPProbeConfig, MBPPTaskConfig
from obfuscation_atlas.training.obfuscation_training import prepare_dataset

SEED = 53
N_TRAIN_PER_CLASS = 200
N_FIXED_PER_CLASS = 200
ON_POLICY_PER_ROUND = 100
ON_POLICY_PER_SOURCE = 50
ON_POLICY_ROUND1_MIXED = 100
N_ROUND1_MIXED_PER_CLASS = N_FIXED_PER_CLASS + ON_POLICY_ROUND1_MIXED
MAX_SEQUENCE_LENGTH = 512


def _source_rng(seed: int, name: str) -> random.Random:
    """Stable per-source RNG (zlib.crc32, not Python's salted str.__hash__)."""
    return random.Random(seed + (zlib.crc32(name.encode()) & 0xFFFFFFFF))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rows_to_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [r["prompt"] for r in rows],
            "completion": [r["completion"] for r in rows],
            "task_id": [r["task_id"] for r in rows],
            "source_model": [r.get("model_key", r.get("model", r.get("source_model", "unknown"))) for r in rows],
        }
    )


def _resolve_sources(round_num: int, trial: int, buffer_dir: Path) -> list[tuple[str, Path | None]]:
    sources: list[tuple[str, Path | None]] = []
    for k in range(round_num - 1, 0, -1):
        name = f"round{k}"
        path = buffer_dir / f"trial{trial}_round{k}_train_generations.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        sources.append((name, path))

    base_path = buffer_dir / "base_train_generations.jsonl"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing {base_path}")
    sources.append(("base_model", base_path))
    sources.append(("on_dataset", None))

    print(f"Sources ({len(sources)} expected {round_num + 1}):")
    for name, path in sources:
        print(f"  {name}: {path}")
    assert len(sources) == round_num + 1
    return sources


def _resolve_on_policy_sources(round_num: int, trial: int, buffer_dir: Path) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    base_path = buffer_dir / "base_train_generations.jsonl"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing {base_path}")
    sources.append(("base_model", base_path))
    for k in range(1, round_num):
        name = f"round{k}"
        path = buffer_dir / f"trial{trial}_round{k}_train_generations.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        sources.append((name, path))

    print(f"On-policy sources ({len(sources)}):")
    for name, path in sources:
        print(f"  {name}: {path}")
    return sources


def _resolve_split_sources(
    round_num: int,
    trial: int,
    buffer_dir: Path,
    *,
    anchor_name: str,
    anchor_path: Path,
) -> list[tuple[str, Path]]:
    """Anchor source (base_model or det0) + all previous rounds' on-policy JSONLs."""
    if not anchor_path.exists():
        raise FileNotFoundError(f"Missing {anchor_path}")
    sources: list[tuple[str, Path]] = [(anchor_name, anchor_path)]
    for k in range(1, round_num):
        name = f"round{k}"
        path = buffer_dir / f"trial{trial}_round{k}_train_generations.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        sources.append((name, path))
    return sources


def _allocate_quotas(n_total: int, n_sources: int) -> list[int]:
    base = n_total // n_sources
    remainder = n_total % n_sources
    return [base + (1 if i < remainder else 0) for i in range(n_sources)]


def _label_pool(rows: list[dict], label: str) -> list[dict]:
    return [r for r in rows if r.get(label)]


def _on_dataset_pool(probe_rows: list[dict], *, label: str) -> list[dict]:
    if label == "deceptive":
        return [
            {
                "prompt": r["prompt"],
                "completion": r["hardcode"],
                "task_id": r["task_id"],
                "source": "on_dataset",
                "source_model": "dataset_hardcode",
            }
            for r in probe_rows
        ]
    return [
        {
            "prompt": r["prompt"],
            "completion": r["code"],
            "task_id": r["task_id"],
            "source": "on_dataset",
            "source_model": "dataset_code",
        }
        for r in probe_rows
    ]


def _fits_length(row: dict, tokenizer, max_length: int) -> bool:
    text = row["prompt"] + row["completion"]
    return len(tokenizer.encode(text, add_special_tokens=False)) <= max_length


def _source_pool(
    source_name: str,
    jsonl_rows: list[dict] | None,
    *,
    label: str,
    probe_pos,
    probe_neg,
) -> list[dict]:
    if source_name == "on_dataset":
        probe_rows = probe_pos if label == "deceptive" else probe_neg
        return _on_dataset_pool(probe_rows, label=label)
    return _label_pool(jsonl_rows or [], label)


def _take_from_pool(
    source_name: str,
    pool: list[dict],
    n: int,
    rng: random.Random,
    tokenizer,
    max_length: int,
    used: set[tuple[str, int, int]],
    *,
    tag_on_policy: bool,
) -> list[dict]:
    shuffled = list(pool)
    rng.shuffle(shuffled)
    taken: list[dict] = []
    for row in shuffled:
        if len(taken) >= n:
            break
        key = (source_name, row["task_id"], row.get("sample_idx", 0))
        if key in used:
            continue
        out = dict(row)
        if tag_on_policy:
            out["source"] = "on_policy"
        if not _fits_length(out, tokenizer, max_length):
            continue
        taken.append(out)
        used.add(key)
    return taken


def _select_rows(
    source_entries: list[tuple[str, list[dict] | None]],
    probe_pos,
    probe_neg,
    *,
    label: str,
    n_total: int,
    quotas: list[int],
    seed: int,
    tokenizer,
    max_length: int,
) -> tuple[list[dict], dict[str, int]]:
    used: set[tuple[str, int, int]] = set()
    selected: list[dict] = []
    counts = {name: 0 for name, _ in source_entries}

    for (name, jsonl_rows), quota in zip(source_entries, quotas, strict=True):
        pool = _source_pool(name, jsonl_rows, label=label, probe_pos=probe_pos, probe_neg=probe_neg)
        rng = _source_rng(seed, name)
        taken = _take_from_pool(
            name, pool, quota, rng, tokenizer, max_length, used, tag_on_policy=(name != "on_dataset")
        )
        selected.extend(taken)
        counts[name] += len(taken)

    need = n_total - len(selected)
    if need > 0:
        for name, jsonl_rows in source_entries:
            if need <= 0:
                break
            pool = _source_pool(name, jsonl_rows, label=label, probe_pos=probe_pos, probe_neg=probe_neg)
            rng = _source_rng(seed, f"{name}_backfill")
            taken = _take_from_pool(
                name, pool, need, rng, tokenizer, max_length, used, tag_on_policy=(name != "on_dataset")
            )
            selected.extend(taken)
            counts[name] += len(taken)
            need -= len(taken)

    if len(selected) < n_total:
        pool_sizes = {
            name: len(_source_pool(name, rows, label=label, probe_pos=probe_pos, probe_neg=probe_neg))
            for name, rows in source_entries
        }
        raise ValueError(
            f"Cannot select {n_total} {label} examples: got {len(selected)}. "
            f"Pool sizes={pool_sizes}, counts={counts}"
        )

    return selected[:n_total], counts


def _take_next_valid(
    source_name: str,
    pool: list[dict],
    index: int,
    used: set[tuple[str, int, int]],
    tokenizer,
    max_length: int,
    *,
    tag_on_policy: bool,
) -> tuple[dict | None, int]:
    while index < len(pool):
        row = pool[index]
        index += 1
        key = (source_name, row["task_id"], row.get("sample_idx", 0))
        if key in used:
            continue
        out = dict(row)
        if tag_on_policy:
            out["source"] = "on_policy"
        if not _fits_length(out, tokenizer, max_length):
            continue
        used.add(key)
        return out, index
    return None, index


def _select_fixed_on_dataset(
    probe_pos,
    probe_neg,
    *,
    label: str,
    seed: int,
    tokenizer,
    max_length: int,
    n: int = N_FIXED_PER_CLASS,
) -> tuple[list[dict], dict[str, int]]:
    used: set[tuple[str, int, int]] = set()
    probe_rows = probe_pos if label == "deceptive" else probe_neg
    pool = _on_dataset_pool(probe_rows, label=label)
    rng = random.Random(seed)
    taken = _take_from_pool("on_dataset", pool, n, rng, tokenizer, max_length, used, tag_on_policy=False)
    if len(taken) < n:
        raise ValueError(f"Cannot select {n} fixed {label} examples: got {len(taken)}. Pool size={len(pool)}")
    return taken, {"on_dataset": len(taken)}


def _round_robin_select(
    source_entries: list[tuple[str, list[dict] | None]],
    probe_pos,
    probe_neg,
    *,
    label: str,
    n_target: int,
    seed: int,
    tokenizer,
    max_length: int,
) -> tuple[list[dict], dict[str, int]]:
    used: set[tuple[str, int, int]] = set()
    counts = {name: 0 for name, _ in source_entries}
    shuffled_pools: dict[str, list[dict]] = {}
    indices: dict[str, int] = {}

    for name, jsonl_rows in source_entries:
        pool = _source_pool(name, jsonl_rows, label=label, probe_pos=probe_pos, probe_neg=probe_neg)
        rng = _source_rng(seed, name)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        shuffled_pools[name] = shuffled
        indices[name] = 0

    active = [name for name, _ in source_entries]
    selected: list[dict] = []
    rotation_idx = 0

    while len(selected) < n_target and active:
        name = active[rotation_idx % len(active)]
        row, indices[name] = _take_next_valid(
            name,
            shuffled_pools[name],
            indices[name],
            used,
            tokenizer,
            max_length,
            tag_on_policy=True,
        )
        if row is None:
            active.remove(name)
            if active:
                rotation_idx = rotation_idx % len(active)
        else:
            selected.append(row)
            counts[name] += 1
            rotation_idx += 1

    if len(selected) < n_target:
        pool_sizes = {
            name: len(_source_pool(name, rows, label=label, probe_pos=probe_pos, probe_neg=probe_neg))
            for name, rows in source_entries
        }
        raise ValueError(
            f"Cannot select {n_target} {label} on-policy examples: got {len(selected)}. "
            f"Pool sizes={pool_sizes}, counts={counts}"
        )

    return selected, counts


def build_train_dataset_fixed_size(
    tokenizer,
    *,
    trial: int,
    round: int,
    manifest_path: Path,
    buffer_dir: Path,
    seed: int = SEED,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    data_seed: int = SEED,
) -> tuple[Dataset, Dataset, dict]:
    """Build 200/class mixed probe train set from roundR..round2, base, and on-dataset sources."""
    if round < 2:
        raise ValueError(f"round must be >= 2, got {round}")

    sources = _resolve_sources(round, trial, buffer_dir)
    source_paths = {name: str(path) if path is not None else None for name, path in sources}

    source_entries: list[tuple[str, list[dict] | None]] = []
    for name, path in sources:
        if path is None:
            source_entries.append((name, None))
        else:
            source_entries.append((name, load_jsonl(path)))

    cfg = CodeGenTaskConfig()
    cfg.data.seed = data_seed
    cfg.data.probe_dataset = MBPPProbeConfig()
    cfg.data.task_dataset = MBPPTaskConfig()
    _, (probe_pos, probe_neg), _ = prepare_dataset(cfg.data, tokenizer=tokenizer)
    assert len(probe_pos) == len(probe_neg)
    print(f"OK: probe train dataset loaded ({len(probe_pos)} pos, {len(probe_neg)} neg)")

    quotas = _allocate_quotas(N_TRAIN_PER_CLASS, round + 1)
    target_quotas = {name: quota for (name, _), quota in zip(sources, quotas, strict=True)}

    deceptive_rows, deceptive_sources = _select_rows(
        source_entries,
        probe_pos,
        probe_neg,
        label="deceptive",
        n_total=N_TRAIN_PER_CLASS,
        quotas=quotas,
        seed=seed,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    honest_rows, honest_sources = _select_rows(
        source_entries,
        probe_pos,
        probe_neg,
        label="honest",
        n_total=N_TRAIN_PER_CLASS,
        quotas=quotas,
        seed=seed + 1,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )

    manifest = {
        "trial": trial,
        "round": round,
        "seed": seed,
        "n_train_per_class": N_TRAIN_PER_CLASS,
        "n_sources": round + 1,
        "target_quotas": target_quotas,
        "source_paths": source_paths,
        "deceptive_sources": deceptive_sources,
        "honest_sources": honest_sources,
        "deceptive": deceptive_rows,
        "honest": honest_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"OK: saved mixed train manifest to {manifest_path}")
    print(f"  deceptive sources: {deceptive_sources}")
    print(f"  honest sources:    {honest_sources}")
    assert len(deceptive_rows) == len(honest_rows) == N_TRAIN_PER_CLASS

    return rows_to_dataset(deceptive_rows), rows_to_dataset(honest_rows), manifest


def build_train_dataset_round_robin(
    tokenizer,
    *,
    trial: int,
    round: int,
    manifest_path: Path,
    buffer_dir: Path,
    seed: int = SEED,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    data_seed: int = SEED,
) -> tuple[Dataset, Dataset, dict]:
    """Build 200 fixed + (round-1)*100/class on-policy probe train set via round-robin."""
    if round < 2:
        raise ValueError(f"round must be >= 2, got {round}")

    n_on_policy = (round - 1) * ON_POLICY_PER_ROUND
    n_total = N_FIXED_PER_CLASS + n_on_policy

    sources = _resolve_on_policy_sources(round, trial, buffer_dir)
    source_paths = {name: str(path) for name, path in sources}

    source_entries: list[tuple[str, list[dict] | None]] = [
        (name, load_jsonl(path)) for name, path in sources
    ]

    cfg = CodeGenTaskConfig()
    cfg.data.seed = data_seed
    cfg.data.probe_dataset = MBPPProbeConfig()
    cfg.data.task_dataset = MBPPTaskConfig()
    _, (probe_pos, probe_neg), _ = prepare_dataset(cfg.data, tokenizer=tokenizer)
    assert len(probe_pos) == len(probe_neg)
    print(f"OK: probe train dataset loaded ({len(probe_pos)} pos, {len(probe_neg)} neg)")

    fixed_deceptive, fixed_deceptive_counts = _select_fixed_on_dataset(
        probe_pos,
        probe_neg,
        label="deceptive",
        seed=seed,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    on_policy_deceptive, on_policy_deceptive_counts = _round_robin_select(
        source_entries,
        probe_pos,
        probe_neg,
        label="deceptive",
        n_target=n_on_policy,
        seed=seed,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    deceptive_rows = fixed_deceptive + on_policy_deceptive
    deceptive_sources = {**fixed_deceptive_counts, **on_policy_deceptive_counts}

    fixed_honest, fixed_honest_counts = _select_fixed_on_dataset(
        probe_pos,
        probe_neg,
        label="honest",
        seed=seed + 1,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    on_policy_honest, on_policy_honest_counts = _round_robin_select(
        source_entries,
        probe_pos,
        probe_neg,
        label="honest",
        n_target=n_on_policy,
        seed=seed + 1,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    honest_rows = fixed_honest + on_policy_honest
    honest_sources = {**fixed_honest_counts, **on_policy_honest_counts}

    manifest = {
        "strategy": "round_robin",
        "trial": trial,
        "round": round,
        "seed": seed,
        "n_train_per_class": n_total,
        "n_fixed_per_class": N_FIXED_PER_CLASS,
        "n_on_policy_per_class": n_on_policy,
        "source_paths": source_paths,
        "deceptive_sources": deceptive_sources,
        "honest_sources": honest_sources,
        "deceptive": deceptive_rows,
        "honest": honest_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"OK: saved round-robin train manifest to {manifest_path}")
    print(f"  deceptive sources: {deceptive_sources}")
    print(f"  honest sources:    {honest_sources}")
    assert len(deceptive_rows) == len(honest_rows) == n_total

    return rows_to_dataset(deceptive_rows), rows_to_dataset(honest_rows), manifest


def build_train_dataset_split_sources(
    tokenizer,
    *,
    trial: int,
    round: int,
    manifest_path: Path,
    buffer_dir: Path,
    base_honest_jsonl: Path,
    det0_deceptive_jsonl: Path,
    seed: int = SEED,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    data_seed: int = SEED,
    on_policy_per_source: int = ON_POLICY_PER_SOURCE,
    n_fixed_per_class: int = N_FIXED_PER_CLASS,
) -> tuple[Dataset, Dataset, dict]:
    """Build n_fixed_per_class fixed + round-robin on-policy (on_policy_per_source/source) with class-specific anchor sources.

    Honest on-policy sources: base_model (base_honest_jsonl) + all previous rounds.
    Deceptive on-policy sources: det0 (det0_deceptive_jsonl) + all previous rounds.
    Each class has the same number of sources (1 anchor + round-1 prior rounds = `round`
    sources), so both classes draw the same on-policy total: `round * on_policy_per_source`.
    """
    if round < 2:
        raise ValueError(f"round must be >= 2, got {round}")

    honest_sources = _resolve_split_sources(
        round, trial, buffer_dir, anchor_name="base_model", anchor_path=base_honest_jsonl
    )
    deceptive_sources = _resolve_split_sources(
        round, trial, buffer_dir, anchor_name="det0", anchor_path=det0_deceptive_jsonl
    )
    print(f"Honest on-policy sources ({len(honest_sources)}):")
    for name, path in honest_sources:
        print(f"  {name}: {path}")
    print(f"Deceptive on-policy sources ({len(deceptive_sources)}):")
    for name, path in deceptive_sources:
        print(f"  {name}: {path}")
    assert len(honest_sources) == len(deceptive_sources) == round
    n_on_policy = len(honest_sources) * on_policy_per_source

    honest_source_entries: list[tuple[str, list[dict] | None]] = [
        (name, load_jsonl(path)) for name, path in honest_sources
    ]
    deceptive_source_entries: list[tuple[str, list[dict] | None]] = [
        (name, load_jsonl(path)) for name, path in deceptive_sources
    ]

    cfg = CodeGenTaskConfig()
    cfg.data.seed = data_seed
    cfg.data.probe_dataset = MBPPProbeConfig(max_train_examples=2 * n_fixed_per_class)
    cfg.data.task_dataset = MBPPTaskConfig()
    _, (probe_pos, probe_neg), _ = prepare_dataset(cfg.data, tokenizer=tokenizer)
    assert len(probe_pos) == len(probe_neg)
    print(f"OK: probe train dataset loaded ({len(probe_pos)} pos, {len(probe_neg)} neg)")

    fixed_deceptive, fixed_deceptive_counts = _select_fixed_on_dataset(
        probe_pos,
        probe_neg,
        label="deceptive",
        seed=seed,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
        n=n_fixed_per_class,
    )
    on_policy_deceptive, on_policy_deceptive_counts = _round_robin_select(
        deceptive_source_entries,
        probe_pos,
        probe_neg,
        label="deceptive",
        n_target=n_on_policy,
        seed=seed,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    deceptive_rows = fixed_deceptive + on_policy_deceptive
    deceptive_source_counts = {**fixed_deceptive_counts, **on_policy_deceptive_counts}

    fixed_honest, fixed_honest_counts = _select_fixed_on_dataset(
        probe_pos,
        probe_neg,
        label="honest",
        seed=seed + 1,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
        n=n_fixed_per_class,
    )
    on_policy_honest, on_policy_honest_counts = _round_robin_select(
        honest_source_entries,
        probe_pos,
        probe_neg,
        label="honest",
        n_target=n_on_policy,
        seed=seed + 1,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    honest_rows = fixed_honest + on_policy_honest
    honest_source_counts = {**fixed_honest_counts, **on_policy_honest_counts}

    n_total = n_fixed_per_class + n_on_policy
    manifest = {
        "strategy": "split_sources",
        "trial": trial,
        "round": round,
        "seed": seed,
        "n_train_per_class": n_total,
        "n_fixed_per_class": n_fixed_per_class,
        "n_on_policy_per_class": n_on_policy,
        "on_policy_per_source": on_policy_per_source,
        "honest_source_paths": {name: str(path) for name, path in honest_sources},
        "deceptive_source_paths": {name: str(path) for name, path in deceptive_sources},
        "deceptive_sources": deceptive_source_counts,
        "honest_sources": honest_source_counts,
        "deceptive": deceptive_rows,
        "honest": honest_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"OK: saved split-sources train manifest to {manifest_path}")
    print(f"  deceptive sources: {deceptive_source_counts}")
    print(f"  honest sources:    {honest_source_counts}")
    assert len(deceptive_rows) == len(honest_rows) == n_total

    return rows_to_dataset(deceptive_rows), rows_to_dataset(honest_rows), manifest


def _select_on_policy_from_jsonl(
    jsonl_rows: list[dict],
    *,
    label: str,
    n: int,
    seed: int,
    tokenizer,
    max_length: int,
) -> list[dict]:
    used: set[tuple[str, int, int]] = set()
    pool = _label_pool(jsonl_rows, label)
    rng = random.Random(seed)
    taken = _take_from_pool(
        "on_policy", pool, n, rng, tokenizer, max_length, used, tag_on_policy=True
    )
    if len(taken) < n:
        raise ValueError(
            f"Cannot select {n} on-policy {label} examples: got {len(taken)}. Pool size={len(pool)}"
        )
    for row in taken:
        row["source_model"] = row.get("model_key", row.get("model", "unknown"))
    return taken


def build_train_dataset_round1_mixed(
    tokenizer,
    *,
    manifest_path: Path,
    base_train_jsonl: Path,
    det0_train_jsonl: Path,
    seed: int = SEED,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    data_seed: int = SEED,
) -> tuple[Dataset, Dataset, dict]:
    """Build 200 on-dataset + 100 on-policy per class (300/class) for round-1 mixed probe train."""
    base_rows = load_jsonl(base_train_jsonl)
    det0_rows = load_jsonl(det0_train_jsonl)
    honest_pool = len(_label_pool(base_rows, "honest"))
    deceptive_pool = len(_label_pool(det0_rows, "deceptive"))
    assert honest_pool >= ON_POLICY_ROUND1_MIXED, (
        f"base train honest pool {honest_pool} < {ON_POLICY_ROUND1_MIXED}"
    )
    assert deceptive_pool >= ON_POLICY_ROUND1_MIXED, (
        f"det0 train deceptive pool {deceptive_pool} < {ON_POLICY_ROUND1_MIXED}"
    )

    cfg = CodeGenTaskConfig()
    cfg.data.seed = data_seed
    cfg.data.probe_dataset = MBPPProbeConfig()
    cfg.data.task_dataset = MBPPTaskConfig()
    _, (probe_pos, probe_neg), _ = prepare_dataset(cfg.data, tokenizer=tokenizer)
    assert len(probe_pos) == len(probe_neg) == N_FIXED_PER_CLASS
    print(f"OK: probe train dataset loaded ({len(probe_pos)} pos, {len(probe_neg)} neg)")

    fixed_deceptive, fixed_deceptive_counts = _select_fixed_on_dataset(
        probe_pos,
        probe_neg,
        label="deceptive",
        seed=seed,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    on_policy_deceptive = _select_on_policy_from_jsonl(
        det0_rows,
        label="deceptive",
        n=ON_POLICY_ROUND1_MIXED,
        seed=seed,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    deceptive_rows = fixed_deceptive + on_policy_deceptive
    deceptive_sources = {**fixed_deceptive_counts, "det0": len(on_policy_deceptive)}

    fixed_honest, fixed_honest_counts = _select_fixed_on_dataset(
        probe_pos,
        probe_neg,
        label="honest",
        seed=seed + 1,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    on_policy_honest = _select_on_policy_from_jsonl(
        base_rows,
        label="honest",
        n=ON_POLICY_ROUND1_MIXED,
        seed=seed + 1,
        tokenizer=tokenizer,
        max_length=max_sequence_length,
    )
    honest_rows = fixed_honest + on_policy_honest
    honest_sources = {**fixed_honest_counts, "base_model": len(on_policy_honest)}

    manifest = {
        "strategy": "round1_mixed",
        "seed": seed,
        "n_train_per_class": N_ROUND1_MIXED_PER_CLASS,
        "n_fixed_per_class": N_FIXED_PER_CLASS,
        "n_on_policy_per_class": ON_POLICY_ROUND1_MIXED,
        "source_paths": {
            "base_train": str(base_train_jsonl),
            "det0_train": str(det0_train_jsonl),
        },
        "deceptive_sources": deceptive_sources,
        "honest_sources": honest_sources,
        "deceptive": deceptive_rows,
        "honest": honest_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"OK: saved round1 mixed train manifest to {manifest_path}")
    print(f"  deceptive sources: {deceptive_sources}")
    print(f"  honest sources:    {honest_sources}")
    assert len(deceptive_rows) == len(honest_rows) == N_ROUND1_MIXED_PER_CLASS
    assert deceptive_sources == {"on_dataset": N_FIXED_PER_CLASS, "on_policy": ON_POLICY_ROUND1_MIXED}
    assert honest_sources == {"on_dataset": N_FIXED_PER_CLASS, "on_policy": ON_POLICY_ROUND1_MIXED}

    return rows_to_dataset(deceptive_rows), rows_to_dataset(honest_rows), manifest

# %%
