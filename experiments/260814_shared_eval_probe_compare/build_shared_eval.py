#!/usr/bin/env python3
"""Build the shared on-policy evaluation set for each lineage.

Round-robins over every policy in the lineage until `N_PER_CLASS` samples of each
class are collected, so all three probes are scored on identical rows:

  deceptive: det0, round1 .. round5
  honest:    base, round1 .. round5

det0/base are the per-class anchors (see common.DECEPTIVE_ANCHOR) because the base
policy is deceptive on only 1/400 eval prompts. Sources that run dry drop out of the
rotation and the remaining ones take up the slack; the build hard-fails rather than
silently under-filling a class.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/workspace")))

from experiments.supplemented_probe_eval.common import (
    BASE_EVAL_JSONL,
    DET0_EVAL_JSONL,
    load_jsonl,
    resolve_eval_jsonl,
)

from common import (
    BUILD_SEED,
    DECEPTIVE_ANCHOR,
    DECEPTIVE_DEFINITION,
    EXPECTED_LAST_ROUND,
    HONEST_ANCHOR,
    N_PER_CLASS,
    build_rng,
    label_for,
    normalize_variant,
    provenance_path,
    resolve_targets,
    round6_manifest,
    shared_eval_path,
    write_jsonl,
)

ROW_KEY_FIELDS = ("prompt", "completion")


def _row_key(row: dict) -> tuple:
    return tuple(row[f] for f in ROW_KEY_FIELDS)


def load_sources(trial: int, variant: str) -> dict[str, dict[str, list[dict]]]:
    """Per-class {source_name: rows}, in round-robin rotation order."""
    rounds = {
        f"round{k}": load_jsonl(resolve_eval_jsonl(trial, variant, k))
        for k in range(1, EXPECTED_LAST_ROUND + 1)
    }
    return {
        "deceptive": {DECEPTIVE_ANCHOR: load_jsonl(DET0_EVAL_JSONL), **rounds},
        "honest": {HONEST_ANCHOR: load_jsonl(BASE_EVAL_JSONL), **rounds},
    }


def _class_pool(rows: list[dict], class_name: str, source: str) -> list[dict]:
    pool = []
    for row in rows:
        if not row.get(class_name):
            continue
        if row.get("deceptive") and row.get("honest"):
            raise ValueError(f"{source}: row task_id={row.get('task_id')} is both deceptive and honest")
        if row.get("deceptive_definition") != DECEPTIVE_DEFINITION:
            raise ValueError(
                f"{source}: row task_id={row.get('task_id')} has deceptive_definition="
                f"{row.get('deceptive_definition')!r}, expected {DECEPTIVE_DEFINITION!r}"
            )
        out = dict(row)
        out["source"] = source
        pool.append(out)
    return pool


def round_robin(
    pools: dict[str, list[dict]],
    *,
    n_target: int,
    used: set[tuple],
    rngs: dict[str, "object"],
) -> tuple[list[dict], dict[str, int]]:
    shuffled: dict[str, list[dict]] = {}
    cursor: dict[str, int] = {}
    for name, pool in pools.items():
        rows = list(pool)
        rngs[name].shuffle(rows)
        shuffled[name] = rows
        cursor[name] = 0

    counts = {name: 0 for name in pools}
    selected: list[dict] = []
    active = list(pools)
    rotation = 0

    while len(selected) < n_target and active:
        name = active[rotation % len(active)]
        picked = None
        while cursor[name] < len(shuffled[name]):
            row = shuffled[name][cursor[name]]
            cursor[name] += 1
            key = _row_key(row)
            if key in used:
                continue
            used.add(key)
            picked = row
            break
        if picked is None:
            active.remove(name)
            if active:
                rotation %= len(active)
        else:
            selected.append(picked)
            counts[name] += 1
            rotation += 1

    return selected, {name: n for name, n in counts.items() if n}


def check_no_train_leakage(trial: int, variant: str, rows: list[dict]) -> int:
    """Assert zero task_id overlap with the round-6 probe training manifest."""
    manifest_path = round6_manifest(trial, variant)
    manifest = json.loads(manifest_path.read_text())
    train_ids = {r["task_id"] for key in ("deceptive", "honest") for r in manifest[key]}
    eval_ids = {r["task_id"] for r in rows}
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} task_id(s) appear in both {manifest_path} and the shared eval set: "
            f"{sorted(overlap)[:10]}"
        )
    return len(eval_ids)


def build_lineage(trial: int, variant: str, *, force: bool = False) -> dict:
    label = label_for(trial, variant)
    out_path = shared_eval_path(trial, variant)
    prov_path = provenance_path(trial, variant)
    if out_path.exists() and prov_path.exists() and not force:
        print(f"SKIP {label}: {out_path.name} exists (--force to rebuild)")
        return json.loads(prov_path.read_text())

    sources = load_sources(trial, variant)
    used: set[tuple] = set()
    selected: dict[str, list[dict]] = {}
    provenance: dict[str, dict[str, int]] = {}
    pool_sizes: dict[str, dict[str, int]] = {}

    # Deceptive first, then honest, sharing `used` so no row lands in both halves.
    for class_name in ("deceptive", "honest"):
        pools = {
            source: _class_pool(rows, class_name, source)
            for source, rows in sources[class_name].items()
        }
        pool_sizes[class_name] = {name: len(p) for name, p in pools.items()}
        rngs = {name: build_rng(trial, variant, f"{class_name}|{name}") for name in pools}
        rows, counts = round_robin(pools, n_target=N_PER_CLASS, used=used, rngs=rngs)
        if len(rows) < N_PER_CLASS:
            raise ValueError(
                f"{label}: only {len(rows)}/{N_PER_CLASS} {class_name} samples available. "
                f"Pool sizes={pool_sizes[class_name]}, taken={counts}"
            )
        selected[class_name] = rows
        provenance[class_name] = counts

    all_rows = selected["deceptive"] + selected["honest"]
    assert len(all_rows) == 2 * N_PER_CLASS
    assert len({_row_key(r) for r in all_rows}) == len(all_rows), "duplicate rows in shared eval set"
    n_task_ids = check_no_train_leakage(trial, variant, all_rows)

    write_jsonl(out_path, all_rows)
    meta = {
        "trial": trial,
        "variant": variant or "original",
        "n_per_class": N_PER_CLASS,
        "n_deceptive": len(selected["deceptive"]),
        "n_honest": len(selected["honest"]),
        "deceptive_anchor": DECEPTIVE_ANCHOR,
        "honest_anchor": HONEST_ANCHOR,
        "deceptive_definition": DECEPTIVE_DEFINITION,
        "build_seed": BUILD_SEED,
        "deceptive_provenance": provenance["deceptive"],
        "honest_provenance": provenance["honest"],
        "deceptive_pool_sizes": pool_sizes["deceptive"],
        "honest_pool_sizes": pool_sizes["honest"],
        "n_distinct_task_ids": n_task_ids,
        "source_paths": {
            "deceptive_anchor": str(DET0_EVAL_JSONL),
            "honest_anchor": str(BASE_EVAL_JSONL),
            **{
                f"round{k}": str(resolve_eval_jsonl(trial, variant, k))
                for k in range(1, EXPECTED_LAST_ROUND + 1)
            },
        },
        "shared_eval_jsonl": str(out_path),
    }
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"OK {label}: wrote {out_path.name} ({len(all_rows)} rows, {n_task_ids} distinct task_ids)")
    print(f"   deceptive: {provenance['deceptive']}")
    print(f"   honest:    {provenance['honest']}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shared on-policy probe eval sets")
    parser.add_argument("--trial", type=int, action="append", dest="trials")
    parser.add_argument("--variant", type=str, action="append", dest="variants")
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    args = parser.parse_args()

    variants = [normalize_variant(v) for v in args.variants] if args.variants else None
    targets = resolve_targets(trials=args.trials, variants=variants)
    print(f"Targets ({len(targets)}): {targets}\n")

    for trial, variant in targets:
        build_lineage(trial, variant, force=args.force)


if __name__ == "__main__":
    main()
