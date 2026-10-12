"""Compare the final probe with an initial-model probe on a shared eval set, for one lineage.
Run after train_final_probe.

  final    the final probe: trained on its training set through the last policy's activations
  initial  a new probe trained on the same texts through the base model's activations

Both are scored on one shared eval set built by round-robin over every policy of the
lineage (the det0/base anchors and every round's policy), so the two AUCs differ only in
whose activations each probe reads.

Usage:
    python -m probe_eval.compare_initial_vs_final --trial 1 --variant adaptive_score

Writes experiments/trials/trial<N>/results/initial_vs_final_probe_<variant>.json (plus the
shared eval set and its provenance next to it) and the initial probe under
rounds_<variant>/round<last+1>/probe_weights_base_model/.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil

import torch

from training.build_dataset import load_eval_task_dataset, rows_to_dataset
from training.common import DATA_SEED, DECEPTIVE_DEFINITION, MODEL_NAME, bootstrap_hf_env, check_disk, load_jsonl, write_jsonl
from training.probe_training import load_base_model, train_or_load_probe_and_eval
from training.run_trial import load_trial_state

from probe_eval.common import (
    BASE_EVAL_JSONL,
    DECEPTIVE_ANCHOR,
    DET0_EVAL_JSONL,
    HONEST_ANCHOR,
    N_PER_CLASS,
    build_rng,
    compare_results_path,
    final_probe_weights,
    initial_probe_activations,
    initial_probe_metrics,
    initial_probe_weights,
    load_policy_model,
    load_probe,
    probe_round_manifest,
    provenance_path,
    resolve_eval_jsonl,
    score_rows,
    shared_eval_path,
    unload_model,
)

# Below this on-dataset AUC the initial probe did not fit its training texts through the base
# model, and the comparison says little.
MIN_ON_DATASET_AUC = 0.99
ROW_KEY_FIELDS = ("prompt", "completion")


# --- 1. shared eval set -------------------------------------------------------


def _row_key(row: dict) -> tuple:
    return tuple(row[field] for field in ROW_KEY_FIELDS)


def load_sources(trial: int, variant: str, last_round: int) -> dict[str, dict[str, list[dict]]]:
    """Per class, {source name: eval rows} in round-robin rotation order."""
    rounds = {f"round{k}": load_jsonl(resolve_eval_jsonl(trial, variant, k)) for k in range(1, last_round + 1)}
    return {
        "deceptive": {DECEPTIVE_ANCHOR: load_jsonl(DET0_EVAL_JSONL), **rounds},
        "honest": {HONEST_ANCHOR: load_jsonl(BASE_EVAL_JSONL), **rounds},
    }


def _class_pool(rows: list[dict], class_name: str, source: str) -> list[dict]:
    pool = []
    for row in rows:
        if not row[class_name]:
            continue
        if row["deceptive"] and row["honest"]:
            raise ValueError(f"{source}: row task_id={row['task_id']} is both deceptive and honest")
        if row["deceptive_definition"] != DECEPTIVE_DEFINITION:
            raise ValueError(
                f"{source}: row task_id={row['task_id']} has deceptive_definition="
                f"{row['deceptive_definition']!r}, expected {DECEPTIVE_DEFINITION!r}"
            )
        out = dict(row)
        out["source"] = source
        pool.append(out)
    return pool


def round_robin(
    pools: dict[str, list[dict]], *, n_target: int, used: set[tuple], rngs: dict[str, random.Random]
) -> tuple[list[dict], dict[str, int]]:
    """Cycle through the sources one row at a time; a source that runs dry drops out."""
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


def check_no_train_leakage(trial: int, variant: str, rows: list[dict], probe_round: int) -> int:
    """No task_id of the shared eval set is in the final probe's training manifest."""
    manifest_path = probe_round_manifest(trial, variant, probe_round)
    manifest = json.loads(manifest_path.read_text())
    train_ids = {row["task_id"] for key in ("deceptive", "honest") for row in manifest[key]}
    eval_ids = {row["task_id"] for row in rows}
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} task_id(s) appear in both {manifest_path} and the shared eval set: {sorted(overlap)[:10]}"
        )
    return len(eval_ids)


def build_shared_eval_set(trial: int, variant: str, *, last_round: int, probe_round: int, force: bool) -> dict:
    """N_PER_CLASS rows per class drawn round-robin over the lineage's policies; the two
    classes never share a row. Returns the provenance metadata."""
    label = f"trial{trial} {variant}"
    out_path = shared_eval_path(trial, variant)
    prov_path = provenance_path(trial, variant)
    if out_path.exists() and prov_path.exists() and not force:
        print(f"SKIP {label}: {out_path.name} exists (--force to rebuild)")
        return json.loads(prov_path.read_text())

    sources = load_sources(trial, variant, last_round)
    used: set[tuple] = set()
    selected: dict[str, list[dict]] = {}
    provenance: dict[str, dict[str, int]] = {}
    pool_sizes: dict[str, dict[str, int]] = {}
    # Deceptive first, then honest, sharing `used` so no row lands in both halves.
    for class_name in ("deceptive", "honest"):
        pools = {source: _class_pool(rows, class_name, source) for source, rows in sources[class_name].items()}
        pool_sizes[class_name] = {name: len(pool) for name, pool in pools.items()}
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
    assert len({_row_key(row) for row in all_rows}) == len(all_rows), "duplicate rows in shared eval set"
    n_task_ids = check_no_train_leakage(trial, variant, all_rows, probe_round)

    write_jsonl(out_path, all_rows)
    meta = {
        "trial": trial,
        "variant": variant,
        "n_per_class": N_PER_CLASS,
        "n_deceptive": len(selected["deceptive"]),
        "n_honest": len(selected["honest"]),
        "deceptive_anchor": DECEPTIVE_ANCHOR,
        "honest_anchor": HONEST_ANCHOR,
        "deceptive_definition": DECEPTIVE_DEFINITION,
        "build_seed": DATA_SEED,
        "deceptive_provenance": provenance["deceptive"],
        "honest_provenance": provenance["honest"],
        "deceptive_pool_sizes": pool_sizes["deceptive"],
        "honest_pool_sizes": pool_sizes["honest"],
        "n_distinct_task_ids": n_task_ids,
        "shared_eval_jsonl": str(out_path),
    }
    prov_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"OK {label}: wrote {out_path.name} ({len(all_rows)} rows, {n_task_ids} distinct task_ids)")
    print(f"   deceptive: {provenance['deceptive']}")
    print(f"   honest:    {provenance['honest']}")
    return meta


# --- 2. initial probe: the final probe's training texts, base-model activations --------


def train_initial_probe(model, tokenizer, trial: int, variant: str, probe_round: int, *, force: bool) -> dict:
    """Train (or load) the initial probe and write probe_metrics_base_model.json."""
    label = f"trial{trial} {variant}"
    weights_dir = initial_probe_weights(trial, variant, probe_round)
    metrics_path = initial_probe_metrics(trial, variant, probe_round)
    if force:
        for path in (weights_dir, initial_probe_activations(trial, variant, probe_round)):
            if path.exists():
                shutil.rmtree(path)
                print(f"OK: deleted {path} (--force)")
    if (weights_dir / "layer_0" / "model.pt").exists() and metrics_path.exists():
        print(f"SKIP {label}: initial probe weights already exist (--force to retrain)")
        return json.loads(metrics_path.read_text())

    manifest_path = probe_round_manifest(trial, variant, probe_round)
    if not manifest_path.exists():
        raise FileNotFoundError(f"{label}: missing round-{probe_round} manifest {manifest_path} (run train_final_probe first)")
    manifest = json.loads(manifest_path.read_text())
    n_per_class = manifest["n_train_per_class"]
    if not (len(manifest["deceptive"]) == len(manifest["honest"]) == n_per_class):
        raise ValueError(
            f"{label}: manifest class counts mismatch (deceptive={len(manifest['deceptive'])}, "
            f"honest={len(manifest['honest'])}, n_train_per_class={n_per_class})"
        )
    print(f"OK: reusing {manifest_path} ({n_per_class}/class)")

    result = train_or_load_probe_and_eval(
        model,
        tokenizer,
        train_pos=rows_to_dataset(manifest["deceptive"]),
        train_neg=rows_to_dataset(manifest["honest"]),
        eval_task_dataset=load_eval_task_dataset(tokenizer, seed=DATA_SEED),
        weights_dir=weights_dir,
        activations_dir=initial_probe_activations(trial, variant, probe_round),
    )
    summary = result["summary"]
    if summary["auc_filtered"] < MIN_ON_DATASET_AUC:
        print(
            f"WARN {label}: initial probe on-dataset AUC={summary['auc_filtered']:.4f} < {MIN_ON_DATASET_AUC}; "
            "the final probe's training set is not linearly separable from base-model activations"
        )
    metrics = {
        "trial": trial,
        "variant": variant,
        "activations_model": "base",
        "model_name": MODEL_NAME,
        "no_follow_up": True,
        "data_seed": DATA_SEED,
        "train_manifest": str(manifest_path),
        "n_train_per_class": n_per_class,
        "filtered_layers": result["filtered_layers"],
        "on_dataset_auc": summary["auc_filtered"],
        "on_dataset_auc_all_layers": summary["auc_all_layers"],
        "on_dataset_accuracy": summary["accuracy"],
        "on_dataset_f1": summary["f1"],
        "on_dataset_mean_scores": summary["mean_scores_layer_averaged"],
        "weights_dir": str(weights_dir),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"OK {label}: initial probe on_dataset_auc={summary['auc_filtered']:.4f}, wrote {metrics_path.name}")
    del result
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


# --- 3. score both probes on the shared set ------------------------------------


def score_probe(model, tokenizer, weights_dir, rows: list[dict], *, activations_model: str) -> dict:
    probe_dict, filtered_layers = load_probe(weights_dir)
    scores = score_rows(model, tokenizer, probe_dict=probe_dict, filtered_layers=filtered_layers, rows=rows)
    scores["weights_dir"] = str(weights_dir)
    scores["activations_model"] = activations_model
    scores["filtered_layers"] = filtered_layers
    del probe_dict
    gc.collect()
    return scores


def run_lineage(trial: int, variant: str, *, force: bool) -> dict:
    """Build the shared eval set, train the initial probe, score both probes; returns the
    written payload."""
    label = f"trial{trial} {variant}"
    state = load_trial_state(trial, variant)
    last_round = state["last_completed_round"]
    probe_round = last_round + 1
    final_weights = final_probe_weights(trial, variant, probe_round)
    if not (final_weights / "layer_0" / "model.pt").exists():
        raise FileNotFoundError(f"{label}: missing final probe weights at {final_weights} (run train_final_probe first)")

    shared_meta = build_shared_eval_set(trial, variant, last_round=last_round, probe_round=probe_round, force=force)
    rows = load_jsonl(shared_eval_path(trial, variant))

    print(f"\n{'=' * 60}\nInitial probe (base model): {label}\n{'=' * 60}")
    model, tokenizer = load_base_model()
    train_initial_probe(model, tokenizer, trial, variant, probe_round, force=force)
    initial = score_probe(model, tokenizer, initial_probe_weights(trial, variant, probe_round), rows, activations_model="base")
    print(f"  initial probe: auc={initial['auc']:.4f} deceptive={initial['deceptive']:.4f} honest={initial['honest']:.4f}")
    unload_model(model)
    del tokenizer

    print(f"\n{'=' * 60}\nFinal probe (round-{last_round} model): {label}\n{'=' * 60}")
    model, tokenizer = load_policy_model(state["round_grpo_ids"][str(last_round)])
    final = score_probe(model, tokenizer, final_weights, rows, activations_model=f"round{last_round}")
    print(f"  final probe:   auc={final['auc']:.4f} deceptive={final['deceptive']:.4f} honest={final['honest']:.4f}")
    unload_model(model)
    del tokenizer

    payload = {
        "trial": trial,
        "variant": variant,
        "probe_round": probe_round,
        "policy_round": last_round,
        "final_model_run_id": state["round_grpo_ids"][str(last_round)],
        "deceptive_definition": DECEPTIVE_DEFINITION,
        "data_seed": DATA_SEED,
        "shared_eval": shared_meta,
        "final": final,
        "initial": initial,
    }
    out_path = compare_results_path(trial, variant)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{label}: final auc={final['auc']:.4f}  initial auc={initial['auc']:.4f}")
    print(f"Wrote {out_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--variant", type=str, required=True, help="Lineage name (e.g. adaptive_score)")
    parser.add_argument("--force", action="store_true", help="Rebuild the eval set and retrain the initial probe")
    args = parser.parse_args()

    bootstrap_hf_env()
    check_disk()
    run_lineage(args.trial, args.variant, force=args.force)


if __name__ == "__main__":
    main()
