"""Shared helpers for supplemented on-policy probe eval."""

from __future__ import annotations

import json
import random
from pathlib import Path

from pipeline.paths import WORKSPACE

WANDB_PROJECT = "lilylixin6-the-harker-school/deception_detection"
MIN_CLASS_COUNT = 20
SUPPLEMENT_SEED = 53
DATA_SEED = 53
DECEPTIVE_DEFINITION = "pass_only_shown"

BASE_EVAL_JSONL = WORKSPACE / "experiments/artifacts/base/base_eval_generations.jsonl"
DET0_EVAL_JSONL = WORKSPACE / "experiments/artifacts/det0/det0_eval_generations.jsonl"

LINEAGES: list[tuple[int, str]] = [
    (4, "v2"),
    (4, "frozen"),
    (4, "early_retraining_v2"),
    (6, ""),
    (6, "frozen"),
    (6, "early_retraining"),
    (8, ""),
    (8, "frozen"),
    (8, "early_retraining"),
]

# (trial, variant, round, run_id) — missing local eval JSONLs fetched from W&B.
MISSING_WANDB_RUNS: list[tuple[int, str, int, str]] = [
    (4, "frozen", 2, "0lhmqg1u"),
    (4, "frozen", 3, "pfvnd8rl"),
    (4, "frozen", 4, "ze1uas04"),
    (4, "frozen", 5, "44oizrwy"),
    (6, "frozen", 2, "vvrkmfj1"),
    (6, "frozen", 3, "ef86iiu8"),
    (6, "frozen", 4, "81amw0vp"),
    (6, "frozen", 5, "o7gpqj7x"),
    (8, "frozen", 2, "od4qc0by"),
    (8, "frozen", 3, "9mkc3tgw"),
    (8, "frozen", 4, "diycqwls"),
    (8, "frozen", 5, "8bc0jgn2"),
    (4, "v2", 5, "rkrfsaml"),
    (4, "early_retraining_v2", 5, "8g8580by"),  # eval-only rerun (training run io4ibs0w crashed)
    (6, "", 5, "94wejikq"),
    (6, "early_retraining", 5, "ge0tg5o1"),
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def trial_dir(trial: int) -> Path:
    return WORKSPACE / f"experiments/trials/trial{trial}"


def buffer_dir(trial: int, variant: str) -> Path:
    tdir = trial_dir(trial)
    if variant:
        return tdir / f"buffer_{variant}"
    return tdir / "buffer"


def rounds_dir(trial: int, variant: str) -> Path:
    tdir = trial_dir(trial)
    if variant:
        return tdir / f"rounds_{variant}"
    return tdir / "rounds"


def variant_label(variant: str) -> str:
    if not variant:
        return "original"
    if variant == "v2":
        return "original"
    if variant in ("early_retraining", "early_retraining_v2"):
        return "early retraining"
    return variant


def load_trial_state(trial: int, variant: str) -> dict:
    tdir = trial_dir(trial)
    name = f"trial_state_{variant}.json" if variant else "trial_state.json"
    path = tdir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing trial state: {path}")
    return json.loads(path.read_text())


def eval_jsonl_candidates(trial: int, variant: str, round_k: int) -> list[Path]:
    """Candidate paths for model-k eval generations, highest priority first."""
    if round_k == 0:
        return [BASE_EVAL_JSONL]
    fname = f"trial{trial}_round{round_k}_eval_generations.jsonl"
    wandb_fname = f"trial{trial}_round{round_k}_eval_generations_wandb.jsonl"
    buf = buffer_dir(trial, variant)
    main_buf = trial_dir(trial) / "buffer"
    return [
        buf / fname,
        buf / wandb_fname,
        main_buf / fname,
        main_buf / wandb_fname,
    ]


def resolve_eval_jsonl(trial: int, variant: str, round_k: int) -> Path:
    for path in eval_jsonl_candidates(trial, variant, round_k):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No eval jsonl for trial{trial} variant={variant or 'original'} model round {round_k}"
    )


def source_eval_rows(trial: int, variant: str, source: str) -> list[dict]:
    if source == "base":
        return load_jsonl(BASE_EVAL_JSONL)
    if source == "det0":
        return load_jsonl(DET0_EVAL_JSONL)
    if source.startswith("round"):
        round_k = int(source.removeprefix("round"))
        return load_jsonl(resolve_eval_jsonl(trial, variant, round_k))
    raise ValueError(f"Unknown source {source!r}")


def source_chain(model_k: int) -> list[str]:
    """Walk-back order for supplementation (current model first)."""
    chain = [f"round{k}" for k in range(model_k, 0, -1)]
    chain.append("base")
    return chain


def top_up_class(
    *,
    trial: int,
    variant: str,
    model_k: int,
    class_name: str,
    rng: random.Random,
    target: int = MIN_CLASS_COUNT,
) -> tuple[list[dict], dict[str, int]]:
    """Use all current-model class rows when >= target; else top up to exactly target."""
    provenance: dict[str, int] = {}
    chain = source_chain(model_k)
    current_source = chain[0]
    current_rows = source_eval_rows(trial, variant, current_source)
    chosen = [r for r in current_rows if r.get(class_name)]
    provenance[current_source] = len(chosen)

    if len(chosen) >= target:
        return chosen, provenance

    used: set[tuple[str, str]] = {(r["prompt"], r["completion"]) for r in chosen}

    for source in chain[1:]:
        if len(chosen) >= target:
            break
        rows = source_eval_rows(trial, variant, source)
        pool = [
            r
            for r in rows
            if r.get(class_name)
            and (r["prompt"], r["completion"]) not in used
        ]
        need = target - len(chosen)
        take = min(need, len(pool))
        if take <= 0:
            provenance[source] = 0
            continue
        picks = rng.sample(pool, take)
        chosen.extend(picks)
        used.update((r["prompt"], r["completion"]) for r in picks)
        provenance[source] = provenance.get(source, 0) + take

    if class_name == "deceptive" and len(chosen) < target:
        pool = [
            r
            for r in source_eval_rows(trial, variant, "det0")
            if r.get(class_name)
            and (r["prompt"], r["completion"]) not in used
        ]
        need = target - len(chosen)
        take = min(need, len(pool))
        if take > 0:
            picks = rng.sample(pool, take)
            chosen.extend(picks)
            provenance["det0"] = take

    if len(chosen) < target:
        raise RuntimeError(
            f"Could not top up {class_name} to {target} for trial{trial} model {model_k}: got {len(chosen)}"
        )
    return chosen, provenance


def build_supplemented_rows(
    *,
    trial: int,
    variant: str,
    model_k: int,
    seed: int = SUPPLEMENT_SEED,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed + trial * 100 + model_k)
    dec_rows, dec_prov = top_up_class(
        trial=trial, variant=variant, model_k=model_k, class_name="deceptive", rng=rng
    )
    hon_rows, hon_prov = top_up_class(
        trial=trial, variant=variant, model_k=model_k, class_name="honest", rng=rng
    )
    rows = dec_rows + hon_rows
    for row in rows:
        row = dict(row)
    meta = {
        "model_k": model_k,
        "n_deceptive": len(dec_rows),
        "n_honest": len(hon_rows),
        "deceptive_provenance": dec_prov,
        "honest_provenance": hon_prov,
        "eval_jsonl": str(resolve_eval_jsonl(trial, variant, model_k)) if model_k > 0 else str(BASE_EVAL_JSONL),
    }
    return rows, meta


def results_path(trial: int, variant: str) -> Path:
    suffix = f"_{variant}" if variant else ""
    return trial_dir(trial) / "results" / f"supplemented_probe_eval{suffix}.json"
