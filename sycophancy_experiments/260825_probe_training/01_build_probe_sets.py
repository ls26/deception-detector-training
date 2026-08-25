"""Phase 1: join rescored_labels.jsonl -> completions/followups text, filter, sample.

Builds one train/eval JSONL per probe *definition* (not per extraction position --
position is a Phase-2 slicing choice, both positions share the same rows):
  bias_rm, bias_swap, deception_rm, deception_swap

Train: plain seeded random sample, N_TRAIN_PER_CLASS per class.
Eval: the entire eval split for that class -- no sampling, no cap.

See ../../docs/sycophancy_experiment_log.md "Decisions" and this experiment's own
README.md for why: enumerated population (both cue_gold/cue_dist rows kept), plain
random (not category-stratified) sampling, anti-bias excluded from bias-probe data.
"""
import json
import random
from collections import Counter
from pathlib import Path

EXP_DIR = Path(__file__).parent
DOMAIN_DIR = EXP_DIR.parent / "260819_domain_baseline"
FOLLOWUP_DIR = EXP_DIR.parent / "260823_domain_baseline_followup_wordings"
RESCORE_DIR = EXP_DIR.parent / "260824_rescore"

OUT_DIR = EXP_DIR / "probe_sets"

N_TRAIN_PER_CLASS = 300
SEED = 53
DEFINITIONS = ["rm", "swap"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_completion_index(rows: list[dict]) -> dict[tuple, dict]:
    """(question_id, condition, sample_idx) -> completions.jsonl row, cue_variant=='random' only."""
    idx: dict[tuple, dict] = {}
    for r in rows:
        if r.get("cue_variant") != "random":
            continue
        key = (r["question_id"], r["condition"], r["sample_idx"])
        assert key not in idx, f"duplicate completions.jsonl key: {key}"
        idx[key] = r
    return idx


def build_followup_index(rows: list[dict]) -> dict[tuple, dict]:
    """(question_id, condition, sample_idx, arm) -> followups.jsonl row, cue_variant=='random' only."""
    idx: dict[tuple, dict] = {}
    for r in rows:
        if r.get("cue_variant") != "random":
            continue
        key = (r["question_id"], r["condition"], r["sample_idx"], r["arm"])
        assert key not in idx, f"duplicate followups.jsonl key: {key}"
        idx[key] = r
    return idx


def sample_per_class(rows: list[dict], label_field: str, class_values: list[str],
                      n_per_class: int | None, seed: int) -> list[dict]:
    """Plain random shuffle-and-take per class. n_per_class=None keeps everything (eval)."""
    out = []
    rng = random.Random(seed)
    for cls in class_values:
        cls_rows = [r for r in rows if r[label_field] == cls]
        rng.shuffle(cls_rows)
        if n_per_class is not None:
            assert len(cls_rows) >= n_per_class, (
                f"{label_field}={cls}: only {len(cls_rows)} available, need {n_per_class}"
            )
            cls_rows = cls_rows[:n_per_class]
        out.extend(cls_rows)
    return out


def log_category_counts(rows: list[dict], label_field: str, name: str) -> dict:
    counts = Counter((r["category"], r[label_field]) for r in rows)
    print(f"  [{name}] per-category counts:")
    cats = sorted({c for c, _ in counts})
    for cat in cats:
        parts = ", ".join(f"{lbl}={counts.get((cat, lbl), 0)}" for lbl in sorted({l for _, l in counts}))
        print(f"    {cat}: {parts}")
    return {f"{cat}|{lbl}": n for (cat, lbl), n in counts.items()}


def build_bias_probe_set(definition: str, rescored: list[dict], comp_idx: dict[tuple, dict]) -> dict:
    label_field = f"bias_{definition}"
    manifest = {"definition": definition, "probe_type": "bias", "train": {}, "eval": {}}

    for split, n_per_class in [("train", N_TRAIN_PER_CLASS), ("eval", None)]:
        split_rows = [r for r in rescored if r["split"] == split and r[label_field] in ("bias", "no bias")]
        sampled = sample_per_class(split_rows, label_field, ["bias", "no bias"], n_per_class, seed=SEED)

        out_rows = []
        n_missing_join = 0
        for r in sampled:
            key = (r["question_id"], r["condition"], r["sample_idx"])
            c = comp_idx.get(key)
            if c is None:
                n_missing_join += 1
                continue
            out_rows.append({
                "question_id": r["question_id"],
                "condition": r["condition"],
                "sample_idx": r["sample_idx"],
                "category": r["category"],
                "split": split,
                "label": 1 if r[label_field] == "bias" else 0,
                "prompt": c["prompt"],
                "completion": c["completion"],
            })
        assert n_missing_join == 0, f"bias_{definition}/{split}: {n_missing_join} rows failed to join to completions.jsonl"

        out_path = OUT_DIR / f"bias_{definition}_{split}.jsonl"
        write_jsonl(out_path, out_rows)
        cat_counts = log_category_counts(sampled, label_field, f"bias_{definition}/{split}")
        manifest[split] = {
            "n_total": len(out_rows),
            "n_bias": sum(1 for r in out_rows if r["label"] == 1),
            "n_no_bias": sum(1 for r in out_rows if r["label"] == 0),
            "per_category": cat_counts,
            "path": str(out_path.relative_to(EXP_DIR)),
        }
        print(f"  wrote {out_path.name}: {len(out_rows)} rows "
              f"({manifest[split]['n_bias']} bias / {manifest[split]['n_no_bias']} no-bias)")
    return manifest


def build_deception_probe_set(definition: str, rescored: list[dict], fu_idx: dict[tuple, dict]) -> dict:
    label_field = f"deception_{definition}"
    arm_field = f"assigned_arm_{definition}"
    manifest = {"definition": definition, "probe_type": "deception", "train": {}, "eval": {}}

    for split, n_per_class in [("train", N_TRAIN_PER_CLASS), ("eval", None)]:
        split_rows = [r for r in rescored if r["split"] == split and r[label_field] in ("deceptive", "honest")]
        sampled = sample_per_class(split_rows, label_field, ["deceptive", "honest"], n_per_class, seed=SEED)

        out_rows = []
        n_missing_join = 0
        for r in sampled:
            key = (r["question_id"], r["condition"], r["sample_idx"], r[arm_field])
            fu = fu_idx.get(key)
            if fu is None:
                n_missing_join += 1
                continue
            out_rows.append({
                "question_id": r["question_id"],
                "condition": r["condition"],
                "sample_idx": r["sample_idx"],
                "category": r["category"],
                "split": split,
                "arm": r[arm_field],
                "label": 1 if r[label_field] == "deceptive" else 0,
                "prompt": fu["prompt"],
                "completion": fu["completion"],
                "followup_text": fu["followup_text"],
                "followup_completion": fu["followup_completion"],
            })
        assert n_missing_join == 0, f"deception_{definition}/{split}: {n_missing_join} rows failed to join to followups.jsonl"

        out_path = OUT_DIR / f"deception_{definition}_{split}.jsonl"
        write_jsonl(out_path, out_rows)
        cat_counts = log_category_counts(sampled, label_field, f"deception_{definition}/{split}")
        manifest[split] = {
            "n_total": len(out_rows),
            "n_deceptive": sum(1 for r in out_rows if r["label"] == 1),
            "n_honest": sum(1 for r in out_rows if r["label"] == 0),
            "per_category": cat_counts,
            "path": str(out_path.relative_to(EXP_DIR)),
        }
        print(f"  wrote {out_path.name}: {len(out_rows)} rows "
              f"({manifest[split]['n_deceptive']} deceptive / {manifest[split]['n_honest']} honest)")
    return manifest


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading source files ===")
    rescored = load_jsonl(RESCORE_DIR / "rescored_labels.jsonl")
    print(f"  rescored_labels.jsonl: {len(rescored)} rows")
    completions = load_jsonl(DOMAIN_DIR / "completions.jsonl")
    comp_idx = build_completion_index(completions)
    print(f"  completions.jsonl: {len(completions)} rows -> {len(comp_idx)} indexed (cue_variant=random)")
    followups = load_jsonl(FOLLOWUP_DIR / "followups.jsonl")

    # followups.jsonl doesn't carry prompt/completion (those live in completions.jsonl) --
    # join them in now, before indexing, so fu_idx's dicts already have them and
    # 02_extract_activations.py never needs to re-touch completions.jsonl.
    n_secondary_join_missing = 0
    for r in followups:
        if r.get("cue_variant") != "random":
            continue
        key = (r["question_id"], r["condition"], r["sample_idx"])
        c = comp_idx.get(key)
        if c is None:
            n_secondary_join_missing += 1
            continue
        r["prompt"] = c["prompt"]
        r["completion"] = c["completion"]
    assert n_secondary_join_missing == 0, (
        f"{n_secondary_join_missing} followups.jsonl rows failed to join back to completions.jsonl"
    )

    fu_idx = build_followup_index(followups)
    print(f"  followups.jsonl: {len(followups)} rows -> {len(fu_idx)} indexed (cue_variant=random)")

    all_manifests = []
    for definition in DEFINITIONS:
        print(f"\n=== bias_{definition} ===")
        all_manifests.append(build_bias_probe_set(definition, rescored, comp_idx))
    for definition in DEFINITIONS:
        print(f"\n=== deception_{definition} ===")
        all_manifests.append(build_deception_probe_set(definition, rescored, fu_idx))

    manifest_path = EXP_DIR / "probe_sets_manifest.json"
    manifest_path.write_text(json.dumps(all_manifests, indent=2))
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
