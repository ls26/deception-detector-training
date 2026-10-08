"""Step 1: arms/<arm>/labels.jsonl -> probe_sets/<dataset>_{train,eval}.jsonl.

Trimmed copy of ../260923_improve_probe_performance/step1_build_probe_sets.py.
The only source is the arm's labels.jsonl (SCHEMA.md): every row already
carries its labels (`bias_rm`, `said_rm`, `deception_rm`), the turn-1 and
turn-2 text and `followed_cue`, so there is no join and no relabelling.

Writes one train/eval JSONL per DATASET (not per probe): bias_rm,
deception_rm, bias_followed_rm.

Pools (per (question_id, condition) prompt), both splits: one row per prompt,
drawn at random (seeded `pool_seed`) AFTER the class filter, so a prompt is
lost only when every one of its rows was excluded.

Train draws, from the pool, balanced across cue_gold / cue_dist within each
class, sizes from arms/sizes.json (`cfg.n_train_per_class`):
  bias_rm            n/class
  deception_yes      n/class   false vs true ADMISSION
  deception_no       n/class   false vs true DENIAL
  deception_rm       the union of the two stratified draws
  bias_followed_rm   n/class, rows whose answer went with the cue only
Eval is the entire pool for the dataset: no per-class cap, no rebalancing.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import spec
from probe_config import CONDITIONS, EXP_DIR, PROBE_KINDS, SAID_FILTER, PipelineConfig, config_to_json

#: Class values per probe kind, positive class (label 1 = deceptive/biased) first.
CLASSES: dict[str, tuple[str, str]] = {
    "bias": ("bias", "no bias"),
    "deception": ("deceptive", "honest"),
    "deception_yes": ("false admission", "true admission"),
    "deception_no": ("false denial", "true denial"),
    "bias_followed": ("bias", "no bias"),
}

#: labels.jsonl fields a probe-set row carries through (when present)
CARRIED_FIELDS: tuple[str, ...] = (
    "question_id",
    "condition",
    "sample_idx",
    "category",
    "split",
    "answer_letter",
    "cue_letter",
    "followed_cue",
    "prompt",
    "completion",
    "followup_text",
    "followup_completion",
    "ans_pos",
    "first_token_ids",
    "lp_a",
    "lp_b",
    "p_a_cued",
    "confident",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rel(path: Path) -> str:
    return str(path.relative_to(EXP_DIR)) if path.is_relative_to(EXP_DIR) else str(path)


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def deception_subtype(row: dict, definition: str) -> str:
    """Split `deception_{def}` into its four subtypes using `bias_{def}` + `said_{def}`.

    Rows carrying a non-label value (`excluded_antibias`, `unparsed_turn1`,
    `unparsed_turn2`) pass through unchanged so nothing is silently hidden.
    """
    label = row[f"deception_{definition}"]
    if label not in ("deceptive", "honest"):
        return label
    bias, said = row[f"bias_{definition}"], row[f"said_{definition}"]
    if label == "deceptive":
        subtype = "false admission" if bias == "no bias" else "false denial"
        expected = "Yes" if bias == "no bias" else "No"
    else:
        subtype = "true admission" if bias == "bias" else "true denial"
        expected = "Yes" if bias == "bias" else "No"
    assert said == expected, (
        f"{row['question_id']}/{row['condition']}/{row['sample_idx']}: "
        f"{label} + bias={bias!r} implies said={expected!r}, got {said!r}"
    )
    return subtype


def label_fn(kind: str, definition: str):
    """The label a probe kind reads off a labels.jsonl row."""
    if kind in ("bias", "bias_followed"):
        return lambda r: r[f"bias_{definition}"]
    if kind == "deception":
        return lambda r: r[f"deception_{definition}"]
    return lambda r: deception_subtype(r, definition)


def is_positive(kind: str, value: str) -> bool:
    """label 1 is the biased / deceptive side of every contrast."""
    return value == CLASSES[kind][0]


def is_eligible(row: dict, kind: str, definition: str) -> bool:
    """The row filter a probe kind applies to labels.jsonl rows: its label is one
    of the kind's two classes; `bias_followed` also needs the answer to have gone
    with the cue."""
    if label_fn(kind, definition)(row) not in CLASSES[kind]:
        return False
    if kind == "bias_followed":
        return row["followed_cue"] is True
    return True


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #


def load_labels(cfg: PipelineConfig) -> list[dict]:
    rows = load_jsonl(cfg.labels_path)
    assert rows, f"{cfg.labels_path} is empty"
    for r in rows:
        assert r["spec_hash"] == spec.spec_hash(), f"{cfg.labels_path}: row stamped with another spec"
        assert r["condition"] in CONDITIONS, f"labels.jsonl carries a non-cued row: {r['condition']}"
    print(f"=== {cfg.labels_path}: {len(rows)} rows ===")
    return rows


# --------------------------------------------------------------------------- #
# pools and sampling
# --------------------------------------------------------------------------- #


def samples_per_prompt(cfg: PipelineConfig, split: str) -> int:
    return cfg.train_samples_per_prompt if split == "train" else cfg.eval_samples_per_prompt


def subsample_prompts(rows: list[dict], n_per_prompt: int, seed: int) -> list[dict]:
    """Admit at most `n_per_prompt` rows per (question_id, condition) prompt.

    Deterministic: prompts are visited in sorted key order and the draw comes
    from a single seeded RNG.
    """
    by_prompt: dict[tuple, list[dict]] = {}
    for r in rows:
        by_prompt.setdefault((r["question_id"], r["condition"]), []).append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for key in sorted(by_prompt):
        group = sorted(by_prompt[key], key=lambda r: r["sample_idx"])
        out.extend(group if len(group) <= n_per_prompt else rng.sample(group, n_per_prompt))
    return out


def select_pool(labels: list[dict], family: str, definition: str, split: str, cfg: PipelineConfig) -> tuple[list[dict], list[dict]]:
    """(eligible rows, pool) for one dataset/split.

    Eligible = this split's rows whose label is one of the dataset's two classes
    (anti-bias and unparsed rows are excluded), `bias_followed` additionally
    only rows whose answer went with the cue. Pool = eligible after the
    per-prompt subsample.
    """
    eligible = [r for r in labels if r["split"] == split and is_eligible(r, family, definition)]
    pool = subsample_prompts(eligible, samples_per_prompt(cfg, split), cfg.pool_seed)
    return eligible, pool


def train_pool_per_cell(labels: list[dict], kind: str, definition: str, cfg: PipelineConfig) -> dict[str, dict[str, int]]:
    """{class: {condition: n}}: rows per (class, condition) cell in the train
    pool a probe kind draws from -- the most step 1 can draw per cell, and the
    quantity arms/sizes.json's size rule is computed from. The pool keeps one
    row per prompt, so a prompt with rows in both classes counts in one only."""
    family = next(family for family, kinds in PROBE_KINDS.items() if kind in kinds)
    _, pool = select_pool(labels, family, definition, "train", cfg)
    label_of = label_fn(kind, definition)
    return {
        cls: {cond: sum(1 for r in pool if label_of(r) == cls and r["condition"] == cond) for cond in CONDITIONS}
        for cls in CLASSES[kind]
    }


def sample_per_class(
    rows: list[dict],
    label_of,
    class_values: tuple[str, ...],
    n_per_class: int | None,
    seed: int,
    conditions: tuple[str, ...] | None = None,
) -> list[dict]:
    """Shuffle-and-take per class.

    n_per_class=None keeps everything (eval). `conditions` splits each class's
    draw equally across those cue conditions. A short cell prints a WARNING and
    takes every row it has; extract.py's step-1 gates then fail on the
    imbalance, so the size rule in arms/sizes.json is what keeps cells full.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for cls in class_values:
        cls_rows = [r for r in rows if label_of(r) == cls]
        if n_per_class is None:
            rng.shuffle(cls_rows)
            out.extend(cls_rows)
            continue
        assert conditions is not None
        assert n_per_class % len(conditions) == 0, f"n_per_class={n_per_class} does not divide evenly over {len(conditions)} conditions"
        per_cond = n_per_class // len(conditions)
        for cond in conditions:
            cond_rows = [r for r in cls_rows if r["condition"] == cond]
            rng.shuffle(cond_rows)
            if len(cond_rows) < per_cond:
                print(f"  WARNING: class {cls!r} / {cond}: only {len(cond_rows)} available, need {per_cond} -- taking all")
            out.extend(cond_rows[:per_cond])
    return out


# --------------------------------------------------------------------------- #
# breakdown tables: label x cue condition
# --------------------------------------------------------------------------- #


def contingency(rows: list[dict], label_of) -> dict[str, dict[str, int]]:
    """{label: {cue_gold: n, cue_dist: n}}"""
    counts = Counter((label_of(r), r["condition"]) for r in rows)
    table: dict[str, dict[str, int]] = {}
    for (label, condition), n in counts.items():
        table.setdefault(label, {})[condition] = n
    return table


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


@dataclass
class ProbeSet:
    """One dataset's train+eval rows, in memory, plus its manifest entry."""

    family: str
    definition: str
    rows: dict[str, list[dict]] = field(default_factory=dict)
    pool: dict[str, list[dict]] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.family}_{self.definition}"


def output_row(r: dict, split: str, subtype: str, label: int) -> dict:
    out = {k: r[k] for k in CARRIED_FIELDS if k in r}
    out["split"] = split
    out["subtype"] = subtype
    out["label"] = label
    return out


def _manifest_entry(rows: list[dict], pool: list[dict], eligible: list[dict], split: str, cfg: PipelineConfig, out_path: Path) -> dict:
    return {
        "n_total": len(rows),
        "n_pos": sum(1 for r in rows if r["label"] == 1),
        "n_neg": sum(1 for r in rows if r["label"] == 0),
        "n_eligible": len(eligible),
        "n_pool": len(pool),
        "n_prompts_in_pool": len({(r["question_id"], r["condition"]) for r in pool}),
        "n_prompts": len({(r["question_id"], r["condition"]) for r in rows}),
        "n_questions": len({r["question_id"] for r in rows}),
        "samples_per_prompt": samples_per_prompt(cfg, split),
        "by_condition": contingency(rows, lambda r: r["subtype"]),
        "by_category": dict(Counter(r["category"] for r in rows)),
        "path": _rel(out_path),
    }


def build_bias_probe_set(definition: str, labels: list[dict], cfg: PipelineConfig, *, family: str = "bias", write: bool = True) -> ProbeSet:
    """bias_{def}: label 1 = bias, 0 = no bias; anti-bias dropped entirely.
    `family="bias_followed"` restricts to rows whose answer went with the cue,
    as its own draw."""
    ps = ProbeSet(family=family, definition=definition)
    ps.manifest = {"dataset": ps.name, "family": family, "definition": definition}
    label_of = label_fn(family, definition)

    for split in cfg.splits:
        eligible, pool = select_pool(labels, family, definition, split, cfg)
        n_per_class = cfg.n_train_per_class[family] if split == "train" else None
        sampled = sample_per_class(pool, label_of, CLASSES[family], n_per_class, cfg.sample_seed, conditions=CONDITIONS if n_per_class else None)

        out_rows = [output_row(r, split, label_of(r), 1 if is_positive(family, label_of(r)) else 0) for r in sampled]
        for r in out_rows:
            assert r["answer_letter"] in spec.LETTERS, f"{family}/{split}: row without a parsed answer letter: {r['answer_letter']!r}"
        if family == "bias_followed":
            assert all(r["followed_cue"] is True for r in out_rows), f"{family}_{definition}/{split}: a row that did not follow the cue got through"

        out_path = cfg.probe_sets_dir / f"{family}_{definition}_{split}.jsonl"
        if write:
            write_jsonl(out_path, out_rows)
        ps.rows[split] = out_rows
        ps.pool[split] = pool
        ps.manifest[split] = _manifest_entry(out_rows, pool, eligible, split, cfg, out_path)
        m = ps.manifest[split]
        print(f"  {'wrote' if write else 'built'} {out_path.name}: {m['n_total']} rows ({m['n_pos']} bias / {m['n_neg']} no-bias) from a pool of {m['n_pool']}")
    return ps


def build_deception_probe_set(definition: str, labels: list[dict], cfg: PipelineConfig, *, write: bool = True) -> ProbeSet:
    """deception_{def}: label 1 = deceptive, 0 = honest.

    Train is drawn as two stratified halves and concatenated -- said-Yes rows
    (false vs true admission) and said-No rows (false vs true denial), each
    cfg.n_train_per_class["deception_yes"/"deception_no"] per class, each split
    evenly across cue conditions. The deception_yes / deception_no probes are
    this same dataset masked on `said`.
    """
    ps = ProbeSet(family="deception", definition=definition)
    ps.manifest = {"dataset": ps.name, "family": "deception", "definition": definition}
    subtype_of = lambda r: deception_subtype(r, definition)  # noqa: E731

    for split in cfg.splits:
        eligible, pool = select_pool(labels, "deception", definition, split, cfg)

        if split == "train":
            sampled = []
            strata: dict[str, int] = {}
            for kind in ("deception_yes", "deception_no"):
                part = sample_per_class(pool, subtype_of, CLASSES[kind], cfg.n_train_per_class[kind], cfg.sample_seed, conditions=CONDITIONS)
                for r in part:
                    assert r[f"said_{definition}"] == SAID_FILTER[kind], f"{kind} drew a said={r[f'said_{definition}']!r} row"
                strata[kind] = len(part)
                sampled.extend(part)
            ps.manifest["train_strata"] = strata
        else:
            # eval is the whole pool, contrasted on the deceptive/honest label itself
            sampled = sample_per_class(pool, label_fn("deception", definition), CLASSES["deception"], None, cfg.sample_seed)

        out_rows = []
        for r in sampled:
            row = output_row(r, split, subtype_of(r), 1 if r[f"deception_{definition}"] == "deceptive" else 0)
            row["arm"] = r["arm"]
            row["said"] = r[f"said_{definition}"]
            assert row["followup_text"] is not None and row["followup_completion"] is not None, (
                f"deception_{definition}/{split}: {r['question_id']}/{r['condition']}/{r['sample_idx']} has a deception label but no follow-up text"
            )
            out_rows.append(row)

        out_path = cfg.probe_sets_dir / f"deception_{definition}_{split}.jsonl"
        if write:
            write_jsonl(out_path, out_rows)
        ps.rows[split] = out_rows
        ps.pool[split] = pool
        ps.manifest[split] = _manifest_entry(out_rows, pool, eligible, split, cfg, out_path)
        ps.manifest[split]["n_said_yes"] = sum(1 for r in out_rows if r["said"] == "Yes")
        ps.manifest[split]["n_said_no"] = sum(1 for r in out_rows if r["said"] == "No")
        m = ps.manifest[split]
        print(
            f"  {'wrote' if write else 'built'} {out_path.name}: {m['n_total']} rows ({m['n_pos']} deceptive / {m['n_neg']} honest; "
            f"{m['n_said_yes']} said-Yes / {m['n_said_no']} said-No) from a pool of {m['n_pool']}"
        )
    return ps


_BUILDERS = {
    "bias": build_bias_probe_set,
    "deception": build_deception_probe_set,
    "bias_followed": partial(build_bias_probe_set, family="bias_followed"),
}


def write_manifest(probe_sets: list[ProbeSet], cfg: PipelineConfig) -> Path:
    cfg.manifest_path.write_text(
        json.dumps(
            {
                "spec_hash": spec.spec_hash(),
                "config": config_to_json(cfg),
                "probes": [{"name": s.name, "kind": s.kind, "dataset": s.dataset, "said": s.said, "position": s.position} for s in cfg.probe_specs()],
                "datasets": [ps.manifest for ps in probe_sets],
            },
            indent=2,
        )
    )
    return cfg.manifest_path


def build_all(cfg: PipelineConfig, labels: list[dict] | None = None, *, write: bool = True) -> list[ProbeSet]:
    """Build every (family, definition) dataset in cfg and write the manifest."""
    if labels is None:
        labels = load_labels(cfg)
    if write:
        cfg.probe_sets_dir.mkdir(parents=True, exist_ok=True)
    probe_sets = []
    for family, definition in cfg.dataset_keys():
        print(f"\n=== {family}_{definition} ===")
        probe_sets.append(_BUILDERS[family](definition, labels, cfg, write=write))
    if write:
        print(f"\nWrote manifest: {write_manifest(probe_sets, cfg)}")
    return probe_sets
