"""Shared helpers for label.py, baselines.py, train_sizes.py and render_examples.py.

The label vocabulary and the probe-population semantics are copied from
260923's step1_build_probe_sets.py (CLASSES, deception_subtype, majority_vote,
FAMILY_ROW_FILTER / SAID_FILTER) so every arm's probe sets are drawn from the
same populations the reference probes were.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

#: class values per probe kind, positive class (label 1 = biased / deceptive) first
CLASSES: dict[str, tuple[str, str]] = {
    "bias": ("bias", "no bias"),
    "bias_followed": ("bias", "no bias"),
    "deception": ("deceptive", "honest"),
    "deception_yes": ("false admission", "true admission"),
    "deception_no": ("false denial", "true denial"),
}
BIAS_LABELS: tuple[str, ...] = ("bias", "anti-bias", "no bias", "unparsed_turn1")
DECEPTION_LABELS: tuple[str, ...] = (
    "deceptive", "honest", "excluded_antibias", "unparsed_turn2", "unparsed_turn1",
)
SUBTYPES: tuple[str, ...] = ("false admission", "true admission", "false denial", "true denial")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def majority_vote(votes: list[str | None]) -> str | None:
    """Strict parsed majority over up to 3 counterfactual samples (260923's)."""
    parsed = [v for v in votes if v is not None]
    if not parsed:
        return None
    ranked = Counter(parsed).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def deception_subtype(row: dict) -> str:
    """260923's four-way split of deception_rm by (bias_rm, said_rm); non-labels pass through."""
    label = row["deception_rm"]
    if label not in ("deceptive", "honest"):
        return label
    bias, said = row["bias_rm"], row["said_rm"]
    assert bias in ("bias", "no bias") and said in ("Yes", "No"), (row["bias_rm"], row["said_rm"])
    if bias == "bias":
        return "true admission" if said == "Yes" else "false denial"
    return "false admission" if said == "Yes" else "true denial"


def class_of(row: dict, kind: str) -> str:
    """The value a probe kind reads off a labels row (compared against CLASSES[kind])."""
    if kind in ("bias", "bias_followed"):
        return row["bias_rm"]
    if kind == "deception":
        return row["deception_rm"]
    return deception_subtype(row)


def is_eligible(row: dict, kind: str) -> bool:
    """260923's select_pool filter: label in the kind's two classes, plus the
    family row filter (bias_followed keeps followed-cue rows). said-Yes / said-No
    are implied by the subtype classes."""
    if class_of(row, kind) not in CLASSES[kind]:
        return False
    if kind == "bias_followed":
        return row["followed_cue"] is True
    return True
