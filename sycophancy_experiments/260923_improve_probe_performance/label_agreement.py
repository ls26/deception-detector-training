"""Agreement of the 3 sampled answers inside each (question_id, condition) cell.

Every question has 3 samples in each of `none`, `cue_gold` and `cue_dist`
(260819 scores.jsonl: one row per cell, `contents[sample_idx]` in G / D / None,
None = no answer letter parsed). Under `rm`, a bias label reads the rollout's
own cued answer against the majority of the `none` cell, so a cell whose 3
samples disagree is where the label is least certain.

The `none` cell carries cue_variant "shared" (it serves both cue wordings), so
`build_scores_index` -- which keeps only cfg.cue_variant -- drops it. This module
builds its own cell index that admits both.
"""

from __future__ import annotations

import json

from probe_config import PipelineConfig

CELL_CONDITIONS: tuple[str, ...] = ("none", "cue_gold", "cue_dist")
SPLITS: tuple[str, ...] = ("train", "eval", "all")


def load_cells(cfg: PipelineConfig) -> dict[tuple[int, str], dict]:
    """(question_id, condition) -> scores.jsonl row, for cfg.cue_variant and the shared `none` cell."""
    # local jsonl reader: step 1 imports this module, so it cannot import step 1's
    with (cfg.domain_dir / "scores.jsonl").open() as f:
        scores_rows = [json.loads(line) for line in f if line.strip()]
    cells: dict[tuple[int, str], dict] = {}
    for row in scores_rows:
        if row["cue_variant"] not in (cfg.cue_variant, "shared"):
            continue
        key = (row["question_id"], row["condition"])
        assert key not in cells, f"duplicate scores.jsonl cell: {key}"
        assert row["n_samples"] == 3 and len(row["contents"]) == 3, f"cell {key} does not have 3 samples"
        cells[key] = row
    return cells


def is_unanimous(contents: list[str | None]) -> bool:
    """All 3 samples parsed and gave the same answer."""
    return None not in contents and len(set(contents)) == 1


def cell_agreement(cfg: PipelineConfig) -> list[dict]:
    """One row per (split, condition): how many cells are unanimous."""
    cells = load_cells(cfg)
    out: list[dict] = []
    for split in SPLITS:
        for condition in CELL_CONDITIONS:
            group = [
                row
                for (_, cell_condition), row in cells.items()
                if cell_condition == condition and (split == "all" or row["split"] == split)
            ]
            n_all_parsed = sum(None not in row["contents"] for row in group)
            n_unanimous = sum(is_unanimous(row["contents"]) for row in group)
            out.append(
                {
                    "split": split,
                    "condition": condition,
                    "n_cells": len(group),
                    "n_all_parsed": n_all_parsed,
                    "n_unanimous": n_unanimous,
                    "frac_unanimous": n_unanimous / len(group),
                    "frac_unanimous_of_parsed": n_unanimous / n_all_parsed,
                }
            )
    return out


def print_cell_agreement(rows: list[dict]) -> None:
    print(f"{'split':6s} {'condition':10s} {'cells':>6s} {'3 parsed':>9s} {'unanimous':>10s} {'frac':>7s} {'frac|parsed':>12s}")
    for r in rows:
        print(
            f"{r['split']:6s} {r['condition']:10s} {r['n_cells']:6d} {r['n_all_parsed']:9d} "
            f"{r['n_unanimous']:10d} {r['frac_unanimous']:7.3f} {r['frac_unanimous_of_parsed']:12.3f}"
        )


def none_unanimous_index(cfg: PipelineConfig) -> dict[int, bool]:
    """question_id -> did its 3 no-cue rollouts all give the same parsed answer."""
    cells = load_cells(cfg)
    return {
        question_id: is_unanimous(row["contents"])
        for (question_id, condition), row in cells.items()
        if condition == "none"
    }
