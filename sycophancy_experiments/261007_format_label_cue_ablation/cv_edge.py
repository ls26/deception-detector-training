"""One more CV cell past a grid edge, for the probes whose chosen cell sits on it.

    python cv_edge.py --arm ref_cot        # then: python cv_select.py --arm ref_cot

Reads each probe's cv/<arm>/<probe>/selection.json. If the chosen learning
rate is max(spec.CV_LEARNING_RATES), the cell (spec.CV_EDGE_LEARNING_RATE,
chosen steps) is run; if the chosen steps are max(spec.CV_MAX_STEPS), the
cell (chosen lr, spec.CV_EDGE_MAX_STEPS); both -> both cells plus (edge lr,
edge steps). Cells are run through cv_sweep.sweep_probe (resumable), one
probe after another; the caller re-runs cv_select.py, which picks over every
complete cell. Prints what it ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import spec  # noqa: E402
from cv_sweep import sweep_probe  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from probe_config import PipelineConfig, arm_config  # noqa: E402


def edge_cells(selection: dict) -> list[tuple[float, int]]:
    chosen = selection["chosen"]
    lr, steps = chosen["learning_rate"], chosen["max_steps"]
    on_lr_edge = lr == max(spec.CV_LEARNING_RATES)
    on_steps_edge = steps == max(spec.CV_MAX_STEPS)
    cells: list[tuple[float, int]] = []
    if on_lr_edge:
        cells.append((spec.CV_EDGE_LEARNING_RATE, steps))
    if on_steps_edge:
        cells.append((lr, spec.CV_EDGE_MAX_STEPS))
    if on_lr_edge and on_steps_edge:
        cells.append((spec.CV_EDGE_LEARNING_RATE, spec.CV_EDGE_MAX_STEPS))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=spec.ARMS)
    parser.add_argument("--arms-dir", type=Path, default=spec.ARMS_DIR)
    args = parser.parse_args()

    bootstrap_hf_env()
    from transformers import AutoTokenizer

    cfg: PipelineConfig = arm_config(args.arm, args.arms_dir.resolve())
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ran: dict[str, list] = {}
    for probe_name in cfg.probe_names():
        selection_path = cfg.cv_dir / probe_name / "selection.json"
        assert selection_path.exists(), f"{selection_path} missing -- run cv_select.py first"
        selection = json.loads(selection_path.read_text())
        assert selection["spec_hash"] == spec.spec_hash(), f"{selection_path} from another spec"
        cells = edge_cells(selection)
        ran[probe_name] = [list(c) for c in cells]
        if not cells:
            print(f"{probe_name}: chosen lr={selection['chosen']['learning_rate']:g} steps={selection['chosen']['max_steps']} is inside the grid, nothing to run")
            continue
        print(f"{probe_name}: chosen lr={selection['chosen']['learning_rate']:g} steps={selection['chosen']['max_steps']} on a grid edge -> cells {cells}")
        for lr, steps in cells:
            sweep_probe(probe_name, cfg, tokenizer, [lr], [steps])

    (cfg.cv_dir / "edge_cells.json").write_text(json.dumps({"spec_hash": spec.spec_hash(), "arm": cfg.arm, "cells_run": ran}, indent=2))
    print(f"\nEDGE DONE {cfg.arm}: {json.dumps(ran)}")


if __name__ == "__main__":
    main()
