"""Append one finished round's row to runs.md.

    python append_row.py --round runs/01_baseline --change "..." --decision "..."

Reads the round's config.json for the activation variant and metrics path, so
it needs nothing from the session that ran the round.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import run_log  # noqa: E402
from probe_config import default_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, help="runs/<NN>_<slug>")
    parser.add_argument("--change", required=True)
    parser.add_argument("--decision", required=True)
    args = parser.parse_args()

    round_dir = Path(args.round)
    if not round_dir.is_absolute():
        round_dir = EXP_DIR / round_dir
    saved = json.loads((round_dir / "config.json").read_text())
    round_cfg = replace(
        default_config(), act_dir=Path(saved["act_dir"]), metrics_path=Path(saved["metrics_path"])
    )
    assert round_cfg.metrics_path.exists(), f"{round_cfg.metrics_path} missing -- round not finished"
    run_log.append_runs_table(round_cfg, args.change, args.decision)


if __name__ == "__main__":
    main()
