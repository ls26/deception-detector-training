"""261104's eval-answers figure with this folder's no-cue ceiling at KL 0.01 in place of 261020's at KL 0.001.

    python plot_results.py     # -> figures/261104/fig1_eval_answers.png and .pdf

Runs 261104's plot_results.py fig1 unchanged (its docstring describes the panels), with two differences: the dash-dot
line is the mean over seeds 42-44 of this folder's runs (tables/nocue_kl/seed_table.json, "mean_sd"), and its legend
entry says KL 0.01. The bars (261104's rounds 1-5) and the dashed line (261103's no-penalty runs) read the same files as
261104's figure. 261104's other two figures do not draw the ceiling and are not redrawn here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = EXP_DIR / "figures" / "261104"
CEILING_TABLE = EXP_DIR / "tables" / "nocue_kl" / "seed_table.json"
PLOT_261104 = EXP_DIR.parent / "261104_iterative_kl_0_01" / "plot_results.py"


def load_plot_261104():
    sys.dont_write_bytecode = True  # no __pycache__ written into 261104's folder
    spec = importlib.util.spec_from_file_location("plot_261104", PLOT_261104)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    plot = load_plot_261104()
    plot.FIG_DIR = FIG_DIR
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot.REFERENCE_LINES["ceiling"]["label"] = "No-cue ceiling: mean of 3 runs trained without the cue (KL 0.01)"
    round_tables = {}
    for seed in plot.SEEDS:
        table = json.loads(plot.ROUND_TABLES[seed].read_text())
        assert list(table["runs"]) == list(plot.ROUNDS), (seed, list(table["runs"]))
        round_tables[seed] = table["mean_sd"]
    references = {"no_penalty": json.loads(plot.NO_PENALTY_TABLE.read_text())["mean_sd"],
                  "ceiling": json.loads(CEILING_TABLE.read_text())["mean_sd"]}
    plot.fig1(round_tables, references)


if __name__ == "__main__":
    main()
