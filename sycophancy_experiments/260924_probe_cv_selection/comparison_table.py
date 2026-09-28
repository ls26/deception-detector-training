"""One comparison table: 260923's re-scored rounds beside this folder's rounds.

    python comparison_table.py --rescored 01_baseline 31_poolall_sgd_lr1e-2 \
        --rounds runs/01_cv_chosen_pool runs/02_cv_chosen_lossrule_pool runs/03_cv_chosen_pool_nu_evalnu \
        --out comparison.md

Every row is one (round, eval filter) pair. Re-scored 260923 rounds come from
rescored/<round>.json (rescore_rounds.py; both eval filters, the round's saved
layers). This folder's rounds come from their probe_metrics.json (the eval
filter they were trained/evaluated with) and, when rescored/<round>.json exists
for them too, the other eval filter as well. Cells are mean ± sd over seeds.
Three tables: auc_mean_sigmoid (the headline), auc_mean_logit, auc_filtered
(260923's headline); then the eval sizes and each round's recipe per probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from run_log import REPORTED_KINDS, probes_by_kind  # noqa: E402

METRICS: tuple[tuple[str, str], ...] = (
    ("auc_mean_sigmoid", "AUC of the mean per-layer sigmoid over the selected layers (headline; the RL pipeline's score)"),
    ("auc_mean_logit", "AUC of the mean per-layer logit over the selected layers"),
    ("auc_filtered", "mean of the per-layer AUCs over the selected layers (260923's headline)"),
)
EVAL_FILTERS: tuple[str, ...] = ("none", "none_unanimous")


def kind_of(probe_name: str) -> str:
    return probe_name.split("_rm_")[0]


def rows_from_rescored(path: Path) -> list[dict]:
    """One row per eval filter, from rescore_rounds.py output."""
    data = json.loads(path.read_text())
    rows = []
    for eval_filter in EVAL_FILTERS:
        cells = {}
        for probe_name, entry in data["aggregated"].items():
            a = entry["saved"][eval_filter]
            cells[kind_of(probe_name)] = {
                **{f"{m}_mean": a[f"{m}_mean"] for m, _ in METRICS},
                **{f"{m}_sd": a[f"{m}_sd"] for m, _ in METRICS},
                "n_eval": a["n_eval"], "n_eval_pos": a["n_eval_pos"], "n_eval_neg": a["n_eval_neg"],
                "layers": entry["filtered_layers_by_seed"][data["seeds"][0]],
            }
        rows.append({"round": data["round"], "eval_filter": eval_filter, "source": "re-scored saved weights", "cells": cells})
    return rows


def row_from_metrics(round_dir: Path) -> dict:
    metrics = json.loads((round_dir / "probe_metrics.json").read_text())
    by_kind = probes_by_kind(metrics)
    cells = {}
    for kind, p in by_kind.items():
        cells[kind] = {
            **{f"{m}_mean": p[f"{m}_mean"] for m, _ in METRICS},
            **{f"{m}_sd": p[f"{m}_sd"] for m, _ in METRICS},
            "n_eval": p["n_eval"], "n_eval_pos": p["n_eval_pos"], "n_eval_neg": p["n_eval_neg"],
            "n_train": p["n_train"], "layers": p["filtered_layers"],
            "learning_rate": p["learning_rate"], "max_steps": p["max_steps"], "epochs": p["epochs"],
            "layer_selection": p["layer_selection"],
        }
    return {
        "round": round_dir.name,
        "eval_filter": metrics["config"]["eval_filter"],
        "source": f"trained here on {Path(metrics['config']['act_dir']).name}",
        "cells": cells,
    }


def fmt(cell: dict, metric: str) -> str:
    return f"{cell[f'{metric}_mean']:.3f} ± {cell[f'{metric}_sd']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescored", nargs="*", default=[], help="names under rescored/ (without .json)")
    parser.add_argument("--rounds", nargs="*", default=[], help="this folder's runs/<NN>_<slug>")
    parser.add_argument("--out", default="comparison.md")
    args = parser.parse_args()

    rows: list[dict] = []
    for name in args.rescored:
        rows.extend(rows_from_rescored(EXP_DIR / "rescored" / f"{name}.json"))
    for name in args.rounds:
        round_dir = EXP_DIR / name
        rows.append(row_from_metrics(round_dir))
        rescored = EXP_DIR / "rescored" / f"{round_dir.name}.json"
        if rescored.exists():
            trained_filter = rows[-1]["eval_filter"]
            rows.extend(r for r in rows_from_rescored(rescored) if r["eval_filter"] != trained_filter)

    header = "| round | eval filter | source | " + " | ".join(REPORTED_KINDS) + " |"
    rule = "|---|---|---|" + "---|" * len(REPORTED_KINDS)
    lines = ["# Comparison: 260923 rounds vs this folder's rounds", ""]
    lines.append("Eval AUC, mean ± sd over seeds. Rounds trained here use their CV-chosen per-probe recipe (table at the end).")
    lines.append("")
    for metric, description in METRICS:
        lines += [f"## {metric}", "", description, "", header, rule]
        for r in rows:
            cells = [fmt(r["cells"][k], metric) if k in r["cells"] else "—" for k in REPORTED_KINDS]
            lines.append(f"| {r['round']} | {r['eval_filter']} | {r['source']} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## eval rows (n, pos/neg)", "", "| eval filter | " + " | ".join(REPORTED_KINDS) + " |", "|---|" + "---|" * len(REPORTED_KINDS)]
    for eval_filter in EVAL_FILTERS:
        r = next((r for r in rows if r["eval_filter"] == eval_filter and all(k in r["cells"] for k in REPORTED_KINDS)), None)
        if r is None:
            continue
        lines.append(f"| {eval_filter} | " + " | ".join(f"{r['cells'][k]['n_eval']} ({r['cells'][k]['n_eval_pos']}/{r['cells'][k]['n_eval_neg']})" for k in REPORTED_KINDS) + " |")
    lines.append("")

    lines += ["## recipe per probe (rounds trained here)", "", "| round | probe | n_train | lr | steps | epochs | layers |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "trained here" not in r["source"]:
            continue
        for k in REPORTED_KINDS:
            if k not in r["cells"]:
                continue
            c = r["cells"][k]
            lines.append(
                f"| {r['round']} | {k} | {c['n_train']} | {c['learning_rate']:g} | {c['max_steps']} | {c['epochs']:.1f} | "
                f"{c['layers'][0]}-{c['layers'][-1]} ({len(c['layers'])}, {c['layer_selection']}) |"
            )
    lines.append("")
    lines += ["## selected layers (re-scored rounds, seed 0)", "", "| round | " + " | ".join(REPORTED_KINDS) + " |", "|---|" + "---|" * len(REPORTED_KINDS)]
    seen = set()
    for r in rows:
        if "trained here" in r["source"] or r["round"] in seen:
            continue
        seen.add(r["round"])
        layer_cells = [
            f"{r['cells'][k]['layers'][0]}-{r['cells'][k]['layers'][-1]} ({len(r['cells'][k]['layers'])})" if k in r["cells"] else "—"
            for k in REPORTED_KINDS
        ]
        lines.append(f"| {r['round']} | " + " | ".join(layer_cells) + " |")
    lines.append("")

    out = EXP_DIR / args.out
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
