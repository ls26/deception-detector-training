"""Cross-validation for one probe: 261007's sweep, selection and edge rule, unchanged.

    python cv_run.py --probe bias_followed_rm_answer
    python cv_run.py --probe bias_followed_rm_answer --dry-run   # dry/, lr 0.03 x DRY_STEPS (runs the edge rule)

Grid: learning rate x steps from 261007's spec.py (0.01, 0.03 x 2000, 4000),
5-fold GroupKFold over train question ids (seed 0), 32 per-layer probes per
fit, held-out logits stored per fold (cv/slice_cot/<probe>/lr*_steps*/fold*.json).
Selection (cv_select.select_probe): for every complete cell and every 4-layer
band, the mean over folds of the held-out AUC of the band's mean sigmoid; the
argmax is the choice. Edge rule (cv_edge.edge_cells): if the chosen lr is the
grid's largest, the cell (0.1, chosen steps) is added; if the chosen steps are
the largest, (chosen lr, 8000); both -> also (0.1, 8000); then the selection
is redone over every complete cell. The eval split is never read.
Writes cv/slice_cot/<probe>/{folds.json, selection.json, edge_cells.json}.
"""

from __future__ import annotations

import argparse
import json

import slice_spec as S
from slice_pipeline import bootstrap_hf_env, cv_edge, cv_select, cv_sweep, slice_config, tokenizer_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, choices=S.PROBES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = S.EXP_DIR / "dry" if args.dry_run else S.EXP_DIR
    bootstrap_hf_env()

    cfg = slice_config(root)
    tokenizer = tokenizer_for(cfg)
    lrs, steps = ([0.03], [S.DRY_STEPS]) if args.dry_run else (list(cfg.cv_learning_rates), list(cfg.cv_max_steps))
    cv_sweep.sweep_probe(args.probe, cfg, tokenizer, lrs, steps)
    selection = cv_select.select_probe(args.probe, cfg)

    edge = cv_edge.edge_cells(selection)
    print(f"edge cells: {edge}")
    for lr, max_steps in edge:
        cv_sweep.sweep_probe(args.probe, cfg, tokenizer, [lr], [max_steps])
    if edge:
        selection = cv_select.select_probe(args.probe, cfg)
    (cfg.cv_dir / args.probe / "edge_cells.json").write_text(json.dumps({"cells_run": [list(c) for c in edge]}, indent=2))
    chosen = selection["chosen"]
    print(f"\nCV DONE {args.probe}: lr={chosen['learning_rate']:g} steps={chosen['max_steps']} layers {chosen['layers']} OOF {chosen['oof_auc_mean']:.4f}")


if __name__ == "__main__":
    main()
