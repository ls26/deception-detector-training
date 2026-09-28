"""Per-probe analysis of one recorded round: slice tables and the 2x2 figure.

    python analyze_round.py --round runs/01_cv_chosen_pool
    python analyze_round.py --round runs/03_cv_chosen_pool_nu_evalnu --cv-variant pool

For each reported probe, `step3.analyze_probe` re-scores the round's seed-0
weights on the cached activations (train and eval, under the round's own
eval_filter and per-probe recipe), prints the combined-score AUC by cue
condition, cue-following, self-report, category and no-cue unanimity, and saves
runs/<round>/analysis/<probe>_analysis.{json,png}. Panel 2 of the figure adds
the per-layer out-of-fold AUC from cv/<cv-variant>/<probe>/selection.json when
that file exists (the chosen (lr, steps) cell), so the CV curve can be read
against the eval curve. Panel 4 uses the round's stored final train loss per
layer. Also writes analysis/slices.md with every slice table. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import step3_train_eval_probes as s3  # noqa: E402
from probe_config import apply_overrides, default_config  # noqa: E402
from run_log import REPORTED_KINDS, probes_by_kind  # noqa: E402


def round_config(round_dir: Path):
    """The round's config, rebuilt from its config.json (paths and the fields that change rows or recipes)."""
    saved = json.loads((round_dir / "config.json").read_text())
    cfg = replace(
        default_config(),
        act_dir=Path(saved["act_dir"]),
        weights_dir=round_dir / "probe_weights" / f"seed{saved['probe_seeds'][0]}",
        train_draw_per_class=saved["train_draw_per_class"],
        holdout_category=saved["holdout_category"],
        eval_filter=saved["eval_filter"],
        combined_score=saved["combined_score"],
    )
    return apply_overrides(cfg, {"per_probe_overrides": saved["per_probe_overrides"]})


def slices_markdown(results: dict[str, dict]) -> str:
    lines = []
    for probe_name, r in results.items():
        lines.append(f"## {probe_name}")
        lines.append("")
        lines.append(
            f"layers {r['filtered_layers'][0]}-{r['filtered_layers'][-1]} ({len(r['filtered_layers'])}), "
            f"eval_filter={r['eval_filter']}, n_eval={r['n_eval']} ({r['n_eval_pos']}/{r['n_eval_neg']}), "
            f"overall AUC ({r['combined_score']}) {r['auc_overall']:.3f}"
        )
        lines.append("")
        for by, rows in r["slices"].items():
            if not rows:
                continue
            lines.append(f"| by {by} | AUC | n | pos | neg |")
            lines.append("|---|---|---|---|---|")
            for row in rows:
                auc = "nan" if row["auc"] != row["auc"] else f"{row['auc']:.3f}"
                lines.append(f"| {row['value']} | {auc} | {row['n']} | {row['n_pos']} | {row['n_neg']} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, help="runs/<NN>_<slug>")
    parser.add_argument("--cv-variant", default="pool", help="cv/<variant>/<probe>/selection.json supplies the OOF curve")
    args = parser.parse_args()

    round_dir = Path(args.round)
    if not round_dir.is_absolute():
        round_dir = EXP_DIR / round_dir
    cfg = round_config(round_dir)
    metrics = json.loads((round_dir / "probe_metrics.json").read_text())
    by_kind = probes_by_kind(metrics)
    analysis_dir = round_dir / "analysis"

    results: dict[str, dict] = {}
    for kind in REPORTED_KINDS:
        if kind not in by_kind:  # a round may hold a subset of the probes (round.py --probes)
            continue
        entry = by_kind[kind]
        probe_name = entry["probe"]
        selection_path = cfg.cv_dir / args.cv_variant / probe_name / "selection.json"
        oof_curve = json.loads(selection_path.read_text())["chosen"]["per_layer_oof_auc"] if selection_path.exists() else None
        result = s3.analyze_probe(
            probe_name,
            cfg,
            train_loss_final_per_layer=entry["train_loss_final_per_layer"],
            oof_auc_per_layer=oof_curve,
            save_dir=analysis_dir,
        )
        result.pop("scores")
        result.pop("figure")
        results[probe_name] = result
        print()
    (analysis_dir / "slices.md").write_text(f"# {round_dir.name}: combined-score AUC by slice (seed 0 weights)\n\n" + slices_markdown(results))
    print(f"wrote {analysis_dir}/<probe>_analysis.{{json,png}} and {analysis_dir / 'slices.md'}")


if __name__ == "__main__":
    main()
