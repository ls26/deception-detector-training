"""Choose (learning rate, max_steps, layer band) per probe from cv_sweep.py's
out-of-fold logits. CPU only, seconds.

    python cv_select.py --arm ref_cot

For every complete cell (all cv_folds fold files, including any cv_edge.py
cell) and every fold, the held-out rows' combined score (`cfg.combined_score`)
is formed over each candidate layer band and its AUC taken; the cell's score
for a band is the mean over folds (fold sd reported; the pooled-over-folds AUC
is stored as well). Candidate bands: every contiguous band of width
spec.CV_BAND_WIDTH. The chosen recipe per probe is the argmax over (cell,
band) of the fold-mean AUC. The eval split is never read.

Writes, under cv/<arm>/:
  <probe>/selection.json   every cell x band, per-layer OOF AUC, the choice
  summary.md               one table per probe
  chosen_overrides.json    {"per_probe_overrides": {...}} for final_fit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import spec  # noqa: E402
import step3_train_eval_probes as s3  # noqa: E402
from probe_config import PipelineConfig, arm_config  # noqa: E402


def band_key(band: tuple[int, int]) -> str:
    return f"{band[0]}-{band[0] + band[1] - 1}"


def candidate_bands(cfg: PipelineConfig) -> list[tuple[int, int]]:
    return [(start, width) for width in cfg.cv_band_widths for start in range(cfg.n_layers - width + 1)]


def load_cells(probe_dir: Path, cfg: PipelineConfig) -> list[dict]:
    """Complete cells only, as {lr, steps, folds: [fold json, ...]} sorted by (lr, steps)."""
    cells = []
    for cell_dir in sorted(p for p in probe_dir.iterdir() if p.is_dir()):
        fold_files = sorted(cell_dir.glob("fold*.json"))
        if len(fold_files) < cfg.cv_folds:
            print(f"  {cell_dir.relative_to(probe_dir.parent)}: {len(fold_files)}/{cfg.cv_folds} folds, skipped")
            continue
        folds = [json.loads(f.read_text()) for f in fold_files]
        for f in folds:
            assert f["spec_hash"] == spec.spec_hash(), f"{cell_dir}: fold file from another spec"
        cells.append({"lr": folds[0]["learning_rate"], "steps": folds[0]["max_steps"], "folds": folds})
    return sorted(cells, key=lambda c: (c["lr"], c["steps"]))


def layer_set_auc(folds: list[dict], layers_per_fold: list[list[int]], method: str) -> dict:
    """Fold-mean, fold-sd and pooled AUC of the combined score over the given layers."""
    per_fold, all_labels, all_scores = [], [], []
    for fold, layers in zip(folds, layers_per_fold):
        labels = np.array(fold["labels"])
        logits = np.array(fold["logits"])[layers]
        scores = s3.combined_score(logits, method)
        per_fold.append(float(roc_auc_score(labels, scores)))
        all_labels.append(labels)
        all_scores.append(scores)
    per_fold = np.array(per_fold)
    return {
        "mean": float(per_fold.mean()),
        "sd": float(per_fold.std(ddof=1)),
        "pooled": float(roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores))),
        "per_fold": per_fold.tolist(),
    }


def score_cell(cell: dict, cfg: PipelineConfig) -> dict:
    folds = cell["folds"]
    method = cfg.combined_score
    n_layers = cfg.n_layers
    per_layer = np.array([[roc_auc_score(np.array(f["labels"]), np.array(f["logits"])[layer]) for layer in range(n_layers)] for f in folds])
    bands = {}
    for band in candidate_bands(cfg):
        layers = s3.band_layers(band, n_layers)
        bands[band_key(band)] = {"band": list(band), **layer_set_auc(folds, [layers] * len(folds), method)}
    best_key = max(bands, key=lambda k: bands[k]["mean"])
    loss_rule_layers = [f["layers_below_threshold"] or list(range(n_layers)) for f in folds]
    return {
        "lr": cell["lr"],
        "steps": cell["steps"],
        "epochs_mean": float(np.mean([f["epochs"] for f in folds])),
        "n_fit_mean": float(np.mean([f["n_fit"] for f in folds])),
        "seconds_mean": float(np.mean([f["seconds"] for f in folds])),
        "per_layer_oof_auc": per_layer.mean(axis=0).tolist(),
        "per_layer_oof_auc_sd": per_layer.std(axis=0, ddof=1).tolist(),
        "bands": bands,
        "best_band": best_key,
        "all_32": layer_set_auc(folds, [list(range(n_layers))] * len(folds), method),
        "loss_rule": {
            **layer_set_auc(folds, loss_rule_layers, method),
            "layers_per_fold": loss_rule_layers,
            "n_layers_per_fold": [len(l) for l in loss_rule_layers],
        },
    }


def select_probe(probe_name: str, cfg: PipelineConfig) -> dict | None:
    probe_dir = cfg.cv_dir / probe_name
    if not probe_dir.exists():
        print(f"{probe_name}: no cv folder yet")
        return None
    cells = load_cells(probe_dir, cfg)
    if not cells:
        print(f"{probe_name}: no complete cells yet")
        return None
    scored = [score_cell(c, cfg) for c in cells]
    chosen_cell = max(scored, key=lambda c: c["bands"][c["best_band"]]["mean"])
    chosen_band = chosen_cell["bands"][chosen_cell["best_band"]]
    folds_info = json.loads((probe_dir / "folds.json").read_text())
    n_train_full = folds_info["n_rows"]
    selection = {
        "spec_hash": spec.spec_hash(),
        "probe": probe_name,
        "arm": cfg.arm,
        "combined_score": cfg.combined_score,
        "n_train_rows": n_train_full,
        "n_train_questions": folds_info["n_questions"],
        "cv_folds": cfg.cv_folds,
        "chosen": {
            "learning_rate": chosen_cell["lr"],
            "max_steps": chosen_cell["steps"],
            "epochs_full_train": chosen_cell["steps"] * cfg.train_cfg.batch_size / n_train_full,
            "layer_band": chosen_band["band"],
            "layers": band_key(tuple(chosen_band["band"])),
            "oof_auc_mean": chosen_band["mean"],
            "oof_auc_sd": chosen_band["sd"],
            "oof_auc_pooled": chosen_band["pooled"],
            "oof_all_32": chosen_cell["all_32"]["mean"],
            "oof_loss_rule": chosen_cell["loss_rule"]["mean"],
            "per_layer_oof_auc": chosen_cell["per_layer_oof_auc"],
        },
        "cells": scored,
    }
    (probe_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    c = selection["chosen"]
    print(
        f"{probe_name}: lr={c['learning_rate']:g} steps={c['max_steps']} ({c['epochs_full_train']:.1f} epochs on full train) "
        f"band {c['layers']}: OOF {c['oof_auc_mean']:.4f} ± {c['oof_auc_sd']:.4f}  [loss rule {c['oof_loss_rule']:.4f}, all 32 {c['oof_all_32']:.4f}]"
    )
    return selection


def probe_table(selection: dict) -> str:
    c = selection["chosen"]
    lines = [
        f"### {selection['probe']}  ({selection['n_train_rows']} train rows, {selection['n_train_questions']} questions)",
        "",
        "Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over "
        "contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).",
        "",
        "| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in selection["cells"]:
        best = cell["bands"][cell["best_band"]]
        chosen = cell["lr"] == c["learning_rate"] and cell["steps"] == c["max_steps"]
        mark = " **←**" if chosen else ""
        lines.append(
            f"| {cell['lr']:g} | {cell['steps']} | {cell['epochs_mean']:.1f} | {cell['best_band']} | "
            f"{best['mean']:.3f} ± {best['sd']:.3f}{mark} | {cell['loss_rule']['mean']:.3f} ± {cell['loss_rule']['sd']:.3f} "
            f"[{'/'.join(str(n) for n in cell['loss_rule']['n_layers_per_fold'])}] | "
            f"{cell['all_32']['mean']:.3f} ± {cell['all_32']['sd']:.3f} | {cell['seconds_mean']:.0f} |"
        )
    lines.append("")
    lines.append(
        f"Chosen: lr {c['learning_rate']:g}, {c['max_steps']} steps ({c['epochs_full_train']:.1f} epochs on the full train "
        f"set), layers {c['layers']}; OOF {c['oof_auc_mean']:.3f} ± {c['oof_auc_sd']:.3f} (pooled {c['oof_auc_pooled']:.3f})."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=spec.ARMS)
    parser.add_argument("--arms-dir", type=Path, default=spec.ARMS_DIR)
    parser.add_argument("--probes", nargs="*", default=None, help="default: every probe of the arm")
    args = parser.parse_args()

    cfg = arm_config(args.arm, args.arms_dir.resolve())
    probes = args.probes if args.probes is not None else cfg.probe_names()

    selections: dict[str, dict] = {}
    for probe_name in probes:
        selection = select_probe(probe_name, cfg)
        if selection is not None:
            selections[probe_name] = selection
    if not selections:
        return

    summary = [
        f"# CV selection, arm {cfg.arm}",
        "",
        f"{cfg.cv_folds}-fold GroupKFold over train question_ids (seed {cfg.cv_seed}); grid lr x steps from cv_sweep.py (plus cv_edge.py cells); "
        f"bands of width {list(cfg.cv_band_widths)}; combined score = {cfg.combined_score}. The eval split was not used. spec_hash {spec.spec_hash()}.",
        "",
        "| probe | lr | steps | epochs (full train) | band | OOF ± fold sd | loss rule | all 32 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for probe_name, s in selections.items():
        c = s["chosen"]
        summary.append(
            f"| {probe_name} | {c['learning_rate']:g} | {c['max_steps']} | {c['epochs_full_train']:.1f} | {c['layers']} | "
            f"{c['oof_auc_mean']:.3f} ± {c['oof_auc_sd']:.3f} | {c['oof_loss_rule']:.3f} | {c['oof_all_32']:.3f} |"
        )
    summary.append("")
    summary.append("## Per probe")
    summary.append("")
    for s in selections.values():
        summary.append(probe_table(s))
    (cfg.cv_dir / "summary.md").write_text("\n".join(summary))

    chosen_overrides = {
        "spec_hash": spec.spec_hash(),
        "per_probe_overrides": {
            probe_name: {
                "train_cfg": {"learning_rate": s["chosen"]["learning_rate"], "max_steps": s["chosen"]["max_steps"]},
                "layer_selection": "band",
                "layer_band": s["chosen"]["layer_band"],
            }
            for probe_name, s in selections.items()
        },
    }
    (cfg.cv_dir / "chosen_overrides.json").write_text(json.dumps(chosen_overrides, indent=2))
    print(f"\nwrote {cfg.cv_dir / 'summary.md'}, {cfg.cv_dir / 'chosen_overrides.json'}")


if __name__ == "__main__":
    main()
