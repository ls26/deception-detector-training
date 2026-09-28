"""Choose (learning rate, max_steps, layer band) per probe from cv_sweep.py's
out-of-fold logits. CPU only, seconds.

    python cv_select.py --variant pool

For every complete cell (all cv_folds fold files) and every fold, the held-out
rows' combined score (`cfg.combined_score`) is formed over each candidate layer
set and its AUC taken; the cell's score for a layer set is the mean over folds
(fold sd reported; the pooled-over-folds AUC is stored as well). Candidate sets:
every contiguous band of a width in `cfg.cv_band_widths` (all starts), the
pipeline's train-loss<0.6 rule applied per fold (that fold's passing layers,
all 32 when none), and all 32 layers. The chosen recipe per probe is the argmax
over (cell, band) of the fold-mean AUC. The eval split is never read.

Writes, under cv/<variant>/:
  <probe>/selection.json   every cell x layer set, per-layer OOF AUC, the choice
  summary.md               tables per probe and the shared-band diagnostic
  chosen_overrides.json    {"per_probe_overrides": {...}} for round.py --set-file
  ../../figures/cv_heatmap_<probe>.png   OOF AUC over lr x steps, best band per cell
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import step3_train_eval_probes as s3  # noqa: E402
from cv_sweep import DEFAULT_PROBE_ORDER  # noqa: E402
from probe_config import PipelineConfig, default_config  # noqa: E402

matplotlib.use("Agg")

#: probes whose bands are averaged for the shared-band diagnostic (the plain
#: deception probe is near chance and would only add noise)
SHARED_BAND_PROBES: tuple[str, ...] = (
    "bias_rm_answer",
    "bias_followed_rm_answer",
    "deception_yes_rm_preyesno",
    "deception_no_rm_preyesno",
)
#: the band 260923's in-sample rounds kept landing on, reported for reference
REFERENCE_BAND: tuple[int, int] = (13, 8)

# figure colors (dataviz reference palette): text tokens for ink, one hue for magnitude
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SEQUENTIAL_CMAP = "Blues"


def band_key(band: tuple[int, int]) -> str:
    return f"{band[0]}-{band[0] + band[1] - 1}"


def candidate_bands(cfg: PipelineConfig) -> list[tuple[int, int]]:
    bands = []
    for width in cfg.cv_band_widths:
        for start in range(cfg.n_layers - width + 1):
            bands.append((start, width))
    return bands


def load_cells(probe_dir: Path, cfg: PipelineConfig) -> list[dict]:
    """Complete cells only, as {lr, steps, folds: [fold json, ...]} sorted by (lr, steps)."""
    cells = []
    for cell_dir in sorted(p for p in probe_dir.iterdir() if p.is_dir()):
        fold_files = sorted(cell_dir.glob("fold*.json"))
        if len(fold_files) < cfg.cv_folds:
            print(f"  {cell_dir.relative_to(probe_dir.parent)}: {len(fold_files)}/{cfg.cv_folds} folds, skipped")
            continue
        folds = [json.loads(f.read_text()) for f in fold_files]
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
    per_layer = np.array(
        [[roc_auc_score(np.array(f["labels"]), np.array(f["logits"])[layer]) for layer in range(n_layers)] for f in folds]
    )
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
        "reference_band": bands[band_key(REFERENCE_BAND)],
    }


def select_probe(probe_name: str, variant_dir: Path, cfg: PipelineConfig) -> dict | None:
    probe_dir = variant_dir / probe_name
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
        "probe": probe_name,
        "variant": variant_dir.name,
        "combined_score": cfg.combined_score,
        "n_train_rows": n_train_full,
        "n_train_questions": folds_info["n_questions"],
        "cv_folds": cfg.cv_folds,
        "chosen": {
            "learning_rate": chosen_cell["lr"],
            "max_steps": chosen_cell["steps"],
            # passes over the full train set at this step count (batch 32)
            "epochs_full_train": chosen_cell["steps"] * cfg.train_cfg.batch_size / n_train_full,
            "layer_band": chosen_band["band"],
            "layers": band_key(tuple(chosen_band["band"])),
            "oof_auc_mean": chosen_band["mean"],
            "oof_auc_sd": chosen_band["sd"],
            "oof_auc_pooled": chosen_band["pooled"],
            "oof_all_32": chosen_cell["all_32"]["mean"],
            "oof_loss_rule": chosen_cell["loss_rule"]["mean"],
            "oof_reference_band": chosen_cell["reference_band"]["mean"],
            "per_layer_oof_auc": chosen_cell["per_layer_oof_auc"],
        },
        "cells": scored,
    }
    (probe_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    c = selection["chosen"]
    print(
        f"{probe_name}: lr={c['learning_rate']:g} steps={c['max_steps']} ({c['epochs_full_train']:.1f} epochs on full train) "
        f"band {c['layers']}: OOF {c['oof_auc_mean']:.4f} ± {c['oof_auc_sd']:.4f}  "
        f"[loss rule {c['oof_loss_rule']:.4f}, all 32 {c['oof_all_32']:.4f}, band {band_key(REFERENCE_BAND)} {c['oof_reference_band']:.4f}]"
    )
    return selection


def probe_table(selection: dict) -> str:
    c = selection["chosen"]
    lines = [
        f"### {selection['probe']}  ({selection['n_train_rows']} train rows, {selection['n_train_questions']} questions)",
        "",
        "Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over "
        "contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold "
        f"(layers passing per fold in brackets); `{band_key(REFERENCE_BAND)}` is 260923's in-sample band.",
        "",
        f"| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | {band_key(REFERENCE_BAND)} | s/fit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in selection["cells"]:
        best = cell["bands"][cell["best_band"]]
        chosen = cell["lr"] == c["learning_rate"] and cell["steps"] == c["max_steps"]
        mark = " **←**" if chosen else ""
        lines.append(
            f"| {cell['lr']:g} | {cell['steps']} | {cell['epochs_mean']:.1f} | {cell['best_band']} | "
            f"{best['mean']:.3f} ± {best['sd']:.3f}{mark} | {cell['loss_rule']['mean']:.3f} ± {cell['loss_rule']['sd']:.3f} "
            f"[{'/'.join(str(n) for n in cell['loss_rule']['n_layers_per_fold'])}] | "
            f"{cell['all_32']['mean']:.3f} ± {cell['all_32']['sd']:.3f} | "
            f"{cell['reference_band']['mean']:.3f} ± {cell['reference_band']['sd']:.3f} | {cell['seconds_mean']:.0f} |"
        )
    lines.append("")
    lines.append(
        f"Chosen: lr {c['learning_rate']:g}, {c['max_steps']} steps ({c['epochs_full_train']:.1f} epochs on the full train "
        f"set), layers {c['layers']}; OOF {c['oof_auc_mean']:.3f} ± {c['oof_auc_sd']:.3f} (pooled {c['oof_auc_pooled']:.3f})."
    )
    lines.append("")
    return "\n".join(lines)


def shared_band_section(selections: dict[str, dict], cfg: PipelineConfig) -> str:
    """One band for all probes: at each probe's chosen (lr, steps), the OOF AUC per
    band, averaged over SHARED_BAND_PROBES; compared with the per-probe choice."""
    present = [p for p in SHARED_BAND_PROBES if p in selections]
    if len(present) < 2:
        return ""
    per_band: dict[str, list[float]] = {}
    for probe_name in present:
        s = selections[probe_name]
        cell = next(c for c in s["cells"] if c["lr"] == s["chosen"]["learning_rate"] and c["steps"] == s["chosen"]["max_steps"])
        for key, band in cell["bands"].items():
            per_band.setdefault(key, []).append(band["mean"])
    shared_key = max(per_band, key=lambda k: float(np.mean(per_band[k])))
    lines = [
        "## Shared band diagnostic",
        "",
        f"Mean OOF AUC over {', '.join(present)} at each probe's chosen (lr, steps), per band: "
        f"best shared band **{shared_key}** ({np.mean(per_band[shared_key]):.4f}).",
        "",
        "| probe | own band | OOF own | OOF shared band | difference |",
        "|---|---|---|---|---|",
    ]
    for probe_name in present:
        s = selections[probe_name]
        cell = next(c for c in s["cells"] if c["lr"] == s["chosen"]["learning_rate"] and c["steps"] == s["chosen"]["max_steps"])
        own = s["chosen"]["oof_auc_mean"]
        shared = cell["bands"][shared_key]["mean"]
        lines.append(f"| {probe_name} | {s['chosen']['layers']} | {own:.3f} | {shared:.3f} | {shared - own:+.3f} |")
    lines.append("")
    return "\n".join(lines)


def plot_heatmap(selection: dict, cfg: PipelineConfig, out_path: Path) -> None:
    """lr x steps grid, three panels (best band / loss rule / all 32), one shared
    single-hue scale; every cell direct-labeled, the chosen cell outlined."""
    cells = selection["cells"]
    lrs = sorted({c["lr"] for c in cells})
    steps_grid = sorted({c["steps"] for c in cells})
    panels = (
        ("best contiguous band", lambda c: c["bands"][c["best_band"]], lambda c: c["best_band"]),
        ("train-loss < 0.6 rule (per fold)", lambda c: c["loss_rule"], lambda c: f"[{'/'.join(str(n) for n in c['loss_rule']['n_layers_per_fold'])}]"),
        ("all 32 layers", lambda c: c["all_32"], lambda c: ""),
    )
    values = np.full((len(panels), len(lrs), len(steps_grid)), np.nan)
    for c in cells:
        i, j = lrs.index(c["lr"]), steps_grid.index(c["steps"])
        for k, (_, getter, _) in enumerate(panels):
            values[k, i, j] = getter(c)["mean"]
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    chosen = selection["chosen"]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8), sharey=True)
    for k, (title, getter, annot) in enumerate(panels):
        ax = axes[k]
        im = ax.imshow(values[k], cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        for c in cells:
            i, j = lrs.index(c["lr"]), steps_grid.index(c["steps"])
            v = getter(c)
            dark_cell = (v["mean"] - vmin) / max(vmax - vmin, 1e-9) > 0.6
            ax.text(
                j, i, f"{v['mean']:.3f}\n±{v['sd']:.3f}\n{annot(c)}",
                ha="center", va="center", fontsize=8, color="white" if dark_cell else INK,
            )
            if k == 0 and c["lr"] == chosen["learning_rate"] and c["steps"] == chosen["max_steps"]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=2))
        ax.set_xticks(range(len(steps_grid)), [str(s) for s in steps_grid])
        ax.set_yticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
        ax.set_xlabel("max_steps", color=INK_SECONDARY)
        if k == 0:
            ax.set_ylabel("learning rate", color=INK_SECONDARY)
        ax.set_title(title, fontsize=10, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label=f"out-of-fold AUC ({cfg.combined_score})")
    fig.suptitle(
        f"{selection['probe']}: 5-fold CV over train questions, {selection['n_train_rows']} rows. "
        f"Chosen lr {chosen['learning_rate']:g}, {chosen['max_steps']} steps, layers {chosen['layers']}",
        fontsize=10, color=INK,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--probes", nargs="*", default=list(DEFAULT_PROBE_ORDER))
    args = parser.parse_args()

    cfg = default_config()
    variant_dir = cfg.cv_dir / args.variant
    cfg.figures_dir.mkdir(exist_ok=True)

    selections: dict[str, dict] = {}
    for probe_name in args.probes:
        selection = select_probe(probe_name, variant_dir, cfg)
        if selection is None:
            continue
        selections[probe_name] = selection
        plot_heatmap(selection, cfg, cfg.figures_dir / f"cv_heatmap_{args.variant}_{probe_name}.png")
    if not selections:
        return

    summary = [
        f"# CV selection on {args.variant}",
        "",
        f"{cfg.cv_folds}-fold GroupKFold over train question_ids (seed {cfg.cv_seed}); grid lr x steps from cv_sweep.py; "
        f"bands of width {list(cfg.cv_band_widths)}; combined score = {cfg.combined_score}. The eval split was not used.",
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
    summary.append(shared_band_section(selections, cfg))
    summary.append("## Per probe")
    summary.append("")
    for s in selections.values():
        summary.append(probe_table(s))
    (variant_dir / "summary.md").write_text("\n".join(summary))

    chosen_overrides = {
        "per_probe_overrides": {
            probe_name: {
                "train_cfg": {"learning_rate": s["chosen"]["learning_rate"], "max_steps": s["chosen"]["max_steps"]},
                "layer_selection": "band",
                "layer_band": s["chosen"]["layer_band"],
            }
            for probe_name, s in selections.items()
        }
    }
    (variant_dir / "chosen_overrides.json").write_text(json.dumps(chosen_overrides, indent=2))
    print(f"\nwrote {variant_dir / 'summary.md'}, {variant_dir / 'chosen_overrides.json'}, figures/cv_heatmap_{args.variant}_*.png")


if __name__ == "__main__":
    main()
