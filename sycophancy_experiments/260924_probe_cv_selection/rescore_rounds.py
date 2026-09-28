"""Re-score 260923's saved probes under this folder's headline, without retraining.

    python rescore_rounds.py
    python rescore_rounds.py --rounds 01_baseline 31_poolall_sgd_lr1e-2 --bands cv/pool/chosen_overrides.json

For each 260923 round (default: 01 = the 260826 baseline config, 10 logreg, 15
LDA, 31 = the AdamW lr 0.01 recipe) and each of its seeds, every reported probe
is scored on the cached eval activations with `step3.score_probe`. Three AUCs
are computed on the round's own saved layer set: `auc_mean_sigmoid` (this
folder's headline), `auc_mean_logit` (260923's `auc_layer_mean_logit`, a check
that the re-scoring reproduces the stored numbers) and `auc_filtered` (260923's
headline, the mean of per-layer AUCs). Each is computed on the full eval set and
on the no-cue-unanimous subset. With `--bands` (cv_select.py's
chosen_overrides.json) the same is also computed on the CV-chosen band per
probe, so an old round can be compared with a new one on equal layers.

Writes rescored/<round>.json (per seed and aggregated) and rescored/comparison.md.
CPU only; a few seconds per round.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import step3_train_eval_probes as s3  # noqa: E402
from probe_config import SOURCE_DIR, default_config  # noqa: E402
from run_log import REPORTED_KINDS, probes_by_kind  # noqa: E402

DEFAULT_ROUNDS: tuple[str, ...] = ("01_baseline", "10_poolall_logreg", "15_poolall_lda05", "31_poolall_sgd_lr1e-2")
EVAL_FILTERS: tuple[str, ...] = ("none", "none_unanimous")
METRICS: tuple[str, ...] = ("auc_mean_sigmoid", "auc_mean_logit", "auc_filtered")
RESCORED_DIR = EXP_DIR / "rescored"


def resolve_round_dir(name: str) -> Path:
    """A bare name is a 260923 round; a path (with '/') is a round of this folder."""
    if "/" in name:
        path = Path(name)
        return path if path.is_absolute() else EXP_DIR / path
    return SOURCE_DIR / "runs" / name


def round_config(round_dir: Path):
    """This folder's config pointed at one round's activations, row draw and
    per-probe recipe (260923 rounds have no per_probe_overrides key)."""
    saved = json.loads((round_dir / "config.json").read_text())
    assert saved.get("holdout_category") is None, f"{round_dir.name} is a category-holdout round; not supported here"
    cfg = replace(
        default_config(),
        act_dir=Path(saved["act_dir"]),
        train_draw_per_class=saved.get("train_draw_per_class"),
    )
    return replace(cfg, per_probe_overrides=saved.get("per_probe_overrides", {}))


def aucs_on_rows(ps: s3.ProbeScores, layers: list[int], mask: np.ndarray) -> dict[str, float]:
    labels = ps.labels["eval"][mask]
    logits = np.stack([ps.scores["eval"][layer][mask] for layer in layers])
    return {
        "auc_mean_sigmoid": s3.combined_auc(labels, logits, "mean_sigmoid"),
        "auc_mean_logit": s3.combined_auc(labels, logits, "mean_logit"),
        "auc_filtered": float(np.mean([roc_auc_score(labels, logits[i]) for i in range(len(layers))])),
        "n_eval": int(mask.sum()),
        "n_eval_pos": int((labels == 1).sum()),
        "n_eval_neg": int((labels == 0).sum()),
    }


def rescore_round(round_dir: Path, bands: dict[str, list[int]] | None) -> dict:
    cfg = round_config(round_dir)
    stored = json.loads((round_dir / "probe_metrics.json").read_text())
    stored_by_kind = probes_by_kind(stored)
    seed_dirs = sorted(p for p in (round_dir / "probe_weights").iterdir() if p.name.startswith("seed"))
    assert seed_dirs, f"no seed folders under {round_dir / 'probe_weights'}"
    # a round may hold a subset of the five probes (round.py --probes)
    kinds_present = [kind for kind in REPORTED_KINDS if kind in stored_by_kind]

    per_seed: dict[str, dict] = {}
    for seed_dir in seed_dirs:
        seed_cfg = replace(cfg, weights_dir=seed_dir)
        per_seed[seed_dir.name] = {}
        for kind in kinds_present:
            probe_name = stored_by_kind[kind]["probe"]
            ps = s3.score_probe(probe_name, seed_cfg, splits=("eval",))
            unanimous = np.array(ps.meta["eval"]["none_unanimous"], dtype=bool)
            masks = {"none": np.ones(ps.n("eval"), dtype=bool), "none_unanimous": unanimous}
            layer_sets = {"saved": ps.filtered_layers}
            if bands is not None:
                layer_sets["cv_band"] = bands[probe_name]
            per_seed[seed_dir.name][probe_name] = {
                "filtered_layers": ps.filtered_layers,
                "cv_band": layer_sets.get("cv_band"),
                "scores": {
                    layer_set_name: {eval_filter: aucs_on_rows(ps, layers, mask) for eval_filter, mask in masks.items()}
                    for layer_set_name, layers in layer_sets.items()
                },
            }

    # aggregate over seeds: mean and sample sd per (probe, layer set, eval filter, metric)
    aggregated: dict[str, dict] = {}
    first = next(iter(per_seed.values()))
    for probe_name, entry in first.items():
        aggregated[probe_name] = {"filtered_layers_by_seed": {s: per_seed[s][probe_name]["filtered_layers"] for s in per_seed}}
        for layer_set_name, by_filter in entry["scores"].items():
            aggregated[probe_name][layer_set_name] = {}
            for eval_filter, values in by_filter.items():
                agg = {k: values[k] for k in ("n_eval", "n_eval_pos", "n_eval_neg")}
                for metric in METRICS:
                    samples = np.array([per_seed[s][probe_name]["scores"][layer_set_name][eval_filter][metric] for s in per_seed])
                    agg[f"{metric}_mean"] = float(samples.mean())
                    agg[f"{metric}_sd"] = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0
                aggregated[probe_name][layer_set_name][eval_filter] = agg

    # the re-scoring must reproduce the round's stored numbers on the saved layers and
    # its own eval filter (260923 rounds store auc_layer_mean_logit, this folder's auc_mean_logit)
    round_eval_filter = stored["config"].get("eval_filter", "none")
    for kind in kinds_present:
        probe_name = stored_by_kind[kind]["probe"]
        s = stored_by_kind[kind]
        r = aggregated[probe_name]["saved"][round_eval_filter]
        mean_logit_key = "auc_layer_mean_logit_mean" if "auc_layer_mean_logit_mean" in s else "auc_mean_logit_mean"
        for stored_key, metric in (("auc_filtered_mean", "auc_filtered"), (mean_logit_key, "auc_mean_logit")):
            assert abs(s[stored_key] - r[f"{metric}_mean"]) < 2e-3, (
                f"{round_dir.name}/{probe_name}: stored {stored_key}={s[stored_key]:.4f} vs re-scored {r[f'{metric}_mean']:.4f}"
            )
    print(f"{round_dir.name}: re-scored {len(per_seed)} seeds; stored auc_filtered / mean-logit AUC reproduced (eval_filter={round_eval_filter})")
    return {
        "round": round_dir.name,
        "source": str(round_dir),
        "trained_eval_filter": round_eval_filter,
        "seeds": list(per_seed),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }


def comparison_markdown(results: list[dict]) -> str:
    lines = ["# 260923 rounds re-scored under the combined-score headline", ""]
    lines.append("Eval AUC, mean ± sd over the round's seeds, on the round's own saved layer set.")
    lines.append("")
    kinds_header = "| round | " + " | ".join(REPORTED_KINDS) + " |"
    rule = "|---|" + "---|" * len(REPORTED_KINDS)
    for eval_filter in EVAL_FILTERS:
        for layer_set_name in ("saved", "cv_band"):
            if not any(layer_set_name in next(iter(r["aggregated"].values())) for r in results):
                continue
            for metric in METRICS:
                lines.append(f"## eval_filter={eval_filter}, layers={layer_set_name}, metric={metric}")
                lines.append("")
                lines.append(kinds_header)
                lines.append(rule)
                for r in results:
                    by_kind = {p.split("_rm_")[0]: p for p in r["aggregated"]}
                    cells = []
                    for kind in REPORTED_KINDS:
                        if kind not in by_kind:
                            cells.append("—")
                            continue
                        a = r["aggregated"][by_kind[kind]][layer_set_name][eval_filter]
                        cells.append(f"{a[f'{metric}_mean']:.3f} ± {a[f'{metric}_sd']:.3f}")
                    lines.append(f"| {r['round']} | " + " | ".join(cells) + " |")
                lines.append("")
    first = results[0]
    lines.append("## eval rows per probe (n, pos/neg)")
    lines.append("")
    lines.append("| eval_filter | " + " | ".join(REPORTED_KINDS) + " |")
    lines.append(rule)
    by_kind = {p.split("_rm_")[0]: p for p in first["aggregated"]}
    for eval_filter in EVAL_FILTERS:
        cells = []
        for kind in REPORTED_KINDS:
            if kind not in by_kind:
                cells.append("—")
                continue
            a = first["aggregated"][by_kind[kind]]["saved"][eval_filter]
            cells.append(f"{a['n_eval']} ({a['n_eval_pos']}/{a['n_eval_neg']})")
        lines.append(f"| {eval_filter} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds", nargs="*", default=list(DEFAULT_ROUNDS),
        help="260923 round names (e.g. 31_poolall_sgd_lr1e-2) or paths to this folder's rounds (e.g. runs/01_cv_chosen_pool)",
    )
    parser.add_argument("--bands", default=None, help="cv_select.py's chosen_overrides.json: also score on each probe's CV band")
    args = parser.parse_args()

    bands = None
    if args.bands is not None:
        chosen = json.loads(Path(args.bands).read_text())["per_probe_overrides"]
        bands = {probe: s3.band_layers(tuple(o["layer_band"]), default_config().n_layers) for probe, o in chosen.items()}

    RESCORED_DIR.mkdir(exist_ok=True)
    results = []
    for round_name in args.rounds:
        round_dir = resolve_round_dir(round_name)
        result = rescore_round(round_dir, bands)
        (RESCORED_DIR / f"{round_dir.name}.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    (RESCORED_DIR / "comparison.md").write_text(comparison_markdown(results))
    print(f"\nwrote {RESCORED_DIR / 'comparison.md'}")
    print(comparison_markdown(results))


if __name__ == "__main__":
    main()
