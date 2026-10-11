"""Slice-trained probes vs the baseline probe, on the same slice eval rows and labels.

    python compare_slice.py              # -> comparison.md, comparison.json
    python compare_slice.py --dry-run    # dry/

Per probe: each probe's AUC of its mean-sigmoid score over its own band;
Δ = slice-trained minus baseline, with 261007's question-level bootstrap
(2000 draws, seed 20261007; both AUCs recomputed on the same resampled
questions; 95% CI from the 2.5 / 97.5 percentiles). One training seed.
"""

from __future__ import annotations

import argparse
import json

import slice_spec as S
from slice_pipeline import bootstrap_delta, ref_spec, slice_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = S.EXP_DIR / "dry" if args.dry_run else S.EXP_DIR

    cfg = slice_config(root)
    slice_scores = json.loads(cfg.eval_scores_path.read_text())
    slice_metrics = {p["probe"]: p for p in json.loads(cfg.metrics_path.read_text())["probes"]}
    baseline_scores = json.loads((root / "runs" / "baseline_on_slice" / "eval_scores.json").read_text())
    baseline_full = {p["probe"]: p for p in json.loads((S.BASELINE_PROBE_RUN_DIR / "probe_metrics.json").read_text())["probes"]}

    results = {}
    for probe_name in S.PROBES:
        arm, ref = slice_scores[probe_name], baseline_scores[probe_name]
        assert list(zip(arm["question_id"], arm["condition"], arm["label"])) == list(zip(ref["question_id"], ref["condition"], ref["label"]))
        delta = bootstrap_delta(ref, arm)
        assert delta["n_shared_questions"] == len(set(arm["question_id"]))
        m, b = slice_metrics[probe_name], baseline_full[probe_name]
        results[probe_name] = {
            "n_train": m["n_train"],
            "n_train_baseline": b["n_train"],
            "n_eval": m["n_eval"],
            "n_eval_pos": m["n_eval_pos"],
            "n_eval_neg": m["n_eval_neg"],
            "n_eval_questions": delta["n_shared_questions"],
            "slice_recipe": {"learning_rate": m["learning_rate"], "max_steps": m["max_steps"], "layers": m["filtered_layers"], "epochs": m["epochs"]},
            "baseline_recipe": {"learning_rate": b["learning_rate"], "max_steps": b["max_steps"], "layers": b["filtered_layers"]},
            "auc_slice_trained": delta["auc_arm_shared"],
            "auc_baseline": delta["auc_ref_shared"],
            "auc_baseline_full_eval": b["auc_combined"],
            "n_eval_baseline_full_eval": b["n_eval"],
            "delta": delta["delta"],
            "ci_low": delta["ci_low"],
            "ci_high": delta["ci_high"],
            "excludes_zero": delta["excludes_zero"],
        }
    (root / "comparison.json").write_text(json.dumps({"bootstrap_draws": ref_spec.BOOTSTRAP_DRAWS, "bootstrap_seed": ref_spec.BOOTSTRAP_SEED, "probes": results}, indent=2))

    lines = [
        "# Slice-trained vs baseline probe on the slice eval rows",
        "",
        f"Slice: questions where at least {S.MIN_AGREE} of {S.N_NOCUE_SAMPLES} no-cue samples give the same answer; labels use that answer. "
        "Both probes are scored on the same slice eval rows with the slice labels; AUC of the mean sigmoid over each probe's own band. "
        f"Δ = slice-trained minus baseline; 95% CI from a question-level bootstrap ({ref_spec.BOOTSTRAP_DRAWS} draws, seed {ref_spec.BOOTSTRAP_SEED}); "
        "`*` marks a CI that excludes 0. One training seed (0).",
        "",
        "| Probe | Eval rows (pos / neg) | Slice-trained AUC | Baseline AUC | Δ [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for probe_name, r in results.items():
        star = "*" if r["excludes_zero"] else ""
        lines.append(
            f"| {probe_name} | {r['n_eval']} ({r['n_eval_pos']} / {r['n_eval_neg']}) | {r['auc_slice_trained']:.3f} | {r['auc_baseline']:.3f} | "
            f"{r['delta']:+.3f}{star} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |"
        )
    lines += [
        "",
        "## Training",
        "",
        "| Probe | Train rows, slice (baseline) | Slice recipe: lr / steps / layers (epochs) | Baseline recipe: lr / steps / layers | Baseline AUC, all eval rows (rows) |",
        "|---|---|---|---|---|",
    ]
    for probe_name, r in results.items():
        sr, br = r["slice_recipe"], r["baseline_recipe"]
        lines.append(
            f"| {probe_name} | {r['n_train']} ({r['n_train_baseline']}) | {sr['learning_rate']:g} / {sr['max_steps']} / {sr['layers'][0]}–{sr['layers'][-1]} ({sr['epochs']:.1f}) | "
            f"{br['learning_rate']:g} / {br['max_steps']} / {br['layers'][0]}–{br['layers'][-1]} | {r['auc_baseline_full_eval']:.3f} ({r['n_eval_baseline_full_eval']}) |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
