"""Why is ref_cot lower than 260924? Two extra fits of the reference arm.

    python ref_decomposition.py --part matched_oldrecipe   # the run's probe sets + activations, 260924's recipe
    python ref_decomposition.py --part full3_oldrecipe     # new labels, 260924's size (3 rows per prompt, largest
                                                           # cue-balanced draw), 260924's recipe; loads the model
    python ref_decomposition.py --part report              # decomp/decomposition.md

Four cells, differing one step at a time:

  A  260924's labels, full size, 260924's recipe     review/parts/reproduction_*.json (the gate)
  B  this run's labels, full size, 260924's recipe   decomp/full3_oldrecipe/        A -> B: the label change
  D  this run's labels, matched size, 260924's recipe decomp/matched_oldrecipe/     B -> D: the size change
  C  this run's labels, matched size, CV recipe      runs/ref_cot/ (the run)        D -> C: the recipe change

"Full size" is 260924's rule applied to this run's labels: every sample of a
prompt is a row and each class draws 2 x its smallest (class, condition) cell.
Each part writes its own decomp/<part>/{arms,runs}/ and touches nothing in the
run's folders (the matched part reads them through a symlink).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import spec  # noqa: E402
import step1_build_probe_sets as s1  # noqa: E402
import step2_extract_activations as s2  # noqa: E402
import step3_train_eval_probes as s3  # noqa: E402
from extract import step1_gates  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from probe_config import DRAWN_KINDS, PipelineConfig, apply_overrides, arm_config, config_to_json  # noqa: E402

DECOMP_DIR = EXP_DIR / "decomp"
ARM = spec.REFERENCE_ARM
CELLS = (("A", "260924 labels, full size, 260924 recipe"), ("B", "run labels, full size, 260924 recipe"),
         ("D", "run labels, matched size, 260924 recipe"), ("C", "run labels, matched size, CV recipe (the run)"))


def old_recipe_overrides() -> dict:
    return {
        "per_probe_overrides": {
            probe: {
                "train_cfg": {"learning_rate": r["learning_rate"], "max_steps": r["max_steps"]},
                "layer_selection": "band",
                "layer_band": r["layer_band"],
            }
            for probe, r in spec.REPRODUCTION_RECIPE.items()
        }
    }


def prepare_arms_dir(part: str) -> Path:
    arms_dir = DECOMP_DIR / part / "arms"
    arm_dir = arms_dir / ARM
    arms_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(spec.ARMS_DIR / "sizes.json", arms_dir / "sizes.json")
    if part == "matched_oldrecipe":
        # the run's probe sets and activations, read in place
        if not arm_dir.exists():
            arm_dir.symlink_to(spec.ARMS_DIR / ARM)
    else:
        arm_dir.mkdir(exist_ok=True)
        if not (arm_dir / "labels.jsonl").exists():
            (arm_dir / "labels.jsonl").symlink_to(spec.ARMS_DIR / ARM / "labels.jsonl")
    return arms_dir


def tokenizer_for(cfg: PipelineConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def fit_old_recipe(cfg: PipelineConfig, tokenizer) -> None:
    cfg = apply_overrides(cfg, old_recipe_overrides())
    assert not cfg.run_dir.exists(), f"{cfg.run_dir} exists"
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "config.json").write_text(json.dumps(config_to_json(cfg), indent=2))
    s3.train_eval_all(cfg, tokenizer)


def part_matched_oldrecipe() -> None:
    cfg = arm_config(ARM, prepare_arms_dir("matched_oldrecipe"))
    assert cfg.act_dir.exists(), f"no activations at {cfg.act_dir}"
    fit_old_recipe(cfg, tokenizer_for(cfg))


def part_full3_oldrecipe() -> None:
    arms_dir = prepare_arms_dir("full3_oldrecipe")
    # every sample is a row (3 per prompt); the draw is the largest cue-balanced one per kind
    counting_cfg = PipelineConfig(arm=ARM, arms_dir=arms_dir, train_samples_per_prompt=spec.N_SAMPLES)
    labels = s1.load_labels(counting_cfg)
    n_train_per_class = {}
    for kind in DRAWN_KINDS:
        cells = s1.train_pool_per_cell(labels, kind, spec.DEFINITION, counting_cfg)
        n_train_per_class[kind] = 2 * min(n for by_condition in cells.values() for n in by_condition.values())
    print("full size per class (3 rows per prompt):", n_train_per_class)
    cfg = PipelineConfig(arm=ARM, arms_dir=arms_dir, train_samples_per_prompt=spec.N_SAMPLES, n_train_per_class=n_train_per_class)
    assert not cfg.act_dir.exists(), f"{cfg.act_dir} exists"

    probe_sets = s1.build_all(cfg, labels)
    # the run's gate also requires one train row per prompt, which this cell deliberately does not have
    failures = [f for f in step1_gates(cfg, probe_sets) if "rows over" not in f]
    assert not failures, failures
    model, tokenizer = s2.load_model_and_tokenizer(cfg)
    s2.extract_all(cfg, model, tokenizer, {})
    del model
    fit_old_recipe(cfg, tokenizer)


def cell_aucs(cell: str) -> dict[str, float] | None:
    if cell == "A":
        out = {}
        for probe in spec.REPRODUCTION_RECIPE:
            path = EXP_DIR / "review" / "parts" / f"reproduction_{probe}.json"
            out[probe] = json.loads(path.read_text())["auc_mean_sigmoid"]
        return out
    metrics_path = {
        "B": DECOMP_DIR / "full3_oldrecipe" / "runs" / ARM / "probe_metrics.json",
        "D": DECOMP_DIR / "matched_oldrecipe" / "runs" / ARM / "probe_metrics.json",
        "C": spec.RUNS_DIR / ARM / "probe_metrics.json",
    }[cell]
    if not metrics_path.exists():
        return None
    return {p["probe"]: p["auc_combined"] for p in json.loads(metrics_path.read_text())["probes"]}


def cell_n_train(cell: str) -> dict[str, int] | None:
    if cell == "A":
        return {p: json.loads((EXP_DIR / "review" / "parts" / f"reproduction_{p}.json").read_text())["n_train"] for p in spec.REPRODUCTION_RECIPE}
    metrics_path = {
        "B": DECOMP_DIR / "full3_oldrecipe" / "runs" / ARM / "probe_metrics.json",
        "D": DECOMP_DIR / "matched_oldrecipe" / "runs" / ARM / "probe_metrics.json",
        "C": spec.RUNS_DIR / ARM / "probe_metrics.json",
    }[cell]
    if not metrics_path.exists():
        return None
    return {p["probe"]: p["n_train"] for p in json.loads(metrics_path.read_text())["probes"]}


def part_report() -> None:
    aucs = {cell: cell_aucs(cell) for cell, _ in CELLS}
    n_train = {cell: cell_n_train(cell) for cell, _ in CELLS}
    lines = [
        "# Reference arm: 260924 -> this run, one change at a time",
        "",
        f"spec_hash {spec.spec_hash()}. Eval AUC of the mean sigmoid over the band, seed 0. Cells:",
        "",
    ]
    for cell, description in CELLS:
        lines.append(f"- **{cell}**: {description}" + ("" if aucs[cell] else " (not run)"))
    present = [cell for cell, _ in CELLS if aucs[cell]]
    lines += ["", "| probe | " + " | ".join(present) + " | A->B labels | B->D size | D->C recipe |", "|---|" + "---|" * (len(present) + 3)]
    for probe in spec.REPRODUCTION_RECIPE:
        values = {cell: aucs[cell][probe] for cell in present}
        steps = []
        for left, right in (("A", "B"), ("B", "D"), ("D", "C")):
            steps.append(f"{values[right] - values[left]:+.3f}" if left in values and right in values else "—")
        lines.append(f"| {probe} | " + " | ".join(f"{values[c]:.3f}" for c in present) + " | " + " | ".join(steps) + " |")
    lines += ["", "Train rows per cell:", "", "| probe | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for probe in spec.REPRODUCTION_RECIPE:
        lines.append(f"| {probe} | " + " | ".join(str(n_train[c][probe]) for c in present) + " |")
    DECOMP_DIR.mkdir(exist_ok=True)
    (DECOMP_DIR / "decomposition.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True, choices=["matched_oldrecipe", "full3_oldrecipe", "report"])
    args = parser.parse_args()
    bootstrap_hf_env()
    {"matched_oldrecipe": part_matched_oldrecipe, "full3_oldrecipe": part_full3_oldrecipe, "report": part_report}[args.part]()
    print(f"DECOMP DONE {args.part}")


if __name__ == "__main__":
    main()
