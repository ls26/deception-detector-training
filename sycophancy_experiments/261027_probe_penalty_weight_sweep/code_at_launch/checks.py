"""The numbers the reviews of the unattended queue read, and the copy of 261013's base answers for scoring with each
run's initial probe. 261024's checks.py for runs with the advantage SD floor.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY checks.py dry --out FILE                       (after dynamics.py and smoke on 261024's runs, see below)
    $PY checks.py smoke RUN_YAML --out FILE
    $PY checks.py advantages RUN_YAML --out FILE
    $PY checks.py probe RUN_YAML LOG --out FILE
    $PY checks.py copy_base OUT_DIR                    -> OUT_DIR/completions.jsonl

Nothing here passes or fails a review; the reviewer applies the rules in CLAUDE.md to these numbers.

dry         The dry run of this folder's two scripts on 261024's saved outputs. Reads dry/dynamics_261024/dynamics.json
            (dynamics.py on 261024's penalty run, which has no floor) and dry/smoke_261024.json (smoke on 261024's
            smoke run) and compares every number of 261024's runs/penalty_s42/dynamics.json and of the steps of its
            checks/smoke.json with them: per section, the numbers compared and every one that differs or is missing.
            Also dynamics.py's mean absolute advantage on 261024's step 0 with floor 0.3 and with no floor.
smoke       Per step of a run: probe scores present exactly on parsed answers, and W&B's reward_mean and
            rewards/detector against the step records (task reward + the penalty: -weight x probe score on answers
            naming the cued option, 0 elsewhere; reward_mean also holds the trainer's -0.1 on answers cut off at the
            length cap).
advantages  Per step of a run: W&B's advantage_abs_mean (the trainer's mean absolute advantage over the step's answers)
            against dynamics.py's recompute from the step records with the run's loss.advantage_std_floor, and the
            recompute with no floor.
probe       A run's initial probe test scores (the trainer's saved scores on the 777 test answers, mean over layers
            13-16 of each layer's sigmoid): class counts, AUC, class means; the saved layers; the training class
            counts the trainer printed in LOG; the difference from 261024's probe-only fit (test scores and weights).
copy_base   261013's base eval cut to the package's eval (8 no-cue samples + cued sample 0 per condition).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

import dynamics

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import check_probe_scoring  # noqa: E402
import compare  # noqa: E402
import eval_checkpoint  # noqa: E402
import spec as S  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
DRY_DIR = EXP_DIR / "dry"
RUN_RESULTS = Path("/workspace/results/run_results")
DIR_261024 = S.SYCOPHANCY_DIR / "261024_probe_penalty_rl"
PROBE_FIT_DIR = Path(json.loads((DIR_261024 / "checks" / "probe_fit_run.json").read_text())["run_dir"])  # 261024's probe-only fit
BASE_261013 = S.SYCOPHANCY_DIR / "261013_noisy_label_rl" / "evals" / "base"
TRAIN_PROBE_PATH = S.PACKAGE_DIR / "data" / "bias_probe_train.jsonl"
WANDB_PROJECT_PATH = "lilylixin6-the-harker-school/deception_detection"
DETECTOR_KEY = check_probe_scoring.DETECTOR_KEY
DRY_FLOOR = 0.3
MISSING = "<missing>"


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def run_config(run_yaml: Path) -> dict:
    return yaml.safe_load(run_yaml.read_text())


def run_dir_and_id(run_yaml: Path) -> tuple[Path, str]:
    """A run's folder and W&B id (run id from its run_metrics json, as run_eval.sh finds it)."""
    config = run_config(run_yaml)
    run_id = json.loads((Path(config["output_dir"]) / f"run_metrics_{config['run_key']}.json").read_text())["run_id"]
    return RUN_RESULTS / config["group"] / f"{config['run_key']}_{run_id}", run_id


def wandb_history(run_id: str, keys: list[str]) -> dict[int, dict]:
    import wandb

    run = wandb.Api().run(f"{WANDB_PROJECT_PATH}/{run_id}")
    return {row["grpo_train_step"]: row for row in run.scan_history(keys=["grpo_train_step", *keys])}


# --------------------------------------------------------------------------- #
# the dry run on 261024's saved outputs
# --------------------------------------------------------------------------- #


def leaf_mismatches(saved, copy, path: str) -> tuple[int, list[dict]]:
    """The number of leaf values of `saved` compared, and each one that the copy's value does not equal exactly."""
    if isinstance(saved, dict):
        children = [(f"{path}.{key}", value, copy.get(key, MISSING) if isinstance(copy, dict) else MISSING) for key, value in saved.items()]
    elif isinstance(saved, list):
        children = [(f"{path}[{i}]", value, copy[i] if isinstance(copy, list) and i < len(copy) else MISSING) for i, value in enumerate(saved)]
    else:
        return 1, ([] if saved == copy else [{"path": path, "saved": saved, "copy": copy}])
    compared, mismatches = 0, []
    for child_path, child_saved, child_copy in children:
        n, child_mismatches = leaf_mismatches(child_saved, child_copy, child_path)
        compared += n
        mismatches += child_mismatches
    return compared, mismatches


def dry() -> dict:
    saved_dynamics = json.loads((DIR_261024 / "runs" / "penalty_s42" / "dynamics.json").read_text())
    copy_dynamics = json.loads((DRY_DIR / "dynamics_261024" / "dynamics.json").read_text())
    saved_smoke = json.loads((DIR_261024 / "checks" / "smoke.json").read_text())
    copy_smoke = json.loads((DRY_DIR / "smoke_261024.json").read_text())
    run = "penalty_s42"  # 261024's run key; 261024's dynamics.json calls it "penalty"
    sections = {
        "training (261024's penalty run)": (saved_dynamics["training"]["penalty"], copy_dynamics["training"][run]),
        **{f"training ({name})": (saved_dynamics["training"][name], copy_dynamics["training"][name])
           for name in ("baseline_s42", "baseline_s43", "baseline_s44")},
        "probe_only": (saved_dynamics["probe_only"], copy_dynamics["probe_only"][run]),
        "text, training answers": (saved_dynamics["text"]["training_steps"], copy_dynamics["text"]["training_steps"][run]),
        "text, final evals (base, baseline_s42)": ({k: saved_dynamics["text"]["evals"][k] for k in ("base", "baseline_s42")},
                                                    {k: copy_dynamics["text"]["evals"][k] for k in ("base", "baseline_s42")}),
        "text, final eval (261024's penalty run)": (saved_dynamics["text"]["evals"]["penalty"], copy_dynamics["text"]["evals"][run]),
        "probe_letters": (saved_dynamics["probe_letters"], copy_dynamics["probe_letters"][run]),
        "probe_data": (saved_dynamics["probe_data"], copy_dynamics["probe_data"]),
        "probe_weights": (saved_dynamics["probe_weights"], copy_dynamics["probe_weights"][run]),
        "smoke steps": (saved_smoke["steps"], copy_smoke["steps"]),
        "smoke penalty_weight": (saved_smoke["penalty_weight"], copy_smoke["penalty_weight"]),
    }
    comparison = {}
    for name, (saved, copy) in sections.items():
        compared, mismatches = leaf_mismatches(saved, copy, name)
        comparison[name] = {"numbers_compared": compared, "mismatches": mismatches}

    run_dir, _ = run_dir_and_id(DIR_261024 / "penalty_s42.yaml")
    weight = run_config(DIR_261024 / "penalty_s42.yaml")["overrides"]["loss.detector"]
    step_0 = compare.training_records(run_dir)[0]
    floor_step_0 = {
        f"floor {floor}": {key: dynamics.step_summary(step_0, weight, floor)[key] for key in ("abs_advantage", "abs_advantage_probe_only")}
        for floor in (DRY_FLOOR, None)
    }
    return {"comparison_with_261024": comparison, "step_0_of_261024_penalty_run": floor_step_0}


# --------------------------------------------------------------------------- #
# a run's step records against W&B
# --------------------------------------------------------------------------- #


def penalty_of(record: dict, weight: float) -> float:
    names_cued_option = record["cue_letter"] is not None and record["parsed_answer"] == record["cue_letter"]
    return -weight * record["probe_score"] if names_cued_option else 0.0


def smoke(run_yaml: Path) -> dict:
    weight = float(run_config(run_yaml)["overrides"]["loss.detector"])
    run_dir, run_id = run_dir_and_id(run_yaml)
    wandb_rows = wandb_history(run_id, ["reward_mean", "rewards/detector"])
    records = compare.training_records(run_dir)

    steps = []
    for step, rows in records.items():
        penalties = [penalty_of(r, weight) for r in rows]
        cut_off = [not r["completion"].endswith(dynamics.EOT) for r in rows]
        recomputed_reward_mean = compare.mean([r["reward"] + p - dynamics.NON_TERMINATION_PENALTY * c for r, p, c in zip(rows, penalties, cut_off)])
        wandb_row = wandb_rows[step + 1]  # file step s is W&B training step s + 1 (261013's check_records.py)
        steps.append({
            "file_step": step,
            "answers": len(rows),
            "parsed": sum(r["parsed_answer"] is not None for r in rows),
            "probe_score_present_iff_parsed_mismatches": sum((r["probe_score"] is None) != (r["parsed_answer"] is None) for r in rows),
            "naming_cued_option": sum(p != 0.0 for p in penalties),
            "cut_off": sum(cut_off),
            "recomputed_reward_mean": recomputed_reward_mean,
            "wandb_reward_mean": wandb_row["reward_mean"],
            "reward_mean_abs_difference": abs(recomputed_reward_mean - wandb_row["reward_mean"]),
            "recomputed_penalty_mean": compare.mean(penalties),
            "wandb_rewards_detector": wandb_row["rewards/detector"],
            "penalty_mean_abs_difference": abs(compare.mean(penalties) - wandb_row["rewards/detector"]),
        })
    return {"run_dir": str(run_dir), "penalty_weight": weight, "steps": steps}


def advantages(run_yaml: Path) -> dict:
    overrides = run_config(run_yaml)["overrides"]
    weight, std_floor = overrides["loss.detector"], overrides["loss.advantage_std_floor"]
    run_dir, run_id = run_dir_and_id(run_yaml)
    wandb_rows = wandb_history(run_id, ["advantage_abs_mean"])
    records = compare.training_records(run_dir)
    assert sorted(wandb_rows) == [step + 1 for step in sorted(records)], (sorted(wandb_rows), sorted(records))

    steps = []
    for step, rows in records.items():
        logged = wandb_rows[step + 1]["advantage_abs_mean"]  # file step s is W&B training step s + 1
        with_floor = dynamics.step_summary(rows, weight, std_floor)["abs_advantage"]
        steps.append({
            "file_step": step,
            "wandb_advantage_abs_mean": logged,
            "recomputed_with_floor": with_floor,
            "abs_difference": abs(logged - with_floor),
            "recomputed_without_floor": dynamics.step_summary(rows, weight, None)["abs_advantage"],
        })
    return {
        "run_dir": str(run_dir),
        "advantage_std_floor": std_floor,
        "steps_compared": len(steps),
        "max_abs_difference": max(s["abs_difference"] for s in steps),
        "per_step": steps,
    }


# --------------------------------------------------------------------------- #
# the probe
# --------------------------------------------------------------------------- #


def class_counts(path: Path) -> list[int]:
    """[biased, unbiased] rows of a probe-set file (label 1 / 0)."""
    labels = [row["label"] for row in compare.read_jsonl(path)]
    return [labels.count(1), labels.count(0)]


def saved_layers(run_dir: Path, subdir: str) -> list[int]:
    return sorted(int(path.name.removeprefix("layer_")) for path in (run_dir / subdir / f"detector_{subdir}__{DETECTOR_KEY}").glob("layer_*"))


def layer_weights(run_dir: Path, subdir: str) -> dict:
    import torch

    folder = run_dir / subdir / f"detector_{subdir}__{DETECTOR_KEY}"
    return {layer: torch.load(folder / f"layer_{layer}" / "model.pt", map_location="cpu") for layer in S.PROBE_LAYERS}


def probe_stats(run_yaml: Path, log: Path) -> dict:
    run_dir, _ = run_dir_and_id(run_yaml)
    scores = check_probe_scoring.saved_test_scores(run_dir)
    printed = set(re.findall(r"Probe dataset \(label_mode=normal\): \((\d+), (\d+)\)", log.read_text()))
    assert len(printed) == 1, f"{log}: probe training counts printed {printed}"
    fit_scores = check_probe_scoring.saved_test_scores(PROBE_FIT_DIR)
    weights, fit_weights = layer_weights(run_dir, S.START_PROBE_SUBDIR), layer_weights(PROBE_FIT_DIR, S.START_PROBE_SUBDIR)
    return {
        "run_dir": str(run_dir),
        "log": str(log),
        "saved_layers": {subdir: saved_layers(run_dir, subdir) for subdir in (S.START_PROBE_SUBDIR, S.FRESH_PROBE_SUBDIR)},
        "train_biased_unbiased": [int(n) for n in printed.pop()],
        "expected_train_biased_unbiased": class_counts(TRAIN_PROBE_PATH),
        "test_biased_unbiased": [len(scores["positive"]), len(scores["negative"])],
        "expected_test_biased_unbiased": class_counts(S.BIAS_PROBE_EVAL_PATH),
        "test_auc": compare.auc(scores["positive"], scores["negative"]),
        "test_mean_biased": compare.mean(scores["positive"]),
        "test_mean_unbiased": compare.mean(scores["negative"]),
        "offline_auc_261010": check_probe_scoring.OFFLINE_AUC,
        "vs_261024_probe_fit": {
            "test_score_max_abs_difference": max(
                check_probe_scoring.max_abs_difference(scores[c], fit_scores[c]) for c in ("positive", "negative")
            ),
            "weight_max_abs_difference": max(
                float((weights[layer][key] - fit_weights[layer][key]).abs().max()) for layer in S.PROBE_LAYERS for key in weights[layer]
            ),
        },
    }


# --------------------------------------------------------------------------- #
# the base model's answers, for scoring with a run's initial probe
# --------------------------------------------------------------------------- #


def copy_base(out_dir: Path) -> None:
    out_path = out_dir / "completions.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    rows = [r for r in compare.read_jsonl(BASE_261013 / "completions.jsonl") if r["condition"] == "none" or r["sample_idx"] == 0]
    assert len(rows) == S.N_QUESTIONS["eval"] * sum(S.N_SAMPLES.values()), len(rows)
    eval_checkpoint.write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("dry").add_argument("--out", type=Path, required=True)
    for name in ("smoke", "advantages"):
        command = commands.add_parser(name)
        command.add_argument("run_yaml", type=Path)
        command.add_argument("--out", type=Path, required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("run_yaml", type=Path)
    probe.add_argument("log", type=Path, help="the RL log holding the trainer's output")
    probe.add_argument("--out", type=Path, required=True)
    commands.add_parser("copy_base").add_argument("out_dir", type=Path)
    args = parser.parse_args()
    if args.command == "dry":
        write_json(args.out, dry())
    elif args.command == "smoke":
        write_json(args.out, smoke(args.run_yaml))
    elif args.command == "advantages":
        write_json(args.out, advantages(args.run_yaml))
    elif args.command == "probe":
        write_json(args.out, probe_stats(args.run_yaml, args.log))
    else:
        copy_base(args.out_dir)


if __name__ == "__main__":
    main()
