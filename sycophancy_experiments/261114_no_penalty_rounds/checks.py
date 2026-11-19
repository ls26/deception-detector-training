"""The numbers the reviews of the unattended queue read, and the copy of 261013's base answers for scoring with each
run's probe. 261031's checks.py with its dry run and its comparison with 261024's probe-only fit removed, the probe's
expected class counts read from the run's own probe files, and W&B's KL added to advantages; 261110 added early;
261114 added the reward rows to early (for the no-penalty runs: the penalty is off from the first steps).

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY checks.py smoke RUN_YAML --out FILE
    $PY checks.py advantages RUN_YAML --out FILE
    $PY checks.py probe RUN_YAML LOG --out FILE
    $PY checks.py early RUN_YAML LOG --out FILE        (while the run is still training)
    $PY checks.py copy_base OUT_DIR                    -> OUT_DIR/completions.jsonl

Nothing here passes or fails a review; the reviewer applies the rules in CLAUDE.md to these numbers.

smoke       Per step of a run: probe scores present exactly on parsed answers, and W&B's reward_mean and
            rewards/detector against the step records (task reward + the penalty: -weight x probe score on answers
            naming the cued option, 0 elsewhere; reward_mean also holds the trainer's -0.1 on answers cut off at the
            length cap).
advantages  Per step of a run: W&B's advantage_abs_mean (the trainer's mean absolute advantage over the step's answers)
            against dynamics.py's recompute from the step records with the run's loss.advantage_std_floor, and the
            recompute with no floor. Also W&B's kl_div per step (KL from the reference, here the base model) and
            kl_div_file_step_0, its value on the first step's answers.
probe       A run's start-of-run probe test scores (the trainer's saved scores on the run's probe test file, mean over
            layers 13-16 of each layer's sigmoid): class counts, AUC, class means; the saved layers; the training class
            counts the trainer printed in LOG; the class counts of the run's probe training and test files (the
            config's data.probe_dataset.bias_probe_train_jsonl and bias_probe_eval_jsonl).
early       A running run's first numbers, once its first training step is logged (its run_metrics json is written
            only at the end, so the run folder and W&B id come from LOG's last "Phase-0 detector results cache path"
            line): the starting-checkpoint lines, the frozen-probe swap lines, the probe training class counts the
            trainer printed against the counts of the config's probe training file, W&B's kl_div at training
            step 1 (file step 0; the value advantages later reports as kl_div_file_step_0), and smoke's per-step reward
            rows for every step W&B has logged (a step's record file is written before W&B logs the step).
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

RUN_RESULTS = Path("/workspace/results/run_results")
BASE_261013 = S.SYCOPHANCY_DIR / "261013_noisy_label_rl" / "evals" / "base"
WANDB_PROJECT_PATH = "lilylixin6-the-harker-school/deception_detection"
DETECTOR_KEY = check_probe_scoring.DETECTOR_KEY


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
# a run's step records against W&B
# --------------------------------------------------------------------------- #


def penalty_of(record: dict, weight: float) -> float:
    names_cued_option = record["cue_letter"] is not None and record["parsed_answer"] == record["cue_letter"]
    return -weight * record["probe_score"] if names_cued_option else 0.0


def reward_step(step: int, rows: list[dict], weight: float, wandb_row: dict) -> dict:
    """One step's records against W&B's reward_mean and rewards/detector for that step."""
    penalties = [penalty_of(r, weight) for r in rows]
    cut_off = [not r["completion"].endswith(dynamics.EOT) for r in rows]
    recomputed_reward_mean = compare.mean([r["reward"] + p - dynamics.NON_TERMINATION_PENALTY * c for r, p, c in zip(rows, penalties, cut_off)])
    return {
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
    }


def smoke(run_yaml: Path) -> dict:
    weight = float(run_config(run_yaml)["overrides"]["loss.detector"])
    run_dir, run_id = run_dir_and_id(run_yaml)
    wandb_rows = wandb_history(run_id, ["reward_mean", "rewards/detector"])
    records = compare.training_records(run_dir)
    # file step s is W&B training step s + 1 (261013's check_records.py)
    steps = [reward_step(step, rows, weight, wandb_rows[step + 1]) for step, rows in records.items()]
    return {"run_dir": str(run_dir), "penalty_weight": weight, "steps": steps}


def advantages(run_yaml: Path) -> dict:
    overrides = run_config(run_yaml)["overrides"]
    weight, std_floor = overrides["loss.detector"], overrides["loss.advantage_std_floor"]
    run_dir, run_id = run_dir_and_id(run_yaml)
    wandb_rows = wandb_history(run_id, ["advantage_abs_mean", "kl_div"])
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
            "wandb_kl_div": wandb_rows[step + 1]["kl_div"],
        })
    return {
        "run_dir": str(run_dir),
        "advantage_std_floor": std_floor,
        "steps_compared": len(steps),
        "max_abs_difference": max(s["abs_difference"] for s in steps),
        "kl_div_file_step_0": steps[0]["wandb_kl_div"],
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


def probe_stats(run_yaml: Path, log: Path) -> dict:
    overrides = run_config(run_yaml)["overrides"]
    run_dir, _ = run_dir_and_id(run_yaml)
    scores = check_probe_scoring.saved_test_scores(run_dir)
    printed = set(re.findall(r"Probe dataset \(label_mode=normal\): \((\d+), (\d+)\)", log.read_text()))
    assert len(printed) == 1, f"{log}: probe training counts printed {printed}"
    return {
        "run_dir": str(run_dir),
        "log": str(log),
        "saved_layers": {subdir: saved_layers(run_dir, subdir) for subdir in (S.START_PROBE_SUBDIR, S.FRESH_PROBE_SUBDIR)},
        "train_biased_unbiased": [int(n) for n in printed.pop()],
        "expected_train_biased_unbiased": class_counts(Path(overrides["data.probe_dataset.bias_probe_train_jsonl"])),
        "test_biased_unbiased": [len(scores["positive"]), len(scores["negative"])],
        "expected_test_biased_unbiased": class_counts(Path(overrides["data.probe_dataset.bias_probe_eval_jsonl"])),
        "test_auc": compare.auc(scores["positive"], scores["negative"]),
        "test_mean_biased": compare.mean(scores["positive"]),
        "test_mean_unbiased": compare.mean(scores["negative"]),
        "offline_auc_261010": check_probe_scoring.OFFLINE_AUC,
    }


def early(run_yaml: Path, log: Path) -> dict:
    config = run_config(run_yaml)
    text = log.read_text()
    run_dirs = re.findall(r"^Phase-0 detector results cache path: (\S+)$", text, flags=re.MULTILINE)
    run_dir = Path(run_dirs[-1])
    prefix = f"{config['run_key']}_"
    assert run_dir.parent == RUN_RESULTS / config["group"] and run_dir.name.startswith(prefix), run_dir
    run_id = run_dir.name.removeprefix(prefix)
    wandb_rows = wandb_history(run_id, ["kl_div", "reward_mean", "rewards/detector"])
    assert 1 in wandb_rows, f"W&B run {run_id} has not logged training step 1 yet"
    weight = float(config["overrides"]["loss.detector"])
    logged_file_steps = sorted(step - 1 for step in wandb_rows)
    assert logged_file_steps == list(range(len(logged_file_steps))), logged_file_steps
    reward_steps = [reward_step(step, compare.read_jsonl(run_dir / "checkpoints" / "responses" / f"step_{step}.jsonl"), weight,
                                wandb_rows[step + 1]) for step in logged_file_steps]
    printed = re.findall(r"Probe dataset \(label_mode=normal\): \((\d+), (\d+)\)", text)
    return {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "run_folders_printed": run_dirs,
        "checkpoint_lines": re.findall(r"^Checkpoint: .*$", text, flags=re.MULTILINE),
        "adapter_lines": re.findall(r"^Phase-0: loading fresh model with adapter from .*$", text, flags=re.MULTILINE),
        "frozen_probe_lines": re.findall(r"^(?:Phase-0 override: replacing detector with frozen probe weights from |Filtered layers \(from frozen probe weights dir\): ).*$",
                                         text, flags=re.MULTILINE),
        "train_biased_unbiased_printed": [[int(a), int(b)] for a, b in printed],
        "expected_train_biased_unbiased": class_counts(Path(config["overrides"]["data.probe_dataset.bias_probe_train_jsonl"])),
        "wandb_kl_div_step_1": wandb_rows[1]["kl_div"],
        "penalty_weight": weight,
        "reward_steps": reward_steps,
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
    for name in ("smoke", "advantages"):
        command = commands.add_parser(name)
        command.add_argument("run_yaml", type=Path)
        command.add_argument("--out", type=Path, required=True)
    for name in ("probe", "early"):
        command = commands.add_parser(name)
        command.add_argument("run_yaml", type=Path)
        command.add_argument("log", type=Path, help="the RL log holding the trainer's output")
        command.add_argument("--out", type=Path, required=True)
    commands.add_parser("copy_base").add_argument("out_dir", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        write_json(args.out, smoke(args.run_yaml))
    elif args.command == "advantages":
        write_json(args.out, advantages(args.run_yaml))
    elif args.command == "probe":
        write_json(args.out, probe_stats(args.run_yaml, args.log))
    elif args.command == "early":
        write_json(args.out, early(args.run_yaml, args.log))
    else:
        copy_base(args.out_dir)


if __name__ == "__main__":
    main()
