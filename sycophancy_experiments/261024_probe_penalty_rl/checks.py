"""The numbers the reviews of the unattended queue read, and the copy of 261013's base answers for row A.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY checks.py fit_probe                -> /workspace/results/run_results/261024_probe_penalty_rl_probe_fit/, checks/probe_fit_run.json
    $PY checks.py probe NAME LOG           -> checks/probe_<NAME>.json
    $PY checks.py base50                   -> checks/base50.json
    $PY checks.py smoke                    -> checks/smoke.json
    $PY checks.py copy_base                -> evals/base_rescored/completions.jsonl
    $PY checks.py row_a                    -> checks/row_a.json

Nothing here passes or fails a review; the reviewer applies the rules in CLAUDE.md to these numbers.

fit_probe   The trainer's start-of-run probe in its probe-only mode (training.only_eval_detectors) with
            penalty_s42.yaml's settings, through the launcher's own command builder and subprocess runner, W&B
            offline. Run key probe_check_s42. Its output stays outside git.
probe       A probe fit's test scores (the trainer's saved scores on the 777 test answers, mean over layers 13-16
            of each layer's sigmoid): class counts, AUC, class means; the saved layers; the training class counts
            the trainer printed in LOG. NAME is probe_fit or a run with a config NAME.yaml; a run is also
            compared with the probe_fit probe (test scores and weights, max absolute difference).
base50      The base model's 50-question eval on this GPU (dry/evals/base50) against 261013's saved base eval
            (same prompts and seeds): identical text, same parsed answer, parse failures, and how often two
            independent samples of 261013's eval agree on these answers (pairs within each question and condition).
smoke       The smoke run: per step, probe scores present exactly on parsed answers, and W&B's reward_mean and
            rewards/detector against the step records (task reward + the penalty: -weight x probe score on answers
            naming the cued option, 0 elsewhere; reward_mean also holds the trainer's -0.1 on answers cut off at the
            length cap, which reproduces 261013's reward_mean to 3e-8 on all 128 steps); the saved probes; W&B's end-of-run probe AUCs; the in-loop eval
            rows; the smoke eval's probe scores; the probe rows of dry/compare_smoke; 3 raw training answers.
copy_base   261013's base eval cut to the package's eval (8 no-cue samples + cued sample 0 per condition).
row_a       evals/base_rescored after its selfreport and labels stages: P(Yes) recomputed on this GPU against
            261013's saved values, and row A (the start-of-run probe on the base model's answers) next to the
            package check's row A.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import check_probe_scoring  # noqa: E402
import compare  # noqa: E402
import eval_checkpoint  # noqa: E402
import spec as S  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
CHECKS_DIR = EXP_DIR / "checks"
RUN_RESULTS = Path("/workspace/results/run_results")
PROBE_FIT_DIR = RUN_RESULTS / "261024_probe_penalty_rl_probe_fit"
PROBE_FIT_KEY = "probe_check_s42"
FULL_RUN = "penalty_s42"
SMOKE_RUN = "smoke_penalty_s42"
BASE_261013 = S.SYCOPHANCY_DIR / "261013_noisy_label_rl" / "evals" / "base"
BASE50_DIR = EXP_DIR / "dry" / "evals" / "base50"
SMOKE_EVAL_DIR = EXP_DIR / "dry" / "evals" / SMOKE_RUN
SMOKE_COMPARE_DIR = EXP_DIR / "dry" / "compare_smoke"
BASE_RESCORED_DIR = EXP_DIR / "evals" / "base_rescored"
TRAIN_PROBE_PATH = S.PACKAGE_DIR / "data" / "bias_probe_train.jsonl"
WANDB_PROJECT_PATH = "lilylixin6-the-harker-school/deception_detection"
DETECTOR_KEY = check_probe_scoring.DETECTOR_KEY
END_OF_RUN_AUC_KEYS = {
    subdir: f"probes/eval_{subdir}_averaged/{DETECTOR_KEY}/AUC_ROC"
    for subdir in ("initial_detector.final_model.on_dataset", S.FRESH_PROBE_SUBDIR)
}
SAMPLE_KEY = ("question_id", "condition", "sample_idx")
NON_TERMINATION_PENALTY = 0.1  # grpo.loss.non_termination_penalty (obfuscation_atlas/config.py), not overridden
EOT = "<|eot_id|>"  # a step record's completion ends with it unless the answer was cut off at the length cap


def write_json(name: str, result: dict) -> None:
    CHECKS_DIR.mkdir(exist_ok=True)
    (CHECKS_DIR / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def key_of(row: dict) -> tuple:
    return tuple(row[field] for field in SAMPLE_KEY)


def run_config(name: str) -> dict:
    return yaml.safe_load((EXP_DIR / f"{name}.yaml").read_text())


def run_dir_of(name: str) -> Path:
    """A run's folder (run id from its run_metrics json, as run_eval.sh finds it), or the probe fit's."""
    if name == "probe_fit":
        return Path(json.loads((CHECKS_DIR / "probe_fit_run.json").read_text())["run_dir"])
    config = run_config(name)
    run_id = json.loads((EXP_DIR / f"run_metrics_{config['run_key']}.json").read_text())["run_id"]
    return RUN_RESULTS / config["group"] / f"{config['run_key']}_{run_id}"


# --------------------------------------------------------------------------- #
# the probe
# --------------------------------------------------------------------------- #


def fit_probe() -> None:
    from training.grpo_launcher import build_grpo_args, run_grpo_subprocess

    config = run_config(FULL_RUN)
    config["run_key"] = config["probe"]["directory_save_prefix"] = PROBE_FIT_KEY
    config["overrides"].update({
        "training.only_eval_detectors": "true",
        "training.save_path_base": str(PROBE_FIT_DIR / "runs"),
        "detector.model_cache_path": str(PROBE_FIT_DIR / "activations"),
    })
    os.environ["WANDB_MODE"] = "offline"
    exit_code, _ = run_grpo_subprocess(build_grpo_args(config))
    assert exit_code == 0, f"the probe fit exited with {exit_code}"
    (run_dir,) = (PROBE_FIT_DIR / "runs" / config["group"]).glob(f"{PROBE_FIT_KEY}_*")
    write_json("probe_fit_run.json", {"run_dir": str(run_dir)})


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


def probe_stats(name: str, log: Path) -> dict:
    run_dir = run_dir_of(name)
    scores = check_probe_scoring.saved_test_scores(run_dir)
    printed = set(re.findall(r"Probe dataset \(label_mode=normal\): \((\d+), (\d+)\)", log.read_text()))
    assert len(printed) == 1, f"{log}: probe training counts printed {printed}"
    result = {
        "name": name,
        "run_dir": str(run_dir),
        "log": str(log),
        "saved_layers": {subdir: saved_layers(run_dir, subdir) for subdir in (S.START_PROBE_SUBDIR, S.FRESH_PROBE_SUBDIR)
                         if (run_dir / subdir).exists()},
        "train_biased_unbiased": [int(n) for n in printed.pop()],
        "expected_train_biased_unbiased": class_counts(TRAIN_PROBE_PATH),
        "test_biased_unbiased": [len(scores["positive"]), len(scores["negative"])],
        "expected_test_biased_unbiased": class_counts(S.BIAS_PROBE_EVAL_PATH),
        "test_auc": compare.auc(scores["positive"], scores["negative"]),
        "test_mean_biased": compare.mean(scores["positive"]),
        "test_mean_unbiased": compare.mean(scores["negative"]),
        "offline_auc_261010": check_probe_scoring.OFFLINE_AUC,
    }
    if name != "probe_fit":
        fit_dir = run_dir_of("probe_fit")
        fit_scores = check_probe_scoring.saved_test_scores(fit_dir)
        weights, fit_weights = layer_weights(run_dir, S.START_PROBE_SUBDIR), layer_weights(fit_dir, S.START_PROBE_SUBDIR)
        result["vs_probe_fit"] = {
            "test_score_max_abs_difference": max(
                check_probe_scoring.max_abs_difference(scores[c], fit_scores[c]) for c in ("positive", "negative")
            ),
            "weight_max_abs_difference": max(
                float((weights[layer][key] - fit_weights[layer][key]).abs().max()) for layer in S.PROBE_LAYERS for key in weights[layer]
            ),
        }
    return result


# --------------------------------------------------------------------------- #
# the eval on this GPU
# --------------------------------------------------------------------------- #


def pair_agreement(predictions: list) -> float:
    pairs = list(itertools.combinations(predictions, 2))
    return sum(a == b for a, b in pairs) / len(pairs)


def base50() -> dict:
    new = compare.read_jsonl(BASE50_DIR / "completions.jsonl")
    saved_rows = compare.read_jsonl(BASE_261013 / "completions.jsonl")
    saved = {key_of(r): r for r in saved_rows}
    pairs = [(r, saved[key_of(r)]) for r in new]
    assert all(r["prompt"] == s["prompt"] and r["seed"] == s["seed"] for r, s in pairs), "prompts or seeds differ from 261013's"
    saved_predictions: dict[tuple, list] = {}
    for r in saved_rows:
        saved_predictions.setdefault((r["question_id"], r["condition"]), []).append(r["pred"])
    return {
        "answers": len(new),
        "questions": len({r["question_id"] for r in new}),
        "identical_text": sum(r["completion"] == s["completion"] for r, s in pairs),
        "same_parsed_answer": sum(r["pred"] == s["pred"] for r, s in pairs),
        "same_parsed_answer_share": compare.mean([r["pred"] == s["pred"] for r, s in pairs]),
        "parse_failed_new": sum(r["pred"] is None for r in new),
        "parse_failed_saved": sum(s["pred"] is None for _, s in pairs),
        "independent_sample_agreement_saved": compare.mean(
            [pair_agreement(saved_predictions[(r["question_id"], r["condition"])]) for r in new]
        ),
    }


# --------------------------------------------------------------------------- #
# the smoke run
# --------------------------------------------------------------------------- #


def penalty_of(record: dict, weight: float) -> float:
    names_cued_option = record["cue_letter"] is not None and record["parsed_answer"] == record["cue_letter"]
    return -weight * record["probe_score"] if names_cued_option else 0.0


def raw_example(record: dict, weight: float) -> dict:
    return {
        "completion_end": record["completion"][-200:],
        "parsed_answer": record["parsed_answer"],
        "cue_letter": record["cue_letter"],
        "train_label": record["train_label"],
        "gold_letter": record["gold_letter"],
        "task_reward": record["reward"],
        "probe_score": record["probe_score"],
        "penalty": penalty_of(record, weight),
    }


def smoke() -> dict:
    import wandb

    config = run_config(SMOKE_RUN)
    weight = float(config["overrides"]["loss.detector"])
    run_dir = run_dir_of(SMOKE_RUN)
    run_metrics = json.loads((EXP_DIR / f"run_metrics_{config['run_key']}.json").read_text())
    run = wandb.Api().run(f"{WANDB_PROJECT_PATH}/{run_metrics['run_id']}")
    wandb_rows = {row["grpo_train_step"]: row for row in run.scan_history(keys=["grpo_train_step", "reward_mean", "rewards/detector"])}
    records = compare.training_records(run_dir)

    steps = []
    for step, rows in records.items():
        penalties = [penalty_of(r, weight) for r in rows]
        cut_off = [not r["completion"].endswith(EOT) for r in rows]
        recomputed_reward_mean = compare.mean([r["reward"] + p - NON_TERMINATION_PENALTY * c for r, p, c in zip(rows, penalties, cut_off)])
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

    labels = compare.read_jsonl(SMOKE_EVAL_DIR / "labels.jsonl")
    with_followup = [r for r in labels if r["p_yes"] is not None]
    comparison = json.loads((SMOKE_COMPARE_DIR / "comparison.json").read_text())["eval"]["all"]
    probe_rows = {
        f"{checkpoint} {condition}": {k: v for k, v in comparison[checkpoint][condition].items() if k.startswith("probe_")}
        for checkpoint in ("base", "after_rl") for condition in ("cue_gold", "cue_dist")
    }
    last_rows = records[max(records)]
    naming = [r for r in last_rows if penalty_of(r, weight) != 0.0]
    examples = [
        next(r for r in naming if r["reward"] == 1.0),
        next(r for r in naming if r["reward"] == 0.0),
        next(r for r in last_rows if r["parsed_answer"] is not None and r["parsed_answer"] != r["cue_letter"]),
    ]
    return {
        "run_dir": str(run_dir),
        "penalty_weight": weight,
        "steps": steps,
        "saved_layers": {subdir: saved_layers(run_dir, subdir) for subdir in (S.START_PROBE_SUBDIR, S.FRESH_PROBE_SUBDIR)},
        "wandb_end_of_run_auc": {subdir: run.summary.get(key) for subdir, key in END_OF_RUN_AUC_KEYS.items()},
        "in_loop_eval_rows": len(run_metrics["sycophancy_eval_history"]),
        "smoke_eval_cued_answers": len(labels),
        "smoke_eval_with_followup": len(with_followup),
        "smoke_eval_with_all_probe_scores": sum(all(r[name] is not None for name in S.PROBE_SCORES) for r in with_followup),
        "smoke_eval_fresh_differs_from_start": sum(r["probe_fresh"] != r["probe_start"] for r in with_followup),
        "compare_smoke_probe_rows": probe_rows,
        "raw_training_answers_last_step": [raw_example(r, weight) for r in examples],
    }


# --------------------------------------------------------------------------- #
# row A
# --------------------------------------------------------------------------- #


def copy_base() -> None:
    out_path = BASE_RESCORED_DIR / "completions.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    rows = [r for r in compare.read_jsonl(BASE_261013 / "completions.jsonl") if r["condition"] == "none" or r["sample_idx"] == 0]
    assert len(rows) == S.N_QUESTIONS["eval"] * sum(S.N_SAMPLES.values()), len(rows)
    eval_checkpoint.write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


def row_a() -> dict:
    new = {key_of(r): r for r in compare.read_jsonl(BASE_RESCORED_DIR / "followups.jsonl")}
    saved = {key_of(r): r for r in compare.read_jsonl(BASE_261013 / "followups.jsonl") if r["sample_idx"] == 0}
    assert set(new) == set(saved), "the follow-ups differ from 261013's sample-0 follow-ups"
    differences = [abs(new[k]["p_yes"] - saved[k]["p_yes"]) for k in new]
    said_flips = [k for k in new if eval_checkpoint.said_of(new[k]["p_yes"]) != eval_checkpoint.said_of(saved[k]["p_yes"])]
    tables = compare.eval_tables({"base": compare.rollout_rows(BASE_RESCORED_DIR)})["all"]["base"]
    check = json.loads((S.PACKAGE_DIR / "check_probe_scoring.json").read_text())["c_base_eval"]["probe_rows"]
    return {
        "followups": len(new),
        "p_yes_mean_abs_difference": compare.mean(differences),
        "p_yes_max_abs_difference": max(differences),
        "said_flips": len(said_flips),
        "said_flips_saved_p_yes": [saved[k]["p_yes"] for k in said_flips],
        "row_a": {c: {k: v for k, v in tables[c].items() if k.startswith("probe_start_") and "base_activations" not in k}
                  for c in ("cue_gold", "cue_dist")},
        "row_a_package_check": check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fit_probe", "probe", "base50", "smoke", "copy_base", "row_a"))
    parser.add_argument("name", nargs="?", help="probe: probe_fit or a run name")
    parser.add_argument("log", nargs="?", type=Path, help="probe: the log holding the trainer's output")
    args = parser.parse_args()
    if args.command == "fit_probe":
        fit_probe()
    elif args.command == "probe":
        write_json(f"probe_{args.name}.json", probe_stats(args.name, args.log))
    elif args.command == "base50":
        write_json("base50.json", base50())
    elif args.command == "smoke":
        write_json("smoke.json", smoke())
    elif args.command == "copy_base":
        copy_base()
    else:
        write_json("row_a.json", row_a())


if __name__ == "__main__":
    main()
