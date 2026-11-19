"""The numbers the reviews of the unattended queues read, and the copy of 261013's base answers for scoring with each
run's initial probe. 261114's checks.py without its advantages mode, with 261013's check_records.py's record checks
adapted to labels redrawn every prompt (task_args.redraw_label_flip_prob), and with the learning rate, the saved
adapters, the composed config and the initial probe's md5 added.

    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh && cd <this folder>
    $PY checks.py config RUN_YAML --out FILE
    $PY checks.py finished RUN_YAML --out FILE          (a finished run: the smoke run or a 400-step run)
    $PY checks.py early RUN_YAML LOG --out FILE         (while the run is still training)
    $PY checks.py probe RUN_YAML LOG --out FILE
    $PY checks.py copy_base OUT_DIR                     -> OUT_DIR/completions.jsonl

Nothing here passes or fails a review; the reviewer applies the rules in the queue's CLAUDE.md to these numbers.

config    RUN_YAML composed through the launcher's overrides (as the trainer will see it): the settings this
          experiment sets or relies on, and the number of training steps the trainer will run
          (min(num_epochs x ceil(1217 / 64), max_steps), grpo_trainer.total_steps).
finished  Per step of a finished run, from its step records (checkpoints/responses/step_N.jsonl, 64 prompts x 8
          answers in batch order):
          - records: the 8 answers of a prompt share its fields; each prompt's train_label and label_flipped equal the
            redraw recomputed from numpy default_rng(training seed + step + 3_000_000) (one draw per prompt in batch
            order: flipped if below redraw_label_flip_prob, label = the other option if flipped, else the correct
            answer); its cue_letter and template_idx equal draw_cue on default_rng(seed + step + 2_000_000) placed
            relative to that label; its served prompt equals the question's no-cue content plus that cue under the
            Llama-3 chat template; every answer's parsed_answer equals the task's parser on its completion and its
            reward equals 1.0 if parsed_answer equals train_label, else 0.0.
          - rewards: W&B's reward_mean and rewards/detector against the records (task reward + the penalty:
            -weight x probe score on answers naming the cued option, 0 elsewhere; reward_mean also holds the trainer's
            -0.1 on answers cut off at the length cap), and probe scores present exactly on parsed answers.
          - W&B's learning_rate per training step (file step s is W&B training step s + 1).
          Also the run's saved adapters (checkpoints/adapters/step_N), whether the last one's adapter_model.bin is
          byte-identical to checkpoints/final's and the first one's differs, the full training states left
          (checkpoints/step_*), and the batch settings the W&B config records.
early     A running run's first numbers, once its first training step is logged (the run folder and W&B id come from
          LOG's last "Phase-0 detector results cache path" line): finished's per-step rows for every step W&B has
          logged, the probe training class counts the trainer printed against the counts of the config's probe
          training file, and W&B's kl_div at training step 1.
probe     A run's initial probe: its test scores (the trainer's saved scores on the run's probe test file, mean over
          layers 13-16 of each layer's sigmoid): class counts, AUC, class means; the training class counts the trainer
          printed in LOG; and the md5 of its files against 261107's record of round 1's probe (the probe 261103's
          seed-42 runs fit, which the old chains used), in the format `find . -type f | sort | xargs md5sum`.
copy_base 261013's base eval cut to the package's eval (8 no-cue samples + cued sample 0 per condition).
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import check_probe_scoring  # noqa: E402
import compare  # noqa: E402
import eval_checkpoint  # noqa: E402
import spec as S  # noqa: E402

from obfuscation_atlas.tasks.multiple_choice import (  # noqa: E402
    draw_cue,
    other_letter,
    parse_answer,
    render_cued_user_content,
)

RUN_RESULTS = Path("/workspace/results/run_results")
BASE_261013 = S.SYCOPHANCY_DIR / "261013_noisy_label_rl" / "evals" / "base"
BIAS_PROBE_TRAIN_PATH = S.PACKAGE_DIR / "data" / "bias_probe_train.jsonl"  # the trainer's default probe files (config.py)
ROUND1_PROBE_MD5 = S.SYCOPHANCY_DIR / "261107_frozen_probe_kl_0_01" / "seed_42" / "checks" / "frozen_probe_md5.txt"
WANDB_PROJECT_PATH = "lilylixin6-the-harker-school/deception_detection"
DETECTOR_KEY = check_probe_scoring.DETECTOR_KEY
EOT = "<|eot_id|>"
NON_TERMINATION_PENALTY = 0.1  # the trainer's penalty on answers cut off at the length cap (grpo_mcq)
CUE_RNG_OFFSET = 2_000_000  # grpo_trainer._prepare_mcq_training_inputs
LABEL_RNG_OFFSET = 3_000_000  # the same, for the label redraw
PROMPTS_PER_STEP = 64
GENERATIONS_PER_PROMPT = 8
GROUP_FIELDS = ("prompt", "question_id", "gold_letter", "train_label", "label_flipped", "cue_letter", "template_idx")


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result if "per_step" not in result else {k: v for k, v in result.items() if k != "per_step"}, indent=2))


def run_config(run_yaml: Path) -> dict:
    return yaml.safe_load(run_yaml.read_text())


def run_dir_and_id(run_yaml: Path) -> tuple[Path, str]:
    """A run's folder and W&B id (run id from its run_metrics json, as run_eval.sh finds it)."""
    config = run_config(run_yaml)
    run_id = json.loads((Path(config["output_dir"]) / f"run_metrics_{config['run_key']}.json").read_text())["run_id"]
    return RUN_RESULTS / config["group"] / f"{config['run_key']}_{run_id}", run_id


def wandb_run(run_id: str):
    import wandb

    return wandb.Api().run(f"{WANDB_PROJECT_PATH}/{run_id}")


def wandb_history(run_id: str, keys: list[str]) -> dict[int, dict]:
    return {row["grpo_train_step"]: row for row in wandb_run(run_id).scan_history(keys=["grpo_train_step", *keys])}


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(S.MODEL_NAME)


def train_questions() -> dict[int, dict]:
    return {q["question_id"]: q for q in compare.read_jsonl(S.QUESTIONS_PATH) if q["split"] == "train"}


# --------------------------------------------------------------------------- #
# a run's step records: the served labels, cues and prompts, the reward, and the penalty against W&B
# --------------------------------------------------------------------------- #


def record_step(step: int, rows: list[dict], questions: dict[int, dict], chat_tokenizer, overrides: dict) -> dict:
    """One step's records against the label redraw, the cue draws and the served prompts recomputed from the seed."""
    assert len(rows) == PROMPTS_PER_STEP * GENERATIONS_PER_PROMPT, f"step {step}: {len(rows)} rows"
    groups = [rows[i : i + GENERATIONS_PER_PROMPT] for i in range(0, len(rows), GENERATIONS_PER_PROMPT)]
    for group in groups:
        assert all(r[f] == group[0][f] for r in group for f in GROUP_FIELDS), f"step {step}: a group mixes prompts"
    heads = [group[0] for group in groups]
    seed, flip_prob, cue_prob = overrides["training.seed"], overrides["task_args.redraw_label_flip_prob"], overrides["task_args.cue_gold_prob"]

    label_rng = np.random.default_rng(seed + step + LABEL_RNG_OFFSET)
    cue_rng = np.random.default_rng(seed + step + CUE_RNG_OFFSET)
    labels_matching = cues_matching = prompts_matching = flipped = 0
    for r in heads:
        q = questions[r["question_id"]]  # a KeyError means the record's question is not a training question
        expected_flipped = bool(label_rng.random() < flip_prob)
        expected_label = other_letter(q["answer"]) if expected_flipped else q["answer"]
        at_label, template_idx = draw_cue(cue_rng, cue_prob)
        expected_cue = expected_label if at_label else other_letter(expected_label)
        expected_prompt = chat_tokenizer.apply_chat_template(
            [{"role": "user", "content": render_cued_user_content(q["user_content_none"], expected_cue, template_idx)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        flipped += expected_flipped
        labels_matching += r["gold_letter"] == q["answer"] and r["train_label"] == expected_label and r["label_flipped"] == expected_flipped
        cues_matching += r["cue_letter"] == expected_cue and r["template_idx"] == template_idx
        prompts_matching += r["prompt"] == expected_prompt

    parsed_matching = sum(r["parsed_answer"] == parse_answer(r["completion"].removesuffix(EOT), letter_only=False) for r in rows)
    reward_matching = sum(r["reward"] == (1.0 if r["parsed_answer"] == r["train_label"] else 0.0) for r in rows)
    return {
        "prompts": len(heads),
        "flipped_prompts": flipped,
        "labels_matching": labels_matching,
        "cues_matching": cues_matching,
        "prompts_matching": prompts_matching,
        "answers": len(rows),
        "parsed_matching": parsed_matching,
        "reward_matching": reward_matching,
    }


def penalty_of(record: dict, weight: float) -> float:
    names_cued_option = record["cue_letter"] is not None and record["parsed_answer"] == record["cue_letter"]
    return -weight * record["probe_score"] if names_cued_option else 0.0


def reward_step(rows: list[dict], weight: float, wandb_row: dict) -> dict:
    """One step's records against W&B's reward_mean and rewards/detector for that step."""
    penalties = [penalty_of(r, weight) for r in rows]
    cut_off = [not r["completion"].endswith(EOT) for r in rows]
    recomputed_reward_mean = compare.mean([r["reward"] + p - NON_TERMINATION_PENALTY * c for r, p, c in zip(rows, penalties, cut_off)])
    return {
        "probe_score_present_iff_parsed_mismatches": sum((r["probe_score"] is None) != (r["parsed_answer"] is None) for r in rows),
        "naming_cued_option": sum(r["cue_letter"] is not None and r["parsed_answer"] == r["cue_letter"] for r in rows),
        "nonzero_penalties": sum(p != 0.0 for p in penalties),
        "cut_off": sum(cut_off),
        "recomputed_reward_mean": recomputed_reward_mean,
        "wandb_reward_mean": wandb_row["reward_mean"],
        "reward_mean_abs_difference": abs(recomputed_reward_mean - wandb_row["reward_mean"]),
        "recomputed_penalty_mean": compare.mean(penalties),
        "wandb_rewards_detector": wandb_row["rewards/detector"],
        "penalty_mean_abs_difference": abs(compare.mean(penalties) - wandb_row["rewards/detector"]),
    }


def step_rows(steps: list[int], records: dict[int, list[dict]], wandb_rows: dict[int, dict], overrides: dict) -> list[dict]:
    """finished's per-step rows for the given file steps (file step s is W&B training step s + 1)."""
    questions, chat_tokenizer, weight = train_questions(), tokenizer(), float(overrides["loss.detector"])
    return [
        {
            "file_step": step,
            "wandb_learning_rate": wandb_rows[step + 1]["learning_rate"],
            **record_step(step, records[step], questions, chat_tokenizer, overrides),
            **reward_step(records[step], weight, wandb_rows[step + 1]),
        }
        for step in steps
    ]


def totals(per_step: list[dict]) -> dict:
    """The per-step rows summed (counts) or maximized (differences) over steps."""
    count_keys = ("prompts", "flipped_prompts", "labels_matching", "cues_matching", "prompts_matching", "answers",
                  "parsed_matching", "reward_matching", "probe_score_present_iff_parsed_mismatches", "naming_cued_option", "nonzero_penalties", "cut_off")
    out = {key: sum(s[key] for s in per_step) for key in count_keys}
    out["flipped_share"] = out["flipped_prompts"] / out["prompts"]
    out["max_reward_mean_abs_difference"] = max(s["reward_mean_abs_difference"] for s in per_step)
    out["max_penalty_mean_abs_difference"] = max(s["penalty_mean_abs_difference"] for s in per_step)
    out["learning_rates_first_6"] = [s["wandb_learning_rate"] for s in per_step[:6]]
    out["learning_rate_values_after_step_5"] = sorted({s["wandb_learning_rate"] for s in per_step[5:]})
    return out


def adapters(run_dir: Path) -> dict:
    checkpoints = run_dir / "checkpoints"
    saved = sorted((p for p in (checkpoints / "adapters").iterdir()), key=lambda p: int(p.name.removeprefix("step_")))
    final_weights = checkpoints / "final" / "adapter_model.bin"
    return {
        "adapter_steps": [int(p.name.removeprefix("step_")) for p in saved],
        "adapter_steps_with_config": [int(p.name.removeprefix("step_")) for p in saved if (p / "adapter_config.json").exists()],
        "last_adapter_identical_to_final": filecmp.cmp(saved[-1] / "adapter_model.bin", final_weights, shallow=False),
        "first_adapter_identical_to_final": filecmp.cmp(saved[0] / "adapter_model.bin", final_weights, shallow=False),
        "full_training_states_left": sorted(p.name for p in checkpoints.glob("step_*")),
    }


def finished(run_yaml: Path) -> dict:
    overrides = run_config(run_yaml)["overrides"]
    run_dir, run_id = run_dir_and_id(run_yaml)
    wandb_rows = wandb_history(run_id, ["reward_mean", "rewards/detector", "learning_rate"])
    records = compare.training_records(run_dir)
    assert sorted(wandb_rows) == [step + 1 for step in sorted(records)], (len(wandb_rows), len(records))
    per_step = step_rows(sorted(records), records, wandb_rows, overrides)
    training = wandb_run(run_id).config["training"]
    return {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "penalty_weight": float(overrides["loss.detector"]),
        "step_files": len(records),
        "wandb_training": {key: training[key] for key in ("batch_size", "grad_accum_steps", "max_steps", "num_epochs", "scheduler", "warmup_steps", "learning_rate")},
        **adapters(run_dir),
        "totals": totals(per_step),
        "per_step": per_step,
    }


def early(run_yaml: Path, log: Path) -> dict:
    config = run_config(run_yaml)
    overrides = config["overrides"]
    text = log.read_text()
    run_dirs = re.findall(r"^Phase-0 detector results cache path: (\S+)$", text, flags=re.MULTILINE)
    run_dir = Path(run_dirs[-1])
    prefix = f"{config['run_key']}_"
    assert run_dir.parent == RUN_RESULTS / config["group"] and run_dir.name.startswith(prefix), run_dir
    run_id = run_dir.name.removeprefix(prefix)
    wandb_rows = wandb_history(run_id, ["kl_div", "reward_mean", "rewards/detector", "learning_rate"])
    assert 1 in wandb_rows, f"W&B run {run_id} has not logged training step 1 yet"
    logged_file_steps = sorted(step - 1 for step in wandb_rows)
    assert logged_file_steps == list(range(len(logged_file_steps))), logged_file_steps
    records = {step: compare.read_jsonl(run_dir / "checkpoints" / "responses" / f"step_{step}.jsonl") for step in logged_file_steps}
    per_step = step_rows(logged_file_steps, records, wandb_rows, overrides)
    printed = re.findall(r"Probe dataset \(label_mode=normal\): \((\d+), (\d+)\)", text)
    return {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "total_steps_lines": re.findall(r"Total number of steps: .*$", text, flags=re.MULTILINE),
        "train_biased_unbiased_printed": [[int(a), int(b)] for a, b in printed],
        "expected_train_biased_unbiased": class_counts(BIAS_PROBE_TRAIN_PATH),
        "wandb_kl_div_step_1": wandb_rows[1]["kl_div"],
        "penalty_weight": float(overrides["loss.detector"]),
        "totals": totals(per_step),
        "per_step": per_step,
    }


# --------------------------------------------------------------------------- #
# the composed config
# --------------------------------------------------------------------------- #


def composed_config(run_yaml: Path) -> dict:
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    sys.path.insert(0, "/workspace/sycophancy-rl")
    from training.grpo_launcher import build_grpo_args

    from obfuscation_atlas.config import register_configs

    args = build_grpo_args(run_config(run_yaml))
    register_configs()
    with initialize(version_base=None):
        cfg = OmegaConf.to_object(compose(config_name=args[3].removeprefix("--config-name="), overrides=args[4:]))
    prompts_per_step = cfg.training.batch_size * cfg.training.grad_accum_steps
    return {
        "loss.detector": cfg.loss.detector,
        "loss.kl": cfg.loss.kl,
        "loss.advantage_std_floor": cfg.loss.advantage_std_floor,
        "training": {key: getattr(cfg.training, key) for key in ("seed", "max_steps", "num_epochs", "scheduler", "warmup_steps",
                                                                 "learning_rate", "batch_size", "grad_accum_steps", "eval_steps")},
        "grpo.optimizer.scheduler_type": cfg.grpo.optimizer.scheduler_type,
        "grpo.logging.adapter_save_steps": cfg.grpo.logging.adapter_save_steps,
        "grpo.logging.save_steps": cfg.grpo.logging.save_steps,
        "task_args": {key: getattr(cfg.task_args, key) for key in ("redraw_label_flip_prob", "cue_gold_prob", "train_cue", "bias_probe_at_answer")},
        "detector.train_every_n_steps": cfg.detector.train_every_n_steps,
        "probe_files": [cfg.data.probe_dataset.bias_probe_train_jsonl, cfg.data.probe_dataset.bias_probe_eval_jsonl],
        "training_steps_the_trainer_will_run": min(cfg.training.num_epochs * math.ceil(S.N_QUESTIONS["train"] / prompts_per_step), cfg.training.max_steps),
    }


# --------------------------------------------------------------------------- #
# the probe
# --------------------------------------------------------------------------- #


def class_counts(path: Path) -> list[int]:
    """[biased, unbiased] rows of a probe-set file (label 1 / 0)."""
    labels = [row["label"] for row in compare.read_jsonl(path)]
    return [labels.count(1), labels.count(0)]


def md5_listing(folder: Path) -> str:
    """`cd folder && find . -type f | sort | xargs md5sum` (C-locale byte order of the paths)."""
    paths = sorted("./" + str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    return "".join(f"{hashlib.md5((folder / p).read_bytes()).hexdigest()}  {p}\n" for p in paths)


def probe_stats(run_yaml: Path, log: Path) -> dict:
    run_dir, _ = run_dir_and_id(run_yaml)
    scores = check_probe_scoring.saved_test_scores(run_dir)
    printed = set(re.findall(r"Probe dataset \(label_mode=normal\): \((\d+), (\d+)\)", log.read_text()))
    assert len(printed) == 1, f"{log}: probe training counts printed {printed}"
    probe_dir = run_dir / S.START_PROBE_SUBDIR / f"detector_{S.START_PROBE_SUBDIR}__{DETECTOR_KEY}"
    return {
        "run_dir": str(run_dir),
        "log": str(log),
        "train_biased_unbiased": [int(n) for n in printed.pop()],
        "expected_train_biased_unbiased": class_counts(BIAS_PROBE_TRAIN_PATH),
        "test_biased_unbiased": [len(scores["positive"]), len(scores["negative"])],
        "expected_test_biased_unbiased": class_counts(S.BIAS_PROBE_EVAL_PATH),
        "test_auc": compare.auc(scores["positive"], scores["negative"]),
        "test_mean_biased": compare.mean(scores["positive"]),
        "test_mean_unbiased": compare.mean(scores["negative"]),
        "md5_identical_to_round1_probe": md5_listing(probe_dir) == ROUND1_PROBE_MD5.read_text(),
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
    for name in ("config", "finished"):
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
    if args.command == "config":
        write_json(args.out, composed_config(args.run_yaml))
    elif args.command == "finished":
        write_json(args.out, finished(args.run_yaml))
    elif args.command == "early":
        write_json(args.out, early(args.run_yaml, args.log))
    elif args.command == "probe":
        write_json(args.out, probe_stats(args.run_yaml, args.log))
    else:
        copy_base(args.out_dir)


if __name__ == "__main__":
    main()
