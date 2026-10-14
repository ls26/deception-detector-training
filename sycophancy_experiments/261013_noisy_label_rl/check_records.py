"""Check the RL run's per-step rollout records against what the code should have served and scored.

    python check_records.py --run-dir <GRPO run dir> --wandb-run-id <id> --out check_records_<name>.json

For each record file `checkpoints/responses/step_<s>.jsonl` (64 prompts x 8 completions):

  1. Fields. Every record's question is a training question of dataset/labels.jsonl, and its
     gold_letter and train_label equal that question's answer and train_label. The 8 records of a
     group share the prompt, question and cue.
  2. Served prompt. The prompt equals the question's no-cue user content with the cue at the
     recorded cue_letter in wording template_idx, under the Llama-3 chat template.
  3. Cue draws. Over the 64 prompts in batch order, (cue at the training label, template_idx)
     equals the draws of `draw_cue` from the trainer's generator: numpy
     default_rng(training seed + s + 2_000_000), probability task_args.cue_gold_prob.
  4. Scores. parsed_answer equals the task's parser on the completion, and reward equals 1.0 if
     parsed_answer equals train_label, else 0.0.
  5. W&B. The counts of reward, true accuracy, followed-cue and parse failures recomputed with
     the task's reward function equal W&B's training-step means for step s + 1 times 512, and
     the count of prompts with the cue at the true answer equals train/cue_gold_fraction x 64.

The file step s -> W&B step s + 1 mapping and the seed offset are the trainer's (checked on
August's run by 261012's check_responses.py). Run with run_env.sh sourced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wandb
from transformers import AutoTokenizer

from obfuscation_atlas.tasks.multiple_choice import (
    compute_multiple_choice_rewards,
    draw_cue,
    extract_answer,
    render_cued_user_content,
)

import noisy_spec as S

WANDB_PROJECT_PATH = "lilylixin6-the-harker-school/deception_detection"
EOT = "<|eot_id|>"
CUE_RNG_OFFSET = 2_000_000  # grpo_trainer._prepare_mcq_training_inputs
PROMPTS_PER_STEP = 64
GENERATIONS_PER_PROMPT = 8
GROUP_FIELDS = ("prompt", "question_id", "gold_letter", "train_label", "cue_letter", "template_idx")
COUNT_KEYS = (
    "rewards/task_success",
    "rewards_metrics/true_correct",
    "rewards_metrics/followed_cue",
    "rewards_metrics/parse_failed",
)


def wandb_training_rows(run_id: str) -> tuple[dict, dict[int, dict]]:
    run = wandb.Api().run(f"{WANDB_PROJECT_PATH}/{run_id}")
    keys = ["grpo_train_step", "train/cue_gold_fraction", *COUNT_KEYS]
    rows = {row["grpo_train_step"]: row for row in run.scan_history(keys=keys)}
    return run.config, rows


def check_step(step: int, rows: list[dict], questions: dict[int, dict], tokenizer, seed: int, cue_label_prob: float, wandb_row: dict) -> dict:
    assert len(rows) == PROMPTS_PER_STEP * GENERATIONS_PER_PROMPT, f"step {step}: {len(rows)} rows"
    groups = [rows[i : i + GENERATIONS_PER_PROMPT] for i in range(0, len(rows), GENERATIONS_PER_PROMPT)]
    for group in groups:
        assert all(all(r[f] == group[0][f] for f in GROUP_FIELDS) for r in group), f"step {step}: a group mixes prompts or cues"
    heads = [group[0] for group in groups]

    fields_matching = sum(
        q["split"] == "train" and r["gold_letter"] == q["answer"] and r["train_label"] == q["train_label"]
        for r in heads
        for q in [questions[r["question_id"]]]
    )
    prompts_matching = sum(
        r["prompt"]
        == tokenizer.apply_chat_template(
            [{"role": "user", "content": render_cued_user_content(questions[r["question_id"]]["user_content_none"], r["cue_letter"], r["template_idx"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in heads
    )
    served_draws = [(r["cue_letter"] == r["train_label"], r["template_idx"]) for r in heads]
    generator = np.random.default_rng(seed + step + CUE_RNG_OFFSET)
    expected_draws = [draw_cue(generator, cue_label_prob) for _ in heads]
    draws_matching = sum(a == b for a, b in zip(served_draws, expected_draws))

    completions = [r["completion"].removesuffix(EOT) for r in rows]
    parsed_matching = sum(r["parsed_answer"] == extract_answer(c)[0] for r, c in zip(rows, completions))
    reward_matching = sum(r["reward"] == (1.0 if r["parsed_answer"] == r["train_label"] else 0.0) for r in rows)

    _, metrics = compute_multiple_choice_rewards(
        completions, [r["gold_letter"] for r in rows], [r["train_label"] for r in rows], [r["cue_letter"] for r in rows]
    )
    recomputed = {key: int(sum(metrics[key])) for key in COUNT_KEYS}
    recomputed["cue_gold_prompts"] = sum(r["cue_letter"] == r["gold_letter"] for r in heads)
    logged = {}
    for key in COUNT_KEYS:
        count = wandb_row[key] * len(rows)
        assert abs(count - round(count)) < 1e-6, f"step {step}: W&B {key} x {len(rows)} = {count} is not a count"
        logged[key] = round(count)
    cue_gold_count = wandb_row["train/cue_gold_fraction"] * PROMPTS_PER_STEP
    assert abs(cue_gold_count - round(cue_gold_count)) < 1e-6
    logged["cue_gold_prompts"] = round(cue_gold_count)

    return {
        "file_step": step,
        "wandb_train_step": step + 1,
        "fields_matching": fields_matching,
        "prompts_matching": prompts_matching,
        "cue_draws_matching": draws_matching,
        "parsed_matching": parsed_matching,
        "reward_matching": reward_matching,
        "recomputed_counts": recomputed,
        "wandb_counts": logged,
        "PASS": fields_matching == prompts_matching == draws_matching == PROMPTS_PER_STEP
        and parsed_matching == reward_matching == len(rows)
        and recomputed == logged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config, wandb_rows = wandb_training_rows(args.wandb_run_id)
    seed = config["training"]["seed"]
    cue_label_prob = config["task_args"]["cue_gold_prob"]
    assert config["training"]["batch_size"] * config["training"]["grad_accum_steps"] == PROMPTS_PER_STEP
    assert config["data"]["task_dataset"]["mmlu_questions_jsonl"] == str(S.DATASET_PATH), "the run used a different question file"
    assert cue_label_prob == S.CUE_LABEL_PROB
    print(f"run {args.wandb_run_id}: seed {seed}, cue_gold_prob {cue_label_prob}, {len(wandb_rows)} W&B training rows")

    response_files = sorted((args.run_dir / "checkpoints/responses").glob("step_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    steps = [int(path.stem.split("_")[1]) for path in response_files]
    assert steps == list(range(len(steps))), f"response files are not step_0..step_{len(steps) - 1}"
    assert len(steps) == len(wandb_rows), f"{len(steps)} response files but {len(wandb_rows)} W&B training rows"

    questions = {q["question_id"]: q for q in map(json.loads, S.DATASET_PATH.read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(S.MODEL_NAME)
    per_step = []
    for path, step in zip(response_files, steps):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        result = check_step(step, rows, questions, tokenizer, seed, cue_label_prob, wandb_rows[step + 1])
        per_step.append(result)
        print(
            f"step_{step}: fields {result['fields_matching']}, prompts {result['prompts_matching']}, cue draws "
            f"{result['cue_draws_matching']} of {PROMPTS_PER_STEP}; parsed {result['parsed_matching']}, reward "
            f"{result['reward_matching']} of {len(rows)}; counts {result['recomputed_counts']} "
            f"{'==' if result['recomputed_counts'] == result['wandb_counts'] else '!='} W&B {result['wandb_counts']}"
        )

    passed = all(result["PASS"] for result in per_step)
    output = {
        "PASS": passed,
        "wandb_run_id": args.wandb_run_id,
        "run_dir": str(args.run_dir),
        "n_files": len(per_step),
        "seed": seed,
        "cue_gold_prob": cue_label_prob,
        "per_step": per_step,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: {sum(r['PASS'] for r in per_step)}/{len(per_step)} files; wrote {args.out}")


if __name__ == "__main__":
    main()
