"""Check the RL run's per-step answer records against what the code should have served and scored.

    python check_records.py --run-dir <GRPO run dir> --wandb-run-id <id> --out check_records_<name>.json
        [--questions <the run's question file>] [--no-cue]

For each record file `checkpoints/responses/step_<s>.jsonl` (64 prompts x 8 answers):

  1. Fields. Every record's question is a training question of dataset/labels.jsonl, and its
     gold_letter and train_label equal that question's answer and train_label. The 8 records of a
     group share the prompt, question and cue.
  2. Served prompt. The prompt equals the question's no-CoT no-cue user content with the cue at the
     recorded cue_letter in wording template_idx, under the Llama-3 chat template, followed by the
     prefill "The answer is (" (whose last token is 320).
  3. Cue draws. Over the 64 prompts in batch order, (cue at the training label, template_idx)
     equals the draws of `draw_cue` from the trainer's generator: numpy
     default_rng(training seed + s + 2_000_000), probability task_args.cue_gold_prob.
  4. One token. Every answer is one generated token: the completion without a trailing <|eot_id|>
     tokenizes to at most one token (empty when the model's one token was the end-of-turn token).
  5. Scores. parsed_answer equals the no-CoT parser (the token is exactly "A" or "B") on the
     completion, and reward equals 1.0 if parsed_answer equals train_label, else 0.0.
  6. W&B. The counts of reward, true accuracy, followed-cue and parse failures recomputed with
     the task's reward function equal W&B's training-step means for step s + 1 times 512, and
     the count of prompts with the cue at the true answer equals train/cue_gold_fraction x 64.

Also reported per step (not checked): unparsed answers and the share of groups whose 8 rewards are equal.
Run-level: the W&B config has the no-CoT settings (prefill, 1-token responses, no non-termination
penalty), the question file, the cue rate and 64 prompts per step. The file step s -> W&B step
s + 1 mapping and the seed offset are the trainer's (checked on August's run by 261012's
check_responses.py). Run with run_env.sh sourced.

--questions (default dataset/labels.jsonl) is the question file the run was configured with, used for
check 1 and the served prompts. --no-cue is for a run with task_args.train_cue false (training prompts
without a cue, 261020): check 2 expects the no-cue user content itself (then the prefill), check 3
becomes "cue_letter and template_idx are None on every prompt", and check 6 drops followed-cue and
train/cue_gold_fraction, which such a run does not log. The W&B config's train_cue must match the flag.
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
    parse_answer,
    render_cued_user_content,
)

import nocot_spec as S

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
NO_CUE_COUNT_KEYS = tuple(key for key in COUNT_KEYS if key != "rewards_metrics/followed_cue")


def wandb_training_rows(run_id: str, no_cue: bool) -> tuple[dict, dict[int, dict]]:
    run = wandb.Api().run(f"{WANDB_PROJECT_PATH}/{run_id}")
    keys = ["grpo_train_step", *NO_CUE_COUNT_KEYS] if no_cue else ["grpo_train_step", "train/cue_gold_fraction", *COUNT_KEYS]
    rows = {row["grpo_train_step"]: row for row in run.scan_history(keys=keys)}
    return run.config, rows


def served_prompt(tokenizer, question: dict, cue_letter: str | None, template_idx: int | None, no_cue: bool) -> str:
    """The prompt the trainer should have served: the no-cue content (plus the recorded cue unless no_cue), then the prefill."""
    if no_cue:
        user_content = question["user_content_none"]
    else:
        user_content = render_cued_user_content(question["user_content_none"], cue_letter, template_idx)
    templated = tokenizer.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
    return templated + S.NOCOT_PREFILL


def is_one_token(tokenizer, completion: str) -> bool:
    """One generated token: a token (an end-of-turn token may follow) or the end-of-turn token itself."""
    return len(tokenizer(completion.removesuffix(EOT), add_special_tokens=False)["input_ids"]) <= 1


def check_step(step: int, rows: list[dict], questions: dict[int, dict], tokenizer, seed: int, cue_label_prob: float, wandb_row: dict, no_cue: bool) -> dict:
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
        r["prompt"] == served_prompt(tokenizer, questions[r["question_id"]], r["cue_letter"], r["template_idx"], no_cue)
        and tokenizer(r["prompt"], add_special_tokens=False)["input_ids"][-1] == S.PREFILL_LAST_TOKEN_ID
        for r in heads
    )
    if no_cue:
        cue_matching = sum(r["cue_letter"] is None and r["template_idx"] is None for r in heads)
    else:
        served_draws = [(r["cue_letter"] == r["train_label"], r["template_idx"]) for r in heads]
        generator = np.random.default_rng(seed + step + CUE_RNG_OFFSET)
        expected_draws = [draw_cue(generator, cue_label_prob) for _ in heads]
        cue_matching = sum(a == b for a, b in zip(served_draws, expected_draws))

    one_token = sum(is_one_token(tokenizer, r["completion"]) for r in rows)
    completions = [r["completion"].removesuffix(EOT) for r in rows]
    parsed_matching = sum(r["parsed_answer"] == parse_answer(c, letter_only=True) for r, c in zip(rows, completions))
    reward_matching = sum(r["reward"] == (1.0 if r["parsed_answer"] == r["train_label"] else 0.0) for r in rows)

    count_keys = NO_CUE_COUNT_KEYS if no_cue else COUNT_KEYS
    _, metrics = compute_multiple_choice_rewards(
        completions,
        [r["gold_letter"] for r in rows],
        [r["train_label"] for r in rows],
        None if no_cue else [r["cue_letter"] for r in rows],
        letter_only=True,
    )
    recomputed = {key: int(sum(metrics[key])) for key in count_keys}
    logged = {}
    for key in count_keys:
        count = wandb_row[key] * len(rows)
        assert abs(count - round(count)) < 1e-6, f"step {step}: W&B {key} x {len(rows)} = {count} is not a count"
        logged[key] = round(count)
    if not no_cue:
        recomputed["cue_gold_prompts"] = sum(r["cue_letter"] == r["gold_letter"] for r in heads)
        cue_gold_count = wandb_row["train/cue_gold_fraction"] * PROMPTS_PER_STEP
        assert abs(cue_gold_count - round(cue_gold_count)) < 1e-6
        logged["cue_gold_prompts"] = round(cue_gold_count)

    return {
        "file_step": step,
        "wandb_train_step": step + 1,
        "fields_matching": fields_matching,
        "prompts_matching": prompts_matching,
        ("no_cue_fields_matching" if no_cue else "cue_draws_matching"): cue_matching,
        "one_token_answers": one_token,
        "parsed_matching": parsed_matching,
        "reward_matching": reward_matching,
        "unparsed": sum(r["parsed_answer"] is None for r in rows),
        "unparsed_completions": sorted({c for r, c in zip(rows, completions) if r["parsed_answer"] is None}),
        "same_reward_groups": sum(len({r["reward"] for r in group}) == 1 for group in groups) / len(groups),
        "recomputed_counts": recomputed,
        "wandb_counts": logged,
        "PASS": fields_matching == prompts_matching == cue_matching == PROMPTS_PER_STEP
        and one_token == parsed_matching == reward_matching == len(rows)
        and recomputed == logged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=S.DATASET_PATH, help="the run's question file (default: this folder's)")
    parser.add_argument("--no-cue", action="store_true", help="the run served training prompts without a cue (task_args.train_cue false)")
    args = parser.parse_args()

    config, wandb_rows = wandb_training_rows(args.wandb_run_id, args.no_cue)
    seed = config["training"]["seed"]
    cue_label_prob = config["task_args"]["cue_gold_prob"]
    assert config["training"]["batch_size"] * config["training"]["grad_accum_steps"] == PROMPTS_PER_STEP
    assert config["data"]["task_dataset"]["mmlu_questions_jsonl"] == str(args.questions), "the run used a different question file"
    assert cue_label_prob == S.CUE_LABEL_PROB
    train_cue = config["task_args"].get("train_cue", True)  # absent in runs made before the setting existed
    assert train_cue == (not args.no_cue), f"the run's task_args.train_cue is {train_cue}"
    assert config["task_args"]["nocot_prefill"] == S.NOCOT_PREFILL, config["task_args"]["nocot_prefill"]
    assert config["grpo"]["tokenizer"]["max_response_length"] == 1, config["grpo"]["tokenizer"]["max_response_length"]
    assert config["grpo"]["loss"]["non_termination_penalty"] is None, config["grpo"]["loss"]["non_termination_penalty"]
    print(f"run {args.wandb_run_id}: seed {seed}, cue_gold_prob {cue_label_prob}, train_cue {train_cue}, no-CoT settings present, {len(wandb_rows)} W&B training rows")

    response_files = sorted((args.run_dir / "checkpoints/responses").glob("step_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    steps = [int(path.stem.split("_")[1]) for path in response_files]
    assert steps == list(range(len(steps))), f"response files are not step_0..step_{len(steps) - 1}"
    assert len(steps) == len(wandb_rows), f"{len(steps)} response files but {len(wandb_rows)} W&B training rows"

    questions = {q["question_id"]: q for q in map(json.loads, args.questions.read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(S.MODEL_NAME)
    per_step = []
    for path, step in zip(response_files, steps):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        result = check_step(step, rows, questions, tokenizer, seed, cue_label_prob, wandb_rows[step + 1], args.no_cue)
        per_step.append(result)
        print(
            f"step_{step}: fields {result['fields_matching']}, prompts {result['prompts_matching']}, "
            f"{'no-cue fields' if args.no_cue else 'cue draws'} "
            f"{result['no_cue_fields_matching' if args.no_cue else 'cue_draws_matching']} of {PROMPTS_PER_STEP}; one-token {result['one_token_answers']}, parsed "
            f"{result['parsed_matching']}, reward {result['reward_matching']} of {len(rows)}; unparsed {result['unparsed']}; "
            f"same-reward groups {result['same_reward_groups']:.3f}; counts {result['recomputed_counts']} "
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
        "unparsed_share": sum(r["unparsed"] for r in per_step) / (len(per_step) * PROMPTS_PER_STEP * GENERATIONS_PER_PROMPT),
        "same_reward_groups_mean": sum(r["same_reward_groups"] for r in per_step) / len(per_step),
        "per_step": per_step,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: {sum(r['PASS'] for r in per_step)}/{len(per_step)} files; unparsed share "
          f"{output['unparsed_share']:.4f}; same-reward groups {output['same_reward_groups_mean']:.3f}; wrote {args.out}")


if __name__ == "__main__":
    main()
