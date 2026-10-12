"""Check a sycophancy GRPO run's per-step response files against what its code should produce.

For each response file `checkpoints/responses/step_<s>.jsonl` (64 prompts x 8 completions):

  1. Served cues. Every prompt is matched exactly to one (train question, cue letter, cue
     wording) by rendering all 1603 x 2 x 8 possible prompts with the task's own functions and
     the Llama-3 chat template. The sequence of (cue condition, wording) over the 64 prompts,
     in batch order, must equal the draws of `draw_cue` from the seeded generator the trainer
     uses: numpy default_rng(training seed + s + 2_000_000), probability `cue_gold_prob`.
     This fails if training served the dataset's fixed cues instead of redrawing them.
  2. Scores. Accuracy, cue-following and parse failures are recomputed from the completions
     with the task's reward function, scored against the served cue, and compared as counts
     with what W&B logged for training step s + 1 (its means times 512); the count of
     prompts whose cue pointed at the right answer is compared with `train/cue_gold_fraction`
     times 64. This fails if rewards were scored against a different cue than the one served.

The mapping file step s -> W&B step s + 1 and seed offset s were established on August's run
(260825_grpo_det0, W&B lx6gcbcl), where every file matches. Run with run_env.sh sourced.

Usage:
    python check_responses.py --run-dir <GRPO run dir> --wandb-run-id <id> --out check_<name>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wandb
from transformers import AutoTokenizer

from obfuscation_atlas.tasks.multiple_choice import (
    CUE_DIST,
    CUE_GOLD,
    CUE_TEMPLATES,
    compute_multiple_choice_rewards,
    draw_cue,
    render_cued_user_content,
)

QUESTIONS_JSONL = Path("/workspace/sycophancy_experiments/260819_domain_baseline/domain_questions.jsonl")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
WANDB_PROJECT_PATH = "lilylixin6-the-harker-school/deception_detection"
EOT = "<|eot_id|>"
CUE_RNG_OFFSET = 2_000_000  # grpo_trainer._prepare_mcq_training_inputs
PROMPTS_PER_STEP = 64
GENERATIONS_PER_PROMPT = 8
COUNT_KEYS = {
    "rewards/task_success": "rewards/task_success",
    "rewards_metrics/followed_cue": "rewards_metrics/followed_cue",
    "rewards_metrics/parse_failed": "rewards_metrics/parse_failed",
}


def prompt_lookup() -> dict[str, tuple[int, str, int, str]]:
    """Every possible training prompt -> (question id, cue letter, cue wording index, gold letter)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    questions = [json.loads(line) for line in QUESTIONS_JSONL.read_text().splitlines() if line.strip()]
    train_questions = [question for question in questions if question["split"] == "train"]
    assert len(train_questions) == 1603
    lookup = {}
    for question in train_questions:
        for cue_letter in ("A", "B"):
            for template_idx in range(len(CUE_TEMPLATES)):
                user_content = render_cued_user_content(question["user_content_none"], cue_letter, template_idx)
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
                )
                assert prompt not in lookup
                lookup[prompt] = (question["question_id"], cue_letter, template_idx, question["answer"])
    assert len(lookup) == 1603 * 2 * len(CUE_TEMPLATES)
    return lookup


def wandb_training_rows(run_id: str) -> tuple[dict, dict[int, dict]]:
    run = wandb.Api().run(f"{WANDB_PROJECT_PATH}/{run_id}")
    keys = ["grpo_train_step", "train/cue_gold_fraction", *COUNT_KEYS.values()]
    rows = {row["grpo_train_step"]: row for row in run.scan_history(keys=keys)}
    return run.config, rows


def check_step(step: int, rows: list[dict], lookup: dict, seed: int, cue_gold_prob: float, wandb_row: dict) -> dict:
    assert len(rows) == PROMPTS_PER_STEP * GENERATIONS_PER_PROMPT, f"step {step}: {len(rows)} rows"
    prompts = [rows[i]["prompt"] for i in range(0, len(rows), GENERATIONS_PER_PROMPT)]
    assert all(row["prompt"] == prompts[i // GENERATIONS_PER_PROMPT] for i, row in enumerate(rows)), f"step {step}: a group mixes prompts"

    served = [lookup[prompt] for prompt in prompts]
    served_draws = [(CUE_GOLD if cue_letter == gold else CUE_DIST, template_idx) for _, cue_letter, template_idx, gold in served]
    generator = np.random.default_rng(seed + step + CUE_RNG_OFFSET)
    expected_draws = [draw_cue(generator, cue_gold_prob) for _ in prompts]
    draws_matching = sum(a == b for a, b in zip(served_draws, expected_draws))

    gold_letters = [served[i // GENERATIONS_PER_PROMPT][3] for i in range(len(rows))]
    cue_letters = [served[i // GENERATIONS_PER_PROMPT][1] for i in range(len(rows))]
    conditions = [served_draws[i // GENERATIONS_PER_PROMPT][0] for i in range(len(rows))]
    completions = [row["completion"].removesuffix(EOT) for row in rows]
    _, metrics = compute_multiple_choice_rewards(completions, gold_letters, cue_letters, conditions)
    recomputed = {key: int(sum(metrics[key])) for key in COUNT_KEYS}
    recomputed["cue_gold_prompts"] = sum(condition == CUE_GOLD for condition, _ in served_draws)

    logged = {}
    for key, wandb_key in COUNT_KEYS.items():
        count = wandb_row[wandb_key] * len(rows)
        assert abs(count - round(count)) < 1e-6, f"step {step}: W&B {wandb_key} x {len(rows)} = {count} is not a count"
        logged[key] = round(count)
    cue_gold_count = wandb_row["train/cue_gold_fraction"] * PROMPTS_PER_STEP
    assert abs(cue_gold_count - round(cue_gold_count)) < 1e-6
    logged["cue_gold_prompts"] = round(cue_gold_count)

    return {
        "file_step": step,
        "wandb_train_step": step + 1,
        "cue_draws_matching": draws_matching,
        "recomputed_counts": recomputed,
        "wandb_counts": logged,
        "PASS": draws_matching == PROMPTS_PER_STEP and recomputed == logged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None, help="dry run: check only the first N files")
    args = parser.parse_args()

    config, wandb_rows = wandb_training_rows(args.wandb_run_id)
    seed = config["training"]["seed"]
    cue_gold_prob = config["task_args"]["cue_gold_prob"]
    assert config["training"]["batch_size"] * config["training"]["grad_accum_steps"] == PROMPTS_PER_STEP
    print(f"run {args.wandb_run_id}: seed {seed}, cue_gold_prob {cue_gold_prob}, {len(wandb_rows)} W&B training rows")

    response_files = sorted((args.run_dir / "checkpoints/responses").glob("step_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    steps = [int(path.stem.split("_")[1]) for path in response_files]
    assert steps == list(range(len(steps))), f"response files are not step_0..step_{len(steps) - 1}"
    assert len(steps) == len(wandb_rows), f"{len(steps)} response files but {len(wandb_rows)} W&B training rows"
    if args.max_files is not None:
        response_files = response_files[: args.max_files]

    lookup = prompt_lookup()
    per_step = []
    for path in response_files:
        step = int(path.stem.split("_")[1])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        result = check_step(step, rows, lookup, seed, cue_gold_prob, wandb_rows[step + 1])
        per_step.append(result)
        print(
            f"step_{step}: cue draws {result['cue_draws_matching']}/{PROMPTS_PER_STEP}, "
            f"counts {result['recomputed_counts']} {'==' if result['recomputed_counts'] == result['wandb_counts'] else '!='} W&B"
        )

    passed = all(result["PASS"] for result in per_step)
    output = {
        "PASS": passed,
        "wandb_run_id": args.wandb_run_id,
        "run_dir": str(args.run_dir),
        "n_files_checked": len(per_step),
        "n_files": len(steps),
        "seed": seed,
        "cue_gold_prob": cue_gold_prob,
        "per_step": per_step,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: {sum(r['PASS'] for r in per_step)}/{len(per_step)} files; wrote {args.out}")


if __name__ == "__main__":
    main()
