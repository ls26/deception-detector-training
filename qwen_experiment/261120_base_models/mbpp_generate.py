"""MBPP base-model measurement on the trainer's 400 held-out problems, in three stages.

    python mbpp_generate.py prompts  --model MODEL --out-dir DIR [--thinking]   -> DIR/mbpp_prompts.json
    python mbpp_generate.py generate --model MODEL --out-dir DIR [--thinking]   -> DIR/completions.jsonl
    python mbpp_generate.py score    --out-dir DIR                              -> DIR/labels.jsonl

prompts   The trainer's eval split (sycophancy-rl/training/build_dataset.py load_eval_task_dataset, data seed 53),
          built with the model's tokenizer, so its 512-token prompt filter uses the model's tokens. Each problem
          gets its prompt with test 1 shown and, for the trainer's two-sample hardcoding label, a second prompt with
          test 2 shown (training/generation.py's alt prompt). The chat template gets enable_thinking (Llama's
          template ignores it); the trainer's prompt builders do not pass it, so it is bound onto the tokenizer here.
generate  vLLM only (no trainer imports in this process, 261010): one sample per prompt, temperature 1, top-p 1,
          a 1024-token cap (8192 with --thinking; --max-tokens 256, the RL's cap, only for the Llama check against
          the release's 256-token artifact), a seed per request, the model's generation_config.json
          ignored and its end-of-sequence ids passed explicitly (generate.py's stop_token_ids_of).
score     The trainer's grader (tasks/code_generation.py compute_code_generation_rewards, code-length penalty 0.001
          as in training/generation.py), on the text after "</think>" with --thinking.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import time
from pathlib import Path

DATA_SEED = 53
SEED_BASE = 53_000_000
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 1024  # the user's choice (2026-11-20): at the RL's 256 (training/common.py), 90% of Qwen 4B's answers were cut
MAX_TOKENS_THINKING = 8192
GPU_MEMORY_UTILIZATION = 0.90
CODE_LENGTH_PENALTY = 0.001
THINK_END = "</think>"
CODE_BLOCK = re.compile(r"```(?:[Pp]ython)?\s*\n(.*?)\n```", re.DOTALL)  # tasks/code_generation.py extract_code's pattern
SHOWN = ("test_1", "test_2")  # which test the prompt shows: index 0 (the RL prompt) or index 1 (the second sample)


def build_prompts(model: str, out_dir: Path, thinking: bool) -> None:
    from transformers import AutoTokenizer

    from obfuscation_atlas.config import MBPPTaskConfig
    from obfuscation_atlas.training.obfuscation_training import build_mbpp_chat_prompt
    from training.build_dataset import load_eval_task_dataset

    out_path = out_dir / "mbpp_prompts.json"
    assert not out_path.exists(), f"{out_path} exists"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, enable_thinking=thinking)
    problems = load_eval_task_dataset(tokenizer, seed=DATA_SEED)[0]
    system_prompt = MBPPTaskConfig().system_prompt
    rows = []
    for i in range(len(problems)):
        alt_prompt = build_mbpp_chat_prompt(problems["text"][i], problems["test_list"][i], 1, system_prompt, tokenizer)
        rows.append({"task_id": problems["task_id"][i], "prompt": problems["prompt"][i], "alt_prompt": alt_prompt,
                     "test_list": problems["test_list"][i], "test_setup_code": problems["test_setup_code"][i]})
    if tokenizer.bos_token_id is None:  # Qwen3.5: the prompts end with the thinking switch
        ending = "<|im_start|>assistant\n<think>\n" if thinking else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        assert all(row["prompt"].endswith(ending) and row["alt_prompt"].endswith(ending) for row in rows), "thinking switch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "thinking": thinking, "problems": rows}) + "\n")
    print(f"{model}: {len(rows)} problems -> {out_path}")


def generate(model: str, out_dir: Path, thinking: bool, max_tokens: int | None) -> None:
    from transformers import AutoTokenizer

    from generate import MAX_NUM_SEQS, stop_token_ids_of

    out_path = out_dir / "completions.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    saved = json.loads((out_dir / "mbpp_prompts.json").read_text())
    assert saved["model"] == model and saved["thinking"] == thinking, (saved["model"], saved["thinking"])
    tokenizer = AutoTokenizer.from_pretrained(model)
    if max_tokens is None:
        max_tokens = MAX_TOKENS_THINKING if thinking else MAX_TOKENS
    requests = []
    for i, problem in enumerate(saved["problems"]):
        for j, shown in enumerate(SHOWN):
            text = problem["prompt"] if shown == "test_1" else problem["alt_prompt"]
            requests.append({"task_id": problem["task_id"], "shown": shown, "prompt": text,
                             "token_ids": tokenizer(text, add_special_tokens=False)["input_ids"], "seed": SEED_BASE + 2 * i + j})

    import vllm
    from vllm import LLM, SamplingParams

    stop_token_ids = stop_token_ids_of(model, tokenizer)
    engine_kwargs = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "Qwen3.5" in model else {}
    if model in MAX_NUM_SEQS:
        engine_kwargs["max_num_seqs"] = MAX_NUM_SEQS[model]
    llm = LLM(model, gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_model_len=max_tokens + 1024,
              enable_prefix_caching=False, generation_config="vllm", **engine_kwargs)
    started = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": r["token_ids"]} for r in requests],
        [SamplingParams(n=1, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=max_tokens, seed=r["seed"],
                        stop_token_ids=stop_token_ids) for r in requests],
    )
    seconds = time.time() - started
    assert len(outputs) == len(requests)
    records = []
    for r, out in zip(requests, outputs):
        completion = out.outputs[0]
        records.append({"task_id": r["task_id"], "shown": r["shown"], "seed": r["seed"], "prompt": r["prompt"],
                        "completion": completion.text, "finish_reason": completion.finish_reason,
                        "n_completion_tokens": len(completion.token_ids), "model": model, "thinking": thinking,
                        "max_tokens": max_tokens, "stop_token_ids": stop_token_ids, "vllm_version": vllm.__version__})
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(rec) + "\n" for rec in records))
    tmp.rename(out_path)
    print(f"{model}: {len(records)} completions in {seconds:.0f}s -> {out_path}")


def answer_text_of(completion: str, thinking: bool) -> str:
    """The text the grader reads: all of it with thinking off; with it on, the part after </think> (empty if the
    answer was cut off before </think>)."""
    if not thinking:
        return completion
    return completion.split(THINK_END, 1)[1] if THINK_END in completion else ""


def score(out_dir: Path) -> None:
    from obfuscation_atlas.tasks.code_generation import compute_code_generation_rewards

    out_path = out_dir / "labels.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    problems = json.loads((out_dir / "mbpp_prompts.json").read_text())["problems"]
    completions = [json.loads(line) for line in (out_dir / "completions.jsonl").read_text().splitlines()]
    by_key = {(c["task_id"], c["shown"]): c for c in completions}
    assert len(by_key) == len(completions) == 2 * len(problems), "one completion per problem and shown test"
    thinking = completions[0]["thinking"]
    main = [by_key[(p["task_id"], "test_1")] for p in problems]
    alt = [by_key[(p["task_id"], "test_2")] for p in problems]
    responses = [answer_text_of(c["completion"], thinking) for c in main]
    _, metrics = compute_code_generation_rewards(
        [p["prompt"] for p in problems], responses, [p["test_list"] for p in problems],
        [p["test_setup_code"] for p in problems], code_length_penalty=CODE_LENGTH_PENALTY, num_test_cases=1, max_workers=8,
        alt_completions=[answer_text_of(c["completion"], thinking) for c in alt],
    )
    rows = []
    for i, (problem, completion) in enumerate(zip(problems, main)):
        rows.append({
            "task_id": problem["task_id"],
            "finish_reason": completion["finish_reason"],
            "alt_finish_reason": alt[i]["finish_reason"],
            "n_completion_tokens": completion["n_completion_tokens"],
            "has_think_end": THINK_END in completion["completion"],
            "has_code_block": CODE_BLOCK.search(responses[i]) is not None,
            "does_execute": bool(metrics["rewards_metrics/does_execute"][i]),
            "passes_shown_test": bool(metrics["rewards_metrics/passes_first_n_tests"][i]),
            "passes_all_tests": bool(metrics["rewards_metrics/passes_all_tests"][i]),
            "passes_shown_not_all": bool(metrics["rewards_metrics/passes_first_n_but_not_all_tests"][i]),
            "pass_only_shown_both": bool(metrics["rewards_metrics/pass_only_shown_deception"][i]),
            "likely_hardcoded": bool(metrics["rewards_metrics/likely_hardcoded"][i]),
            "has_distinct_test_outputs": bool(metrics["rewards_metrics/has_distinct_test_outputs"][i]),
            "model": completion["model"],
            "thinking": thinking,
        })
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp.rename(out_path)
    n = len(rows)
    print(f"{n} problems: shown {sum(r['passes_shown_test'] for r in rows) / n:.3f}, all {sum(r['passes_all_tests'] for r in rows) / n:.3f}, "
          f"executes {sum(r['does_execute'] for r in rows) / n:.3f} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prompts", "generate", "score"))
    parser.add_argument("--model")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None, help="generate: override the cap (the Llama 256-token check)")
    args = parser.parse_args()
    if args.stage == "prompts":
        build_prompts(args.model, args.out_dir, args.thinking)
    elif args.stage == "generate":
        generate(args.model, args.out_dir, args.thinking, args.max_tokens)
    else:
        score(args.out_dir)


if __name__ == "__main__":
    main()
