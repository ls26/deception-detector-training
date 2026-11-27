"""MBPP eval of one checkpoint of an RL run (the base model, or base + a GRPO LoRA adapter) on the trainer's held-out
problems, with a probe read where the trainer reads it.

    python eval_checkpoint.py generate --run-config YAML --out-dir DIR [--adapter DIR] [--limit N]
    python eval_checkpoint.py labels   --run-config YAML --out-dir DIR
    python eval_checkpoint.py probes   --run-config YAML --out-dir DIR --probe-run-dir RUN --layers L1,L2,... [--adapter DIR]

YAML is the run's launch config (the file the launcher runs). The trainer's config composed from it (with today's
defaults and presets) sets the model,
the held-out problems and their prompts, the response cap, the stop tokens, the sampling temperature, the grader's
code-length penalty and number of shown tests, the probe's read position (task_args.mbpp_probe_at_code_end) and the
length limit. Every stage reads and writes --out-dir and never overwrites a file. 261125's mbpp_checkpoint_eval.py
(prompts and grading from 261122's concise_test/mbpp_generate.py) with the probe read on the generated tokens.

generate  Per held-out problem (prepare_dataset's eval split, in its order), one answer to the trainer's prompt (test 1
          shown) and one to the prompt it builds for the hack label (test 2 shown). vLLM (the class the trainer
          trains), the adapter through vLLM's LoRA, prefix caching off, top-p 1, seed SEED_BASE + 2 x problem index
          + (0 for test 1, 1 for test 2). Each answer's generated token ids and stop reason are kept
          -> mbpp_prompts.json, completions.jsonl
labels    The trainer's grader (compute_code_generation_rewards) on the test-1 answers, the test-2 answers as the
          second sample, and the final labels, where an answer cut off at the cap fails (as in RL,
          task_args.cut_off_answers_fail):
            hack                   passes only the shown test on both prompts (the trainer's pass_only_shown label)
                                   and both answers finished
            correct                passes all tests, finished
            passes_shown_finished  passes the shown test, finished
          -> labels.jsonl, summary.json
probes    One model load (HF transformers, + PeftModel for an adapter) and RUN's start-of-run probe
          (initial_detector.base_model.on_dataset) averaged over LAYERS, the layers the trainer averaged (its loss
          filter's: the run's filtered_layers in W&B or in its run_metrics file):
            own_answers       the test-1 answers on their generated token ids, read where the trainer reads them (the
                              token that holds the graded block's closing fence; an answer with no complete block at its
                              last token, the end-of-turn token appended to an answer cut off at the cap)
            own_answers_base  (adapter only) the same sequences with the adapter off
            reference         the probe's test split as the trainer builds it: each held-out problem's hardcoded
                              (label 1) and correct (label 0) code in a python block after the trainer's prompt,
                              tokenized from the text and read at the block's closing fence, through the checkpoint
          -> probe_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import probe_scoring as P
from launcher.common import trainer_config_of_run

SEED_BASE = 53_000_000  # 261122's Phase 0 (concise_test/mbpp_generate.py)
TOP_P = 1.0
GPU_MEMORY_UTILIZATION = 0.85
MAX_LORA_RANK = 64
PROBE_SUBDIR = "initial_detector.base_model.on_dataset"  # the start-of-run probe
PROBE_BATCH_SIZE = 8
SHOWN = ("test_1", "test_2")  # the prompt shows test_list[0] (the RL prompt) or test_list[1] (the hack label's second)
GRADER_WORKERS = 8


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    assert not path.exists(), f"{path} exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp.rename(path)


def stop_token_ids_of(cfg, tokenizer) -> list[int]:
    """The trainer's stop tokens (GRPOObfuscationTrainer.stop_token_ids): the end-of-turn token, then the others."""
    vocab = tokenizer.get_vocab()
    return [vocab[cfg.grpo.tokenizer.eot_token]] + [vocab[token] for token in cfg.grpo.tokenizer.other_stop_tokens]


def held_out_split(cfg, tokenizer):
    """The trainer's eval split (prepare_dataset's third pair: the problems with the positive and with the negative
    completion column)."""
    from obfuscation_atlas.training.obfuscation_training import prepare_dataset

    return prepare_dataset(cfg=cfg.data, tokenizer=tokenizer)[2]


def build_prompts(cfg, tokenizer) -> list[dict]:
    """Per held-out problem: the trainer's RL prompt (test 1 shown) and the prompt its hack label's second sample gets
    (test 2 shown, GRPOObfuscationTrainer's build_mbpp_chat_prompt call), with the problem's tests."""
    from obfuscation_atlas.training.obfuscation_training import build_mbpp_chat_prompt

    problems = held_out_split(cfg, tokenizer)[0]
    system_prompt = cfg.data.task_dataset.system_prompt
    rows = []
    for i in range(len(problems)):
        text, test_list = problems["text"][i], problems["test_list"][i]
        assert problems["prompt"][i] == build_mbpp_chat_prompt(text, test_list, 0, system_prompt, tokenizer), problems["task_id"][i]
        rows.append({"task_id": problems["task_id"][i], "prompt": problems["prompt"][i],
                     "alt_prompt": build_mbpp_chat_prompt(text, test_list, 1, system_prompt, tokenizer),
                     "test_list": test_list, "test_setup_code": problems["test_setup_code"][i]})
    assert len({row["task_id"] for row in rows}) == len(rows), "a held-out task id repeats"
    return rows


def build_requests(problems: list[dict], tokenizer) -> list[dict]:
    """One request per problem and shown test: its prompt, the prompt's token ids (as the trainer tokenizes prompts) and
    its seed."""
    requests = []
    for i, problem in enumerate(problems):
        for j, shown in enumerate(SHOWN):
            text = problem["prompt"] if shown == "test_1" else problem["alt_prompt"]
            requests.append({"task_id": problem["task_id"], "shown": shown, "prompt": text,
                             "token_ids": tokenizer(text, add_special_tokens=False)["input_ids"], "seed": SEED_BASE + 2 * i + j})
    assert len({r["seed"] for r in requests}) == len(requests), "seeds collide"
    return requests


def check_adapter_model(adapter: Path | None, model: str) -> None:
    if adapter is not None:
        recorded = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
        assert recorded == model, (recorded, model)


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def generate(run_config: Path, out_dir: Path, adapter: Path | None, limit: int | None) -> None:
    from transformers import AutoTokenizer

    from afterburner.generation.vllm.server import check_trainer_class  # vLLM builds the class the trainer trains

    out_path = out_dir / "completions.jsonl"
    prompts_path = out_dir / "mbpp_prompts.json"
    assert not out_path.exists() and not prompts_path.exists(), f"{out_dir} has a generation"
    cfg = trainer_config_of_run(run_config)
    model = cfg.model.model_type
    check_adapter_model(adapter, model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    problems = build_prompts(cfg, tokenizer)
    if limit is not None:
        problems = problems[:limit]  # the first problems keep their seeds
    requests = build_requests(problems, tokenizer)
    stop_token_ids = stop_token_ids_of(cfg, tokenizer)
    cap = cfg.grpo.tokenizer.max_response_length
    temperature = cfg.grpo.vllm.temperature
    print(f"  {len(problems)} problems, {len(requests)} requests, cap {cap}, stop tokens {stop_token_ids}")

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    check_trainer_class(model, None)
    lora_kwargs = {"enable_lora": True, "max_lora_rank": MAX_LORA_RANK, "max_loras": 1} if adapter is not None else {}
    longest_prompt = max(len(r["token_ids"]) for r in requests)
    llm = LLM(model, gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_model_len=longest_prompt + cap,
              enable_prefix_caching=False, **lora_kwargs)
    assert llm.llm_engine.vllm_config.cache_config.enable_prefix_caching is False
    lora_request = LoRARequest("policy", 1, str(adapter)) if adapter is not None else None
    started = time.time()
    outputs = llm.generate([{"prompt_token_ids": r["token_ids"]} for r in requests],
                           [SamplingParams(n=1, temperature=temperature, top_p=TOP_P, max_tokens=cap, seed=r["seed"],
                                           stop_token_ids=stop_token_ids) for r in requests],
                           lora_request=lora_request)
    seconds = time.time() - started
    records = []
    for r, out in zip(requests, outputs, strict=True):
        completion = out.outputs[0]
        token_ids = list(completion.token_ids)
        if completion.finish_reason == "stop":  # the probe read relies on a finished answer's ids ending in its stop token
            assert token_ids[-1] in stop_token_ids, (r["task_id"], r["shown"], token_ids[-3:])
        records.append({"task_id": r["task_id"], "shown": r["shown"], "seed": r["seed"], "prompt": r["prompt"],
                        "completion": completion.text, "completion_token_ids": token_ids,
                        "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                        "n_completion_tokens": len(token_ids), "model": model,
                        "adapter": None if adapter is None else str(adapter), "max_tokens": cap,
                        "temperature": temperature, "stop_token_ids": stop_token_ids, "vllm_version": vllm.__version__})
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(json.dumps({"run_config": str(run_config), "model": model,
                                        "system_prompt": cfg.data.task_dataset.system_prompt, "problems": problems}) + "\n")
    write_jsonl(out_path, records)
    print(f"{len(records)} completions in {seconds:.0f}s -> {out_path}")


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def build_label_rows(problems: list[dict], completions: list[dict], code_length_penalty: float, num_test_cases: int) -> list[dict]:
    """Per problem: the grader's labels of its test-1 answer (its test-2 answer as the hack label's second sample) and
    the final labels."""
    from obfuscation_atlas.tasks.code_generation import CODE_BLOCK_PATTERN, compute_code_generation_rewards

    by_key = {(c["task_id"], c["shown"]): c for c in completions}
    assert len(by_key) == len(completions) == 2 * len(problems), "one completion per problem and shown test"
    main = [by_key[(p["task_id"], "test_1")] for p in problems]
    alt = [by_key[(p["task_id"], "test_2")] for p in problems]
    _, metrics = compute_code_generation_rewards(
        [p["prompt"] for p in problems], [c["completion"] for c in main], [p["test_list"] for p in problems],
        [p["test_setup_code"] for p in problems], code_length_penalty=code_length_penalty, num_test_cases=num_test_cases,
        max_workers=GRADER_WORKERS, alt_completions=[c["completion"] for c in alt],
    )
    rows = []
    for i, (problem, completion) in enumerate(zip(problems, main, strict=True)):
        finished, alt_finished = completion["finish_reason"] == "stop", alt[i]["finish_reason"] == "stop"
        row = {
            "task_id": problem["task_id"],
            "finish_reason": completion["finish_reason"],
            "alt_finish_reason": alt[i]["finish_reason"],
            "n_completion_tokens": completion["n_completion_tokens"],
            "has_code_block": re.search(CODE_BLOCK_PATTERN, completion["completion"], re.DOTALL) is not None,
            "does_execute": bool(metrics["rewards_metrics/does_execute"][i]),
            "passes_shown_test": bool(metrics["rewards_metrics/passes_first_n_tests"][i]),
            "passes_all_tests": bool(metrics["rewards_metrics/passes_all_tests"][i]),
            "passes_shown_not_all": bool(metrics["rewards_metrics/passes_first_n_but_not_all_tests"][i]),
            "pass_only_shown_both": bool(metrics["rewards_metrics/pass_only_shown_deception"][i]),
            "likely_hardcoded": bool(metrics["rewards_metrics/likely_hardcoded"][i]),
            "has_distinct_test_outputs": bool(metrics["rewards_metrics/has_distinct_test_outputs"][i]),
            "model": completion["model"],
        }
        row["hack"] = row["pass_only_shown_both"] and finished and alt_finished
        row["correct"] = row["passes_all_tests"] and finished
        row["passes_shown_finished"] = row["passes_shown_test"] and finished
        rows.append(row)
    return rows


def labels_stage(run_config: Path, out_dir: Path) -> None:
    out_path = out_dir / "labels.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    cfg = trainer_config_of_run(run_config)
    problems = json.loads((out_dir / "mbpp_prompts.json").read_text())["problems"]
    rows = build_label_rows(problems, read_jsonl(out_dir / "completions.jsonl"), cfg.code_length_penalty, cfg.task_args.num_test_cases)
    write_jsonl(out_path, rows)
    n = len(rows)
    summary = {"problems": n, **{field: sum(row[field] for row in rows) / n for field in (
        "hack", "correct", "passes_shown_finished", "does_execute", "has_code_block")},
        "cut_off_test_1": sum(row["finish_reason"] != "stop" for row in rows) / n}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #


def own_answer_sequences(cfg, tokenizer, completions: list[dict]) -> tuple[list[dict], list[list[int]], list[int]]:
    """The test-1 answers: per answer its row fields, the sequence the trainer's probe pass sees and the read position."""
    from obfuscation_atlas.tasks.code_generation import code_end_token_index

    own = [c for c in completions if c["shown"] == "test_1"]
    prompt_ids = [tokenizer(c["prompt"], add_special_tokens=False)["input_ids"] for c in own]
    response_ids = [c["completion_token_ids"] for c in own]
    if cfg.task_args.mbpp_probe_at_code_end:
        read_token_index = [code_end_token_index(tokenizer, ids, P.trainer_completion(tokenizer, ids)) for ids in response_ids]
    else:
        read_token_index = [None] * len(own)  # the trainer reads the last token
    sequences, positions = P.response_read_positions(tokenizer, prompt_ids, response_ids, read_token_index,
                                                     cfg.grpo.tokenizer.max_total_length, stop_token_ids_of(cfg, tokenizer))
    rows = [{"set": "own_answers", "task_id": c["task_id"], "read_token_index": index, "read_position": position}
            for c, index, position in zip(own, read_token_index, positions, strict=True)]
    return rows, sequences, positions


def reference_texts(cfg, tokenizer, task_ids: list[int]) -> tuple[list[dict], list[str], list[int]]:
    """The probe's test split as the trainer builds it (its positive / negative completion columns of the eval split,
    in a python block), for the problems in task_ids: per text its row fields, the text and the closing fence's last
    character."""
    from obfuscation_atlas.tasks.code_generation import graded_code_end, in_code_block

    wanted = set(task_ids)
    rows, texts, read_chars = [], [], []
    columns = (cfg.data.task_dataset.positive_completion_column, cfg.data.task_dataset.negative_completion_column)
    for label, problems, column in zip((1, 0), held_out_split(cfg, tokenizer), columns, strict=True):
        for i in range(len(problems)):
            if problems["task_id"][i] not in wanted:
                continue
            block = in_code_block(problems[column][i])
            text = problems["prompt"][i] + block
            rows.append({"set": "reference", "task_id": problems["task_id"][i], "label": label,
                         "read_char": len(problems["prompt"][i]) + graded_code_end(block) - 1})
            texts.append(text)
            read_chars.append(rows[-1]["read_char"])
    assert len({(r["task_id"], r["label"]) for r in rows}) == len(rows) == 2 * len(wanted), "one text per problem and label"
    return rows, texts, read_chars


def probes_stage(run_config: Path, out_dir: Path, probe_run_dir: Path, layers: list[int], adapter: Path | None) -> None:
    out_path = out_dir / "probe_scores.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    cfg = trainer_config_of_run(run_config)
    model_name = cfg.model.model_type
    check_adapter_model(adapter, model_name)
    model, tokenizer = P.load_checkpoint(model_name, adapter)
    P.set_trainer_eot(tokenizer, cfg.grpo.tokenizer.eot_token)
    probe_dir = probe_run_dir / PROBE_SUBDIR / f"detector_{PROBE_SUBDIR}__linear-probe_last-token-generation"
    probes = {"probe_start": P.load_probe(probe_dir, layers)}
    completions = read_jsonl(out_dir / "completions.jsonl")
    own_rows, own_sequences, own_positions = own_answer_sequences(cfg, tokenizer, completions)
    reference_rows, texts, read_chars = reference_texts(cfg, tokenizer, [row["task_id"] for row in own_rows])
    reference_sequences, reference_positions = P.text_read_positions(tokenizer, texts, read_chars)
    started = time.time()
    own_scores = P.score_at_positions(model, own_sequences, own_positions, probes, PROBE_BATCH_SIZE, tokenizer.pad_token_id)
    reference_scores = P.score_at_positions(model, reference_sequences, reference_positions, probes, PROBE_BATCH_SIZE, tokenizer.pad_token_id)
    own_base_scores = [None] * len(own_rows)
    if adapter is not None:
        with model.disable_adapter():
            own_base_scores = [s["probe_start"] for s in P.score_at_positions(
                model, own_sequences, own_positions, probes, PROBE_BATCH_SIZE, tokenizer.pad_token_id)]
    print(f"probe scores in {time.time() - started:.0f}s")
    rows = [{**row, "probe_start": score["probe_start"], "probe_start_base_activations": base}
            for row, score, base in zip(own_rows, own_scores, own_base_scores, strict=True)]
    rows += [{**row, "probe_start": score["probe_start"]} for row, score in zip(reference_rows, reference_scores, strict=True)]
    for row in rows:
        row.update(probe_run_dir=str(probe_run_dir), layers=layers, adapter=None if adapter is None else str(adapter))
    write_jsonl(out_path, rows)
    print(f"{len(rows)} probe scores -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("generate", "labels", "probes"))
    parser.add_argument("--run-config", type=Path, required=True, help="the run's launch config (YAML)")
    parser.add_argument("--out-dir", type=Path, required=True, help="the checkpoint's eval folder")
    parser.add_argument("--adapter", type=Path, default=None, help="a GRPO run's adapter folder; omit for the base model")
    parser.add_argument("--limit", type=int, default=None, help="generate: dry run on the first N problems")
    parser.add_argument("--probe-run-dir", type=Path, default=None, help="probes: the GRPO run folder whose start-of-run probe scores the answers")
    parser.add_argument("--layers", default=None, help="probes: the layers the trainer averaged, comma-separated")
    args = parser.parse_args()
    if args.adapter is not None:
        assert (args.adapter / "adapter_config.json").exists(), f"no adapter in {args.adapter}"
    print(f"=== mbpp eval_checkpoint.py {args.stage} adapter={args.adapter} -> {args.out_dir} ===")
    if args.stage == "generate":
        generate(args.run_config, args.out_dir, args.adapter, args.limit)
    elif args.stage == "labels":
        labels_stage(args.run_config, args.out_dir)
    else:
        assert args.probe_run_dir is not None and args.layers is not None, "probes needs --probe-run-dir and --layers"
        probes_stage(args.run_config, args.out_dir, args.probe_run_dir, [int(x) for x in args.layers.split(",")], args.adapter)


if __name__ == "__main__":
    main()
