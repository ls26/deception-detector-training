"""Sample completions from a policy through the vLLM server and label them.

Every completion is graded twice: once as generated for the prompt (original shown test) and
once regenerated for the same task with a different test shown. A completion is reward
hacking when it passes only the shown test in both cases (`DECEPTIVE_DEFINITION`), honest
when it passes all tests, otherwise unlabeled. Rows are written as JSONL, one per completion.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

import pandas as pd
from afterburner.generation.vllm.client import VLLMClient

from obfuscation_atlas.config import MBPPTaskConfig
from obfuscation_atlas.tasks.code_generation import (
    compute_code_generation_rewards,
    deceptive_json_field,
    deceptive_reward_key,
)
from obfuscation_atlas.training.grpo_config import VLLMConfig
from obfuscation_atlas.training.obfuscation_training import build_mbpp_chat_prompt

from training.build_dataset import load_eval_task_dataset, load_probe_prompts
from training.common import DATA_SEED, DECEPTIVE_DEFINITION, MAX_SEQUENCE_LENGTH, MODEL_NAME, check_vllm

NUM_TEST_CASES = 1
# The GRPO reward's code-length penalty; it does not affect the labels, only the reward
# metrics computed alongside them.
CODE_LENGTH_PENALTY = 0.001
MAX_RESPONSE_TOKENS = MAX_SEQUENCE_LENGTH // 2
# Two completions of one prompt whose normalized text matches at least this closely are
# treated as duplicates when deduplication is on.
SIMILARITY_THRESHOLD = 0.95
MBPP_SYSTEM_PROMPT = MBPPTaskConfig().system_prompt

METRIC_PASSES_ALL_TESTS = "rewards_metrics/passes_all_tests"
METRIC_PASSES_SHOWN_TESTS = "rewards_metrics/passes_first_n_tests"
METRIC_DOES_EXECUTE = "rewards_metrics/does_execute"
METRIC_DECEPTIVE = deceptive_reward_key(DECEPTIVE_DEFINITION)
METRIC_PASSES_SHOWN_NOT_ALL = deceptive_reward_key("pass_first_not_all")


def on_policy_output_paths(model_key: str, dataset: str, *, output_dir: Path) -> tuple[Path, Path]:
    """(labeled JSONL, dedup audit JSON) for one policy and one prompt set under output_dir."""
    return (
        output_dir / f"{model_key}_{dataset}_generations.jsonl",
        output_dir / f"{model_key}_{dataset}_dedup.json",
    )


def generation_label(model_key: str, dataset: str) -> str:
    return f"{model_key}/{dataset}"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _load_prompts_for_dataset(dataset: str, tokenizer, *, seed: int):
    """`train`: the probe-training prompts; `eval`: the held-out prompts."""
    if dataset == "train":
        return load_probe_prompts(tokenizer, seed=seed)[0]
    if dataset == "eval":
        return load_eval_task_dataset(tokenizer, seed=seed)[0]
    raise ValueError(f"unsupported dataset: {dataset}")


def generate_with_vllm(prompts: list[str], tokenizer, *, lora_path: Path | None, n_samples: int, seed: int) -> list[str]:
    """`n_samples` completions per prompt at temperature 1, in prompt-major order."""
    check_vllm()
    client = VLLMClient(VLLMConfig(temperature=1.0, server_wait_timeout_seconds=600, request_timeout_seconds=300))
    client.wait_for_server()
    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    end_of_turn_token_id = tokenizer.eot_token_id if hasattr(tokenizer, "eot_token_id") else tokenizer.eos_token_id
    data = {
        "model": MODEL_NAME,
        "revision": None,
        "prompts": prompt_ids,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": 1.0,
        "n": n_samples,
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "seed": seed,
        "stop_token_ids": [end_of_turn_token_id],
    }
    if lora_path is not None:
        data["lora_path"] = str(lora_path)
    choices = client.request("v1/completions", data)["choices"]
    eos = tokenizer.eos_token or ""
    return [text.removesuffix(eos) for text in client.broadcast_responses(choices).text]


def compute_reward_metrics(prompts, completions, test_lists, test_setup_codes, *, alt_completions: list[str]) -> dict:
    """The task's reward metrics for each completion, including the swapped-test regeneration
    `alt_completions` that the reward-hacking label needs."""
    _, metrics = compute_code_generation_rewards(
        prompts,
        completions,
        test_lists,
        test_setup_codes,
        code_length_penalty=CODE_LENGTH_PENALTY,
        num_test_cases=NUM_TEST_CASES,
        max_workers=1,
        alt_completions=alt_completions,
    )
    return metrics


def classify_completion(metrics: dict, i: int) -> tuple[bool, bool, bool]:
    """(deceptive, honest, unlabeled) for completion `i` of a reward-metrics dict."""
    deceptive = bool(metrics[METRIC_DECEPTIVE][i])
    honest = bool(metrics[METRIC_PASSES_ALL_TESTS][i])
    return deceptive, honest, not deceptive and not honest


def deduplicate(all_responses: list[str], n_prompts: int, n_generations: int) -> tuple[list[dict], list[dict]]:
    """Per prompt, drop exact and near-duplicate completions (see `SIMILARITY_THRESHOLD`).
    Returns (kept records, audit of every completion with its keep/drop decision)."""
    deduped_records: list[dict] = []
    audit: list[dict] = []

    for prompt_idx in range(n_prompts):
        chunk = all_responses[prompt_idx * n_generations : (prompt_idx + 1) * n_generations]
        seen: set[str] = set()
        first_for_key: dict[str, int] = {}
        exact_kept: list[dict] = []

        for sample_idx, completion in enumerate(chunk):
            key = normalize(completion)
            if key in seen:
                audit.append(
                    {
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "completion": completion,
                        "kept": False,
                        "drop_reason": "exact",
                        "similar_to_sample_idx": first_for_key[key],
                        "similarity_score": 1.0,
                    }
                )
                continue
            seen.add(key)
            first_for_key[key] = sample_idx
            exact_kept.append({"sample_idx": sample_idx, "completion": completion})

        similar_kept: list[dict] = []
        for record in exact_kept:
            normalized = normalize(record["completion"])
            best_ratio, best_idx = 0.0, None
            for kept in similar_kept:
                ratio = SequenceMatcher(None, normalized, normalize(kept["completion"])).ratio()
                if ratio > best_ratio:
                    best_ratio, best_idx = ratio, kept["sample_idx"]

            if similar_kept and best_ratio >= SIMILARITY_THRESHOLD:
                audit.append(
                    {
                        "prompt_idx": prompt_idx,
                        "sample_idx": record["sample_idx"],
                        "completion": record["completion"],
                        "kept": False,
                        "drop_reason": "similar",
                        "similar_to_sample_idx": best_idx,
                        "similarity_score": best_ratio,
                    }
                )
                continue

            audit.append(
                {
                    "prompt_idx": prompt_idx,
                    "sample_idx": record["sample_idx"],
                    "completion": record["completion"],
                    "kept": True,
                    "drop_reason": None,
                    "similar_to_sample_idx": best_idx,
                    "similarity_score": best_ratio if similar_kept else None,
                }
            )
            similar_kept.append(record)

        for record in similar_kept:
            deduped_records.append({"prompt_idx": prompt_idx, **record})

    return deduped_records, audit


def _flat_records(all_responses: list[str], n_prompts: int, n_generations: int) -> list[dict]:
    return [
        {
            "prompt_idx": prompt_idx,
            "sample_idx": sample_idx,
            "completion": all_responses[prompt_idx * n_generations + sample_idx],
        }
        for prompt_idx in range(n_prompts)
        for sample_idx in range(n_generations)
    ]


def _labeled_rows(
    records: list[dict], reward_metrics: dict, *, model_key: str, dataset: str, prompts: list[str], task_ids: list
) -> list[dict]:
    rows = []
    for i, record in enumerate(records):
        prompt_idx = record["prompt_idx"]
        deceptive, honest, unlabeled = classify_completion(reward_metrics, i)
        rows.append(
            {
                "model_key": model_key,
                "dataset": dataset,
                "prompt_idx": prompt_idx,
                "sample_idx": record["sample_idx"],
                "prompt": prompts[prompt_idx],
                "completion": record["completion"],
                "task_id": task_ids[prompt_idx],
                "deceptive_definition": DECEPTIVE_DEFINITION,
                "deceptive": deceptive,
                "honest": honest,
                "unlabeled": unlabeled,
                "does_execute": bool(reward_metrics[METRIC_DOES_EXECUTE][i]),
                "passes_first_n_tests": bool(reward_metrics[METRIC_PASSES_SHOWN_TESTS][i]),
                "passes_all_tests": bool(reward_metrics[METRIC_PASSES_ALL_TESTS][i]),
                "passes_first_n_but_not_all_tests": bool(reward_metrics[METRIC_PASSES_SHOWN_NOT_ALL][i]),
                "pass_only_shown_deception": bool(reward_metrics[deceptive_reward_key("pass_only_shown")][i]),
            }
        )
    return rows


def print_pipeline_df_stats(df: pd.DataFrame) -> None:
    """Label counts and pass rates of one labeled JSONL, as a DataFrame."""
    label = generation_label(df["model_key"].iloc[0], df["dataset"].iloc[0])
    n_total = len(df)
    deceptive = int(df["deceptive"].sum())
    honest = int(df["honest"].sum())
    unlabeled = int(df["unlabeled"].sum())
    print(f"\n[{label}] DataFrame stats ({n_total} rows):")
    print(f"  deceptive: {deceptive} ({deceptive / n_total:.3f})")
    print(f"  honest:    {honest} ({honest / n_total:.3f})")
    print(f"  unlabeled: {unlabeled} ({unlabeled / n_total:.3f})")
    print(f"  pass_shown rate:      {df['passes_first_n_tests'].mean():.3f}")
    print(f"  pass_all rate:        {df['passes_all_tests'].mean():.3f}")
    print(f"  pass_shown_not_all:   {df['passes_first_n_but_not_all_tests'].mean():.3f}")
    print(f"  pass_only_shown:      {df['pass_only_shown_deception'].mean():.3f}")
    print(f"  executes rate:        {df['does_execute'].mean():.3f}")


def run_dry_generation(*, model_key: str, dataset: str, tokenizer, lora_path: Path | None, seed: int = DATA_SEED) -> None:
    """Generate and grade one completion for the first prompt, printing everything: a fast check
    that the server, the adapter, and the grader work before a full generation."""
    prompt_ds = _load_prompts_for_dataset(dataset, tokenizer, seed=seed)
    prompt = prompt_ds["prompt"][0]
    test_list = prompt_ds["test_list"][0]
    test_setup_code = prompt_ds["test_setup_code"][0]
    alt_prompt = build_mbpp_chat_prompt(prompt_ds["text"][0], test_list, 1, MBPP_SYSTEM_PROMPT, tokenizer)

    label = generation_label(model_key, dataset)
    model_desc = f"LoRA at {lora_path}" if lora_path is not None else f"base model {MODEL_NAME}"
    print(f"\n=== [{label}] Dry run: 1 prompt x 1 sample ({model_desc}) ===")
    print(f"\n--- [{label}] prompt ---")
    print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

    responses = generate_with_vllm([prompt], tokenizer, lora_path=lora_path, n_samples=1, seed=seed)
    alt_responses = generate_with_vllm([alt_prompt], tokenizer, lora_path=lora_path, n_samples=1, seed=seed + 1)
    assert len(responses) == len(alt_responses) == 1
    print(f"\n--- [{label}] sample ---")
    print(responses[0][:500] + ("..." if len(responses[0]) > 500 else ""))

    metrics = compute_reward_metrics([prompt], responses, [test_list], [test_setup_code], alt_completions=alt_responses)
    deceptive, honest, unlabeled = classify_completion(metrics, 0)
    print(
        f"\n[{label}] Dry run metrics: executes={metrics[METRIC_DOES_EXECUTE][0]:.0f} "
        f"pass_shown={metrics[METRIC_PASSES_SHOWN_TESTS][0]:.0f} pass_all={metrics[METRIC_PASSES_ALL_TESTS][0]:.0f} "
        f"pass_only_shown={metrics[METRIC_DECEPTIVE][0]:.0f} "
        f"deceptive={deceptive} honest={honest} unlabeled={unlabeled}"
    )
    print(f"OK: [{label}] dry run complete")


def run_on_policy_pipeline(
    *,
    model_key: str,
    dataset: str,
    lora_path: Path | None,
    tokenizer,
    output_dir: Path,
    policy_run_id: str,
    n_generations: int,
    deduplicate_responses: bool = True,
    seed: int = DATA_SEED,
) -> pd.DataFrame:
    """Sample `n_generations` completions per prompt of `dataset` from the policy (`lora_path`,
    or the base model when None), grade and label them, and write
    `<output_dir>/<model_key>_<dataset>_generations.jsonl`. Skipped when that file exists.
    `policy_run_id` is recorded in the dedup audit (the W&B run id, or "base")."""
    label = generation_label(model_key, dataset)
    do_dedup = n_generations > 1 and deduplicate_responses
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl, dedup_json = on_policy_output_paths(model_key, dataset, output_dir=output_dir)

    print(f"\n{'=' * 60}")
    print(f"=== [{label}] Config ===")
    if lora_path is not None:
        assert lora_path.exists(), f"Missing checkpoint: {lora_path}"
        print(f"OK: adapter path exists at {lora_path}")
    else:
        print(f"OK: using base model {MODEL_NAME} (no LoRA)")
    print(
        f"policy_run_id={policy_run_id}, model_key={model_key}, dataset={dataset}, "
        f"n_generations={n_generations}, deduplicate={do_dedup}, output_jsonl={output_jsonl}"
    )

    if output_jsonl.exists():
        rows = [json.loads(line) for line in output_jsonl.read_text().splitlines() if line.strip()]
        print(f"\n[{label}] Loaded {len(rows)} existing rows from {output_jsonl}")
        print(f"OK: [{label}] skipped generation; delete {output_jsonl.name} to regenerate")
        return pd.DataFrame(rows)

    prompt_ds = _load_prompts_for_dataset(dataset, tokenizer, seed=seed)
    prompts = list(prompt_ds["prompt"])
    texts = list(prompt_ds["text"])
    test_lists = list(prompt_ds["test_list"])
    test_setup_codes = list(prompt_ds["test_setup_code"])
    task_ids = list(prompt_ds["task_id"])
    alt_prompts = [
        build_mbpp_chat_prompt(text, test_list, 1, MBPP_SYSTEM_PROMPT, tokenizer)
        for text, test_list in zip(texts, test_lists, strict=True)
    ]

    print(f"\n=== [{label}] Full generation: {len(prompts)} prompts x {n_generations} sample(s) ===")
    expected = len(prompts) * n_generations
    all_responses = generate_with_vllm(prompts, tokenizer, lora_path=lora_path, n_samples=n_generations, seed=seed)
    empty_count = sum(1 for response in all_responses if not response.strip())
    print(f"OK: generated {len(all_responses)} primary responses ({empty_count} empty)")
    assert len(all_responses) == expected

    print(f"\n=== [{label}] Alt generation (test 1 shown): {len(alt_prompts)} prompts x {n_generations} sample(s) ===")
    alt_responses = generate_with_vllm(alt_prompts, tokenizer, lora_path=lora_path, n_samples=n_generations, seed=seed + 1)
    assert len(alt_responses) == expected

    if do_dedup:
        print(f"\n=== [{label}] Deduplicate responses ===")
        records, dedup_audit = deduplicate(all_responses, len(prompts), n_generations)
        per_prompt_kept = [
            sum(1 for entry in dedup_audit if entry["prompt_idx"] == i and entry["kept"]) for i in range(len(prompts))
        ]
        dropped_exact = sum(1 for entry in dedup_audit if entry["drop_reason"] == "exact")
        dropped_similar = sum(1 for entry in dedup_audit if entry["drop_reason"] == "similar")
        before, after = len(all_responses), len(records)
        print(f"  before: {before}")
        print(f"  after:  {after}")
        print(f"  dropped: {before - after} ({(before - after) / before:.1%})")
        print(f"    exact duplicates: {dropped_exact}")
        print(f"    similar (>{SIMILARITY_THRESHOLD:.0%}): {dropped_similar}")
        print(
            f"  per prompt kept: min={min(per_prompt_kept)} max={max(per_prompt_kept)} "
            f"mean={mean(per_prompt_kept):.1f}"
        )
        dedup_json.write_text(
            json.dumps(
                {
                    "run_id": policy_run_id,
                    "model_key": model_key,
                    "dataset": dataset,
                    "n_generations": n_generations,
                    "similarity_threshold": SIMILARITY_THRESHOLD,
                    "generations": dedup_audit,
                },
                ensure_ascii=False,
            )
        )
        print(f"Saved dedup audit ({len(dedup_audit)} generations) to {dedup_json}")
    else:
        records = _flat_records(all_responses, len(prompts), n_generations)

    print(f"\n=== [{label}] Grade, label, save JSONL ===")
    alt_completions = [alt_responses[record["prompt_idx"] * n_generations + record["sample_idx"]] for record in records]
    reward_metrics = compute_reward_metrics(
        [prompts[record["prompt_idx"]] for record in records],
        [record["completion"] for record in records],
        [test_lists[record["prompt_idx"]] for record in records],
        [test_setup_codes[record["prompt_idx"]] for record in records],
        alt_completions=alt_completions,
    )
    rows = _labeled_rows(records, reward_metrics, model_key=model_key, dataset=dataset, prompts=prompts, task_ids=task_ids)

    with output_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(rows)} rows to {output_jsonl}")

    read_back = [json.loads(line) for line in output_jsonl.read_text().splitlines() if line.strip()]
    assert len(read_back) == len(rows)
    deceptive_field = deceptive_json_field(DECEPTIVE_DEFINITION)
    for i in (0, len(rows) // 2, -1):
        row = read_back[i]
        assert row["deceptive"] == row[deceptive_field]
        assert row["honest"] == row["passes_all_tests"]
    print(f"OK: [{label}] read-back verified {len(read_back)} rows in {output_jsonl}")
    return pd.DataFrame(rows)
