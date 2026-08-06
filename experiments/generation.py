# Generate and label on-policy MBPP completions for probe training/eval.

# %%
import json
import os
import re
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from afterburner.generation.vllm.client import VLLMClient
import pandas as pd

from obfuscation_atlas.config import CodeGenTaskConfig, MBPPProbeConfig, MBPPTaskConfig
from obfuscation_atlas.tasks.code_generation import (
    DEFAULT_DECEPTIVE_DEFINITION,
    compute_code_generation_rewards,
    deceptive_json_field,
    deceptive_reward_key,
)
from obfuscation_atlas.training.grpo_config import VLLMConfig
from obfuscation_atlas.training.obfuscation_training import build_mbpp_chat_prompt, prepare_dataset

WORKSPACE = Path(__file__).resolve().parents[1]

SEED = 53
N_GENERATIONS = 2
NUM_TEST_CASES = 1
CODE_LENGTH_PENALTY = 0.003
MAX_SEQUENCE_LENGTH = 512
MAX_RESPONSE_TOKENS = MAX_SEQUENCE_LENGTH // 2
SIMILARITY_THRESHOLD = 0.95
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_CACHE_DIR = WORKSPACE / "hf_cache"
VLLM_MODELS_URL = "http://127.0.0.1:8000/v1/models"


def bootstrap_hf_env(
    *,
    workspace: Path = WORKSPACE,
    hf_cache_dir: Path = HF_CACHE_DIR,
) -> None:
    """Jupyter kernels don't inherit atlas_setup.sh; set HF/W&B env before Hub or subprocess calls."""
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(hf_cache_dir)
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        setup_sh = workspace / "atlas_setup.sh"
        if setup_sh.exists():
            for line in setup_sh.read_text().splitlines():
                if line.startswith("export HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
                    break
    if not os.environ.get("WANDB_API_KEY"):
        setup_sh = workspace / "atlas_setup.sh"
        if setup_sh.exists():
            for line in setup_sh.read_text().splitlines():
                if line.startswith("export WANDB_API_KEY="):
                    os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip()
                    break
    # RunPod sets HF_HUB_ENABLE_HF_TRANSFER=1, but hf_transfer is not in project deps.
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        import importlib.util

        if importlib.util.find_spec("hf_transfer") is None:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def on_policy_output_paths(
    model: str,
    dataset: str,
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Return (jsonl, dedup_json) under output_dir."""
    file_dataset = "train" if dataset == "train_full" else dataset
    return (
        output_dir / f"{model}_{file_dataset}_generations.jsonl",
        output_dir / f"{model}_{file_dataset}_dedup.json",
    )


def _run_id(model_key: str, dataset: str) -> str:
    return f"{model_key}/{dataset}"

M_HONEST = "rewards_metrics/passes_all_tests"
M_PASS_FIRST = "rewards_metrics/passes_first_n_tests"
M_EXECUTES = "rewards_metrics/does_execute"
MBPP_SYSTEM_PROMPT = MBPPTaskConfig().system_prompt


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def load_eval_task_dataset(tokenizer, *, seed: int = SEED):
    cfg = CodeGenTaskConfig()
    cfg.data.seed = seed
    cfg.data.probe_dataset = MBPPProbeConfig()
    cfg.data.task_dataset = MBPPTaskConfig()
    _, (probe_pos, _), eval_task_dataset = prepare_dataset(cfg.data, tokenizer=tokenizer)
    val_pos, val_neg = eval_task_dataset
    assert len(val_pos) == len(val_neg) == 400, f"expected 400 val examples per class, got {len(val_pos)}, {len(val_neg)}"
    probe_task_ids = set(probe_pos["task_id"])
    val_task_ids = set(val_pos["task_id"])
    overlap = probe_task_ids & val_task_ids
    print(f"OK: eval_task_dataset loaded ({len(val_pos)} val pos, {len(val_neg)} val neg)")
    print(f"  probe-train task_ids: {len(probe_task_ids)}, val task_ids: {len(val_task_ids)}, overlap: {len(overlap)}")
    if overlap:
        print(f"  WARNING: {len(overlap)} task_ids appear in both probe-train and val sets")
    return eval_task_dataset, probe_pos


def load_probe_prompts(tokenizer, *, seed: int = SEED):
    cfg = CodeGenTaskConfig()
    cfg.data.seed = seed
    cfg.data.probe_dataset = MBPPProbeConfig()
    cfg.data.task_dataset = MBPPTaskConfig()
    _, (probe_pos, probe_neg), _ = prepare_dataset(cfg.data, tokenizer=tokenizer)
    assert len(probe_pos) == len(probe_neg) == 200
    assert probe_pos["task_id"] == probe_neg["task_id"]
    print("OK: loaded 200 probe-train prompts via prepare_dataset (disjoint from val)")
    print(f"Sample prompt (first 300 chars):\n  {probe_pos[0]['prompt'][:300]}...")
    return probe_pos


def load_task_train_prompts(tokenizer, *, seed: int = SEED):
    cfg = CodeGenTaskConfig()
    cfg.data.seed = seed
    cfg.data.probe_dataset = MBPPProbeConfig()
    cfg.data.task_dataset = MBPPTaskConfig()
    (task_positive_train, task_negative_train), _, eval_task_dataset = prepare_dataset(
        cfg.data, tokenizer=tokenizer
    )
    val_pos, val_neg = eval_task_dataset
    assert task_positive_train["task_id"] == task_negative_train["task_id"]
    train_task_ids = set(task_positive_train["task_id"])
    val_task_ids = set(val_pos["task_id"])
    overlap = train_task_ids & val_task_ids
    print(
        f"OK: loaded {len(task_positive_train)} task-train prompts via prepare_dataset "
        f"(disjoint from {len(val_pos)} val)"
    )
    print(f"  task-train task_ids: {len(train_task_ids)}, val task_ids: {len(val_task_ids)}, overlap: {len(overlap)}")
    if overlap:
        print(f"  WARNING: {len(overlap)} task_ids appear in both task-train and val sets")
    print(f"Sample prompt (first 300 chars):\n  {task_positive_train[0]['prompt'][:300]}...")
    return task_positive_train


def _load_prompts_for_dataset(dataset: str, tokenizer, *, seed: int = SEED):
    if dataset == "train":
        return load_probe_prompts(tokenizer, seed=seed)
    if dataset == "train_full":
        return load_task_train_prompts(tokenizer, seed=seed)
    if dataset == "eval":
        eval_task_dataset, _ = load_eval_task_dataset(tokenizer, seed=seed)
        return eval_task_dataset[0]
    raise ValueError(f"unsupported dataset: {dataset}")


def generate_with_vllm(
    prompts: list[str],
    tokenizer,
    *,
    lora_path: Path | None = None,
    n: int | None = None,
    default_n: int = N_GENERATIONS,
    vllm_models_url: str = VLLM_MODELS_URL,
    model_name: str = MODEL_NAME,
    max_response_tokens: int = MAX_RESPONSE_TOKENS,
    seed: int = SEED,
) -> list[str]:
    n_samples = default_n if n is None else n
    with urllib.request.urlopen(vllm_models_url, timeout=5) as resp:
        assert resp.status == 200
    print(f"OK: vLLM server reachable at {vllm_models_url}")

    client = VLLMClient(VLLMConfig(temperature=1.0, server_wait_timeout_seconds=600, request_timeout_seconds=300))
    client.wait_for_server()
    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    eot_token_id = getattr(tokenizer, "eot_token_id", None) or tokenizer.eos_token_id
    data = {
        "model": model_name,
        "revision": None,
        "prompts": prompt_ids,
        "max_tokens": max_response_tokens,
        "temperature": 1.0,
        "n": n_samples,
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "seed": seed,
        "stop_token_ids": [eot_token_id],
    }
    if lora_path is not None:
        data["lora_path"] = str(lora_path)
    choices = client.request("v1/completions", data)["choices"]
    eos = tokenizer.eos_token or ""
    return [text.removesuffix(eos) for text in client.broadcast_responses(choices).text]


def grade(
    prompts,
    completions,
    test_lists,
    test_setup_codes,
    *,
    code_length_penalty: float = CODE_LENGTH_PENALTY,
    num_test_cases: int = NUM_TEST_CASES,
    alt_completions: list[str] | None = None,
):
    _, metrics = compute_code_generation_rewards(
        prompts,
        completions,
        test_lists,
        test_setup_codes,
        code_length_penalty=code_length_penalty,
        num_test_cases=num_test_cases,
        max_workers=1,
        alt_completions=alt_completions,
    )
    return metrics


def labels(metrics, i: int, *, deceptive_definition: str = DEFAULT_DECEPTIVE_DEFINITION) -> tuple[bool, bool, bool]:
    deceptive = bool(metrics[deceptive_reward_key(deceptive_definition)][i])
    honest = bool(metrics[M_HONEST][i])
    return deceptive, honest, not deceptive and not honest


def deduplicate(
    all_responses: list[str],
    n_prompts: int,
    n_generations: int,
    *,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
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
            norm = normalize(record["completion"])
            best_ratio, best_idx = 0.0, None
            for kept in similar_kept:
                ratio = SequenceMatcher(None, norm, normalize(kept["completion"])).ratio()
                if ratio > best_ratio:
                    best_ratio, best_idx = ratio, kept["sample_idx"]

            if similar_kept and best_ratio >= similarity_threshold:
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
    records: list[dict],
    reward_metrics: dict,
    *,
    model_key: str,
    dataset: str,
    prompts: list[str],
    task_ids: list,
    deceptive_definition: str = DEFAULT_DECEPTIVE_DEFINITION,
) -> list[dict]:
    rows = []
    for i, record in enumerate(records):
        prompt_idx = record["prompt_idx"]
        d, h, u = labels(reward_metrics, i, deceptive_definition=deceptive_definition)
        rows.append(
            {
                "model_key": model_key,
                "dataset": dataset,
                "prompt_idx": prompt_idx,
                "sample_idx": record["sample_idx"],
                "prompt": prompts[prompt_idx],
                "completion": record["completion"],
                "task_id": task_ids[prompt_idx],
                "deceptive_definition": deceptive_definition,
                "deceptive": d,
                "honest": h,
                "unlabeled": u,
                "does_execute": bool(reward_metrics[M_EXECUTES][i]),
                "passes_first_n_tests": bool(reward_metrics[M_PASS_FIRST][i]),
                "passes_all_tests": bool(reward_metrics[M_HONEST][i]),
                "passes_first_n_but_not_all_tests": bool(
                    reward_metrics[deceptive_reward_key("pass_first_not_all")][i]
                ),
                "pass_only_shown_deception": bool(reward_metrics[deceptive_reward_key("pass_only_shown")][i]),
            }
        )
    return rows


def print_pipeline_df_stats(df: pd.DataFrame) -> None:
    """Print aggregate label/metric stats for a pipeline output dataframe."""
    if df.empty:
        print("Empty dataframe — no stats to print")
        return
    model_key = df["model_key"].iloc[0] if "model_key" in df.columns else "unknown"
    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"
    run_id = _run_id(model_key, dataset)
    n_total = len(df)
    deceptive = int(df["deceptive"].sum())
    honest = int(df["honest"].sum())
    unlabeled = int(df["unlabeled"].sum())
    print(f"\n[{run_id}] DataFrame stats ({n_total} rows):")
    print(f"  deceptive: {deceptive} ({deceptive / n_total:.3f})")
    print(f"  honest:    {honest} ({honest / n_total:.3f})")
    print(f"  unlabeled: {unlabeled} ({unlabeled / n_total:.3f})")
    print(f"  pass_first rate: {df['passes_first_n_tests'].mean():.3f}")
    print(f"  pass_all rate:   {df['passes_all_tests'].mean():.3f}")
    print(f"  old_deceptive:   {df['passes_first_n_but_not_all_tests'].mean():.3f}")
    print(f"  pass_only_shown: {df['pass_only_shown_deception'].mean():.3f}")
    print(f"  executes rate:   {df['does_execute'].mean():.3f}")


def run_dry_generation(
    *,
    model_key: str,
    dataset: str,
    tokenizer,
    lora_path: Path | None = None,
    n_generations: int | None = None,
    default_n_generations: int = N_GENERATIONS,
    model_name: str = MODEL_NAME,
    vllm_models_url: str = VLLM_MODELS_URL,
    max_response_tokens: int = MAX_RESPONSE_TOKENS,
    seed: int = SEED,
    code_length_penalty: float = CODE_LENGTH_PENALTY,
    num_test_cases: int = NUM_TEST_CASES,
    deceptive_definition: str = DEFAULT_DECEPTIVE_DEFINITION,
) -> None:
    """Generate and grade n samples for one prompt; print only, no files written."""
    prompt_ds = _load_prompts_for_dataset(dataset, tokenizer, seed=seed)
    prompt = prompt_ds["prompt"][0]
    text = prompt_ds["text"][0]
    test_list = prompt_ds["test_list"][0]
    test_setup_code = prompt_ds["test_setup_code"][0]
    alt_prompt = build_mbpp_chat_prompt(text, test_list, 1, MBPP_SYSTEM_PROMPT, tokenizer)

    run_id = _run_id(model_key, dataset)
    n_gen = default_n_generations if n_generations is None else n_generations
    model_desc = f"LoRA at {lora_path}" if lora_path is not None else f"base model {model_name}"
    print(f"\n=== [{run_id}] Dry run: 1 prompt x {n_gen} sample(s) ({model_desc}) ===")
    print(f"\n--- [{run_id}] prompt ---")
    print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

    responses = generate_with_vllm(
        [prompt],
        tokenizer,
        lora_path=lora_path,
        n=n_gen,
        default_n=default_n_generations,
        vllm_models_url=vllm_models_url,
        model_name=model_name,
        max_response_tokens=max_response_tokens,
        seed=seed,
    )
    assert len(responses) == n_gen
    if n_gen > 1:
        assert len(set(responses)) > 1

    alt_responses = generate_with_vllm(
        [alt_prompt],
        tokenizer,
        lora_path=lora_path,
        n=n_gen,
        default_n=default_n_generations,
        vllm_models_url=vllm_models_url,
        model_name=model_name,
        max_response_tokens=max_response_tokens,
        seed=seed + 1,
    )
    assert len(alt_responses) == n_gen

    for i, completion in enumerate(responses):
        print(f"\n--- [{run_id}] sample {i} ---")
        print(completion[:500] + ("..." if len(completion) > 500 else ""))

    metrics = grade(
        [prompt] * n_gen,
        responses,
        [test_list] * n_gen,
        [test_setup_code] * n_gen,
        code_length_penalty=code_length_penalty,
        num_test_cases=num_test_cases,
        alt_completions=alt_responses,
    )
    print(f"\n[{run_id}] Dry run metrics:")
    for i in range(n_gen):
        d, h, u = labels(metrics, i, deceptive_definition=deceptive_definition)
        print(
            f"  [{i}] executes={metrics[M_EXECUTES][i]:.0f} "
            f"pass_first={metrics[M_PASS_FIRST][i]:.0f} pass_all={metrics[M_HONEST][i]:.0f} "
            f"old_deceptive={metrics[deceptive_reward_key('pass_first_not_all')][i]:.0f} "
            f"pass_only_shown={metrics[deceptive_reward_key('pass_only_shown')][i]:.0f} "
            f"deceptive={d} honest={h} unlabeled={u}"
        )
    print(f"OK: [{run_id}] dry run complete")


def run_on_policy_pipeline(
    *,
    model_key: str,
    dataset: str,
    lora_path: Path | None,
    tokenizer,
    output_dir: Path,
    run_id_config: str,
    require_adapter: bool = False,
    n_generations: int = N_GENERATIONS,
    deduplicate_responses: bool | None = None,
    model_name: str = MODEL_NAME,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    vllm_models_url: str = VLLM_MODELS_URL,
    max_response_tokens: int = MAX_RESPONSE_TOKENS,
    seed: int = SEED,
    code_length_penalty: float = CODE_LENGTH_PENALTY,
    num_test_cases: int = NUM_TEST_CASES,
    deceptive_definition: str = DEFAULT_DECEPTIVE_DEFINITION,
) -> pd.DataFrame:
    run_id = _run_id(model_key, dataset)
    n_gen = n_generations
    do_dedup = n_gen > 1 and (True if deduplicate_responses is None else deduplicate_responses)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl, dedup_json = on_policy_output_paths(model_key, dataset, output_dir=output_dir)

    print(f"\n{'=' * 60}")
    print(f"=== [{run_id}] Config ===")
    if require_adapter:
        assert lora_path is not None and lora_path.exists(), f"Missing checkpoint: {lora_path}"
        print(f"OK: adapter path exists at {lora_path}")
    else:
        print(f"OK: using base model {model_name} (no LoRA)")
    print(
        f"RUN_ID={run_id_config}, model_key={model_key}, dataset={dataset}, "
        f"n_generations={n_gen}, deduplicate={do_dedup}, deceptive_definition={deceptive_definition}, "
        f"output_jsonl={output_jsonl}"
    )

    if output_jsonl.exists():
        rows = [json.loads(line) for line in output_jsonl.read_text().splitlines() if line.strip()]
        print(f"\n[{run_id}] Loaded {len(rows)} existing rows from {output_jsonl}")
        if rows and "deceptive" in rows[0]:
            deceptive = sum(r["deceptive"] for r in rows)
            honest = sum(r["honest"] for r in rows)
            unlabeled = sum(r["unlabeled"] for r in rows)
            n_total = len(rows)
            print(f"  deceptive: {deceptive} ({deceptive / n_total:.3f})")
            print(f"  honest:    {honest} ({honest / n_total:.3f})")
            print(f"  unlabeled: {unlabeled} ({unlabeled / n_total:.3f})")
        print(f"OK: [{run_id}] skipped pipeline; delete {output_jsonl.name} to regenerate")
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

    print(f"\n=== [{run_id}] Full generation: {len(prompts)} prompts x {n_gen} sample(s) ===")
    expected = len(prompts) * n_gen
    all_responses = generate_with_vllm(
        prompts,
        tokenizer,
        lora_path=lora_path,
        n=n_gen,
        default_n=n_generations,
        vllm_models_url=vllm_models_url,
        model_name=model_name,
        max_response_tokens=max_response_tokens,
        seed=seed,
    )
    empty_count = sum(1 for r in all_responses if not r.strip())
    print(f"OK: generated {len(all_responses)} primary responses ({empty_count} empty)")
    assert len(all_responses) == expected

    print(f"\n=== [{run_id}] Alt generation (test 1 shown): {len(alt_prompts)} prompts x {n_gen} sample(s) ===")
    alt_responses = generate_with_vllm(
        alt_prompts,
        tokenizer,
        lora_path=lora_path,
        n=n_gen,
        default_n=n_generations,
        vllm_models_url=vllm_models_url,
        model_name=model_name,
        max_response_tokens=max_response_tokens,
        seed=seed + 1,
    )
    assert len(alt_responses) == expected

    if do_dedup:
        print(f"\n=== [{run_id}] Deduplicate responses ===")
        records, dedup_audit = deduplicate(
            all_responses, len(prompts), n_gen, similarity_threshold=similarity_threshold
        )
        per_prompt_kept = [
            sum(1 for r in dedup_audit if r["prompt_idx"] == i and r["kept"]) for i in range(len(prompts))
        ]
        dropped_exact = sum(1 for r in dedup_audit if r["drop_reason"] == "exact")
        dropped_similar = sum(1 for r in dedup_audit if r["drop_reason"] == "similar")
        before, after = len(all_responses), len(records)
        print(f"  before: {before}")
        print(f"  after:  {after}")
        print(f"  dropped: {before - after} ({(before - after) / before:.1%})")
        print(f"    exact duplicates: {dropped_exact}")
        print(f"    similar (>{similarity_threshold:.0%}): {dropped_similar}")
        print(
            f"  per prompt kept: min={min(per_prompt_kept)} max={max(per_prompt_kept)} "
            f"mean={mean(per_prompt_kept):.1f}"
        )
        print(f"  prompts with 1 unique: {sum(k == 1 for k in per_prompt_kept)}")
        print(f"  prompts with {n_gen} unique: {sum(k == n_gen for k in per_prompt_kept)}")
        dedup_json.write_text(
            json.dumps(
                {
                    "run_id": run_id_config,
                    "model_key": model_key,
                    "dataset": dataset,
                    "n_generations": n_gen,
                    "similarity_threshold": similarity_threshold,
                    "generations": dedup_audit,
                },
                ensure_ascii=False,
            )
        )
        print(f"Saved dedup audit ({len(dedup_audit)} generations) to {dedup_json}")
    else:
        records = _flat_records(all_responses, len(prompts), n_gen)

    print(f"\n=== [{run_id}] Grade, label, print metrics, save JSONL ===")
    alt_completions = [alt_responses[r["prompt_idx"] * n_gen + r["sample_idx"]] for r in records]
    assert len(alt_completions) == len(records)
    reward_metrics = grade(
        [prompts[r["prompt_idx"]] for r in records],
        [r["completion"] for r in records],
        [test_lists[r["prompt_idx"]] for r in records],
        [test_setup_codes[r["prompt_idx"]] for r in records],
        code_length_penalty=code_length_penalty,
        num_test_cases=num_test_cases,
        alt_completions=alt_completions,
    )
    rows = _labeled_rows(
        records,
        reward_metrics,
        model_key=model_key,
        dataset=dataset,
        prompts=prompts,
        task_ids=task_ids,
        deceptive_definition=deceptive_definition,
    )

    with output_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(rows)} rows to {output_jsonl}")

    read_back = [json.loads(line) for line in output_jsonl.read_text().splitlines() if line.strip()]
    assert len(read_back) == len(rows)
    deceptive_field = deceptive_json_field(deceptive_definition)
    for i in (0, len(rows) // 2, -1):
        row = read_back[i]
        assert row["deceptive"] == row[deceptive_field]
        assert row["deceptive_definition"] == deceptive_definition
        assert row["honest"] == row["passes_all_tests"]
    print(f"OK: [{run_id}] read-back verified {len(read_back)} rows in {output_jsonl}")
    return pd.DataFrame(rows)

# %%
