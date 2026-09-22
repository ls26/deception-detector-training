#!/usr/bin/env python3
"""Fetch missing on-policy eval generations from W&B stage-F on_policy_responses tables."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/workspace")))

import wandb

from obfuscation_atlas.tasks.code_generation import deceptive_reward_key
from obfuscation_atlas.training.obfuscation_training import HONEST_REWARD_KEY
from pipeline.generation import M_EXECUTES, M_HONEST, M_PASS_FIRST

from experiments.supplemented_probe_eval.common import (
    BASE_EVAL_JSONL,
    DECEPTIVE_DEFINITION,
    MISSING_WANDB_RUNS,
    WANDB_PROJECT,
    buffer_dir,
    load_jsonl,
    write_jsonl,
)


def bootstrap_env() -> None:
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    secrets_env = Path("/workspace/.secrets.env")
    if secrets_env.exists():
        for line in secrets_env.read_text().splitlines():
            if line.startswith("export ") and "=" in line:
                key, _, val = line.removeprefix("export ").partition("=")
                if key in ("WANDB_API_KEY", "HF_TOKEN", "HF_HOME") and key not in os.environ:
                    os.environ[key] = val.strip()


def reference_prompt_index() -> tuple[list[str], dict[str, dict]]:
    ref_rows = load_jsonl(BASE_EVAL_JSONL)
    prompts = [r["prompt"] for r in ref_rows]
    by_prompt = {r["prompt"]: r for r in ref_rows}
    return prompts, by_prompt


def wandb_table_to_rows(df, *, run_id: str, ref_by_prompt: dict[str, dict]) -> list[dict]:
    deceptive_key = deceptive_reward_key(DECEPTIVE_DEFINITION)
    required = {
        "prompt",
        "completion",
        deceptive_key,
        HONEST_REWARD_KEY,
        M_EXECUTES,
        M_PASS_FIRST,
        "rewards_metrics/passes_first_n_but_not_all_tests",
        "rewards_metrics/pass_only_shown_deception",
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Run {run_id}: W&B table missing columns {sorted(missing_cols)}")

    rows: list[dict] = []
    for i, record in df.iterrows():
        prompt = record["prompt"]
        if prompt not in ref_by_prompt:
            raise ValueError(f"Run {run_id}: row {i} prompt not in reference eval set")
        ref = ref_by_prompt[prompt]
        deceptive = bool(record[deceptive_key])
        honest = bool(record[HONEST_REWARD_KEY])
        rows.append(
            {
                "model_key": f"wandb_{run_id}",
                "dataset": "eval",
                "prompt_idx": ref["prompt_idx"],
                "sample_idx": 0,
                "prompt": prompt,
                "completion": record["completion"],
                "task_id": ref["task_id"],
                "deceptive_definition": DECEPTIVE_DEFINITION,
                "deceptive": deceptive,
                "honest": honest,
                "unlabeled": not deceptive and not honest,
                "does_execute": bool(record[M_EXECUTES]),
                "passes_first_n_tests": bool(record[M_PASS_FIRST]),
                "passes_all_tests": bool(record[HONEST_REWARD_KEY]),
                "passes_first_n_but_not_all_tests": bool(
                    record["rewards_metrics/passes_first_n_but_not_all_tests"]
                ),
                "pass_only_shown_deception": bool(record["rewards_metrics/pass_only_shown_deception"]),
                "source": f"wandb:{run_id}",
            }
        )
    return rows


def fetch_one(trial: int, variant: str, round_num: int, run_id: str, *, dry_run: bool = False) -> Path:
    out_path = (
        buffer_dir(trial, variant)
        / f"trial{trial}_round{round_num}_eval_generations_wandb.jsonl"
    )
    if out_path.exists() and not dry_run:
        print(f"SKIP existing {out_path}")
        return out_path

    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    df = None
    for artifact in run.logged_artifacts():
        if "on_policy_responses" in artifact.name:
            table = artifact.get("on_policy_responses")
            df = table.get_dataframe()
            print(f"Fetched {artifact.name} for {run_id} ({len(df)} rows)")
            break
    if df is None:
        raise RuntimeError(f"No on_policy_responses artifact for run {run_id}")

    ref_prompts, ref_by_prompt = reference_prompt_index()
    if len(df) != 400:
        raise ValueError(f"Run {run_id}: expected 400 rows, got {len(df)}")
    if list(df["prompt"]) != ref_prompts:
        # Order may differ — validate set equality instead.
        if set(df["prompt"]) != set(ref_prompts):
            raise ValueError(f"Run {run_id}: prompt set mismatch vs reference eval set")
        df = df.set_index("prompt").loc[ref_prompts].reset_index()

    rows = wandb_table_to_rows(df, run_id=run_id, ref_by_prompt=ref_by_prompt)
    n_dec = sum(r["deceptive"] for r in rows)
    n_hon = sum(r["honest"] for r in rows)
    print(
        f"trial{trial} {variant or 'original'} round{round_num} ({run_id}): "
        f"deceptive={n_dec} honest={n_hon} -> {out_path.name}"
    )
    if not dry_run:
        write_jsonl(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bootstrap_env()
    failures: list[str] = []
    for trial, variant, round_num, run_id in MISSING_WANDB_RUNS:
        try:
            fetch_one(trial, variant, round_num, run_id, dry_run=args.dry_run)
        except Exception as exc:
            failures.append(f"trial{trial} {variant or 'original'} r{round_num} {run_id}: {exc}")
            print(f"ERROR: {failures[-1]}")

    if failures:
        raise SystemExit(f"{len(failures)} fetch failures:\n" + "\n".join(failures))
    print(f"OK: fetched {len(MISSING_WANDB_RUNS)} W&B eval generation files")


if __name__ == "__main__":
    main()
