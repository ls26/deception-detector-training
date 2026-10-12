"""Produce the shared on-policy starting data:

    experiments/artifacts/base/base_{train,eval}_generations.jsonl   base model
    experiments/artifacts/det0/det0_{train,eval}_generations.jsonl   no-probe-penalty model

`base` is the honest anchor and `det0` the reward-hacking anchor of every probe training
set; the eval files are the anchors for probe_eval. Needs the vLLM server and a completed
no-probe-penalty baseline run (experiments/baselines/no_probe_penalty.yaml). Each file is
skipped if it already exists.

Usage:
    python -m training.make_artifacts                                  # det0 from no_probe_penalty_s42
    python -m training.make_artifacts --det0-run-key no_probe_penalty_s43
    python -m training.make_artifacts --only base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from training.build_dataset import ON_POLICY_ROUND1_MIXED
from training.common import EVAL_SAMPLES, EXPERIMENTS, MODEL_NAME, bootstrap_hf_env, check_disk, check_jsonl_rows, check_vllm
from training.generation import on_policy_output_paths, print_pipeline_df_stats, run_on_policy_pipeline
from training.round_config import adapter_path_for, checkpoint_final_for_id

BASE_TRAIN_GENERATIONS = 10  # near-duplicate completions are dropped (dedup on)
DET0_TRAIN_GENERATIONS = 3  # kept exactly as sampled (dedup off)
DEFAULT_DET0_RUN_KEY = "no_probe_penalty_s42"

ARTIFACTS_DIR = EXPERIMENTS / "artifacts"
BASELINE_EXPORT_DIR = EXPERIMENTS / "baselines" / "no_probe_penalty"


def det0_run_id_from_export(run_key: str) -> str:
    export_path = BASELINE_EXPORT_DIR / f"run_metrics_{run_key}.json"
    if not export_path.exists():
        raise FileNotFoundError(
            f"Missing {export_path}: run `python -m training.run_grpo "
            "experiments/baselines/no_probe_penalty.yaml` first, or pass --det0-run-id"
        )
    return json.loads(export_path.read_text())["run_id"]


def remove_outputs(model_key: str, output_dir: Path) -> None:
    for dataset in ("train", "eval"):
        for path in on_policy_output_paths(model_key, dataset, output_dir=output_dir):
            if path.exists():
                path.unlink()
                print(f"OK: deleted {path} (--force)")


def make_base(tokenizer) -> None:
    output_dir = ARTIFACTS_DIR / "base"
    for dataset, n_generations in (("train", BASE_TRAIN_GENERATIONS), ("eval", 1)):
        df = run_on_policy_pipeline(
            model_key="base",
            dataset=dataset,
            lora_path=None,
            tokenizer=tokenizer,
            output_dir=output_dir,
            policy_run_id="base",
            n_generations=n_generations,
        )
        print_pipeline_df_stats(df)

    train_path, _ = on_policy_output_paths("base", "train", output_dir=output_dir)
    honest = check_jsonl_rows(train_path, None, label_fields=("honest",))["counts"]["honest"]
    if honest < ON_POLICY_ROUND1_MIXED:
        raise ValueError(
            f"{train_path}: only {honest} honest rows, need >= {ON_POLICY_ROUND1_MIXED} for the round-1 probe training set"
        )
    eval_path, _ = on_policy_output_paths("base", "eval", output_dir=output_dir)
    check_jsonl_rows(eval_path, EVAL_SAMPLES)
    print(f"OK: base artifacts ready ({honest} honest train rows, {EVAL_SAMPLES} eval rows)")


def make_det0(tokenizer, run_id: str) -> None:
    output_dir = ARTIFACTS_DIR / "det0"
    adapter_path = adapter_path_for(checkpoint_final_for_id(run_id))
    for dataset, n_generations in (("train", DET0_TRAIN_GENERATIONS), ("eval", 1)):
        df = run_on_policy_pipeline(
            model_key="det0",
            dataset=dataset,
            lora_path=adapter_path,
            tokenizer=tokenizer,
            output_dir=output_dir,
            policy_run_id=run_id,
            n_generations=n_generations,
            deduplicate_responses=False,
        )
        print_pipeline_df_stats(df)

    train_path, _ = on_policy_output_paths("det0", "train", output_dir=output_dir)
    deceptive = check_jsonl_rows(train_path, None, label_fields=("deceptive",))["counts"]["deceptive"]
    if deceptive < ON_POLICY_ROUND1_MIXED:
        raise ValueError(
            f"{train_path}: only {deceptive} reward-hacking rows, need >= {ON_POLICY_ROUND1_MIXED}. "
            f"This no-penalty run ({run_id}) did not reward-hack enough; rerun with --force and "
            "--det0-run-key for another seed"
        )
    eval_path, _ = on_policy_output_paths("det0", "eval", output_dir=output_dir)
    check_jsonl_rows(eval_path, EVAL_SAMPLES)
    print(f"OK: det0 artifacts ready ({deceptive} reward-hacking train rows, {EVAL_SAMPLES} eval rows)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    det0_source = parser.add_mutually_exclusive_group()
    det0_source.add_argument(
        "--det0-run-key",
        default=DEFAULT_DET0_RUN_KEY,
        help=f"Baseline run whose export gives the det0 run id (default: {DEFAULT_DET0_RUN_KEY})",
    )
    det0_source.add_argument("--det0-run-id", help="W&B run id of the no-probe-penalty run to sample det0 from")
    parser.add_argument("--only", choices=("base", "det0"), help="Make only one of the two artifact sets")
    parser.add_argument("--force", action="store_true", help="Delete the selected artifacts first and regenerate")
    args = parser.parse_args(argv)

    bootstrap_hf_env()
    check_vllm()
    check_disk()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if args.only != "det0":
        if args.force:
            remove_outputs("base", ARTIFACTS_DIR / "base")
        make_base(tokenizer)
    if args.only != "base":
        run_id = args.det0_run_id or det0_run_id_from_export(args.det0_run_key)
        print(f"det0 source: run {run_id}")
        if args.force:
            remove_outputs("det0", ARTIFACTS_DIR / "det0")
        make_det0(tokenizer, run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
