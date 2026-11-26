"""Rounds 2..N of one trial lineage: generate and label the previous round's policy
completions, build the probe training set and train the round's probe (unless the probe is
frozen), then run GRPO against it and record the new run id in the lineage's state file.

Usage:
    python -m training.run_trial --init --trial 1 --variant adaptive_score
    python -m training.run_trial --trial 1 --variant adaptive_score --num-rounds 4

A lineage is `experiments/trials/trial<N>/trial_config_<variant>.yaml` plus the files the
rounds write next to it: `buffer_<variant>/` (labeled completions), `rounds_<variant>/round<k>/`
(probe manifest, weights, metrics, exported run metrics), and `trial_state_<variant>.json`.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from training import grpo_launcher
from training.build_dataset import build_train_dataset_split_sources
from training.common import (
    DATA_SEED,
    EVAL_SAMPLES,
    EXPERIMENTS,
    REPO_ROOT,
    bootstrap_hf_env,
    check_disk,
    check_jsonl_rows,
    check_vllm,
    resolve_relative_path,
)
from training.generation import on_policy_output_paths, print_pipeline_df_stats, run_dry_generation, run_on_policy_pipeline
from training.probe_training import load_final_policy_model, run_probe_train_eval_pipeline
from training.round_config import RoundConfig, checkpoint_final_for_id

# Every trial config must set these; the GRPO hyperparameters among them must match the
# round-1 run the lineage branches from (experiments/trials/trial<N>/round1.yaml).
REQUIRED_TRIAL_CONFIG_KEYS = {
    "training_seed",
    "loss_kl",
    "loss_detector",
    "mixed_reward_pass_all_prob",
    "code_length_penalty",
    "kl_reference_from_base",
    "gen_n_generations",
    "base_honest_jsonl",
    "det0_deceptive_jsonl",
    "on_policy_per_source",
}
# Optional keys: `freeze_probe_manifest` + `frozen_probe_weights_dir` (frozen probe);
# `early_stop_threshold_from_train_min`, `early_stop_threshold_floor`,
# `early_stop_detector_patience`, `early_stop_min_steps` (score-based adaptive retraining);
# `early_stop_auc_threshold`, `early_stop_auc_min_class_count`, `early_stop_auc_window_size`
# (AUC-based adaptive retraining); `ground_truth_penalty` (the label replaces the probe score
# in the penalty); `grpo_overrides` (raw Hydra overrides, for short test runs).
EARLY_STOP_KEYS = (
    "early_stop_threshold_from_train_min",
    "early_stop_threshold_floor",
    "early_stop_detector_patience",
    "early_stop_min_steps",
    "early_stop_auc_threshold",
    "early_stop_auc_min_class_count",
    "early_stop_auc_window_size",
)


def stage_header(stage: str, cfg: RoundConfig) -> None:
    print(f"\n{'─' * 60}")
    print(f"  [{cfg.run_key}] {stage}")
    print(f"{'─' * 60}")


def trial_config_path(trial: int, variant: str) -> Path:
    return EXPERIMENTS / "trials" / f"trial{trial}" / f"trial_config_{variant}.yaml"


def load_trial_config(trial: int, variant: str) -> dict:
    path = trial_config_path(trial, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing trial config: {path}")
    config = yaml.safe_load(path.read_text())
    missing = REQUIRED_TRIAL_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"{path}: missing required keys: {sorted(missing)}")
    return config


def trial_state_path(trial: int, variant: str) -> Path:
    return EXPERIMENTS / "trials" / f"trial{trial}" / f"trial_state_{variant}.json"


def load_trial_state(trial: int, variant: str) -> dict:
    path = trial_state_path(trial, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing trial state: {path} (run --init first)")
    state = json.loads(path.read_text())
    if state["trial"] != trial:
        raise ValueError(f"trial_state trial={state['trial']!r} != requested {trial}")
    for key in ("previous_run_id", "last_completed_round", "round_grpo_ids"):
        if key not in state:
            raise ValueError(f"{path}: missing required key {key!r}")
    return state


def save_trial_state(state: dict, variant: str) -> None:
    path = trial_state_path(state["trial"], variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def init_trial_state(trial: int, round1_run_id: str, variant: str) -> dict:
    state = {
        "trial": trial,
        "round1_run_id": round1_run_id,
        "last_completed_round": 1,
        "previous_run_id": round1_run_id,
        "round_grpo_ids": {"1": round1_run_id},
    }
    save_trial_state(state, variant)
    return state


def update_state_after_round(state: dict, round_number: int, grpo_run_id: str, variant: str) -> dict:
    state["last_completed_round"] = round_number
    state["previous_run_id"] = grpo_run_id
    state["round_grpo_ids"][str(round_number)] = grpo_run_id
    save_trial_state(state, variant)
    print(f"OK: trial state updated (last_completed_round={round_number}, previous_run_id={grpo_run_id})")
    return state


def preflight(cfg: RoundConfig) -> None:
    """The previous round's checkpoint exists, vLLM is up, and there is enough disk."""
    if cfg.round < 2:
        raise ValueError(f"run_trial runs rounds >= 2, got {cfg.round}")
    if not cfg.adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter: {cfg.adapter_path}")
    check_vllm()
    check_disk()
    print(f"OK: preflight passed for {cfg.run_key} (previous_run_id={cfg.previous_run_id})")


def generate_eval_completions(cfg: RoundConfig, seed: int, tokenizer) -> Path:
    """One labeled completion per held-out prompt from the policy that starts this round,
    written to `buffer_<variant>/<producer_key>_eval_generations.jsonl`."""
    stage_header("On-policy EVAL generation", cfg)
    eval_df = run_on_policy_pipeline(
        model_key=cfg.producer_key,
        dataset="eval",
        lora_path=cfg.adapter_path,
        tokenizer=tokenizer,
        output_dir=cfg.buffer,
        policy_run_id=cfg.previous_run_id,
        n_generations=1,
        seed=seed,
    )
    print_pipeline_df_stats(eval_df)
    check_jsonl_rows(cfg.eval_jsonl, EVAL_SAMPLES)
    print(f"OK: eval on-policy {cfg.eval_jsonl.name} ({EVAL_SAMPLES} rows)")
    return cfg.eval_jsonl


def generate_train_completions(cfg: RoundConfig, trial_cfg: dict, tokenizer) -> Path:
    """`gen_n_generations` labeled completions per probe-training prompt from the policy that
    starts this round, written to `buffer_<variant>/<producer_key>_train_generations.jsonl`."""
    stage_header("On-policy TRAIN generation", cfg)
    train_df = run_on_policy_pipeline(
        model_key=cfg.producer_key,
        dataset="train",
        lora_path=cfg.adapter_path,
        tokenizer=tokenizer,
        output_dir=cfg.buffer,
        policy_run_id=cfg.previous_run_id,
        n_generations=trial_cfg["gen_n_generations"],
        seed=DATA_SEED,
    )
    print_pipeline_df_stats(train_df)
    train_path, _ = on_policy_output_paths(cfg.producer_key, "train", output_dir=cfg.buffer)
    counts = check_jsonl_rows(train_path, None, label_fields=("deceptive", "honest"))["counts"]
    if counts["deceptive"] == 0 or counts["honest"] == 0:
        print(
            f"WARN: {train_path.name}: deceptive={counts['deceptive']}, honest={counts['honest']} -- "
            "one class is empty; the probe training set draws it from the other sources"
        )
    print(f"OK: train on-policy {train_path.name}")
    return train_path


def train_round_probe(cfg: RoundConfig, trial_cfg: dict, *, force: bool) -> None:
    """Build this round's probe training set from the buffer and train the probe on the
    activations of the policy that starts the round."""
    if force:
        for path in (cfg.probe_weights, cfg.probe_activations):
            if path.exists():
                shutil.rmtree(path)
                print(f"OK: deleted {path} (--force)")

    stage_header("Load final policy model", cfg)
    model, tokenizer = load_final_policy_model(checkpoint_final=cfg.checkpoint_final, adapter_path=cfg.adapter_path)

    stage_header("Build probe train manifest", cfg)
    train_pos, train_neg, manifest = build_train_dataset_split_sources(
        tokenizer,
        trial=cfg.trial,
        round_number=cfg.round,
        manifest_path=cfg.probe_train_dataset,
        buffer_dir=cfg.buffer,
        base_honest_jsonl=resolve_relative_path(trial_cfg["base_honest_jsonl"], REPO_ROOT),
        det0_deceptive_jsonl=resolve_relative_path(trial_cfg["det0_deceptive_jsonl"], REPO_ROOT),
        on_policy_per_source=trial_cfg["on_policy_per_source"],
        seed=DATA_SEED,
    )

    stage_header("Probe train + eval", cfg)
    run_probe_train_eval_pipeline(
        model,
        tokenizer,
        train_pos=train_pos,
        train_neg=train_neg,
        manifest=manifest,
        weights_dir=cfg.probe_weights,
        activations_dir=cfg.probe_activations,
        metrics_path=cfg.probe_metrics,
        manifest_path=cfg.probe_train_dataset,
        val_jsonl=cfg.eval_jsonl,
        run_id=cfg.previous_run_id,
    )
    probe_metrics = json.loads(cfg.probe_metrics.read_text())
    print(f"OK: probe train/eval (on_dataset_auc={probe_metrics['on_dataset_auc']:.4f})")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def prepare_round_probe(cfg: RoundConfig, trial_cfg: dict, *, force: bool) -> None:
    """Everything a round needs before GRPO: the previous policy's labeled completions and,
    unless the probe is frozen, this round's probe."""
    cfg.round_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_hf_env()
    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint_final)
    if "frozen_probe_weights_dir" in trial_cfg:
        # The round-1 probe is the penalty in every round; only the eval completions are
        # generated, so probe_eval can score the frozen probe on the current policy.
        for key in ("frozen_probe_weights_dir", "freeze_probe_manifest"):
            path = resolve_relative_path(trial_cfg[key], REPO_ROOT)
            if not path.exists():
                raise FileNotFoundError(f"{key}: {path} does not exist")
        print("OK: frozen probe -- skipping probe training")
        generate_eval_completions(cfg, DATA_SEED, tokenizer)
        return

    stage_header("Dry generation", cfg)
    run_dry_generation(model_key=cfg.producer_key, dataset="train", tokenizer=tokenizer, lora_path=cfg.adapter_path)
    generate_train_completions(cfg, trial_cfg, tokenizer)
    generate_eval_completions(cfg, DATA_SEED, tokenizer)
    train_round_probe(cfg, trial_cfg, force=force)


def launch_config_for_round(cfg: RoundConfig, trial_cfg: dict) -> dict:
    """The grpo_launcher config for this round's GRPO run."""
    overrides: dict[str, object] = {
        "loss.kl": trial_cfg["loss_kl"],
        "loss.detector": trial_cfg["loss_detector"],
        "training.seed": trial_cfg["training_seed"],
        "data.seed": DATA_SEED,
        "task_args.mixed_reward_pass_all_prob": trial_cfg["mixed_reward_pass_all_prob"],
        "code_length_penalty": trial_cfg["code_length_penalty"],
    }
    if trial_cfg["kl_reference_from_base"]:
        overrides["++grpo.model.kl_reference_from_base"] = "true"
    if trial_cfg.get("ground_truth_penalty", False):
        overrides["ground_truth_detector_penalty"] = "true"
    if "frozen_probe_weights_dir" in trial_cfg:
        overrides["++detector.frozen_probe_weights_dir"] = str(
            resolve_relative_path(trial_cfg["frozen_probe_weights_dir"], REPO_ROOT)
        )
    for key in EARLY_STOP_KEYS:
        if key in trial_cfg:
            overrides[key] = trial_cfg[key]
    overrides.update(trial_cfg.get("grpo_overrides", {}))

    if "freeze_probe_manifest" in trial_cfg:
        manifest = str(resolve_relative_path(trial_cfg["freeze_probe_manifest"], REPO_ROOT))
    else:
        manifest = str(cfg.probe_train_dataset)
    return {
        "run_key": cfg.run_key,
        "group": cfg.wandb_group,
        "output_dir": str(cfg.round_dir),
        "probe": {"manifest": manifest, "directory_save_prefix": cfg.run_key, "retrain": True},
        "checkpoint": str(cfg.checkpoint_final),
        "overrides": overrides,
        "early_stop": any(key in trial_cfg for key in EARLY_STOP_KEYS),
    }


def run_single_round(cfg: RoundConfig, trial_cfg: dict, *, force: bool) -> str:
    """Prepare the round's probe, run GRPO against it, and return the new W&B run id."""
    prepare_round_probe(cfg, trial_cfg, force=force)
    stage_header("GRPO", cfg)
    if force and cfg.run_metrics.exists():
        cfg.run_metrics.unlink()
    return grpo_launcher.launch(launch_config_for_round(cfg, trial_cfg))


def run_trial(trial: int, variant: str, *, num_rounds: int, start_round: int | None, force: bool) -> None:
    state = load_trial_state(trial, variant)
    trial_cfg = load_trial_config(trial, variant)
    if start_round is None:
        start_round = state["last_completed_round"] + 1
    for round_number in range(start_round, start_round + num_rounds):
        cfg = RoundConfig(trial=trial, round=round_number, previous_run_id=state["previous_run_id"], variant=variant)
        print(f"\n{'=' * 60}\n=== Round {round_number} ({cfg.run_key}) ===\n{'=' * 60}")
        if cfg.run_metrics.exists() and not force:
            grpo_run_id = json.loads(cfg.run_metrics.read_text())["run_id"]
            print(f"SKIP: {cfg.run_metrics.name} exists (use --force to re-run)")
        else:
            preflight(cfg)
            grpo_run_id = run_single_round(cfg, trial_cfg, force=force)
        if state["last_completed_round"] < round_number or force:
            state = update_state_after_round(state, round_number, grpo_run_id, variant)


def round1_run_metrics_path(trial: int) -> Path:
    return EXPERIMENTS / "trials" / f"trial{trial}" / "rounds" / "round1" / f"run_metrics_trial{trial}_round1.json"


def read_round1_run_id(trial: int) -> str:
    path = round1_run_metrics_path(trial)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}: run `python -m training.run_grpo experiments/trials/trial{trial}/round1.yaml` "
            "first, or pass --round1-run-id"
        )
    return json.loads(path.read_text())["run_id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Lineage name: experiments/trials/trial<N>/trial_config_<variant>.yaml (e.g. adaptive_score, frozen)",
    )
    parser.add_argument("--init", action="store_true", help="Create trial_state_<variant>.json from the round-1 run")
    parser.add_argument(
        "--round1-run-id",
        type=str,
        help="Round-1 GRPO run id (with --init); default: read from rounds/round1/run_metrics_trial<N>_round1.json",
    )
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--start-round", type=int, default=None, help="Default: the round after the last completed one")
    parser.add_argument("--force", action="store_true", help="Re-run rounds whose run metrics exist; retrain probes")
    args = parser.parse_args(argv)

    if args.init:
        round1_run_id = args.round1_run_id or read_round1_run_id(args.trial)
        checkpoint_final = checkpoint_final_for_id(round1_run_id)
        if not checkpoint_final.is_dir():
            raise FileNotFoundError(f"Round 1 checkpoint missing: {checkpoint_final}")
        init_trial_state(args.trial, round1_run_id, args.variant)
        print(f"OK: initialized {trial_state_path(args.trial, args.variant)}")
        return 0

    run_trial(args.trial, args.variant, num_rounds=args.num_rounds, start_round=args.start_round, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
