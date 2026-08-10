#!/usr/bin/env python3
"""Multi-round experiment orchestrator for trials/trial{N}/."""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer

from pipeline.build_dataset import (
    build_train_dataset_round_robin,
    build_train_dataset_split_sources,
    rows_to_dataset,
)
from pipeline.extract_run_results import print_all_results
from pipeline.generation import (
    bootstrap_hf_env,
    on_policy_output_paths,
    print_pipeline_df_stats,
    run_dry_generation,
    run_on_policy_pipeline,
)
from pipeline.grpo_launcher import (
    DEFAULT_MAX_GRPO_RETRIES,
    DEFAULT_MIN_VAL_DOES_EXECUTE,
    GrpoLaunchError,
    check_degeneracy,
    check_disk,
    check_vllm,
    cleanup_activation_caches,
    cleanup_crashed_run_dir,
    cleanup_intermediates,
    export_wandb_metrics,
    rename_degenerate_metrics,
    resolve_run_id,
    run_grpo_subprocess,
    verify_run_complete,
)
from pipeline.paths import EXPERIMENTS, WORKSPACE
from pipeline.probe_training import FOLLOW_UP, load_final_policy_model, run_probe_train_eval_pipeline
from pipeline.round_config import RoundConfig, resolve_checkpoint_final

EVAL_SAMPLES = 400
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
GRPO_RUN_ID_RE = re.compile(r"/runs/([a-z0-9]+)")


def _stage_header(stage: str, cfg: RoundConfig) -> None:
    print(f"\n{'─' * 60}")
    print(f"  [{cfg.run_key}] {stage}")
    print(f"{'─' * 60}")


# Round where calibration_logits are first computed (round 1 has no calibration data yet).
# Used by `fix_calibration_slope` to anchor the frozen (z_d - z_h) denominator.
FIXED_CALIBRATION_ANCHOR_ROUND = 2

# Required keys every trials/trial{N}/trial_config.yaml must define. Mismatches
# here silently invalidate cross-round comparisons (see FULL_TRIAL_RUNBOOK.md
# "Known pitfalls" / "Hyperparameter mismatch between round 1 and automated rounds").
REQUIRED_TRIAL_CONFIG_KEYS = {
    "training_seed",
    "data_seed",
    "loss_kl",
    "loss_detector",
    "nofu",
    "mixed_reward_pass_all_prob",
    "code_length_penalty",
    "kl_reference_from_base",
    "dataset_strategy",
    "gen_dataset",
    "gen_n_generations",
    "base_honest_jsonl",
    "det0_deceptive_jsonl",
    "on_policy_per_source",
    "n_fixed_per_class",
    "needs_base_backfill",
    "use_calibrated_penalty",
}


def trial_config_path(trial: int, variant: str = "") -> Path:
    trial_dir = EXPERIMENTS / "trials" / f"trial{trial}"
    if variant:
        variant_path = trial_dir / f"trial_config_{variant}.yaml"
        if not variant_path.exists():
            raise FileNotFoundError(
                f"Missing variant trial config: {variant_path}. "
                f"Create trial_config_{variant}.yaml (no silent fallback to trial_config.yaml)."
            )
        return variant_path
    return trial_dir / "trial_config.yaml"


def load_trial_config(trial: int, variant: str = "") -> dict:
    path = trial_config_path(trial, variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing trial config: {path}. Create trials/trial{trial}/trial_config.yaml "
            "before running automation (see FULL_TRIAL_RUNBOOK.md § Known pitfalls)."
        )
    config = yaml.safe_load(path.read_text())
    missing = REQUIRED_TRIAL_CONFIG_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"{path}: missing required keys: {sorted(missing)}")
    return config


def trial_state_path(trial: int, variant: str = "") -> Path:
    trial_dir = EXPERIMENTS / "trials" / f"trial{trial}"
    name = f"trial_state_{variant}.json" if variant else "trial_state.json"
    return trial_dir / name


def load_trial_state(trial: int, variant: str = "") -> dict:
    path = trial_state_path(trial, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing trial state: {path} (run --init first)")
    state = json.loads(path.read_text())
    if state["trial"] != trial:
        raise ValueError(f"trial_state trial={state['trial']!r} != requested {trial}")
    if "previous_run_id" not in state:
        raise ValueError(f"{path}: missing required key 'previous_run_id'")
    if "last_completed_round" not in state:
        raise ValueError(f"{path}: missing required key 'last_completed_round'")
    return state


def save_trial_state(state: dict, variant: str = "") -> None:
    path = trial_state_path(state["trial"], variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def init_trial_state(trial: int, round1_run_id: str, variant: str = "") -> dict:
    state = {
        "trial": trial,
        "round1_run_id": round1_run_id,
        "last_completed_round": 1,
        "previous_run_id": round1_run_id,
        "round_grpo_ids": {"1": round1_run_id},
    }
    save_trial_state(state, variant)
    return state


def train_jsonl_path(cfg: RoundConfig, gen_dataset: str) -> Path:
    path, _ = on_policy_output_paths(cfg.producer_key, gen_dataset, output_dir=cfg.buffer)
    return path


def checkpoint_dir_for_run_id(
    workspace: Path, run_id: str, *, trial: int | None = None
) -> Path:
    return resolve_checkpoint_final(workspace, run_id, trial=trial)


def check_jsonl_rows(
    path: Path, expected: int | None, *, label_fields: tuple[str, ...] = ()
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if expected is not None:
        if len(rows) != expected:
            raise ValueError(f"{path.name}: expected {expected} rows, got {len(rows)}")
    elif len(rows) == 0:
        raise ValueError(f"{path.name}: expected at least 1 row, got 0")
    counts = {}
    for field in label_fields:
        counts[field] = sum(bool(r.get(field)) for r in rows)
    return {"rows": rows, "counts": counts}


def preflight(cfg: RoundConfig, state: dict, trial_cfg: dict) -> None:
    if cfg.round < 2:
        raise ValueError(f"automation runs round >= 2, got {cfg.round}")
    if not cfg.adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter: {cfg.adapter_path}")
    if not cfg.checkpoint_final.exists():
        raise FileNotFoundError(f"Missing checkpoint: {cfg.checkpoint_final}")
    if trial_cfg["needs_base_backfill"]:
        base_train = cfg.buffer / "base_train_generations.jsonl"
        if not base_train.exists():
            raise FileNotFoundError(f"Missing base backfill: {base_train}")
    check_vllm()
    check_disk(min_free_gb=50, path=cfg.workspace)
    if state["previous_run_id"] != cfg.previous_run_id:
        raise ValueError(
            f"state previous_run_id={state['previous_run_id']!r} != "
            f"cfg.previous_run_id={cfg.previous_run_id!r}"
        )
    print(f"OK: preflight passed for {cfg.run_key} (previous_run_id={cfg.previous_run_id})")


def resolve_grpo_run_id(cfg: RoundConfig, stdout: str | None = None) -> str:
    if stdout:
        match = GRPO_RUN_ID_RE.search(stdout)
        if match:
            run_id = match.group(1)
            ckpt = checkpoint_dir_for_run_id(cfg.workspace, run_id, trial=cfg.trial)
            if ckpt.exists():
                return run_id
            print(
                f"WARN: GRPO run id {run_id} parsed from stdout but checkpoint missing "
                f"({ckpt}); falling back to W&B API"
            )

    run_id = resolve_run_id(cfg.run_key, None)
    ckpt = checkpoint_dir_for_run_id(cfg.workspace, run_id, trial=cfg.trial)
    if not ckpt.exists():
        raise GrpoLaunchError(f"W&B run {run_id} found but checkpoint missing: {ckpt}")
    return run_id


def run_grpo(cfg: RoundConfig, trial_cfg: dict) -> str:
    args = [
        "python",
        "-m",
        "obfuscation_atlas.scripts.train_grpo_model_against_probe",
        "+model_preset=llama3_8b",
        f"loss.kl={trial_cfg['loss_kl']}",
        f"loss.detector={trial_cfg['loss_detector']}",
        "+data/probe_dataset=mbpp_probe",
        f"training.seed={trial_cfg['training_seed']}",
        f"data.seed={trial_cfg['data_seed']}",
    ]
    if trial_cfg["nofu"]:
        args += [
            "data.probe_dataset.congruent_follow_up_prompts=[]",
            "data.probe_dataset.incongruent_follow_up_prompts=[]",
            "data.task_dataset.congruent_follow_up_prompts=[]",
            "data.task_dataset.incongruent_follow_up_prompts=[]",
        ]
    args += [
        "grpo.model.attn_implementation=sdpa",
        f"training.save_path_base={cfg.workspace / 'results/run_results'}",
        "detector.model_cache_path=/workspace/results/activations",
        f"detector.directory_save_prefix={cfg.run_key}",
        f"++detector.mixed_probe_manifest_path={trial_cfg.get('freeze_probe_manifest') or cfg.probe_train_dataset}",
        f"++grpo.model.reference_adapter_path={cfg.checkpoint_final}",
        "++training.resume_from_checkpoint=False",
        "corrected_detector_gradient=false",
        "misalignment_eval.enabled=false",
        "wandb.project=deception_detection",
        f"wandb.group={cfg.wandb_group}",
        f"wandb.run_name_prefix={cfg.run_key}",
        f"task_args.deceptive_definition={cfg.deceptive_definition}",
    ]
    if trial_cfg["mixed_reward_pass_all_prob"] is not None:
        args.append(f"task_args.mixed_reward_pass_all_prob={trial_cfg['mixed_reward_pass_all_prob']}")
    if trial_cfg.get("code_length_penalty") is not None:
        args.append(f"code_length_penalty={trial_cfg['code_length_penalty']}")
    if trial_cfg.get("kl_reference_from_base", False):
        args.append("++grpo.model.kl_reference_from_base=true")
    if trial_cfg.get("ground_truth_penalty", False):
        args.append("ground_truth_detector_penalty=true")
    if trial_cfg.get("use_calibrated_penalty", False):
        cal_path = cfg.probe_metrics
        # GRPO Phase-0 always trains its own probe from scratch, which has a different
        # logit scale than the external pipeline's probe that computed calibration_logits
        # below (same data/hyperparameters, but different init/shuffling). Freezing GRPO
        # onto this same probe keeps the penalty's z_agg on the same scale as z_h/z_d --
        # see docs/EXPERIMENT_LOG.md methodology entry on the calibrated-probe-mismatch fix.
        frozen_weights_for_calibration = cfg.probe_weights
        cal_data = json.loads(cal_path.read_text()).get("calibration_logits")
        if not cal_data or cal_data.get("z_h") is None:
            # Current round couldn't compute calibration_logits (e.g. a near-fully-deceptive
            # policy produced zero honest on-policy eval samples) -- fall back to the previous
            # round's z_h so the calibrated penalty still has a numerator to work with.
            prev_round_dir = cfg.round_dir.parent / f"round{cfg.round - 1}"
            prev_path = prev_round_dir / "probe_metrics.json"
            if not prev_path.exists():
                raise FileNotFoundError(f"No fallback calibration available: {prev_path}")
            print(
                f"WARN: calibration_logits missing/invalid for round {cfg.round}; "
                f"falling back to round {cfg.round - 1}'s ({prev_path})"
            )
            cal_path = prev_path
            frozen_weights_for_calibration = prev_round_dir / "probe_weights"
        args.append(f"++detector.calibration_logits_path={cal_path}")
        args.append(f"++detector.frozen_probe_weights_dir={frozen_weights_for_calibration}")
        if trial_cfg.get("fix_calibration_slope", False) and cfg.round > FIXED_CALIBRATION_ANCHOR_ROUND:
            anchor_metrics_path = cfg.round_dir.parent / f"round{FIXED_CALIBRATION_ANCHOR_ROUND}" / "probe_metrics.json"
            if not anchor_metrics_path.exists():
                raise FileNotFoundError(f"fix_calibration_slope: missing {anchor_metrics_path}")
            anchor_cal = json.loads(anchor_metrics_path.read_text())["calibration_logits"]
            fixed_denominator = anchor_cal["z_d"] - anchor_cal["z_h"]
            args.append(f"++detector.fixed_calibration_denominator={fixed_denominator}")
    if not trial_cfg.get("scale_rewards", True):
        args.append("loss.scale_rewards=false")
    if trial_cfg.get("frozen_probe_weights_dir"):
        args.append(f"++detector.frozen_probe_weights_dir={trial_cfg['frozen_probe_weights_dir']}")
    if trial_cfg.get("early_stop_detector_threshold") is not None:
        args.append(f"early_stop_detector_threshold={trial_cfg['early_stop_detector_threshold']}")
    if trial_cfg.get("early_stop_detector_patience") is not None:
        args.append(f"early_stop_detector_patience={trial_cfg['early_stop_detector_patience']}")
    if trial_cfg.get("early_stop_min_steps") is not None:
        args.append(f"early_stop_min_steps={trial_cfg['early_stop_min_steps']}")
    if trial_cfg.get("early_stop_threshold_from_train_min", False):
        args.append("early_stop_threshold_from_train_min=true")
    if trial_cfg.get("early_stop_threshold_floor") is not None:
        args.append(f"early_stop_threshold_floor={trial_cfg['early_stop_threshold_floor']}")
    if trial_cfg.get("early_stop_auc_threshold") is not None:
        args.append(f"early_stop_auc_threshold={trial_cfg['early_stop_auc_threshold']}")
    if trial_cfg.get("early_stop_auc_min_class_count") is not None:
        args.append(f"early_stop_auc_min_class_count={trial_cfg['early_stop_auc_min_class_count']}")
    if trial_cfg.get("early_stop_auc_window_size") is not None:
        args.append(f"early_stop_auc_window_size={trial_cfg['early_stop_auc_window_size']}")

    exit_code, output = run_grpo_subprocess(args)
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, args, output)
    return resolve_grpo_run_id(cfg, output)


def build_or_reuse_probe_dataset(
    cfg: RoundConfig,
    trial_cfg: dict,
    tokenizer,
    *,
    train_check: dict,
) -> tuple[Dataset, Dataset, dict, bool]:
    """Build a fresh mixed probe train manifest, or carry forward the previous
    round's manifest unchanged if this round's on-policy generation didn't
    produce at least `min_on_policy_per_class` samples of both classes.

    Returns (train_pos, train_neg, manifest, skipped_expansion).
    `skipped_expansion` is True when the manifest was carried forward unchanged.

    `min_on_policy_per_class` defaults to 0 (always rebuild -- prior behavior).
    See docs/FULL_TRIAL_RUNBOOK.md for the conditional-retraining rationale.
    """
    min_on_policy = trial_cfg.get("min_on_policy_per_class", 0)
    deceptive_count = train_check["counts"]["deceptive"]
    honest_count = train_check["counts"]["honest"]
    threshold_met = min_on_policy <= 0 or (
        deceptive_count >= min_on_policy and honest_count >= min_on_policy
    )

    if not threshold_met:
        prev_manifest_path = cfg.round_dir.parent / f"round{cfg.round - 1}" / "probe_train_dataset.json"
        if prev_manifest_path.exists():
            shutil.copy2(prev_manifest_path, cfg.probe_train_dataset)
            manifest = json.loads(cfg.probe_train_dataset.read_text())
            train_pos = rows_to_dataset(manifest["deceptive"])
            train_neg = rows_to_dataset(manifest["honest"])
            print(
                f"SKIP EXPANSION: latest on-policy deceptive={deceptive_count}, honest={honest_count} "
                f"< min_on_policy_per_class={min_on_policy}; reusing {prev_manifest_path} "
                f"(round {cfg.round - 1}) unchanged -- probe will still be retrained on this data "
                "with the current model's activations"
            )
            return train_pos, train_neg, manifest, True
        print(
            f"WARN: latest on-policy deceptive={deceptive_count}, honest={honest_count} "
            f"< min_on_policy_per_class={min_on_policy}, but no previous manifest at "
            f"{prev_manifest_path} to reuse -- building fresh manifest instead"
        )

    data_seed = trial_cfg["data_seed"]
    if trial_cfg["dataset_strategy"] == "split_sources":
        pos, neg, m = build_train_dataset_split_sources(
            tokenizer,
            trial=cfg.trial,
            round=cfg.round,
            manifest_path=cfg.probe_train_dataset,
            buffer_dir=cfg.buffer,
            base_honest_jsonl=Path(trial_cfg["base_honest_jsonl"]),
            det0_deceptive_jsonl=Path(trial_cfg["det0_deceptive_jsonl"]),
            on_policy_per_source=trial_cfg["on_policy_per_source"],
            n_fixed_per_class=trial_cfg["n_fixed_per_class"],
            data_seed=data_seed,
        )
        return pos, neg, m, False
    if trial_cfg["dataset_strategy"] == "round_robin":
        pos, neg, m = build_train_dataset_round_robin(
            tokenizer,
            trial=cfg.trial,
            round=cfg.round,
            manifest_path=cfg.probe_train_dataset,
            buffer_dir=cfg.buffer,
            data_seed=data_seed,
        )
        return pos, neg, m, False
    raise ValueError(f"Unknown dataset_strategy: {trial_cfg['dataset_strategy']!r}")


def run_single_round(
    cfg: RoundConfig,
    *,
    dry_run: bool = False,
    mock_grpo_run_id: str | None = None,
    force: bool = False,
) -> str:
    cfg.round_dir.mkdir(parents=True, exist_ok=True)
    trial_cfg = load_trial_config(cfg.trial, cfg.variant)
    data_seed = trial_cfg["data_seed"]

    frozen_weights_dir = trial_cfg.get("frozen_probe_weights_dir")
    frozen_manifest = trial_cfg.get("freeze_probe_manifest")
    if frozen_weights_dir or frozen_manifest:
        # Truly frozen probe: the same externally-trained probe weights are reused for
        # every round (see docs/FULL_TRIAL_RUNBOOK.md and train_grpo_model_against_probe.py's
        # frozen_probe_weights_dir override). No new on-policy data feeds into probe
        # training, so generation and the external probe pipeline are skipped entirely.
        if not (frozen_weights_dir and frozen_manifest):
            raise ValueError(
                "frozen_probe_weights_dir and freeze_probe_manifest must both be set together"
            )
        if trial_cfg.get("use_calibrated_penalty", False):
            raise ValueError(
                "frozen_probe_weights_dir is incompatible with use_calibrated_penalty "
                "(calibration logits come from external probe training, which is skipped)"
            )
        if not Path(frozen_weights_dir).exists():
            raise FileNotFoundError(f"Frozen probe weights not found: {frozen_weights_dir}")
        if not Path(frozen_manifest).exists():
            raise FileNotFoundError(f"Frozen manifest not found: {frozen_manifest}")
        print(
            "OK: frozen probe -- skipping generation + probe training "
            f"(weights={frozen_weights_dir}, manifest={frozen_manifest})"
        )
    else:
        if force:
            if cfg.probe_weights.exists():
                shutil.rmtree(cfg.probe_weights)
                print(f"OK: deleted {cfg.probe_weights} (--force)")
            if cfg.probe_activations.exists():
                shutil.rmtree(cfg.probe_activations)
                print(f"OK: deleted {cfg.probe_activations} (--force)")

        _stage_header("MBPP dataset prep + dry generation", cfg)
        bootstrap_hf_env()
        tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint_final)

        run_dry_generation(
            model_key=cfg.producer_key,
            dataset=trial_cfg["gen_dataset"],
            tokenizer=tokenizer,
            lora_path=cfg.adapter_path,
            n_generations=1,
            deceptive_definition=cfg.deceptive_definition,
            seed=data_seed,
        )

        _stage_header("On-policy TRAIN generation", cfg)
        train_df = run_on_policy_pipeline(
            model_key=cfg.producer_key,
            dataset=trial_cfg["gen_dataset"],
            lora_path=cfg.adapter_path,
            tokenizer=tokenizer,
            require_adapter=True,
            n_generations=trial_cfg["gen_n_generations"],
            output_dir=cfg.buffer,
            run_id_config=cfg.previous_run_id,
            deceptive_definition=cfg.deceptive_definition,
            seed=data_seed,
        )
        print_pipeline_df_stats(train_df)
        train_path = train_jsonl_path(cfg, trial_cfg["gen_dataset"])
        train_check = check_jsonl_rows(train_path, None, label_fields=("deceptive", "honest"))
        if train_check["counts"]["deceptive"] == 0 or train_check["counts"]["honest"] == 0:
            print(
                f"WARN: {train_path.name}: deceptive={train_check['counts']['deceptive']}, "
                f"honest={train_check['counts']['honest']} -- one class is empty; "
                f"round-robin will draw from other sources"
            )
        print(f"OK: train on-policy {train_path.name} ({len(train_check['rows'])} rows)")

        _stage_header("On-policy EVAL generation", cfg)
        eval_df = run_on_policy_pipeline(
            model_key=cfg.producer_key,
            dataset="eval",
            lora_path=cfg.adapter_path,
            tokenizer=tokenizer,
            require_adapter=True,
            n_generations=1,
            output_dir=cfg.buffer,
            run_id_config=cfg.previous_run_id,
            deceptive_definition=cfg.deceptive_definition,
            seed=data_seed,
        )
        print_pipeline_df_stats(eval_df)
        eval_path = cfg.buffer / f"{cfg.producer_key}_eval_generations.jsonl"
        check_jsonl_rows(eval_path, EVAL_SAMPLES)
        print(f"OK: eval on-policy {eval_path.name} ({EVAL_SAMPLES} rows)")

        if trial_cfg["needs_base_backfill"]:
            base_path = cfg.buffer / "base_train_generations.jsonl"
            if not base_path.exists():
                base_train_df = run_on_policy_pipeline(
                    model_key="base",
                    dataset="train_full",
                    lora_path=None,
                    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL),
                    n_generations=15,
                    output_dir=cfg.buffer,
                    run_id_config="base",
                    deceptive_definition=cfg.deceptive_definition,
                    seed=data_seed,
                )
                print_pipeline_df_stats(base_train_df)
            if not base_path.exists():
                raise FileNotFoundError(f"Missing {base_path}")
            print(f"OK: base backfill {base_path.name}")
        else:
            print("OK: base backfill not required (dataset_strategy uses external anchor sources)")

        _stage_header("Build mixed probe train manifest", cfg)
        train_pos, train_neg, manifest, skipped_expansion = build_or_reuse_probe_dataset(
            cfg, trial_cfg, tokenizer, train_check=train_check
        )
        manifest_data = json.loads(cfg.probe_train_dataset.read_text())
        n_per_class = manifest_data["n_train_per_class"]
        if not (
            len(manifest_data["deceptive"]) == len(manifest_data["honest"]) == n_per_class
        ):
            raise ValueError(
                f"probe train manifest class counts mismatch: "
                f"deceptive={len(manifest_data['deceptive'])}, "
                f"honest={len(manifest_data['honest'])}, n_train_per_class={n_per_class}"
            )
        print(f"OK: probe train manifest {cfg.probe_train_dataset.name} ({n_per_class}/class)")

        if skipped_expansion and trial_cfg.get("carry_over_probe_weights", False):
            prev_round_dir = cfg.round_dir.parent / f"round{cfg.round - 1}"
            prev_weights = prev_round_dir / "probe_weights"
            prev_metrics = prev_round_dir / "probe_metrics.json"
            if not prev_weights.exists():
                raise FileNotFoundError(f"carry_over_probe_weights: missing {prev_weights}")
            if not prev_metrics.exists():
                raise FileNotFoundError(f"carry_over_probe_weights: missing {prev_metrics}")
            if cfg.probe_weights.exists():
                shutil.rmtree(cfg.probe_weights)
                print(f"OK: removed existing {cfg.probe_weights} before carry-over")
            shutil.copytree(prev_weights, cfg.probe_weights)
            shutil.copy2(prev_metrics, cfg.probe_metrics)
            trial_cfg["frozen_probe_weights_dir"] = str(cfg.probe_weights)
            carried_metrics = json.loads(cfg.probe_metrics.read_text())
            cal = carried_metrics.get("calibration_logits", {})
            print(
                f"CARRY OVER PROBE WEIGHTS: copied weights + metrics from round {cfg.round - 1} "
                f"(on_dataset_auc={carried_metrics.get('on_dataset_auc', '?')}, "
                f"calibration z_h={cal.get('z_h', '?')}, z_d={cal.get('z_d', '?')})"
            )
        else:
            _stage_header("Load final policy model", cfg)
            model, tokenizer = load_final_policy_model(
                checkpoint_final=cfg.checkpoint_final,
                adapter_path=cfg.adapter_path,
            )

            _stage_header("Probe train + eval (activations + linear probe)", cfg)
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
                follow_up=[] if trial_cfg["nofu"] else FOLLOW_UP,
                deceptive_definition=cfg.deceptive_definition,
                seed=data_seed,
            )
            probe_metrics = json.loads(cfg.probe_metrics.read_text())
            if "on_dataset_auc" not in probe_metrics:
                raise ValueError("probe_metrics.json missing on_dataset_auc")
            if not (cfg.probe_weights / "layer_0" / "config.json").exists():
                raise FileNotFoundError(
                    f"Missing probe weights config: {cfg.probe_weights / 'layer_0' / 'config.json'}"
                )
            print(f"OK: probe train/eval (on_dataset_auc={probe_metrics['on_dataset_auc']:.4f})")

            _stage_header("Free GPU memory", cfg)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if dry_run:
        grpo_run_id = mock_grpo_run_id or cfg.previous_run_id
        ckpt = checkpoint_dir_for_run_id(cfg.workspace, grpo_run_id, trial=cfg.trial)
        if not ckpt.exists():
            raise GrpoLaunchError(f"Dry-run mock id {grpo_run_id} has no checkpoint: {ckpt}")
        print(f"OK: dry-run skipping GRPO/export; mock grpo_run_id={grpo_run_id}")
        return grpo_run_id

    # Retry on crash (non-zero exit) or degenerate collapse (exit 0 but
    # val_does_execute below threshold -- see docs/EXPERIMENT_LOG.md #13).
    # Both are stochastic instabilities that a same-config retry can resolve;
    # generation/probe training above are already cached on disk, so only
    # the GRPO step itself repeats.
    #
    # NOTE: Intentionally parallel to grpo_launcher.launch()'s retry loop.
    # Update both together if the crash/degeneracy policy changes.
    max_retries = trial_cfg.get("max_grpo_retries")
    if max_retries is None:
        max_retries = DEFAULT_MAX_GRPO_RETRIES
    min_val_does_execute = trial_cfg.get("min_val_does_execute")
    if min_val_does_execute is None:
        min_val_does_execute = DEFAULT_MIN_VAL_DOES_EXECUTE
    max_attempts = max_retries + 1
    attempt_summaries: list[str] = []

    _stage_header("GRPO subprocess", cfg)
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            check_vllm()
            check_disk(min_free_gb=50, path=cfg.workspace)

        try:
            grpo_run_id = run_grpo(cfg, trial_cfg)
        except subprocess.CalledProcessError as e:
            print(f"CRASH: {cfg.run_key} attempt {attempt} failed with exit code {e.returncode}")
            cleanup_crashed_run_dir(
                e.output or "", group=cfg.wandb_group, run_key=cfg.run_key
            )
            attempt_summaries.append(f"attempt {attempt}: crashed (exit={e.returncode})")
            if attempt < max_attempts:
                continue
            raise GrpoLaunchError(
                f"{cfg.run_key}: GRPO subprocess failed after {max_attempts} attempt(s): "
                + "; ".join(attempt_summaries)
            ) from e
        print(f"OK: GRPO finished run_id={grpo_run_id}")

        # Early-stopped runs can legitimately have far fewer than 128 logged
        # response steps, since training may stop well before max_steps.
        early_stop_enabled = (
            trial_cfg.get("early_stop_detector_threshold") is not None
            or trial_cfg.get("early_stop_threshold_from_train_min", False)
            or trial_cfg.get("early_stop_auc_threshold") is not None
        )
        min_responses = 1 if early_stop_enabled else 128
        verify_run_complete(
            grpo_run_id,
            min_responses=min_responses,
            group=cfg.wandb_group,
            run_key=cfg.run_key,
        )
        cleanup_intermediates(
            grpo_run_id, group=cfg.wandb_group, run_key=cfg.run_key
        )
        cleanup_activation_caches(grpo_run_id, run_key=cfg.run_key)
        print(f"OK: intermediates cleaned for {grpo_run_id}")

        export_wandb_metrics(
            grpo_run_id,
            cfg.run_metrics,
            max_retries=3,
            expected_run_id=grpo_run_id,
        )
        print(f"OK: exported {cfg.run_metrics}")

        is_degenerate, val_does_execute = check_degeneracy(cfg.run_metrics, min_val_does_execute)
        if is_degenerate:
            print(
                f"DEGENERATE: {cfg.run_key} attempt {attempt} run_id={grpo_run_id} "
                f"val_does_execute={val_does_execute:.4f} < {min_val_does_execute} "
                "-- collapse, see docs/EXPERIMENT_LOG.md #13"
            )
            rename_degenerate_metrics(cfg.run_metrics, attempt)
            attempt_summaries.append(
                f"attempt {attempt}: degenerate (run_id={grpo_run_id}, val_does_execute={val_does_execute:.4f})"
            )
            if attempt < max_attempts:
                continue
            raise GrpoLaunchError(
                f"{cfg.run_key}: GRPO run degenerate after {max_attempts} attempt(s): "
                + "; ".join(attempt_summaries)
            )

        print_all_results(cfg.run_metrics)
        return grpo_run_id

    raise GrpoLaunchError(
        f"{cfg.run_key}: exhausted all attempts without success (max_grpo_retries={max_retries})"
    )


def update_state_after_round(
    state: dict, round_num: int, grpo_run_id: str, variant: str = ""
) -> dict:
    state["last_completed_round"] = round_num
    state["previous_run_id"] = grpo_run_id
    round_ids = state.setdefault("round_grpo_ids", {})
    round_ids[str(round_num)] = grpo_run_id
    save_trial_state(state, variant)
    print(f"OK: trial state updated (last_completed_round={round_num}, previous_run_id={grpo_run_id})")
    return state


def run_trial(
    trial: int,
    *,
    num_rounds: int,
    start_round: int | None,
    force: bool,
    dry_run: bool,
    mock_grpo_run_id: str | None,
    variant: str = "",
) -> None:
    state = load_trial_state(trial, variant)
    if start_round is None:
        start_round = state["last_completed_round"] + 1
    end_round = start_round + num_rounds - 1

    for round_num in range(start_round, end_round + 1):
        if round_num < 2:
            raise ValueError(f"round must be >= 2, got {round_num}")

        cfg = RoundConfig(
            trial=trial,
            round=round_num,
            previous_run_id=state["previous_run_id"],
            variant=variant,
        )
        print(f"\n{'=' * 60}\n=== Round {round_num} ({cfg.run_key}) ===\n{'=' * 60}")

        if cfg.run_metrics.exists() and not force:
            exported = json.loads(cfg.run_metrics.read_text())
            grpo_run_id = exported["run_id"]
            print(f"SKIP: {cfg.run_metrics.name} exists (use --force to re-run)")
            if state["last_completed_round"] < round_num:
                state = update_state_after_round(state, round_num, grpo_run_id, variant)
            continue

        preflight(cfg, state, load_trial_config(trial, variant))
        grpo_run_id = run_single_round(
            cfg,
            dry_run=dry_run,
            mock_grpo_run_id=mock_grpo_run_id,
            force=force,
        )
        state = update_state_after_round(state, round_num, grpo_run_id, variant)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-round experiment orchestrator")
    parser.add_argument("--trial", type=int, default=2)
    parser.add_argument(
        "--variant",
        type=str,
        default="",
        help=(
            "Optional named branch within a trial (e.g. 'calibrated'). Redirects "
            "rounds/buffer to rounds_{variant}/buffer_{variant} and state/config to "
            "trial_state_{variant}.json / trial_config_{variant}.yaml."
        ),
    )
    parser.add_argument("--init", action="store_true", help="Create trial_state.json")
    parser.add_argument("--round1-run-id", type=str, help="Bootstrap GRPO run id (with --init)")
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--start-round", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even if run_metrics exists")
    parser.add_argument("--dry-run", action="store_true", help="Skip GRPO and export; mock handoff")
    parser.add_argument("--mock-grpo-run-id", type=str, default=None)
    args = parser.parse_args(argv)

    if args.init:
        if not args.round1_run_id:
            parser.error("--init requires --round1-run-id")
        ckpt = checkpoint_dir_for_run_id(
            WORKSPACE, args.round1_run_id, trial=args.trial
        )
        if not ckpt.exists():
            raise FileNotFoundError(f"Round 1 checkpoint missing: {ckpt}")
        init_trial_state(args.trial, args.round1_run_id, args.variant)
        print(f"OK: initialized {trial_state_path(args.trial, args.variant)}")
        return 0

    run_trial(
        args.trial,
        num_rounds=args.num_rounds,
        start_round=args.start_round,
        force=args.force,
        dry_run=args.dry_run,
        mock_grpo_run_id=args.mock_grpo_run_id,
        variant=args.variant,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
