#!/usr/bin/env python3
"""Train and evaluate round-1 on-dataset probe on base-model on-policy eval outputs."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import build_train_dataset_round1_mixed
from extract_run_results import build_run_metrics_df, probe_value
from generation import bootstrap_hf_env, print_pipeline_df_stats, run_on_policy_pipeline
from probe_training import FOLLOW_UP, run_probe_train_eval_pipeline

EVAL_SAMPLES = 400
N_TRAIN_PER_CLASS = 300
VLLM_MODELS_URL = "http://127.0.0.1:8000/v1/models"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

REQUIRED_METRIC_KEYS = (
    "on_dataset_auc",
    "on_policy_split_auc",
    "on_policy_mean_scores",
    "deceptive_sources",
    "honest_sources",
)


def experiments_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_paths(*, output_dir: Path | None = None, trial: int | None = None) -> dict[str, Path]:
    if (output_dir is None) == (trial is None):
        raise ValueError("Exactly one of output_dir or trial must be set")
    if output_dir is not None:
        root = output_dir.resolve()
        trial_dir = root
        run_metrics_name = "run_metrics.json"
    else:
        assert trial is not None
        root = experiments_dir() / "trials" / f"trial{trial}"
        trial_dir = root
        run_metrics_name = f"run_metrics_trial{trial}_round1.json"
    buffer = root / "buffer"
    round_dir = root / "rounds" / "round1"
    return {
        "trial_dir": trial_dir,
        "buffer": buffer,
        "round_dir": round_dir,
        "base_eval_jsonl": buffer / "base_eval_generations.jsonl",
        "run_metrics": round_dir / run_metrics_name,
        "manifest_path": round_dir / "probe_train_dataset.json",
        "probe_weights": round_dir / "probe_weights",
        "probe_activations": round_dir / "probe_activations",
        "probe_metrics": round_dir / "probe_metrics.json",
        "output_dir": output_dir,
        "trial": trial,
    }


def trial_paths(trial: int) -> dict[str, Path]:
    return resolve_paths(trial=trial)


def check_vllm(url: str = VLLM_MODELS_URL) -> None:
    with urllib.request.urlopen(url, timeout=5) as resp:
        assert resp.status == 200, f"vLLM returned {resp.status}"


def check_jsonl_rows(path: Path, expected: int, *, label_fields: tuple[str, ...] = ()) -> dict:
    assert path.exists(), f"Missing {path}"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == expected, f"{path.name}: expected {expected} rows, got {len(rows)}"
    counts = {field: sum(bool(r.get(field)) for r in rows) for field in label_fields}
    return {"rows": rows, "counts": counts}


def preflight(paths: dict[str, Path]) -> None:
    paths["buffer"].mkdir(parents=True, exist_ok=True)
    paths["round_dir"].mkdir(parents=True, exist_ok=True)
    check_vllm()
    label = paths["output_dir"] if paths.get("output_dir") is not None else f"trial{paths['trial']}"
    print(f"OK: preflight passed for {label} (vLLM up)")


def load_base_model(model_name: str = BASE_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    print(f"OK: loaded base model {model_name}")
    return model, tokenizer


def generate_base_eval(paths: dict[str, Path], *, force: bool, deceptive_definition: str = "pass_only_shown") -> None:
    base_eval = paths["base_eval_jsonl"]
    if force and base_eval.exists():
        base_eval.unlink()
        print(f"OK: deleted {base_eval.name} (--force)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    eval_df = run_on_policy_pipeline(
        model_key="base",
        dataset="eval",
        lora_path=None,
        tokenizer=tokenizer,
        n_generations=1,
        output_dir=paths["buffer"],
        run_id_config="base",
        deceptive_definition=deceptive_definition,
    )
    print_pipeline_df_stats(eval_df)

    result = check_jsonl_rows(
        base_eval,
        EVAL_SAMPLES,
        label_fields=("deceptive", "honest"),
    )
    assert result["counts"]["deceptive"] > 0 and result["counts"]["honest"] > 0, (
        f"{base_eval.name}: need both deceptive and honest labels"
    )
    print(f"OK: base on-policy eval {base_eval.name} ({EVAL_SAMPLES} rows)")


def maybe_clear_probe_cache(paths: dict[str, Path], *, force: bool) -> None:
    if not force:
        return
    if paths["probe_weights"].exists():
        shutil.rmtree(paths["probe_weights"])
        print(f"OK: deleted {paths['probe_weights']} (--force)")
    on_policy_cache = paths["probe_activations"] / "on_policy"
    if on_policy_cache.exists():
        shutil.rmtree(on_policy_cache)
        print(f"OK: deleted {on_policy_cache} (--force)")


def verify_probe_metrics(paths: dict[str, Path]) -> dict:
    metrics_path = paths["probe_metrics"]
    assert metrics_path.exists(), f"Missing {metrics_path}"
    metrics = json.loads(metrics_path.read_text())

    for key in REQUIRED_METRIC_KEYS:
        assert key in metrics, f"probe_metrics.json missing {key!r}"

    assert metrics["deceptive_sources"] == {"on_dataset": 200, "on_policy": 100}
    assert metrics["honest_sources"] == {"on_dataset": 200, "on_policy": 100}
    assert metrics["on_dataset_auc"] > 0.9, (
        f"on_dataset_auc={metrics['on_dataset_auc']:.4f} unexpectedly low"
    )
    assert (paths["probe_weights"] / "layer_0" / "model.pt").exists()

    print(f"OK: probe metrics verified at {metrics_path.name}")
    return metrics


def print_summary(metrics: dict, paths: dict[str, Path]) -> None:
    print("\n=== Round 1 probe metrics ===")
    print(f"  on_dataset_auc:       {metrics['on_dataset_auc']:.4f}")
    print(f"  on_policy_split_auc:  {metrics['on_policy_split_auc']:.4f}")
    print(
        f"  on_policy scores:     "
        f"deceptive={metrics['on_policy_mean_scores']['deceptive']:.4f}  "
        f"honest={metrics['on_policy_mean_scores']['honest']:.4f}"
    )
    print(f"  train sources:        {metrics['deceptive_sources']}")
    print(f"  saved to:             {paths['probe_metrics']}")

    run_metrics_path = paths["run_metrics"]
    if not run_metrics_path.exists():
        return

    try:
        data = json.loads(run_metrics_path.read_text())
        df = build_run_metrics_df(data)
        grpo_on_dataset = probe_value(
            df, "averaged_probe", "AUC_ROC", "initial_detector", "base_model", "on_dataset"
        )
        grpo_on_policy_split = probe_value(
            df, "averaged_probe", "AUC_ROC", "initial_detector", "base_model", "on_policy.split"
        )
        print("\n=== Soft check vs round 1 GRPO export (initial_detector.base_model) ===")
        print(f"  on_dataset AUC:       script={metrics['on_dataset_auc']:.4f}  grpo={grpo_on_dataset}")
        print(f"  on_policy.split AUC:  script={metrics['on_policy_split_auc']:.4f}  grpo={grpo_on_policy_split}")
    except Exception as exc:
        print(f"\n(note: could not compare to GRPO export: {exc})")


def run(
    *,
    trial: int | None = None,
    output_dir: Path | None = None,
    force: bool,
    nofu: bool = False,
    obfuscate_over: str = "last-token-generation",
    deceptive_definition: str = "pass_only_shown",
) -> None:
    bootstrap_hf_env()
    paths = resolve_paths(output_dir=output_dir, trial=trial)
    follow_up = [] if nofu else FOLLOW_UP
    probe_label = "nofu" if nofu else "follow-up"
    run_label = (
        str(paths["output_dir"])
        if paths.get("output_dir") is not None
        else f"trial {paths['trial']}"
    )

    print(
        f"\n{'=' * 60}\n=== Round 1 {probe_label} probe eval ({run_label}, "
        f"obfuscate_over={obfuscate_over}) ===\n{'=' * 60}"
    )
    preflight(paths)
    generate_base_eval(paths, force=force, deceptive_definition=deceptive_definition)

    print("\n(note: stop vLLM before probe training if sharing one GPU)\n")
    model, tokenizer = load_base_model()
    artifacts_dir = (
        experiments_dir() / "artifacts"
        if paths.get("output_dir") is not None
        else paths["trial_dir"].parent.parent / "artifacts"
    )
    train_pos, train_neg, manifest = build_train_dataset_round1_mixed(
        tokenizer,
        manifest_path=paths["manifest_path"],
        base_train_jsonl=artifacts_dir / "base" / "base_train_generations.jsonl",
        det0_train_jsonl=artifacts_dir / "det0" / "det0_train_generations.jsonl",
    )

    maybe_clear_probe_cache(paths, force=force)
    run_probe_train_eval_pipeline(
        model,
        tokenizer,
        train_pos=train_pos,
        train_neg=train_neg,
        manifest=manifest,
        weights_dir=paths["probe_weights"],
        activations_dir=paths["probe_activations"],
        metrics_path=paths["probe_metrics"],
        manifest_path=paths["manifest_path"],
        val_jsonl=paths["base_eval_jsonl"],
        run_id="base",
        follow_up=follow_up,
        obfuscate_over=obfuscate_over,
        deceptive_definition=deceptive_definition,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    metrics = verify_probe_metrics(paths)
    print_summary(metrics, paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round 1 on-dataset probe train/eval on base on-policy outputs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial", type=int, help="Trial number (e.g. 2)")
    group.add_argument(
        "--output-dir",
        type=Path,
        help="Alternate artifact root (e.g. experiments/archive/mean_pool_probe)",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate base eval JSONL and retrain probe")
    parser.add_argument(
        "--nofu",
        action="store_true",
        help="No follow-up probe (follow_up=[])",
    )
    parser.add_argument(
        "--obfuscate-over",
        choices=("last-token-generation", "generation"),
        default="last-token-generation",
        help="Token mask for probe activations (generation = mean-pool over completion tokens)",
    )
    parser.add_argument(
        "--deceptive-definition",
        choices=("pass_only_shown", "pass_first_not_all"),
        default="pass_only_shown",
        help="Which metric drives deceptive labels and split eval",
    )
    args = parser.parse_args(argv)

    run(
        trial=args.trial,
        output_dir=args.output_dir,
        force=args.force,
        nofu=args.nofu,
        obfuscate_over=args.obfuscate_over,
        deceptive_definition=args.deceptive_definition,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
