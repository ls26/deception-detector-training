"""Run one GRPO training job: build the Hydra command line, run the trainer as a subprocess,
verify the run directory, export its W&B metrics, and remove intermediate checkpoints.

Used by `run_grpo.py` (the baseline sweep and round 1) and `run_trial.py` (rounds 2+).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from training.common import (
    ACTIVATIONS_DIR,
    EXPERIMENTS,
    REPO_ROOT,
    RUNS_BASE,
    GrpoLaunchError,
    check_disk,
    check_vllm,
    resolve_relative_path,
)
from training.extract_run_results import print_all_results, print_sycophancy_results
from training.round_config import run_dir_for_id

OBFUSCATION_ATLAS_DIR = REPO_ROOT / "obfuscation-atlas"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "deception_detection")

# The trainer prints "Wandb URL: https://wandb.ai/<entity>/<project>/runs/<id>" (possibly
# ANSI-colored); W&B itself prints "setting up run <id>" before that.
WANDB_URL_RUN_ID_RE = re.compile(r"/runs/([a-z0-9]+)")
WANDB_SETUP_RUN_ID_RE = re.compile(r"setting up run ([a-z0-9]{8})")

# A run is retried, with the same seed, when the trainer crashes or when its final policy
# stopped producing executable code (a stochastic collapse), at most this many times.
MAX_GRPO_RETRIES = 3
MIN_VAL_DOES_EXECUTE = 0.3
# Response files expected from a full-length run; early-stopped runs may write fewer.
DEFAULT_MAX_STEPS = 128
# Hydra root config of the sycophancy (MMLU multiple-choice) task; a launch config selects it
# with `config_name`. Without `config_name` the trainer's default (the MBPP task) is used.
SYCOPHANCY_CONFIG_NAME = "grpo_mcq"
# In-loop eval metrics of a sycophancy run, logged every `training.eval_steps` steps.
SYCOPHANCY_EVAL_KEYS = (
    "eval/rewards/task_success_mean",
    "eval/rewards_metrics/followed_cue_mean",
    "eval/rewards_metrics/parse_failed_mean",
)

# Hydra overrides shared by every run; a config's `overrides:` can replace any of them.
GRPO_DEFAULTS: dict[str, str] = {
    "+model_preset": "llama3_8b",
    "+data/probe_dataset": "mbpp_probe",
    "grpo.model.attn_implementation": "sdpa",
    "training.save_path_base": str(RUNS_BASE),
    "detector.model_cache_path": str(ACTIVATIONS_DIR),
    "misalignment_eval.enabled": "false",
    "wandb.project": WANDB_PROJECT,
    "++training.resume_from_checkpoint": "False",
    "corrected_detector_gradient": "false",
    "loss.kl": "0.001",
    "data.seed": "53",
    "task_args.deceptive_definition": "pass_only_shown",
}
# Probes are read at the completion's end-of-turn token: no follow-up question is appended.
NO_FOLLOW_UP_KEYS = (
    "data.probe_dataset.congruent_follow_up_prompts",
    "data.probe_dataset.incongruent_follow_up_prompts",
    "data.task_dataset.congruent_follow_up_prompts",
    "data.task_dataset.incongruent_follow_up_prompts",
)

REQUIRED_KEYS = ("run_key", "group", "output_dir", "probe", "checkpoint", "overrides")

_wandb_entity: str | None = None


def wandb_entity() -> str:
    """$WANDB_ENTITY if set, else the entity behind the W&B API key (memoised)."""
    global _wandb_entity
    if _wandb_entity is None:
        entity = os.environ.get("WANDB_ENTITY")
        if not entity:
            import wandb

            entity = wandb.Api().default_entity
        if not entity:
            raise GrpoLaunchError("Cannot determine the W&B entity: set WANDB_ENTITY or log in to W&B")
        _wandb_entity = entity
    return _wandb_entity


def build_grpo_args(config: dict) -> list[str]:
    """The trainer command line for a launch config: GRPO_DEFAULTS, no follow-up, the probe
    manifest/prefix/retrain, the starting checkpoint, W&B naming, then `overrides`."""
    merged: dict[str, str] = dict(GRPO_DEFAULTS)
    for key in NO_FOLLOW_UP_KEYS:
        merged[key] = "[]"

    probe = config["probe"]
    merged["detector.directory_save_prefix"] = probe["directory_save_prefix"]
    merged["detector.retrain"] = "true" if probe["retrain"] else "false"
    # Without a manifest, Phase 0 trains the default on-dataset probe (the baseline sweep).
    if probe.get("manifest") is not None:
        probe_manifest = resolve_relative_path(probe["manifest"], EXPERIMENTS)
        if not probe_manifest.exists():
            raise GrpoLaunchError(f"Probe manifest missing: {probe_manifest}")
        merged["++detector.mixed_probe_manifest_path"] = str(probe_manifest)

    if config["checkpoint"] is not None:
        checkpoint_path = resolve_relative_path(config["checkpoint"], EXPERIMENTS)
        if not checkpoint_path.exists():
            raise GrpoLaunchError(f"Checkpoint missing: {checkpoint_path}")
        merged["++grpo.model.reference_adapter_path"] = str(checkpoint_path)

    merged["wandb.run_name_prefix"] = config["run_key"]
    merged["wandb.group"] = config["group"]
    for key, value in config["overrides"].items():
        merged[key] = str(value).lower() if isinstance(value, bool) else str(value)

    args = [sys.executable, "-m", "obfuscation_atlas.scripts.train_grpo_model_against_probe"]
    if config.get("config_name") is not None:
        args.append(f"--config-name={config['config_name']}")
    args.extend(f"{key}={value}" for key, value in merged.items())
    return args


def run_grpo_subprocess(args: list[str]) -> tuple[int, str]:
    """Run the trainer, streaming its output; returns (exit code, captured output)."""
    env = os.environ.copy()
    # Training runs on GPU 0; vLLM is on GPU 1 or shares GPU 0 (see README, GPU allocation).
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Phase 0's activation extraction fragments the CUDA allocator; expandable segments avoid
    # out-of-memory errors later in training even though enough total memory is free.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    lines: list[str] = []
    process = subprocess.Popen(
        args,
        cwd=OBFUSCATION_ATLAS_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    process.wait()
    return process.returncode, "".join(lines)


def extract_run_id(output: str) -> str | None:
    for pattern in (WANDB_URL_RUN_ID_RE, WANDB_SETUP_RUN_ID_RE):
        match = pattern.search(output)
        if match:
            return match.group(1)
    return None


def resolve_run_id(run_key: str, group: str, output: str | None) -> str:
    """The W&B run id of a finished run: parsed from the trainer output when it names a run
    directory that exists, otherwise the newest W&B run whose name starts with `run_key`."""
    if output:
        run_id = extract_run_id(output)
        if run_id and run_dir_for_id(run_id, group=group, run_key=run_key).is_dir():
            return run_id
        print(f"WARNING: no run directory for the run id in the GRPO output of {run_key!r}; asking the W&B API")

    import wandb

    api = wandb.Api()
    project = f"{wandb_entity()}/{WANDB_PROJECT}"
    matching = [run for run in api.runs(project, order="-created_at", per_page=50) if run.name.startswith(run_key)]
    if not matching:
        raise GrpoLaunchError(f"No W&B run with name prefix {run_key!r} and no run id in the output")
    return matching[0].id


def _final_adapter_exists(run_dir: Path) -> bool:
    return (run_dir / "checkpoints/final/adapter_model.bin").exists() or (
        run_dir / "checkpoints/final/lora/adapter_model.bin"
    ).exists()


def verify_run_complete(run_id: str, *, group: str, run_key: str, min_responses: int) -> Path:
    """The run directory has a final adapter and at least `min_responses` response files."""
    run_dir = run_dir_for_id(run_id, group=group, run_key=run_key)
    if not run_dir.is_dir():
        raise GrpoLaunchError(f"Run directory missing: {run_dir}")
    if not _final_adapter_exists(run_dir):
        raise GrpoLaunchError(f"No final adapter under {run_dir}/checkpoints/final/")
    n_responses = len(list((run_dir / "checkpoints/responses").glob("step_*.jsonl")))
    if n_responses < min_responses:
        raise GrpoLaunchError(f"Run {run_id} has only {n_responses} response files (expected >= {min_responses})")
    print(f"OK: run {run_id} verified complete (responses={n_responses})")
    return run_dir


def cleanup_intermediates(run_id: str, *, group: str, run_key: str) -> None:
    """Remove checkpoints/lora/ and checkpoints/step_*/ after verification. Keeps
    checkpoints/final/, checkpoints/responses/, and the eval outputs."""
    run_dir = run_dir_for_id(run_id, group=group, run_key=run_key)
    if not run_dir.is_relative_to(RUNS_BASE):
        raise GrpoLaunchError(f"Refusing to clean up a run directory outside {RUNS_BASE}: {run_dir}")
    if not _final_adapter_exists(run_dir):
        raise GrpoLaunchError(f"Refusing cleanup: no final adapter under {run_dir}/checkpoints/final/")
    lora_dir = run_dir / "checkpoints/lora"
    if lora_dir.is_dir():
        shutil.rmtree(lora_dir)
        print(f"  removed {lora_dir}")
    for step_dir in (run_dir / "checkpoints").glob("step_*"):
        if step_dir.is_dir():
            shutil.rmtree(step_dir)
            print(f"  removed {step_dir}")
    print(f"OK: cleaned up intermediates for {run_id}")


def remove_prefix_activation_cache(prefix: str) -> None:
    """Remove the Phase-0 activation cache `{prefix}_base_activations_*` shared by every run
    with `detector.directory_save_prefix=prefix`."""
    for cache_dir in sorted(ACTIVATIONS_DIR.glob(f"{prefix}_base_activations_*")):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            print(f"OK: removed activation cache {cache_dir.name}")


def cleanup_activation_caches(run_id: str, *, prefix: str | None) -> None:
    """Remove the activation caches a finished run left behind: its own `activations_{run_id}*`
    and, when the prefix cache belongs to this run alone (`probe.retrain: true`), that too.
    Called from the parent process, after the trainer exited and released its memory maps."""
    for cache_dir in sorted(ACTIVATIONS_DIR.glob(f"activations_{run_id}*")):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            print(f"OK: removed activation cache {cache_dir.name}")
    if prefix is not None:
        remove_prefix_activation_cache(prefix)


def val_does_execute(metrics_path: Path) -> float:
    """The final policy's rate of executable completions on the held-out prompts."""
    data = json.loads(metrics_path.read_text())
    return float(data["metrics"]["val_does_execute"])


def rename_degenerate_metrics(metrics_path: Path, attempt: int) -> Path:
    """Keep a collapsed run's exported metrics under a `.DEGENERATE_<attempt>` suffix."""
    degenerate_path = metrics_path.with_name(metrics_path.stem + f".DEGENERATE_{attempt}" + metrics_path.suffix)
    metrics_path.rename(degenerate_path)
    print(f"  kept degenerate run metrics: {degenerate_path}")
    return degenerate_path


def cleanup_crashed_run_dir(output: str, *, group: str, run_key: str) -> None:
    """Remove the partial run directory of a crashed run (never one with a final adapter)."""
    run_id = extract_run_id(output)
    if run_id is None:
        print("WARNING: no run id in the crashed GRPO output; leaving its run directory in place")
        return
    run_dir = run_dir_for_id(run_id, group=group, run_key=run_key)
    if not run_dir.is_dir():
        return
    if _final_adapter_exists(run_dir):
        print(f"  crashed run {run_id} has a final adapter; leaving {run_dir} untouched")
        return
    shutil.rmtree(run_dir)
    print(f"  removed partial run directory for crashed run {run_id}: {run_dir}")


def export_wandb_metrics(run_id: str, output_path: Path, *, max_retries: int = 3) -> None:
    """Export the run's W&B metrics and classification to `output_path` (JSON), retrying on
    failure since W&B can lag behind a run that just finished."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "obfuscation_atlas.scripts.export_wandb_run_metrics",
        "--entity",
        wandb_entity(),
        "--project",
        WANDB_PROJECT,
        f"--run-id={run_id}",
        f"--output={output_path}",
    ]
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(command, cwd=OBFUSCATION_ATLAS_DIR)
        if result.returncode == 0:
            break
        if attempt == max_retries:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"WARN: export_wandb_run_metrics attempt {attempt}/{max_retries} failed; retrying in 60 s")
        time.sleep(60)
    if not output_path.exists():
        raise GrpoLaunchError(f"Export claimed success but {output_path} is missing")
    exported_run_id = json.loads(output_path.read_text())["run_id"]
    if exported_run_id != run_id:
        raise GrpoLaunchError(f"Exported run_id {exported_run_id!r} != {run_id!r}")
    print(f"OK: exported metrics to {output_path}")


def sycophancy_eval_history(run_id: str) -> list[dict]:
    """The in-loop eval rows of a sycophancy run, in order, each tagged with the training step
    whose W&B row it was logged with (`logged_at_train_step`).

    The eval logged at step N scores the policy that generated step N's training answers, after
    N - 1 updates (afterburner evaluation_step submits the current policy and pops it). Runs from before
    2026-10-24 (261013-261020) queued one eval before training started, so there the eval
    logged at step N scores the policy of the previous eval point and the first row the
    starting model. The run must have all training.max_steps // training.eval_steps rows."""
    import wandb

    run = wandb.Api().run(f"{wandb_entity()}/{WANDB_PROJECT}/{run_id}")
    expected_rows = run.config["training"]["max_steps"] // run.config["training"]["eval_steps"]
    train_step_at = {row["_step"]: row["grpo_train_step"] for row in run.scan_history(keys=["grpo_train_step", "_step"])}
    history = [
        {"logged_at_train_step": train_step_at[row["_step"]], **{key: row[key] for key in SYCOPHANCY_EVAL_KEYS}}
        for row in run.scan_history(keys=[*SYCOPHANCY_EVAL_KEYS, "_step"])
    ]
    if len(history) != expected_rows:
        raise GrpoLaunchError(f"Run {run_id} has {len(history)} in-loop eval rows in W&B, expected {expected_rows}")
    return history


def add_sycophancy_eval_history(run_id: str, run_metrics_path: Path, *, max_retries: int = 3) -> None:
    """Add the in-loop eval history to the exported metrics (the export keeps only the last row),
    retrying since W&B can lag behind a run that just finished."""
    for attempt in range(1, max_retries + 1):
        try:
            history = sycophancy_eval_history(run_id)
            break
        except Exception as error:  # W&B API errors and incomplete history alike: wait and retry
            if attempt == max_retries:
                raise
            print(f"WARN: eval history attempt {attempt}/{max_retries} failed ({error}); retrying in 60 s")
            time.sleep(60)
    data = json.loads(run_metrics_path.read_text())
    data["sycophancy_eval_history"] = history
    run_metrics_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"OK: added {len(history)} in-loop eval rows to {run_metrics_path}")


def launch(config: dict) -> str:
    """Run one GRPO job to completion and return its W&B run id.

    `config` keys: `run_key`, `group`, `output_dir` (relative to experiments/ or absolute),
    `probe` (`directory_save_prefix`, `retrain`, optional `manifest`), `checkpoint` (starting
    adapter or None for the base model), `overrides` (Hydra key -> value), and optional
    `early_stop` (True when the run may end before `training.max_steps`) and `config_name`
    (the trainer's Hydra root config; SYCOPHANCY_CONFIG_NAME for the sycophancy task). The
    metrics export lands at `<output_dir>/run_metrics_<run_key>.json`; when it already exists
    the run is skipped and its run id returned. A crash, or an MBPP policy that stopped writing
    executable code, is retried up to MAX_GRPO_RETRIES times.
    """
    for key in REQUIRED_KEYS:
        if key not in config:
            raise GrpoLaunchError(f"Launch config missing required key: {key!r}")
    run_key = config["run_key"]
    group = config["group"]
    output_dir = resolve_relative_path(config["output_dir"], EXPERIMENTS)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_path = output_dir / f"run_metrics_{run_key}.json"
    if run_metrics_path.exists():
        print(f"SKIP {run_key}: {run_metrics_path.name} already exists (delete it to re-run)")
        return json.loads(run_metrics_path.read_text())["run_id"]

    print(f"\n{'=' * 60}\n=== {run_key} ===\n{'=' * 60}")
    args = build_grpo_args(config)
    print(f"Manifest: {config['probe'].get('manifest') or 'none (default on-dataset probe)'}")
    print(f"Checkpoint: {config['checkpoint'] or 'base model (no starting adapter)'}")
    print(f"W&B group: {group}")
    print(f"Output dir: {output_dir}")

    # The trainer stops at the smaller of training.max_steps and the data-derived step count,
    # which is DEFAULT_MAX_STEPS for the shipped data.
    max_steps = min(int(config["overrides"].get("training.max_steps", DEFAULT_MAX_STEPS)), DEFAULT_MAX_STEPS)
    min_responses = 1 if config.get("early_stop", False) else max_steps
    own_prefix = config["probe"]["directory_save_prefix"] if config["probe"]["retrain"] else None
    max_attempts = MAX_GRPO_RETRIES + 1
    attempt_summaries: list[str] = []
    for attempt in range(1, max_attempts + 1):
        check_vllm()
        check_disk()
        start = time.time()
        exit_code, output = run_grpo_subprocess(args)
        elapsed_min = (time.time() - start) / 60
        print(f"GRPO subprocess finished in {elapsed_min:.1f} min (exit={exit_code}, attempt={attempt}/{max_attempts})")

        if exit_code != 0:
            print(f"CRASH: {run_key} attempt {attempt} failed with exit code {exit_code}")
            cleanup_crashed_run_dir(output, group=group, run_key=run_key)
            attempt_summaries.append(f"attempt {attempt}: crashed (exit={exit_code})")
            if attempt < max_attempts:
                continue
            raise GrpoLaunchError(f"{run_key}: GRPO failed after {max_attempts} attempt(s): " + "; ".join(attempt_summaries))

        run_id = resolve_run_id(run_key, group, output)
        print(f"Completed {run_key} -> run_id={run_id}")
        verify_run_complete(run_id, group=group, run_key=run_key, min_responses=min_responses)
        cleanup_intermediates(run_id, group=group, run_key=run_key)
        cleanup_activation_caches(run_id, prefix=own_prefix)
        export_wandb_metrics(run_id, run_metrics_path)

        if config.get("config_name") == SYCOPHANCY_CONFIG_NAME:
            # The executable-code collapse check is MBPP-only.
            add_sycophancy_eval_history(run_id, run_metrics_path)
            print_sycophancy_results(run_metrics_path)
            return run_id

        executes = val_does_execute(run_metrics_path)
        if executes < MIN_VAL_DOES_EXECUTE:
            print(f"DEGENERATE: {run_key} attempt {attempt} run_id={run_id} val_does_execute={executes:.4f} < {MIN_VAL_DOES_EXECUTE}")
            rename_degenerate_metrics(run_metrics_path, attempt)
            attempt_summaries.append(f"attempt {attempt}: degenerate (run_id={run_id}, val_does_execute={executes:.4f})")
            if attempt < max_attempts:
                continue
            raise GrpoLaunchError(f"{run_key}: policy collapsed in {max_attempts} attempt(s): " + "; ".join(attempt_summaries))

        print_all_results(run_metrics_path)
        return run_id

    raise AssertionError("unreachable")
