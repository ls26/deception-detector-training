"""Shared GRPO launch infrastructure: build the Hydra command, run it, verify
completion, export W&B metrics, and clean up intermediates.

Used by `run_ablation.py` (single runs + sweeps defined in YAML configs) and
intended to be adoptable by `run_trial.py` in a future pass. See
docs/FULL_GRPO_RUN.md and docs/iterative_training_experiments.md for the
methodology this codifies.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
OBFUSCATION_ATLAS_DIR = WORKSPACE / "obfuscation-atlas"

WANDB_ENTITY = "lilylixin6-the-harker-school"
WANDB_PROJECT = "deception_detection"
VLLM_MODELS_URL = "http://127.0.0.1:8000/v1/models"

RUNS_BASE = WORKSPACE / "results/run_results/full_grpo_runs"
GRPO_RUN_ID_RE = re.compile(r"setting up run ([a-z0-9]{8})|trigger_none_model_([a-z0-9]+)_layers_32")

# Defaults for the crash/degeneracy retry loop in `launch()` (and mirrored in
# run_trial.py's run_single_round()). See docs/EXPERIMENT_LOG.md #13: the
# degenerate collapse is a stochastic instability, so retrying with the same
# config/seed can succeed. 0.3 catches genuine collapses (val_does_execute
# dropping toward 0) while letting merely-mediocre-but-valid runs through.
DEFAULT_MAX_GRPO_RETRIES = 3
DEFAULT_MIN_VAL_DOES_EXECUTE = 0.3

# Hydra overrides applied to every GRPO run unless a config's `overrides:`
# (or top-level `nofu:`) explicitly replaces them. See plan section 7 for the
# rationale behind each of these.
GRPO_DEFAULTS: dict[str, str] = {
    "+model_preset": "llama3_8b",
    "+data/probe_dataset": "mbpp_probe",
    "grpo.model.attn_implementation": "sdpa",
    "training.save_path_base": str(WORKSPACE / "results/run_results"),
    "detector.model_cache_path": "/workspace/results/activations",
    "misalignment_eval.enabled": "false",
    "wandb.project": WANDB_PROJECT,
    "wandb.group": "full_grpo_runs",
    "++training.resume_from_checkpoint": "False",
    "corrected_detector_gradient": "false",
    "loss.kl": "0.001",
    "data.seed": "53",
    "task_args.deceptive_definition": "pass_only_shown",
}

NOFU_KEYS = (
    "data.probe_dataset.congruent_follow_up_prompts",
    "data.probe_dataset.incongruent_follow_up_prompts",
    "data.task_dataset.congruent_follow_up_prompts",
    "data.task_dataset.incongruent_follow_up_prompts",
)

REQUIRED_KEYS = ("run_key", "output_dir", "probe", "overrides")


class GrpoLaunchError(RuntimeError):
    """Raised when a GRPO run fails preflight, training, or verification."""


def _resolve(path_str: str) -> Path:
    """Resolve a path relative to experiments/, unless already absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (WORKSPACE / "experiments" / path_str)


def check_vllm(url: str = VLLM_MODELS_URL) -> None:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            status = resp.status
    except (urllib.error.URLError, OSError) as e:
        raise GrpoLaunchError(f"vLLM not reachable at {url}: {e}") from e
    if status != 200:
        raise GrpoLaunchError(f"vLLM returned status {status} at {url}")
    print("OK: vLLM reachable")


def check_disk(min_free_gb: int = 50, path: Path = WORKSPACE) -> None:
    stat = os.statvfs(path)
    free_gb = stat.f_bavail * stat.f_frsize / (1024**3)
    print(f"Disk free on {path}: {free_gb:.1f}G (need >= {min_free_gb}G)")
    if free_gb < min_free_gb:
        raise GrpoLaunchError(f"Insufficient disk space: {free_gb:.1f}G < {min_free_gb}G")


def preflight(*, check_vllm_reachable: bool = True, min_free_gb: int = 50) -> None:
    if check_vllm_reachable:
        check_vllm()
    check_disk(min_free_gb=min_free_gb)


def build_grpo_args(config: dict) -> list[str]:
    """Merge GRPO_DEFAULTS + nofu + probe settings + checkpoint + overrides
    into the Hydra override list for train_grpo_model_against_probe."""
    merged: dict[str, str] = dict(GRPO_DEFAULTS)

    nofu = config.get("nofu", True)
    if nofu:
        for key in NOFU_KEYS:
            merged[key] = "[]"

    probe = config["probe"]
    if "directory_save_prefix" not in probe:
        raise GrpoLaunchError("Config probe section missing required key: 'directory_save_prefix'")
    merged["detector.directory_save_prefix"] = probe["directory_save_prefix"]
    merged["detector.retrain"] = "true" if probe.get("retrain", False) else "false"
    # `manifest` is optional: some runs just cache/reuse the standard mbpp_probe
    # dataset via directory_save_prefix + retrain=false, with no custom mixed
    # manifest (e.g. the det1_nofu sweep).
    manifest = probe.get("manifest")
    if manifest:
        probe_manifest = _resolve(manifest)
        if not probe_manifest.exists():
            raise GrpoLaunchError(f"Probe manifest missing: {probe_manifest}")
        merged["++detector.mixed_probe_manifest_path"] = str(probe_manifest)

    checkpoint = config.get("checkpoint")
    if checkpoint:
        checkpoint_path = _resolve(checkpoint)
        if not checkpoint_path.exists():
            raise GrpoLaunchError(f"Checkpoint missing: {checkpoint_path}")
        merged["++grpo.model.reference_adapter_path"] = str(checkpoint_path)

    merged["wandb.run_name_prefix"] = config["run_key"]

    # User overrides win over every default above, including nofu/probe/checkpoint keys.
    for key, value in config.get("overrides", {}).items():
        merged[key] = str(value)

    args = [
        "python",
        "-m",
        "obfuscation_atlas.scripts.train_grpo_model_against_probe",
    ]
    for key, value in merged.items():
        args.append(f"{key}={value}")
    return args


def run_grpo_subprocess(args: list[str]) -> tuple[int, str]:
    """Run the GRPO subprocess with CUDA_VISIBLE_DEVICES=0 (vLLM occupies GPU 1
    per the workspace GPU-pinning rule). Streams output live and also returns
    the captured text so the run_id can be parsed from it afterward."""
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Phase-0's activation-extraction inference loop fragments the CUDA caching
    # allocator's memory pool; on a shared/tight GPU budget this can leave enough
    # total free memory but no contiguous block large enough for later training
    # allocations (e.g. the log_softmax reference-logprobs OOM). Expandable
    # segments let the allocator grow/shrink instead of hoarding fixed-size
    # fragmented blocks, eliminating that failure mode.
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
    for match in GRPO_RUN_ID_RE.finditer(output):
        run_id = match.group(1) or match.group(2)
        if run_id:
            return run_id
    return None


def resolve_run_id(run_key: str, output: str | None = None) -> str:
    if output:
        run_id = extract_run_id(output)
        if run_id:
            return run_id
        print(f"WARNING: could not parse run_id from GRPO output for {run_key!r}; falling back to W&B API")

    import wandb

    api = wandb.Api()
    project = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    matching = [
        run
        for run in api.runs(project, order="-created_at", per_page=50)
        if run.name.startswith(run_key)
    ]
    if not matching:
        raise GrpoLaunchError(f"No W&B run with name prefix {run_key!r} and no run_id in output")
    return matching[0].id


def run_dir_for_id(run_id: str) -> Path:
    return RUNS_BASE / f"trigger_none_model_{run_id}_layers_32"


def _final_adapter_exists(run_dir: Path) -> bool:
    return (run_dir / "checkpoints/final/adapter_model.bin").exists() or (
        run_dir / "checkpoints/final/lora/adapter_model.bin"
    ).exists()


def _count_response_steps(run_dir: Path) -> int:
    responses_dir = run_dir / "checkpoints/responses"
    if not responses_dir.is_dir():
        return 0
    return len(list(responses_dir.glob("step_*.jsonl")))


def verify_run_complete(run_id: str, *, min_responses: int = 0) -> Path:
    run_dir = run_dir_for_id(run_id)
    if not run_dir.is_dir():
        raise GrpoLaunchError(f"Run directory missing: {run_dir}")
    if not _final_adapter_exists(run_dir):
        raise GrpoLaunchError(f"No final adapter under {run_dir}/checkpoints/final/")
    n_resp = _count_response_steps(run_dir)
    if n_resp < min_responses:
        raise GrpoLaunchError(
            f"Run {run_id} has only {n_resp} response JSONLs (expected >= {min_responses})"
        )
    print(f"OK: run {run_id} verified complete (responses={n_resp})")
    return run_dir


def cleanup_intermediates(run_id: str) -> None:
    """Remove checkpoints/lora/ and checkpoints/step_*/ after verification.
    Never touches checkpoints/final/, checkpoints/responses/, or eval artifacts."""
    run_dir = run_dir_for_id(run_id)
    if not run_dir.is_relative_to(RUNS_BASE):
        raise GrpoLaunchError(f"Refusing to clean up run_dir outside {RUNS_BASE}: {run_dir}")
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

    print(f"OK: cleaned up intermediates for {run_id} (kept final/, responses/, eval artifacts)")


def check_degeneracy(metrics_path: Path, min_val_does_execute: float) -> tuple[bool, float | None]:
    """Inspect an exported run_metrics.json for a degenerate collapse (exit
    code 0 but the policy stopped producing executable code -- see
    docs/EXPERIMENT_LOG.md #13). Returns (is_degenerate, val_does_execute)."""
    data = json.loads(metrics_path.read_text())
    val_does_execute = data.get("metrics", {}).get("val_does_execute")
    if val_does_execute is None:
        print(f"WARNING: {metrics_path.name} missing metrics.val_does_execute; skipping degeneracy check")
        return False, None
    val_does_execute = float(val_does_execute)
    is_degenerate = val_does_execute < min_val_does_execute
    return is_degenerate, val_does_execute


def rename_degenerate_metrics(metrics_path: Path, attempt: int) -> Path:
    """Preserve a degenerate run's exported metrics under a `.DEGENERATE_{n}`
    suffix instead of deleting them, so the run can still be inspected later."""
    degenerate_path = metrics_path.with_name(
        metrics_path.stem + f".DEGENERATE_{attempt}" + metrics_path.suffix
    )
    metrics_path.rename(degenerate_path)
    print(f"  kept degenerate run metrics: {degenerate_path}")
    return degenerate_path


def cleanup_crashed_run_dir(output: str) -> None:
    """Best-effort cleanup of a crashed (non-zero exit) run's partial
    directory to reclaim disk space. Never removes a directory that has a
    final adapter -- if one somehow exists despite the non-zero exit, leave
    it alone for manual inspection rather than guessing."""
    run_id = extract_run_id(output)
    if not run_id:
        print("WARNING: could not parse run_id from crashed GRPO output; skipping directory cleanup")
        return
    run_dir = run_dir_for_id(run_id)
    if not run_dir.is_dir():
        print(f"  no partial run directory found for crashed run {run_id}")
        return
    if _final_adapter_exists(run_dir):
        print(f"  crashed run {run_id} unexpectedly has a final adapter; leaving {run_dir} untouched")
        return
    shutil.rmtree(run_dir)
    print(f"  removed partial run directory for crashed run {run_id}: {run_dir}")


def export_wandb_metrics(run_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python",
            "-m",
            "obfuscation_atlas.scripts.export_wandb_run_metrics",
            "--entity",
            WANDB_ENTITY,
            "--project",
            WANDB_PROJECT,
            f"--run-id={run_id}",
            f"--output={output_path}",
        ],
        cwd=OBFUSCATION_ATLAS_DIR,
        check=True,
    )
    if not output_path.exists():
        raise GrpoLaunchError(f"Export claimed success but {output_path} is missing")
    print(f"OK: exported metrics to {output_path}")


def launch(config: dict, *, min_free_gb: int = 50) -> str:
    """Full lifecycle: preflight -> build args -> run GRPO -> verify -> export
    -> cleanup. Retries on crash (non-zero exit) or degenerate collapse (exit
    0 but val_does_execute below threshold -- see docs/EXPERIMENT_LOG.md #13),
    since both are stochastic instabilities that a same-config retry can
    resolve. Returns the W&B run_id. Raises GrpoLaunchError if every attempt
    fails.

    Optional config keys: `max_grpo_retries` (default 3) and
    `min_val_does_execute` (default 0.3, ignored if `export_metrics=False`
    since there would be no metrics to inspect)."""
    for key in REQUIRED_KEYS:
        if key not in config:
            raise GrpoLaunchError(f"Config missing required key: {key!r}")

    run_key = config["run_key"]
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_path = output_dir / f"run_metrics_{run_key}.json"
    export_metrics = config.get("export_metrics", True)

    if export_metrics and run_metrics_path.exists():
        print(f"SKIP {run_key}: {run_metrics_path.name} already exists (delete it to re-run)")
        return json.loads(run_metrics_path.read_text())["run_id"]

    print(f"\n{'=' * 60}\n=== {run_key} ===\n{'=' * 60}")

    args = build_grpo_args(config)
    print(f"Manifest: {config['probe'].get('manifest') or 'none (default mbpp_probe dataset)'}")
    print(f"Checkpoint: {config.get('checkpoint') or 'base model (no reference adapter)'}")
    print(f"Output dir: {output_dir}")

    max_retries = config.get("max_grpo_retries")
    if max_retries is None:
        max_retries = DEFAULT_MAX_GRPO_RETRIES
    min_val_does_execute = config.get("min_val_does_execute")
    if min_val_does_execute is None:
        min_val_does_execute = DEFAULT_MIN_VAL_DOES_EXECUTE
    max_attempts = max_retries + 1
    attempt_summaries: list[str] = []

    for attempt in range(1, max_attempts + 1):
        preflight(min_free_gb=min_free_gb)

        start = time.time()
        exit_code, output = run_grpo_subprocess(args)
        elapsed_min = (time.time() - start) / 60
        print(
            f"GRPO subprocess finished in {elapsed_min:.1f} min "
            f"(exit={exit_code}, attempt={attempt}/{max_attempts})"
        )

        if exit_code != 0:
            print(f"CRASH: {run_key} attempt {attempt} failed with exit code {exit_code}")
            cleanup_crashed_run_dir(output)
            attempt_summaries.append(f"attempt {attempt}: crashed (exit={exit_code})")
            if attempt < max_attempts:
                continue
            raise GrpoLaunchError(
                f"{run_key}: GRPO subprocess failed after {max_attempts} attempt(s): "
                + "; ".join(attempt_summaries)
            )

        run_id = resolve_run_id(run_key, output)
        print(f"Completed {run_key} -> run_id={run_id}")
        verify_run_complete(run_id, min_responses=config.get("min_responses", 0))

        if export_metrics:
            export_wandb_metrics(run_id, run_metrics_path)
            is_degenerate, val_does_execute = check_degeneracy(run_metrics_path, min_val_does_execute)
            if is_degenerate:
                print(
                    f"DEGENERATE: {run_key} attempt {attempt} run_id={run_id} "
                    f"val_does_execute={val_does_execute:.4f} < {min_val_does_execute} "
                    "-- collapse, see docs/EXPERIMENT_LOG.md #13"
                )
                cleanup_intermediates(run_id)
                rename_degenerate_metrics(run_metrics_path, attempt)
                attempt_summaries.append(
                    f"attempt {attempt}: degenerate (run_id={run_id}, val_does_execute={val_does_execute:.4f})"
                )
                if attempt < max_attempts:
                    continue
                raise GrpoLaunchError(
                    f"{run_key}: GRPO run degenerate after {max_attempts} attempt(s): "
                    + "; ".join(attempt_summaries)
                )

        if config.get("cleanup", True):
            cleanup_intermediates(run_id)

        return run_id

    raise GrpoLaunchError(f"{run_key}: exhausted all attempts without success (max_grpo_retries={max_retries})")


if __name__ == "__main__":
    print("This module is a library. Use run_ablation.py to launch runs from a YAML config.")
    sys.exit(1)
