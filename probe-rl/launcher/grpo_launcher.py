"""Run one GRPO training job from a launch config: build the trainer's command line, run it, check the run folder,
export its W&B metrics, and only then write `<output_dir>/run_metrics_<run_key>.json`.

That file marks the run as done: a launch whose file exists is skipped. A crash keeps the run folder (its saved
training states stay resumable) and marks it with a CRASHED file; such a folder does not stop a later launch of the same
run key, nor does the folder of an MBPP attempt whose policy collapsed. Any other run folder without that file stops the
launch instead of training again: the run is still training (perhaps on another pod), was killed before the launcher
could mark it (a pod reboot), or finished training but its checks or export failed, which `finish_run` completes
(`python -m launcher.run_grpo <config> --finish-run-id <id>`).
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

from launcher.common import ACTIVATIONS_DIR, OBFUSCATION_ATLAS_DIR, RUNS_BASE, GrpoLaunchError, check_disk, check_vllm, run_dir

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "deception_detection")

# The trainer prints "Wandb URL: https://wandb.ai/<entity>/<project>/runs/<id>" (possibly ANSI-colored); W&B itself
# prints "setting up run <id>" before that, so a run that crashes early can still be named.
WANDB_URL_RUN_ID_RE = re.compile(r"/runs/([a-z0-9]+)")
WANDB_SETUP_RUN_ID_RE = re.compile(r"setting up run ([a-z0-9]{8})")

# MBPP: a run whose final policy stopped writing executable code (a stochastic collapse) is run again with the same
# seed, at most this many more times. A crash is never retried.
MAX_COLLAPSE_RETRIES = 3
# Written into the run folder of a crashed run; a marked folder does not stop a later launch of its run key.
CRASHED_MARKER = "CRASHED"
MIN_VAL_DOES_EXECUTE = 0.3
# Hydra root config of the sycophancy (MMLU multiple-choice) task, selected with `config_name`. Without it the trainer
# runs its default task, MBPP.
SYCOPHANCY_CONFIG_NAME = "grpo_mcq"
# In-loop eval metrics of a sycophancy run, logged every `training.eval_steps` steps.
SYCOPHANCY_EVAL_KEYS = (
    "eval/rewards/task_success_mean",
    "eval/rewards_metrics/followed_cue_mean",
    "eval/rewards_metrics/parse_failed_mean",
)
# The end-of-run eval metrics a finished run's export must hold.
SYCOPHANCY_FINAL_EVAL_KEYS = (
    "final_eval/rewards/task_success",
    "final_eval/rewards_metrics/followed_cue",
    "final_eval/rewards_metrics/parse_failed",
)
MBPP_FINAL_EVAL_KEYS = ("val_does_execute",)

# Hydra overrides shared by every run, after the probe dataset group and the run's model preset (build_grpo_args); a
# config's `overrides:` can replace any of them.
GRPO_DEFAULTS: dict[str, str] = {
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
# MBPP's probe dataset group; sycophancy configs replace it in their overrides.
DEFAULT_PROBE_DATASET = "mbpp_probe"
# Probes are read at the completion itself: no follow-up question is appended.
NO_FOLLOW_UP_KEYS = (
    "data.probe_dataset.congruent_follow_up_prompts",
    "data.probe_dataset.incongruent_follow_up_prompts",
    "data.task_dataset.congruent_follow_up_prompts",
    "data.task_dataset.incongruent_follow_up_prompts",
)

REQUIRED_KEYS = {"run_key", "group", "output_dir", "probe", "checkpoint", "overrides"}
OPTIONAL_KEYS = {"config_name"}
BATCH_KEYS = {"training.batch_size", "training.grad_accum_steps"}

# Qwen3.5-9B answers per forward/backward pass, by the training GPU: what fits next to a vLLM server at 0.40 of the
# GPU with 1024-token answers (qwen_experiment/261123_9b_speed/results.md). Every size keeps 64 answers per optimizer
# step, so the update is the same; only speed and memory differ. Any other GPU stops the launch.
QWEN_9B_ANSWERS_PER_PASS = {"H100": 4, "H200": 8}

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


def check_config(config: dict) -> None:
    """The launch config's structure: its keys, absolute paths, a probe trained at the start of the run, and the
    overrides every run must set."""
    if not REQUIRED_KEYS <= config.keys() <= REQUIRED_KEYS | OPTIONAL_KEYS:
        raise GrpoLaunchError(
            f"Launch config keys {sorted(config)}: required {sorted(REQUIRED_KEYS)}, optional {sorted(OPTIONAL_KEYS)}"
        )
    if set(config["probe"]) != {"directory_save_prefix", "retrain"} or config["probe"]["retrain"] is not True:
        raise GrpoLaunchError(f"probe must be {{directory_save_prefix: ..., retrain: true}}, got {config['probe']}")
    if not Path(config["output_dir"]).is_absolute():
        raise GrpoLaunchError(f"output_dir must be an absolute path, got {config['output_dir']!r}")
    if config["checkpoint"] is not None and not Path(config["checkpoint"]).is_absolute():
        raise GrpoLaunchError(f"checkpoint must be null or an absolute path, got {config['checkpoint']!r}")
    for key in ("+model_preset", "training.max_steps"):
        if key not in config["overrides"]:
            raise GrpoLaunchError(f"overrides must set {key}")
    batch_keys_set = BATCH_KEYS & config["overrides"].keys()
    if batch_keys_set and batch_keys_set != BATCH_KEYS:
        raise GrpoLaunchError(f"overrides set {sorted(batch_keys_set)}: set both {sorted(BATCH_KEYS)} or neither")


def training_gpu_name() -> str:
    """nvidia-smi's name of the GPU the trainer will run on: CUDA_VISIBLE_DEVICES, or GPU 0 as in run_grpo_subprocess."""
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", device],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def qwen_9b_answers_per_pass(gpu_name: str) -> int:
    """Qwen3.5-9B answers per pass for a GPU name such as "NVIDIA H200"; stops on a GPU not in the table."""
    matches = [answers for gpu, answers in QWEN_9B_ANSWERS_PER_PASS.items() if gpu in gpu_name]
    if len(matches) != 1:
        raise GrpoLaunchError(
            f"No Qwen3.5-9B answers-per-pass setting for GPU {gpu_name!r} (known: {sorted(QWEN_9B_ANSWERS_PER_PASS)}); "
            "set training.batch_size and training.grad_accum_steps in the run's overrides"
        )
    return matches[0]


def build_grpo_args(config: dict) -> list[str]:
    """The trainer command line for a launch config: the probe dataset group, the model preset, GRPO_DEFAULTS, no
    follow-up, the probe prefix, the starting checkpoint, W&B naming, then `overrides`. A Qwen3.5-9B run whose
    overrides don't set the batch gets the answers per pass of its GPU."""
    overrides = config["overrides"]
    # The model preset comes after the probe dataset group so that a preset's data settings (the Qwen3.5 presets'
    # length limits) win over the group's.
    merged: dict[str, str] = {"+data/probe_dataset": DEFAULT_PROBE_DATASET, "+model_preset": overrides["+model_preset"]}
    merged.update(GRPO_DEFAULTS)
    for key in NO_FOLLOW_UP_KEYS:
        merged[key] = "[]"
    merged["detector.directory_save_prefix"] = config["probe"]["directory_save_prefix"]
    merged["detector.retrain"] = "true"
    if config["checkpoint"] is not None:
        if not Path(config["checkpoint"]).exists():
            raise GrpoLaunchError(f"Checkpoint missing: {config['checkpoint']}")
        merged["++grpo.model.reference_adapter_path"] = str(config["checkpoint"])
    merged["wandb.run_name_prefix"] = config["run_key"]
    merged["wandb.group"] = config["group"]
    for key, value in overrides.items():
        merged[key] = str(value).lower() if isinstance(value, bool) else str(value)

    if merged["+model_preset"] == "qwen3_5_9b" and not BATCH_KEYS & overrides.keys():
        gpu_name = training_gpu_name()
        answers_per_pass = qwen_9b_answers_per_pass(gpu_name)
        merged["training.batch_size"] = str(answers_per_pass)
        merged["training.grad_accum_steps"] = str(64 // answers_per_pass)
        print(f"Qwen3.5-9B on {gpu_name}: {answers_per_pass} answers per forward/backward pass")

    args = [sys.executable, "-m", "obfuscation_atlas.scripts.train_grpo_model_against_probe"]
    if config.get("config_name") is not None:
        args.append(f"--config-name={config['config_name']}")
    args.extend(f"{key}={value}" for key, value in merged.items())
    return args


def run_grpo_subprocess(args: list[str]) -> tuple[int, str]:
    """Run the trainer, streaming its output; returns (exit code, captured output)."""
    env = os.environ.copy()
    # Training runs on GPU 0 unless the caller chose one; vLLM is on GPU 1 or shares GPU 0 (README, GPU allocation).
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Phase 0's activation extraction fragments the CUDA allocator; expandable segments avoid
    # out-of-memory errors later in training even though enough total memory is free.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    lines: list[str] = []
    process = subprocess.Popen(
        args, cwd=OBFUSCATION_ATLAS_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
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


def existing_run_dirs(group: str, run_key: str) -> list[Path]:
    """The run folders of this group and run key, of any attempt (`<run_key>_<8-character W&B id>`)."""
    group_dir = RUNS_BASE / group
    if not group_dir.is_dir():
        return []
    pattern = re.compile(re.escape(run_key) + r"_[a-z0-9]{8}")
    return sorted(path for path in group_dir.iterdir() if path.is_dir() and pattern.fullmatch(path.name))


def final_adapter_exists(folder: Path) -> bool:
    return (folder / "checkpoints/final/adapter_model.bin").exists() or (
        folder / "checkpoints/final/lora/adapter_model.bin"
    ).exists()


def resumed_from_step(config: dict) -> int:
    """The step a run starts at: that of the saved state its `grpo.logging.resume_state_dir` names, else 0."""
    resume_state_dir = config["overrides"].get("grpo.logging.resume_state_dir")
    if resume_state_dir is None:
        return 0
    return int(json.loads((Path(resume_state_dir) / "state.json").read_text())["step"])


def expected_steps(config: dict) -> int:
    """The number of response files a finished run writes: its `training.max_steps`, less the steps of the state it
    resumed from. The trainer also stops at its data-derived count (num_epochs x steps per epoch); a run that count
    stopped earlier fails verification (raise training.num_epochs, as 261117 did)."""
    return int(config["overrides"]["training.max_steps"]) - resumed_from_step(config)


def verify_run_complete(folder: Path, expected_responses: int) -> None:
    """The run folder has a final adapter and exactly `expected_responses` response files."""
    if not folder.is_dir():
        raise GrpoLaunchError(f"Run directory missing: {folder}")
    if not final_adapter_exists(folder):
        raise GrpoLaunchError(f"No final adapter under {folder}/checkpoints/final/")
    n_responses = len(list((folder / "checkpoints/responses").glob("step_*.jsonl")))
    if n_responses != expected_responses:
        raise GrpoLaunchError(f"{folder} has {n_responses} response files, expected {expected_responses}")
    print(f"OK: {folder.name} verified complete (responses={n_responses})")


def cleanup_intermediates(folder: Path) -> None:
    """Remove checkpoints/lora/ and all but the newest checkpoints/step_*/ after verification. Keeps
    checkpoints/final/, checkpoints/adapters/, checkpoints/responses/, the eval outputs, and the newest
    saved full training state (adapter, optimizer, scheduler), from which a run can be extended. That state is
    saved every training.save_steps steps, so it is the last step's only when save_steps divides the step count
    (a 128-step run with save_steps 10 keeps step_120)."""
    if not folder.is_relative_to(RUNS_BASE):
        raise GrpoLaunchError(f"Refusing to clean up a run directory outside {RUNS_BASE}: {folder}")
    if not final_adapter_exists(folder):
        raise GrpoLaunchError(f"Refusing cleanup: no final adapter under {folder}/checkpoints/final/")
    lora_dir = folder / "checkpoints/lora"
    if lora_dir.is_dir():
        shutil.rmtree(lora_dir)
        print(f"  removed {lora_dir}")
    step_dirs = sorted(
        (path for path in (folder / "checkpoints").glob("step_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("step_")),
    )
    for step_dir in step_dirs[:-1]:
        shutil.rmtree(step_dir)
        print(f"  removed {step_dir}")
    if step_dirs:
        print(f"  kept {step_dirs[-1]} (the last full training state)")
    print(f"OK: cleaned up intermediates of {folder.name}")


def cleanup_activation_caches(run_id: str, prefix: str) -> None:
    """Remove the activation caches a finished run left behind: its own `activations_{run_id}*` and its start-of-run
    probe's `{prefix}_base_activations_*`. Called from the parent process, after the trainer exited and released its
    memory maps."""
    for cache_dir in [*ACTIVATIONS_DIR.glob(f"activations_{run_id}*"), *ACTIVATIONS_DIR.glob(f"{prefix}_base_activations_*")]:
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            print(f"OK: removed activation cache {cache_dir.name}")


def export_wandb_metrics(run_id: str, output_path: Path, required_keys: tuple[str, ...], *, max_retries: int = 3) -> None:
    """Export the run's W&B metrics and classification to `output_path` (JSON), retrying on failure, or when a
    required metric (one of the run's final metrics) is missing or empty in the export, since W&B can lag behind a
    run that just finished (with wandb 0.31 the end-of-run metrics reach the server after the run exits)."""
    command = [
        sys.executable, "-m", "obfuscation_atlas.scripts.export_wandb_run_metrics",
        "--entity", wandb_entity(), "--project", WANDB_PROJECT, f"--run-id={run_id}", f"--output={output_path}",
    ]
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(command, cwd=OBFUSCATION_ATLAS_DIR)
        if result.returncode == 0:
            if not output_path.exists():
                raise GrpoLaunchError(f"Export claimed success but {output_path} is missing")
            metrics = json.loads(output_path.read_text())["metrics"]
            missing = [key for key in required_keys if metrics.get(key) is None]
            if not missing:
                break
            if attempt == max_retries:
                raise GrpoLaunchError(f"{run_id}: the export still lacks {missing} after {max_retries} attempts")
            print(f"WARN: export attempt {attempt}/{max_retries} lacks {missing}; retrying in 60 s")
        elif attempt == max_retries:
            raise GrpoLaunchError(f"{run_id}: export_wandb_run_metrics failed {max_retries} times (exit {result.returncode})")
        else:
            print(f"WARN: export_wandb_run_metrics attempt {attempt}/{max_retries} failed; retrying in 60 s")
        time.sleep(60)
    exported_run_id = json.loads(output_path.read_text())["run_id"]
    if exported_run_id != run_id:
        raise GrpoLaunchError(f"Exported run_id {exported_run_id!r} != {run_id!r}")
    print(f"OK: exported metrics to {output_path}")


def sycophancy_eval_history(run_id: str, steps_before: int) -> list[dict]:
    """The in-loop eval rows of a sycophancy run, in order, each tagged with the training step
    whose W&B row it was logged with (`logged_at_train_step`).

    The eval logged at step N scores the policy that generated step N's training answers, after
    N - 1 updates (afterburner evaluation_step submits the current policy and pops it). Runs from before
    2026-10-24 (261013-261020) queued one eval before training started, so there the eval
    logged at step N scores the policy of the previous eval point and the first row the
    starting model. The run must have a row for every multiple of training.eval_steps from steps_before (the step
    a resumed run starts at, else 0) up to training.max_steps."""
    import wandb

    run = wandb.Api().run(f"{wandb_entity()}/{WANDB_PROJECT}/{run_id}")
    eval_steps = run.config["training"]["eval_steps"]
    expected_rows = run.config["training"]["max_steps"] // eval_steps - steps_before // eval_steps
    train_step_at = {row["_step"]: row["grpo_train_step"] for row in run.scan_history(keys=["grpo_train_step", "_step"])}
    history = [
        {"logged_at_train_step": train_step_at[row["_step"]], **{key: row[key] for key in SYCOPHANCY_EVAL_KEYS}}
        for row in run.scan_history(keys=[*SYCOPHANCY_EVAL_KEYS, "_step"])
    ]
    if len(history) != expected_rows:
        raise GrpoLaunchError(f"Run {run_id} has {len(history)} in-loop eval rows in W&B, expected {expected_rows}")
    return history


def add_sycophancy_eval_history(run_id: str, metrics_path: Path, steps_before: int, *, max_retries: int = 3) -> None:
    """Add the in-loop eval history to the exported metrics (the export keeps only the last row), retrying while W&B
    still lacks rows of a run that just finished."""
    for attempt in range(1, max_retries + 1):
        try:
            history = sycophancy_eval_history(run_id, steps_before)
            break
        except Exception as error:  # W&B API errors and incomplete history alike: wait and retry, then raise
            if attempt == max_retries:
                raise
            print(f"WARN: eval history attempt {attempt}/{max_retries} failed ({error}); retrying in 60 s")
            time.sleep(60)
    data = json.loads(metrics_path.read_text())
    data["sycophancy_eval_history"] = history
    metrics_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"OK: added {len(history)} in-loop eval rows to {metrics_path}")


def print_sycophancy_results(metrics_path: Path) -> None:
    """Print one exported sycophancy run: the post-training eval on the held-out questions (accuracy, cue agreement
    rate, parse failures) and the in-loop eval history."""
    data = json.loads(metrics_path.read_text())
    metrics = data["metrics"]
    print(f"run_id={data['run_id']}")
    print("\nFinal eval (held-out questions, one sample each):")
    for key in SYCOPHANCY_FINAL_EVAL_KEYS:
        print(f"  {key}: {metrics[key]:.4f}")
    print("\nIn-loop eval (accuracy / cue agreement rate / parse failures; each row scores the policy that")
    print("generated that step's training answers; in runs from before 2026-10-24, 261013-261020, the")
    print("policy from the previous eval point, the first row the starting model):")
    for row in data["sycophancy_eval_history"]:
        print(
            f"  logged at step {row['logged_at_train_step']}: {row['eval/rewards/task_success_mean']:.4f} / "
            f"{row['eval/rewards_metrics/followed_cue_mean']:.4f} / "
            f"{row['eval/rewards_metrics/parse_failed_mean']:.4f}"
        )


def run_metrics_path_of(config: dict) -> Path:
    return Path(config["output_dir"]) / f"run_metrics_{config['run_key']}.json"


def finish_run(config: dict, run_id: str) -> bool:
    """The steps after training: verify the run folder, clean it up, export the metrics (plus, for sycophancy, the
    in-loop eval history) to a partial file, check them, and only then move that file to `run_metrics_<run_key>.json`.
    Returns False when an MBPP run's policy collapsed; its metrics are kept as `...DEGENERATE_<run id>.json`."""
    folder = run_dir(config["group"], config["run_key"], run_id)
    verify_run_complete(folder, expected_steps(config))
    cleanup_intermediates(folder)
    cleanup_activation_caches(run_id, config["probe"]["directory_save_prefix"])
    final_path = run_metrics_path_of(config)
    partial_path = final_path.with_name(f"run_metrics_{config['run_key']}.partial.json")
    is_sycophancy = config.get("config_name") == SYCOPHANCY_CONFIG_NAME
    export_wandb_metrics(run_id, partial_path, SYCOPHANCY_FINAL_EVAL_KEYS if is_sycophancy else MBPP_FINAL_EVAL_KEYS)
    if is_sycophancy:
        add_sycophancy_eval_history(run_id, partial_path, resumed_from_step(config))
        print_sycophancy_results(partial_path)
    else:
        executes = float(json.loads(partial_path.read_text())["metrics"]["val_does_execute"])
        print(f"{config['run_key']} run_id={run_id}: val_does_execute={executes:.4f}")
        if executes < MIN_VAL_DOES_EXECUTE:
            degenerate_path = final_path.with_name(f"run_metrics_{config['run_key']}.DEGENERATE_{run_id}.json")
            partial_path.rename(degenerate_path)
            print(f"DEGENERATE: {config['run_key']} run_id={run_id} val_does_execute={executes:.4f} < {MIN_VAL_DOES_EXECUTE}")
            print(f"  kept its metrics as {degenerate_path}")
            return False
    partial_path.rename(final_path)
    print(f"OK: {final_path} written; {config['run_key']} is done")
    return True


def launch(config: dict) -> str:
    """Run one GRPO job to completion and return its W&B run id (see the module docstring for the skip and stop
    rules). A crash stops the launch and keeps the run folder, marked CRASHED; an MBPP policy that stopped writing
    executable code is run again, at most MAX_COLLAPSE_RETRIES more times."""
    check_config(config)
    run_key, group = config["run_key"], config["group"]
    final_path = run_metrics_path_of(config)
    if final_path.exists():
        print(f"SKIP {run_key}: {final_path.name} already exists (to train again, move it and the run folder away)")
        return json.loads(final_path.read_text())["run_id"]
    for folder in existing_run_dirs(group, run_key):
        folder_run_id = folder.name.removeprefix(run_key + "_")
        crashed = (folder / CRASHED_MARKER).exists()
        collapsed = final_path.with_name(f"run_metrics_{run_key}.DEGENERATE_{folder_run_id}.json").exists()
        if not crashed and not collapsed:
            raise GrpoLaunchError(
                f"{run_key}: run folder {folder} exists but {final_path.name} doesn't, and the folder is not marked "
                f"{CRASHED_MARKER}. The run is still training (perhaps on another pod), was killed before the launcher could "
                f"mark it, or finished training but its checks or export failed: finish the last with `python -m "
                f"launcher.run_grpo <config> --finish-run-id {folder_run_id}`, or move the folder away to train again."
            )

    print(f"\n{'=' * 60}\n=== {run_key} ===\n{'=' * 60}")
    args = build_grpo_args(config)
    print(f"Checkpoint: {config['checkpoint'] or 'base model (no starting adapter)'}")
    print(f"W&B group: {group}")
    print(f"Output dir: {config['output_dir']}")
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    max_attempts = MAX_COLLAPSE_RETRIES + 1
    for attempt in range(1, max_attempts + 1):
        check_vllm()
        check_disk()
        start = time.time()
        exit_code, output = run_grpo_subprocess(args)
        print(f"GRPO subprocess finished in {(time.time() - start) / 60:.1f} min (exit={exit_code}, attempt={attempt}/{max_attempts})")
        run_id = extract_run_id(output)
        if exit_code != 0:
            if run_id is None or not run_dir(group, run_key, run_id).is_dir():
                raise GrpoLaunchError(f"CRASH: {run_key} attempt {attempt} exited with {exit_code}; no run folder to keep")
            folder = run_dir(group, run_key, run_id)
            (folder / CRASHED_MARKER).write_text(f"exit code {exit_code}, attempt {attempt}, {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
            raise GrpoLaunchError(f"CRASH: {run_key} attempt {attempt} exited with {exit_code}; its run folder is kept and marked {CRASHED_MARKER}: {folder}")
        if run_id is None or not run_dir(group, run_key, run_id).is_dir():
            raise GrpoLaunchError(f"{run_key}: no run folder for the run id in the trainer's output ({run_id!r})")
        print(f"Completed {run_key} -> run_id={run_id}")
        if finish_run(config, run_id):
            return run_id
    raise GrpoLaunchError(f"{run_key}: policy collapsed in {max_attempts} attempt(s); see the DEGENERATE files in {config['output_dir']}")
