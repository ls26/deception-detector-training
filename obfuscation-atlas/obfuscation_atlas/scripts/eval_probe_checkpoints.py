"""Batch probe evaluation for downloaded HF checkpoints.

Runs eval-only post-train probe pipeline per checkpoint. Skips checkpoints
with existing run_metrics_{SHORT_NAME}.json exports (never overwrites).
Supports resuming after crashes via export-only retry from .state files.

Requires vLLM server running separately on GPU 1.

Usage:
    # Full default sweep (kl 1e-4/1e-3/1e-2 × det 1/3/10 × general/MBPP × seeds 1-3)
    python -m obfuscation_atlas.scripts.eval_probe_checkpoints

    # Single checkpoint
    python -m obfuscation_atlas.scripts.eval_probe_checkpoints --kl 0.001 --det 3 --probe mbpp --seed 2

    # First unevaluated checkpoint only
    python -m obfuscation_atlas.scripts.eval_probe_checkpoints --limit 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from obfuscation_atlas.scripts.download_checkpoints import (
    CheckpointSpec,
    build_download_list,
    discover_available_repos,
)

DEFAULT_KL_COEFS = [0.0001, 0.001, 0.01]
DEFAULT_DET_COEFS = [1, 3, 10]
DEFAULT_PROBES = ["diverse_deception_probe", "mbpp_probe"]
DEFAULT_SEEDS = [1, 2, 3]

PROBE_ALIASES = {
    "general": "diverse_deception_probe",
    "ddp": "diverse_deception_probe",
    "diverse_deception": "diverse_deception_probe",
    "diverse_deception_probe": "diverse_deception_probe",
    "mbpp": "mbpp_probe",
    "mbpp_probe": "mbpp_probe",
}

KL_SHORT = {0.0001: "1e-4", 0.001: "1e-3", 0.01: "1e-2", 0.1: "1e-1", 1.0: "1e-0"}
PROBE_SHORT = {"diverse_deception_probe": "ddp", "mbpp_probe": "mbpp"}

BASE_PROBE = {
    "mbpp_probe": {
        "prefix": "base-mbpp",
        "glob": "base_results_and_detectors_*mbpp-hardcode*",
    },
    "diverse_deception_probe": {
        "prefix": "base-ddp",
        "glob": "base_results_and_detectors_*local-diverse_deception*",
    },
}

PROBE_EVAL_SAVE_BASE = Path("/workspace/results/run_results")
ACTIVATION_CACHE = Path("/workspace/results/activations")
HF_EVALS_DIR = PROBE_EVAL_SAVE_BASE / "hf_checkpoint_evals"
PROBE_RESULTS_DIR = HF_EVALS_DIR / "hf_probe_eval_runs" / "extracted_results"
STATE_DIR = PROBE_RESULTS_DIR / ".state"
LOG_DIR = PROBE_EVAL_SAVE_BASE / "logs"
MBPP_SUMMARY = HF_EVALS_DIR / "hf_task_eval_metrics" / "summary.json"
MBPP_JSONL_DIR = HF_EVALS_DIR / "hf_task_eval_metrics"
SWEEP_STATUS_PATH = PROBE_RESULTS_DIR / "sweep_status.json"
DEFAULT_HF_HOME = Path("/workspace/hf_cache")

WANDB_ENTITY = "lilylixin6-the-harker-school"
WANDB_PROJECT = "deception_detection"
WANDB_GROUP = "hf_probe_eval"
VLLM_MODELS_URL = "http://127.0.0.1:8000/v1/models"
MIN_DISK_GB = 30
DET0_DEFAULT_PROBE = "diverse_deception_probe"
_ACTIVE_DET0_PROBE: str = DET0_DEFAULT_PROBE
EXPECTED_MANIFEST_COUNT = 5
MANIFEST_NAME = "f1_and_recall_plot_initial_detector.base_model.on_dataset__manifest.json"
WANDB_RUN_RE = re.compile(r"deception_detection/runs/([a-z0-9]{8})")


@dataclass
class RunPlanEntry:
    spec: CheckpointSpec
    probe_dataset: str
    adapter_path: str | None = None
    export_only: bool = False
    run_id: str | None = None


@dataclass
class SweepStatus:
    completed: list[str] = field(default_factory=list)
    skipped_done: list[dict] = field(default_factory=list)
    skipped_missing_hf: list[dict] = field(default_factory=list)
    skipped_invalid: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)


def normalize_probe(name: str) -> str:
    key = name.strip().lower()
    if key not in PROBE_ALIASES:
        raise argparse.ArgumentTypeError(
            f"Unknown probe {name!r}. Use: general, mbpp, diverse_deception_probe, mbpp_probe"
        )
    return PROBE_ALIASES[key]


def kl_short(kl: float) -> str:
    return KL_SHORT.get(kl, f"{kl:g}")


def probe_short(probe: str | None) -> str:
    if probe is None:
        return "na"
    return PROBE_SHORT.get(probe, probe.replace("_probe", ""))


def short_name(spec: CheckpointSpec, *, det0_probe: str | None = None) -> str:
    probe_for_det0 = det0_probe if det0_probe is not None else _ACTIVE_DET0_PROBE
    if spec.det == 0:
        return f"kl{kl_short(spec.kl)}-det0-s{spec.seed}-{probe_short(probe_for_det0)}"
    return f"kl{kl_short(spec.kl)}-det{spec.det:g}-s{spec.seed}-{probe_short(spec.probe)}"


def probe_dataset_for_spec(spec: CheckpointSpec, det0_probe: str) -> str:
    return det0_probe if spec.det == 0 else spec.probe  # type: ignore[return-value]


def spec_dict(spec: CheckpointSpec, **extra) -> dict:
    return {
        "short_name": short_name(spec),
        "kl": spec.kl,
        "det": spec.det,
        "seed": spec.seed,
        "probe": spec.probe,
        "repo_id": spec.repo_id,
        **extra,
    }


def run_metrics_path(name: str) -> Path:
    return PROBE_RESULTS_DIR / f"run_metrics_{name}.json"


def load_mbpp_summary() -> dict[str, dict]:
    with open(MBPP_SUMMARY) as f:
        return {e["checkpoint"]: e for e in json.load(f)}


def write_task_metrics_json(checkpoint_name: str, summary_entry: dict, out_path: Path) -> None:
    jsonl_path = MBPP_JSONL_DIR / f"{checkpoint_name}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"MBPP JSONL missing for avg_score: {jsonl_path}")
    scores = [json.loads(line)["score"] for line in jsonl_path.open()]
    payload = {
        "checkpoint": checkpoint_name,
        "pass_first_test_rate": summary_entry["pass_first_test_rate"],
        "pass_all_tests_rate": summary_entry["pass_all_tests_rate"],
        "deceptive_rate": summary_entry["deceptive_rate"],
        "hardcoded_rate": summary_entry["hardcoded_rate"],
        "executes_rate": summary_entry["executes_rate"],
        "n_examples": summary_entry["n_examples"],
        "avg_score": sum(scores) / len(scores),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def has_adapter_files(lora_path: str) -> bool:
    p = Path(lora_path)
    return (p / "adapter_config.json").exists() and (
        (p / "adapter_model.bin").exists() or (p / "adapter_model.safetensors").exists()
    )


def _common_train_args() -> list[str]:
    return [
        sys.executable,
        "-m",
        "obfuscation_atlas.scripts.train_grpo_model_against_probe",
        "+model_preset=llama3_8b",
        f"training.save_path_base={PROBE_EVAL_SAVE_BASE}",
        f"detector.model_cache_path={ACTIVATION_CACHE}",
        f"wandb.project={WANDB_PROJECT}",
        f"wandb.group={WANDB_GROUP}",
        "grpo.model.attn_implementation=sdpa",
        "corrected_detector_gradient=false",
        "misalignment_eval.enabled=false",
    ]


def build_eval_command(spec: CheckpointSpec, adapter_path: str, probe_dataset: str) -> list[str]:
    return _common_train_args() + [
        f"loss.kl={spec.kl:g}",
        f"loss.detector={spec.det:g}",
        f"+data/probe_dataset={probe_dataset}",
        f"training.seed={spec.seed}",
        "++training.only_eval_from_other_run=True",
        "++training.resume_from_checkpoint=False",
        f"++grpo.model.reference_adapter_path={adapter_path}",
        f"wandb.run_name_prefix={short_name(spec)}",
    ]


def build_base_probe_command(probe_dataset: str) -> list[str]:
    return _common_train_args() + [
        f"+data/probe_dataset={probe_dataset}",
        "++training.only_eval_detectors=True",
        "++training.resume_from_checkpoint=False",
        f"wandb.run_name_prefix={BASE_PROBE[probe_dataset]['prefix']}",
    ]


def parse_wandb_run_id(output: str) -> str | None:
    matches = WANDB_RUN_RE.findall(output)
    return matches[-1] if matches else None


def find_base_probe_manifest(probe_dataset: str) -> Path | None:
    stage = "initial_detector.base_model.on_dataset"
    for base_dir in PROBE_EVAL_SAVE_BASE.glob(BASE_PROBE[probe_dataset]["glob"]):
        # Nested layout first, then legacy flat layout
        for manifest in (base_dir / stage / MANIFEST_NAME, base_dir / MANIFEST_NAME):
            if manifest.exists():
                return manifest
    return None


def _gpu0_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def run_preflight() -> None:
    checks: list[tuple[str, bool, str]] = []

    try:
        import obfuscation_atlas  # noqa: F401

        checks.append(("venv (import obfuscation_atlas)", True, "ok"))
    except ImportError as e:
        checks.append(("venv (import obfuscation_atlas)", False, str(e)))

    wandb_ok = bool(os.environ.get("WANDB_API_KEY"))
    checks.append(("WANDB_API_KEY", wandb_ok, "set" if wandb_ok else "missing"))

    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
        checks.append(("HF_HOME", True, f"defaulted to {DEFAULT_HF_HOME}"))
    else:
        checks.append(("HF_HOME", True, os.environ["HF_HOME"]))

    free_gb = shutil.disk_usage("/workspace").free / (1024**3)
    checks.append(
        ("disk /workspace", free_gb >= MIN_DISK_GB, f"{free_gb:.1f} GB free (need >= {MIN_DISK_GB} GB)")
    )

    vllm_ok, vllm_msg = False, "not reachable"
    try:
        with urllib.request.urlopen(VLLM_MODELS_URL, timeout=5) as resp:
            vllm_ok, vllm_msg = resp.status == 200, f"HTTP {resp.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        vllm_msg = str(e)
    checks.append(("vLLM server", vllm_ok, vllm_msg))
    checks.append(("MBPP summary", MBPP_SUMMARY.exists(), str(MBPP_SUMMARY)))

    print("=== Pre-flight checks ===")
    failed = False
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        failed |= not ok
    if failed:
        raise SystemExit("Pre-flight checks failed. Fix issues above before running.")
    print("Pre-flight checks passed.\n")


def stream_subprocess(cmd: list[str], log_path: Path, env: dict[str, str]) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"# command: {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
            lines.append(line)
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output="".join(lines))
    return "".join(lines)


def ensure_base_probe_cache(probe_dataset: str) -> None:
    manifest = find_base_probe_manifest(probe_dataset)
    if manifest is not None:
        print(f"  Base probe cache OK for {probe_dataset}: {manifest.parent.name}")
        return

    prefix = BASE_PROBE[probe_dataset]["prefix"]
    print(f"  Base probe cache missing for {probe_dataset}; running Phase 0...")
    output = stream_subprocess(
        build_base_probe_command(probe_dataset),
        LOG_DIR / f"phase0_{prefix}.log",
        _gpu0_env(),
    )
    if run_id := parse_wandb_run_id(output):
        print(f"  Phase 0 W&B run: {run_id}")

    manifest = find_base_probe_manifest(probe_dataset)
    if manifest is None:
        raise RuntimeError(f"Phase 0 failed: base probe manifest still missing for {probe_dataset}")
    print(f"  Phase 0 complete: {manifest.parent.name}")


def load_state(name: str) -> dict | None:
    path = STATE_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def write_state(name: str, run_id: str, checkpoint_name: str, probe_dataset: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{name}.json").write_text(
        json.dumps(
            {
                "short_name": name,
                "run_id": run_id,
                "checkpoint_name": checkpoint_name,
                "probe_dataset": probe_dataset,
                "completed_at_phase1": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )


def clear_state(name: str) -> None:
    path = STATE_DIR / f"{name}.json"
    if path.exists():
        path.unlink()


def build_run_plan(
    args: argparse.Namespace,
    mbpp_summary: dict[str, dict],
) -> tuple[list[RunPlanEntry], SweepStatus]:
    found, missing_hf = build_download_list(
        discover_available_repos(HfApi()), args.kl, args.det, args.probe, args.seed
    )

    status = SweepStatus(skipped_missing_hf=[spec_dict(s) for s in missing_hf])
    to_run: list[RunPlanEntry] = []

    for spec in found:
        name = short_name(spec)
        if run_metrics_path(name).exists():
            status.skipped_done.append(spec_dict(spec, reason="export exists"))
            continue

        ckpt_name = spec.repo_id.split("/")[-1]
        if ckpt_name not in mbpp_summary:
            status.skipped_invalid.append(spec_dict(spec, reason="no MBPP summary"))
            continue

        probe_ds = probe_dataset_for_spec(spec, args.det0_probe)
        if saved := load_state(name):
            if saved.get("run_id"):
                to_run.append(
                    RunPlanEntry(spec, probe_ds, export_only=True, run_id=saved["run_id"])
                )
                continue

        try:
            adapter_path = snapshot_download(repo_id=spec.repo_id, local_files_only=True)
        except Exception as e:
            status.skipped_invalid.append(spec_dict(spec, reason=f"adapter missing: {e}"))
            continue

        if not has_adapter_files(adapter_path):
            status.skipped_invalid.append(spec_dict(spec, reason="adapter files incomplete"))
            continue

        to_run.append(RunPlanEntry(spec, probe_ds, adapter_path=adapter_path))

    if args.limit is not None:
        to_run = to_run[: args.limit]
    return to_run, status


def print_skip_summary(status: SweepStatus, to_run_count: int) -> None:
    print(
        f"Plan: to_run={to_run_count} | "
        f"skipped_done={len(status.skipped_done)} | "
        f"skipped_missing_hf={len(status.skipped_missing_hf)} | "
        f"skipped_invalid={len(status.skipped_invalid)}"
    )
    for label, items in (
        ("Skipped (done)", status.skipped_done),
        ("Skipped (missing on HF)", status.skipped_missing_hf),
        ("Skipped (invalid)", status.skipped_invalid),
    ):
        if not items:
            continue
        print(f"\n{label} ({len(items)}):")
        for item in items[:10]:
            print(f"  {item['short_name']}: {item.get('reason', item.get('repo_id', ''))}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")


def verify_phase1(run_id: str, log_output: str | None = None) -> None:
    candidates = [
        PROBE_EVAL_SAVE_BASE / WANDB_GROUP / f"trigger_none_model_{run_id}_layers_32",
        *PROBE_EVAL_SAVE_BASE.glob(f"{WANDB_GROUP}/*_{run_id}"),
        HF_EVALS_DIR / "hf_probe_eval_runs" / f"trigger_none_model_{run_id}_layers_32",
    ]
    art_dir = next((p for p in candidates if p.is_dir()), candidates[0])
    if not art_dir.exists():
        raise FileNotFoundError(f"Phase 1 artifacts dir missing: {art_dir}")

    manifests = list(art_dir.glob("f1_and_recall_plot_*__manifest.json"))
    if not manifests:
        # Nested per-stage layout: {art_dir}/{stage}/f1_and_recall_plot_*__manifest.json
        manifests = list(art_dir.glob("*/f1_and_recall_plot_*__manifest.json"))
    if len(manifests) != EXPECTED_MANIFEST_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_MANIFEST_COUNT} manifest files, found {len(manifests)} in {art_dir}"
        )
    if log_output and "Loaded eval results and detectors" not in log_output:
        print("  WARNING: log does not show base probe loaded from cache")
    print(f"  Phase 1 verified: {len(manifests)} manifests in {art_dir.name}")


def verify_export(path: Path, expected_pass_first: float) -> None:
    data = json.loads(path.read_text())
    if "on_policy_responses" in str(data):
        raise ValueError("Export contains on_policy_responses artifacts")

    metrics = data.get("metrics", {})
    val_first = metrics.get("val_passes_first_n_tests")
    if val_first is None or abs(val_first - expected_pass_first) >= 0.002:
        raise ValueError(f"val_passes_first_n_tests={val_first}, expected ~{expected_pass_first}")

    val_success = metrics.get("val_success")
    if val_success is None or float(val_success) <= 0:
        raise ValueError(f"val_success invalid: {val_success}")

    policy_type = data.get("classification", {}).get("policy_type")
    if not policy_type:
        raise ValueError("classification.policy_type is empty")

    print(
        f"  Export verified: val_passes_first_n_tests={val_first}, "
        f"val_success={val_success}, policy_type={policy_type!r}"
    )


def run_export(
    run_id: str,
    name: str,
    checkpoint_name: str,
    mbpp_summary: dict,
    expected_pass_first: float,
) -> Path:
    final_path = run_metrics_path(name)
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {final_path}")

    task_metrics_path = PROBE_RESULTS_DIR / f"task_metrics_{name}.json"
    write_task_metrics_json(checkpoint_name, mbpp_summary[checkpoint_name], task_metrics_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=PROBE_RESULTS_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "obfuscation_atlas.scripts.export_wandb_run_metrics",
                "--entity",
                WANDB_ENTITY,
                "--project",
                WANDB_PROJECT,
                "--run-id",
                run_id,
                "--task-metrics-json",
                str(task_metrics_path),
                "--output",
                str(tmp_path),
            ],
            check=True,
        )
        verify_export(tmp_path, expected_pass_first)
        if final_path.exists():
            raise FileExistsError(f"Export appeared during write: {final_path}")
        tmp_path.replace(final_path)
        return final_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def process_checkpoint(
    entry: RunPlanEntry,
    mbpp_summary: dict[str, dict],
    status: SweepStatus,
    index: int,
    total: int,
) -> None:
    spec = entry.spec
    name = short_name(spec)
    ckpt_name = spec.repo_id.split("/")[-1]
    expected_pass_first = mbpp_summary[ckpt_name]["pass_first_test_rate"]
    mode = "export-only retry" if entry.export_only else "full"
    print(f"\n=== [{index}/{total}] {name} ({mode}) ===")

    try:
        if entry.export_only:
            run_id = entry.run_id
            assert run_id is not None
            print(f"  Resuming export for W&B run: {run_id}")
            verify_phase1(run_id)
        else:
            assert entry.adapter_path is not None
            output = stream_subprocess(
                build_eval_command(spec, entry.adapter_path, entry.probe_dataset),
                LOG_DIR / f"{name}.log",
                _gpu0_env(),
            )
            run_id = parse_wandb_run_id(output)
            if run_id is None:
                raise RuntimeError("Could not parse W&B run id from Phase 1 output")
            print(f"  W&B run: {run_id}")
            verify_phase1(run_id, output)
            write_state(name, run_id, ckpt_name, entry.probe_dataset)

        out_path = run_export(run_id, name, ckpt_name, mbpp_summary, expected_pass_first)
        clear_state(name)
        status.completed.append(name)
        print(f"  Exported: {out_path}")
    except Exception as e:
        if not entry.export_only:
            clear_state(name)
        status.failed.append(spec_dict(spec, reason=str(e)))
        print(f"  ERROR: {e}")


def write_sweep_status(status: SweepStatus, to_run_count: int) -> None:
    PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_STATUS_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "to_run_count": to_run_count,
                "completed": status.completed,
                "skipped_done": status.skipped_done,
                "skipped_missing_hf": status.skipped_missing_hf,
                "skipped_invalid": status.skipped_invalid,
                "failed": status.failed,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote sweep status: {SWEEP_STATUS_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch probe eval for HF checkpoints (resumable, no overwrite)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--kl", type=float, nargs="+", default=DEFAULT_KL_COEFS)
    parser.add_argument("--det", type=float, nargs="+", default=DEFAULT_DET_COEFS)
    parser.add_argument("--probe", type=str, nargs="+", default=DEFAULT_PROBES)
    parser.add_argument("--seed", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--det0-probe", type=str, default=DET0_DEFAULT_PROBE)
    parser.add_argument("--limit", type=int, default=None, help="Max checkpoints to evaluate")
    args = parser.parse_args()
    args.probe = list(dict.fromkeys(normalize_probe(p) for p in args.probe))
    args.det0_probe = normalize_probe(args.det0_probe)
    return args


def main() -> int:
    global _ACTIVE_DET0_PROBE
    args = parse_args()
    _ACTIVE_DET0_PROBE = args.det0_probe
    run_preflight()

    mbpp_summary = load_mbpp_summary()
    to_run, status = build_run_plan(args, mbpp_summary)
    print_skip_summary(status, len(to_run))

    if to_run:
        print("\nTo evaluate:")
        for entry in to_run:
            mode = "export-only" if entry.export_only else "full"
            print(
                f"  RUN  {short_name(entry.spec)} [{mode}] "
                f"probe_dataset={entry.probe_dataset} repo={entry.spec.repo_id}"
            )
    else:
        write_sweep_status(status, 0)
        print("\nNothing to evaluate.")
        return 0

    print("\n=== Ensuring base probe caches ===")
    for probe_ds in sorted({entry.probe_dataset for entry in to_run}):
        ensure_base_probe_cache(probe_ds)

    for i, entry in enumerate(to_run, 1):
        process_checkpoint(entry, mbpp_summary, status, i, len(to_run))

    write_sweep_status(status, len(to_run))
    print(f"\nDone. Completed: {len(status.completed)} | Failed: {len(status.failed)}")
    return 1 if status.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
