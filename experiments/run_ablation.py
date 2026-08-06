#!/usr/bin/env python3
"""Launch a single GRPO ablation run or a sweep of runs from a YAML config.

Usage:
    python experiments/run_ablation.py experiments/configs/det0_ceiling.yaml
    python experiments/run_ablation.py experiments/configs/det0_seed_sweep.yaml

See experiments/configs/ for schema examples (single run vs sweep) and
docs/FULL_GRPO_RUN.md for the underlying GRPO methodology.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

import grpo_launcher as gl


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise gl.GrpoLaunchError(f"{path}: top-level YAML must be a mapping")
    return config


def build_run_config(sweep_config: dict, run_entry: dict) -> dict:
    """Merge a sweep's shared settings with one `runs:` entry into a
    single-run config dict suitable for grpo_launcher.launch()."""
    output_base = sweep_config["output_dir"]
    output_subdir = run_entry.get("output_subdir")
    output_dir = f"{output_base}/{output_subdir}" if output_subdir else output_base

    overrides = dict(sweep_config.get("shared_overrides", {}))
    overrides.update(run_entry.get("overrides", {}))

    return {
        "run_key": run_entry["run_key"],
        "output_dir": output_dir,
        "probe": sweep_config["probe"],
        "checkpoint": run_entry.get("checkpoint", sweep_config.get("checkpoint")),
        "nofu": run_entry.get("nofu", sweep_config.get("nofu", True)),
        "overrides": overrides,
        "cleanup": sweep_config.get("cleanup", True),
        "export_metrics": sweep_config.get("export_metrics", True),
        "min_responses": run_entry.get("min_responses", sweep_config.get("min_responses", 0)),
        "max_grpo_retries": run_entry.get("max_grpo_retries", sweep_config.get("max_grpo_retries")),
        "min_val_does_execute": run_entry.get(
            "min_val_does_execute", sweep_config.get("min_val_does_execute")
        ),
    }


def run_single(config: dict) -> int:
    try:
        gl.launch(config)
    except gl.GrpoLaunchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


def run_sweep(config: dict) -> int:
    on_failure = config.get("on_failure", "continue")
    if on_failure not in ("continue", "stop"):
        raise gl.GrpoLaunchError(f"on_failure must be 'continue' or 'stop', got {on_failure!r}")

    runs = config["runs"]
    print(f"Sweep {config.get('sweep_name', '(unnamed)')}: {len(runs)} run(s), on_failure={on_failure}")

    completed: list[str] = []
    failed: list[str] = []
    for run_entry in runs:
        run_config = build_run_config(config, run_entry)
        try:
            run_id = gl.launch(run_config)
            completed.append(f"{run_config['run_key']} -> {run_id}")
        except gl.GrpoLaunchError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            failed.append(run_config["run_key"])
            if on_failure == "stop":
                print(f"ABORT: on_failure=stop, halting sweep after {run_config['run_key']}", file=sys.stderr)
                break

    print("\n=== Sweep done ===")
    print("Completed:")
    for entry in completed:
        print(f"  {entry}")
    if failed:
        print("Failed/skipped:", file=sys.stderr)
        for run_key in failed:
            print(f"  {run_key}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a YAML config (single run or sweep)")
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    if "runs" in config:
        return run_sweep(config)
    return run_single(config)


if __name__ == "__main__":
    sys.exit(main())
