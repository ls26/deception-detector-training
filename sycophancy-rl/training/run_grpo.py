"""Launch a single GRPO run or a sweep of runs from a YAML config.

Usage:
    python -m training.run_grpo experiments/trials/trial1/round1.yaml       # single run
    python -m training.run_grpo experiments/baselines/no_probe_penalty.yaml  # sweep

See those two files for the schema: a single-run config has `run_key` and `overrides`; a
sweep config has a `runs:` list of per-run `run_key`/`overrides` entries and shares
`shared_overrides` and everything else. `group` (the W&B group and the run-directory nest
under results/run_results) defaults to the YAML's parent folder name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from training import grpo_launcher
from training.common import GrpoLaunchError, bootstrap_hf_env


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise GrpoLaunchError(f"{path}: top-level YAML must be a mapping")
    if "runs" not in config and "shared_overrides" in config:
        raise GrpoLaunchError(
            f"{path}: `shared_overrides` is only read for sweeps (`runs:`); "
            "a single-run config must put its Hydra overrides under `overrides:`"
        )
    config.setdefault("group", path.resolve().parent.name)
    return config


def build_run_config(sweep_config: dict, run_entry: dict) -> dict:
    """One launch config from a sweep config and one of its `runs:` entries."""
    overrides = dict(sweep_config.get("shared_overrides", {}))
    overrides.update(run_entry.get("overrides", {}))
    return {
        "run_key": run_entry["run_key"],
        "group": sweep_config["group"],
        "output_dir": sweep_config["output_dir"],
        "probe": sweep_config["probe"],
        "checkpoint": sweep_config["checkpoint"],
        "overrides": overrides,
    }


def run_sweep(config: dict) -> int:
    """Launch every run in turn; a failed run is reported and the sweep continues."""
    runs = config["runs"]
    print(f"Sweep {config['sweep_name']}: {len(runs)} run(s)")
    completed: list[str] = []
    failed: list[str] = []
    for run_entry in runs:
        run_config = build_run_config(config, run_entry)
        try:
            run_id = grpo_launcher.launch(run_config)
            completed.append(f"{run_config['run_key']} -> {run_id}")
        except GrpoLaunchError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            failed.append(run_config["run_key"])

    # With `retrain: false` every run of the sweep reuses one Phase-0 activation cache under
    # the shared prefix; remove it once the last run is done.
    if not config["probe"]["retrain"]:
        grpo_launcher.remove_prefix_activation_cache(config["probe"]["directory_save_prefix"])

    print("\n=== Sweep done ===")
    print("Completed:")
    for entry in completed:
        print(f"  {entry}")
    if failed:
        print("Failed:", file=sys.stderr)
        for run_key in failed:
            print(f"  {run_key}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to a YAML config (single run or sweep)")
    args = parser.parse_args(argv)
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    bootstrap_hf_env()
    config = load_config(args.config)
    if "runs" in config:
        return run_sweep(config)
    try:
        grpo_launcher.launch(config)
    except GrpoLaunchError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
