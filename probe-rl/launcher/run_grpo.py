"""Launch one GRPO run from a YAML launch config, or finish one whose training ended but whose checks or metrics
export did not.

    python -m launcher.run_grpo <config.yaml>
    python -m launcher.run_grpo <config.yaml> --finish-run-id <W&B run id>

A launch config has `run_key`, `group` (the W&B group, and the run folder's parent under RUNS_BASE), `output_dir`
(absolute; `run_metrics_<run_key>.json` lands there), `probe` (`directory_save_prefix`, `retrain: true`),
`checkpoint` (an absolute adapter path, or null for the base model), `overrides` (Hydra key -> value; must set
`+model_preset` and `training.max_steps`) and, for the sycophancy task, `config_name: grpo_mcq`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from launcher import grpo_launcher
from launcher.common import GrpoLaunchError, bootstrap_hf_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="the launch config (YAML)")
    parser.add_argument("--finish-run-id", help="finish this run of the config instead of training")
    args = parser.parse_args(argv)

    bootstrap_hf_env()
    config = yaml.safe_load(args.config.read_text())
    try:
        if args.finish_run_id is None:
            grpo_launcher.launch(config)
        else:
            grpo_launcher.check_config(config)
            if grpo_launcher.run_metrics_path_of(config).exists():
                raise GrpoLaunchError(f"{grpo_launcher.run_metrics_path_of(config)} already exists: the run is done")
            if not grpo_launcher.finish_run(config, args.finish_run_id):
                raise GrpoLaunchError(f"{config['run_key']}: the policy of run {args.finish_run_id} collapsed")
    except GrpoLaunchError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
