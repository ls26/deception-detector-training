"""Compare the full trainer config for August's command in the code copy and in the private code.

Hydra composes the config for the exact trainer arguments August's run recorded
(260825_grpo_det0/run_metrics_syco_det0_s42.json, `metrics.command`) and prints it without
running (`--cfg job --resolve`), once with the copy's package and once with the private
package (/workspace/obfuscation-atlas, whose config.py is unchanged since 12631d5). The keys
that may differ are listed in expected_config_keys.json: settings of features the release
removed, and two defaults the GRPO path never reads. Both composed configs are saved.

Usage:
    python check_config.py [--out check_config.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

RUN_DIR = Path(__file__).resolve().parent
AUGUST_METRICS = Path("/workspace/sycophancy_experiments/260825_grpo_det0/run_metrics_syco_det0_s42.json")
COPY = Path("/workspace/sycophancy-rl")
PRIVATE_ATLAS = Path("/workspace/obfuscation-atlas")
TRAINER_MODULE = "obfuscation_atlas.scripts.train_grpo_model_against_probe"


def composed_config(atlas_dir: Path, python_path: str, trainer_args: list[str]) -> dict:
    env = dict(os.environ, PYTHONPATH=python_path)
    printed = subprocess.run(
        [sys.executable, "-m", TRAINER_MODULE, *trainer_args, "--cfg", "job", "--resolve"],
        cwd=atlas_dir, env=env, capture_output=True, text=True, check=True,
    ).stdout
    return yaml.safe_load(printed)


def flatten(tree, prefix: str = "") -> dict:
    if not isinstance(tree, dict):
        return {prefix.removesuffix("."): tree}
    flat = {}
    for key, value in tree.items():
        flat.update(flatten(value, f"{prefix}{key}."))
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RUN_DIR / "check_config.json")
    args = parser.parse_args()

    trainer_args = json.loads(AUGUST_METRICS.read_text())["metrics"]["command"].split()
    copy_python_path = f"{COPY}:{COPY}/obfuscation-atlas:{COPY}/obfuscation-atlas/third_party/afterburner/src"
    copy_config = composed_config(COPY / "obfuscation-atlas", copy_python_path, trainer_args)
    private_config = composed_config(PRIVATE_ATLAS, "/workspace", trainer_args)
    (args.out.parent / "composed_config_copy.json").write_text(json.dumps(copy_config, indent=2) + "\n")
    (args.out.parent / "composed_config_private.json").write_text(json.dumps(private_config, indent=2) + "\n")

    copy_flat, private_flat = flatten(copy_config), flatten(private_config)
    found = {
        "only_in_private": sorted(set(private_flat) - set(copy_flat)),
        "only_in_copy": sorted(set(copy_flat) - set(private_flat)),
        "differ": sorted(key for key in set(copy_flat) & set(private_flat) if copy_flat[key] != private_flat[key]),
    }
    expected = json.loads((RUN_DIR / "expected_config_keys.json").read_text())
    passed = found == expected
    for name in found:
        print(f"{name}: {found[name]}" + ("" if found[name] == expected[name] else f"   EXPECTED {expected[name]}"))
    differing_values = {key: {"private": private_flat[key], "copy": copy_flat[key]} for key in found["differ"]}
    print(f"differing values: {differing_values}")
    print(f"keys compared: {len(set(copy_flat) & set(private_flat))}")

    result = {"PASS": passed, "found": found, "expected": expected, "differing_values": differing_values}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: wrote {args.out}")


if __name__ == "__main__":
    main()
