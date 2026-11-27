"""Paths, the launcher's error, and the checks a run needs before it starts.

GRPO run folders, activation caches and the HF cache live in the volume's shared folders, like every other run on
it. A run's folder is `RUNS_BASE/<group>/<run_key>_<W&B run id>/`.
"""

from __future__ import annotations

import importlib.util
import os
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OBFUSCATION_ATLAS_DIR = REPO_ROOT / "obfuscation-atlas"
HF_CACHE_DIR = Path("/workspace/hf_cache")
RUNS_BASE = Path("/workspace/results/run_results")
ACTIVATIONS_DIR = Path("/workspace/results/activations")
VLLM_MODELS_URL = "http://127.0.0.1:8000/v1/models"
# Free disk required before a GRPO run starts (MIN_FREE_GB overrides it for a queue that knows its runs are small).
MIN_FREE_GB = int(os.environ.get("MIN_FREE_GB", "50"))


class GrpoLaunchError(RuntimeError):
    """A preflight check, GRPO run, or its verification failed."""


def run_dir(group: str, run_key: str, run_id: str) -> Path:
    """The folder of one run (it exists once the trainer has started)."""
    return RUNS_BASE / group / f"{run_key}_{run_id}"


def bootstrap_hf_env() -> None:
    """Point HF_HOME at the volume's cache unless the caller already set it."""
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    # Some container images set HF_HUB_ENABLE_HF_TRANSFER=1, but hf_transfer is not a dependency.
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1" and importlib.util.find_spec("hf_transfer") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def check_vllm(url: str = VLLM_MODELS_URL) -> None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            status = response.status
    except (urllib.error.URLError, OSError) as error:
        raise GrpoLaunchError(f"vLLM not reachable at {url}: {error}") from error
    if status != 200:
        raise GrpoLaunchError(f"vLLM returned status {status} at {url}")
    print("OK: vLLM reachable")


def check_disk(min_free_gb: int = MIN_FREE_GB) -> None:
    stat = os.statvfs(REPO_ROOT)
    free_gb = stat.f_bavail * stat.f_frsize / (1024**3)
    print(f"Disk free on {REPO_ROOT}: {free_gb:.1f}G (need >= {min_free_gb}G)")
    if free_gb < min_free_gb:
        raise GrpoLaunchError(f"Insufficient disk space: {free_gb:.1f}G < {min_free_gb}G")


def model_of_run(run_config: Path) -> str:
    """The model an RL run trains: its config's `+model_preset` as the trainer's config.py defines it."""
    import yaml
    from hydra.core.config_store import ConfigStore

    import obfuscation_atlas.config  # noqa: F401  (registers the presets)

    preset = yaml.safe_load(run_config.read_text())["overrides"]["+model_preset"]
    return ConfigStore.instance().repo["model_preset"][f"{preset}.yaml"].node["model"]["model_type"]


TRAINER_DEFAULT_CONFIG_NAME = "grpo_code_gen"  # @hydra.main in train_grpo_model_against_probe.py (MBPP)


def trainer_config_of_run(run_config: Path):
    """The trainer's full config for an RL run, composed by Hydra from the overrides the launcher gives the trainer
    (launcher.grpo_launcher.trainer_overrides), with today's code: its defaults and presets, not a record of the run.
    The batch a Qwen3.5-9B run gets from its GPU is left at the preset's (an eval does not read it)."""
    import yaml
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    from launcher.grpo_launcher import trainer_overrides
    from obfuscation_atlas.config import register_configs

    config = yaml.safe_load(run_config.read_text())
    overrides = [f"{key}={value}" for key, value in trainer_overrides(config).items()]
    register_configs()
    with initialize(version_base=None):
        cfg = compose(config_name=config.get("config_name") or TRAINER_DEFAULT_CONFIG_NAME, overrides=overrides)
    return OmegaConf.to_object(cfg)
