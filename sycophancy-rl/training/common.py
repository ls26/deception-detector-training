"""Constants and small helpers shared by `training/` and `probe_eval/`.

`REPO_ROOT` is the checkout; `experiments/` (configs and per-trial outputs) lives directly
under it. GRPO run directories, activation caches and the HF cache live in the private
workspace's shared folders, like every other run on this volume.
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
HF_CACHE_DIR = Path("/workspace/hf_cache")
RUNS_BASE = Path("/workspace/results/run_results")
ACTIVATIONS_DIR = Path("/workspace/results/activations")

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
VLLM_MODELS_URL = "http://127.0.0.1:8000/v1/models"
# Held-out evaluation prompts (one completion each per policy) and the token budget of one
# prompt + completion when probing; both match the vendored MBPP dataset configuration.
EVAL_SAMPLES = 400
MAX_SEQUENCE_LENGTH = 512
# Seed for every data draw outside GRPO itself: dataset splits, generation sampling, probe
# training-set selection, and the probe_eval eval sets.
DATA_SEED = 53
# Reward-hacking label: the completion passes only the shown test under both the original and
# the swapped test (`pass_only_shown` in obfuscation_atlas.tasks.code_generation).
DECEPTIVE_DEFINITION = "pass_only_shown"
# Free disk required before a GRPO run or a generation step starts.
MIN_FREE_GB = int(os.environ.get("MIN_FREE_GB", "50"))


class GrpoLaunchError(RuntimeError):
    """A preflight check, GRPO run, or its verification failed."""


def bootstrap_hf_env() -> None:
    """Point HF_HOME at the repo's cache unless the caller already set it."""
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


def resolve_relative_path(path_str: str, base: Path) -> Path:
    """A config path: absolute as given, otherwise relative to `base`."""
    path = Path(path_str)
    return path if path.is_absolute() else base / path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def check_jsonl_rows(path: Path, expected: int | None, *, label_fields: tuple[str, ...] = ()) -> dict:
    """Read a labeled JSONL, require `expected` rows (or at least one when None), and count
    the rows where each of `label_fields` is true. Returns {"rows": [...], "counts": {...}}."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    rows = load_jsonl(path)
    if expected is not None:
        if len(rows) != expected:
            raise ValueError(f"{path.name}: expected {expected} rows, got {len(rows)}")
    elif len(rows) == 0:
        raise ValueError(f"{path.name}: expected at least 1 row, got 0")
    counts = {field: sum(bool(row.get(field)) for row in rows) for field in label_fields}
    return {"rows": rows, "counts": counts}
