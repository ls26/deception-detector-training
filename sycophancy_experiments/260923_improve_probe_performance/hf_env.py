"""HF Hub env for interactive kernels that don't inherit dotfiles/env.sh."""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = Path("/workspace")
HF_CACHE_DIR = WORKSPACE / "hf_cache"
SECRETS_ENV = WORKSPACE / ".secrets.env"


def bootstrap_hf_env() -> None:
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    if not SECRETS_ENV.exists():
        return
    for line in SECRETS_ENV.read_text().splitlines():
        if line.startswith("export HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
            break
