"""261007's probe pipeline, imported unchanged, and this experiment's PipelineConfig.

Imported by run_pipeline.py, cv_run.py, final_fit.py and compare_slice.py.
261007's folder goes first on sys.path, so `spec` and `domain_imports` resolve
to 261007's: its step 2 imports its own `domain_imports` by that name. A
process that imports this module must not import this folder's
domain_imports.py (generate_nocue_train.py and slice_labels.py use that one).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import slice_spec as S

sys.path.insert(0, str(S.REF_DIR))

import cv_edge  # noqa: E402
import cv_select  # noqa: E402
import cv_sweep  # noqa: E402
import spec as ref_spec  # noqa: E402
import step1_build_probe_sets as s1  # noqa: E402
import step2_extract_activations as s2  # noqa: E402
import step3_train_eval_probes as s3  # noqa: E402
from compare import bootstrap_delta  # noqa: E402
from extract import step1_gates  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from probe_config import POSITION_OF_FAMILY, PipelineConfig, apply_overrides, config_to_json  # noqa: E402

assert Path(ref_spec.__file__).parent == S.REF_DIR, ref_spec.__file__
assert Path(sys.modules["domain_imports"].__file__).parent == S.REF_DIR, sys.modules["domain_imports"].__file__


def train_sizes_path(root: Path) -> Path:
    return root / "arms" / "train_sizes.json"


def slice_config(root: Path, n_train_per_class: dict[str, int] | None = None) -> PipelineConfig:
    """The pipeline config of this experiment under `root` (the folder, or dry/).

    Everything not set here is 261007's default, as for the baseline probe:
    pool and draw seeds 53, one eval row per prompt, the CV grid and folds,
    trainer, band width and mean-sigmoid score. Train sizes come from
    arms/train_sizes.json (written by run_pipeline.py) unless given.
    """
    if n_train_per_class is None:
        n_train_per_class = json.loads(train_sizes_path(root).read_text())["n_train_per_class"]
    cfg = PipelineConfig(
        arm=S.ARM,
        arms_dir=root / "arms",
        datasets=S.DATASETS,
        probe_kinds=S.PROBE_KINDS,
        train_samples_per_prompt=S.TRAIN_SAMPLES_PER_PROMPT,
        n_train_per_class=n_train_per_class,
    )
    assert set(cfg.probe_names()) == set(S.PROBES), cfg.probe_names()
    return cfg


def tokenizer_for(cfg: PipelineConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
