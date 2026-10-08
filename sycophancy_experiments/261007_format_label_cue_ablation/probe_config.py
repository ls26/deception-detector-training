"""One config object holding every knob of the probe pipeline, per arm.

Adapted from ../260924_probe_cv_selection/probe_config.py. Every fixed value
comes from spec.py; the only per-arm inputs are `arms/<arm>/labels.jsonl` and
the train sizes in `arms/sizes.json`. The procedure is the same for every arm.

Structure (removal definition `rm` only):

  DATASETS (one probe_sets/ file pair and one activation tensor pair each):
    bias_rm           turn-1 rollouts, label 1 = bias, 0 = no bias
    deception_rm      turn-2 rollouts, label 1 = deceptive, 0 = honest
    bias_followed_rm  turn-1 rollouts again, only rows whose answer went WITH
                      the cue -- its own draw, not a subset of bias_rm

  PROBES:
    bias_rm_answer                  bias family at the answer-letter token
    bias_followed_rm_answer
    deception_rm_preyesno           all deception rows, last token of the turn-2 prompt
    deception_yes_rm_preyesno       rows where said == "Yes"
    deception_no_rm_preyesno        rows where said == "No"

The three deception probes share one dataset and one activation file and
differ only by the `said` row mask.

`arms_dir` defaults to spec.ARMS_DIR; every script takes `--arms-dir` so the
fixture under tests_fixture/ runs the identical code. cv/ and runs/ live next
to the arms dir (EXP/cv, EXP/runs for the real run).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

import spec
from obfuscation_atlas.config import BaseTrainConfig

EXP_DIR = spec.EXP_DIR

#: Probe kinds served by each dataset family.
PROBE_KINDS: dict[str, tuple[str, ...]] = {
    "bias": ("bias",),
    "deception": ("deception", "deception_yes", "deception_no"),
    "bias_followed": ("bias_followed",),
}

#: Self-report token a probe kind restricts its rows to. None = all rows.
SAID_FILTER: dict[str, str | None] = {
    "bias": None,
    "deception": None,
    "deception_yes": "Yes",
    "deception_no": "No",
    "bias_followed": None,  # the dataset already holds only followed-cue rows
}

#: Cue conditions, in the order tables and balanced draws use.
CONDITIONS: tuple[str, ...] = spec.CUED_CONDITIONS

#: extraction position per family: the answer letter for the bias family; for
#: deception the last token of the turn-2 prompt, where P(Yes) and `said` are read
POSITION_OF_FAMILY: dict[str, str] = {"bias": "answer", "bias_followed": "answer", "deception": "preyesno"}

#: probe kinds whose train draw step 1 makes directly (`deception` is the union
#: of the two strata draws)
DRAWN_KINDS: tuple[str, ...] = ("bias", "deception_yes", "deception_no", "bias_followed")


@dataclass(frozen=True)
class ProbeSpec:
    """One trained probe: which dataset it reads, which rows, which position."""

    name: str  # e.g. "deception_yes_rm_preyesno"
    kind: str  # "bias" | "deception" | "deception_yes" | "deception_no" | "bias_followed"
    definition: str  # "rm"
    position: str  # "answer" | "preyesno"
    dataset: str  # probe_sets/activations basename, e.g. "deception_rm"
    family: str  # "bias" | "deception" | "bias_followed"
    said: str | None  # row filter on the self-report token


def _default_train_cfg() -> BaseTrainConfig:
    return BaseTrainConfig(
        max_steps=spec.CV_MAX_STEPS[0],
        batch_size=spec.PROBE_BATCH_SIZE,
        learning_rate=spec.CV_LEARNING_RATES[0],
        warmup_steps=0,
        scheduler="constant",
    )


def sizes_path(arms_dir: Path) -> Path:
    return arms_dir / "sizes.json"


def n_train_per_class_for(arm: str, arms_dir: Path) -> tuple[dict[str, int], tuple[str, ...]]:
    """(n_train_per_class, probe_kinds) for one arm from arms/sizes.json.

    rule "matched"      -> matched_n_per_class, the same for every arm
    rule "unmatched"    -> this arm's own full_n_per_class
    rule "insufficient" -> the probe kind is dropped from probe_kinds
    """
    sizes = json.loads(sizes_path(arms_dir).read_text())
    assert sizes["spec_hash"] == spec.spec_hash(), f"{sizes_path(arms_dir)} was written under another spec"
    n_train_per_class: dict[str, int] = {}
    probe_kinds: list[str] = []
    for kind in DRAWN_KINDS:
        entry = sizes["per_probe"][kind]
        rule = entry["rule"]
        assert rule in ("matched", "unmatched", "insufficient"), f"{kind}: unknown size rule {rule!r}"
        if rule == "insufficient":
            continue
        probe_kinds.append(kind)
        if rule == "matched":
            n_train_per_class[kind] = entry["matched_n_per_class"]
        else:
            n_train_per_class[kind] = entry["per_arm"][arm]["full_n_per_class"]
    # the unstratified deception probe trains on the union of the two strata
    # draws (SCHEMA.md: sizes.json has no entry of its own for it), so it exists
    # exactly when both strata do
    if "deception_yes" in probe_kinds and "deception_no" in probe_kinds:
        probe_kinds.append("deception")
    probe_kinds = [kind for kind in spec.PROBE_KINDS if kind in probe_kinds]
    for kind in n_train_per_class:
        assert n_train_per_class[kind] % len(CONDITIONS) == 0, (
            f"{kind}: n_train_per_class {n_train_per_class[kind]} does not split over {CONDITIONS}"
        )
    return n_train_per_class, tuple(probe_kinds)


@dataclass
class PipelineConfig:
    arm: str
    arms_dir: Path = spec.ARMS_DIR

    # --- what to build ---
    datasets: tuple[str, ...] = ("bias", "deception", "bias_followed")
    definitions: tuple[str, ...] = (spec.DEFINITION,)
    splits: tuple[str, ...] = ("train", "eval")
    probe_kinds: tuple[str, ...] = spec.PROBE_KINDS

    # --- step 1: pools + sampling ---
    #: rows per (question_id, condition) prompt admitted into each split's pool,
    #: applied AFTER the class filter so a prompt is never lost to a pick
    #: landing on an excluded row
    train_samples_per_prompt: int = spec.TRAIN_SAMPLES_PER_PROMPT
    eval_samples_per_prompt: int = spec.EVAL_SAMPLES_PER_PROMPT
    pool_seed: int = spec.POOL_SEED
    #: per-class train sizes keyed by DRAWN_KINDS, from arms/sizes.json
    n_train_per_class: dict[str, int] = field(default_factory=dict)
    sample_seed: int = spec.SAMPLE_SEED

    # --- step 2: model + extraction ---
    model_name: str = spec.MODEL_NAME
    n_layers: int = spec.N_LAYERS
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    batch_size: int = 16
    #: bfloat16 is lossless: the model runs in bfloat16, so every residual value
    #: is bfloat16-representable. Step 3 upcasts to float32 at load.
    act_save_dtype: torch.dtype = torch.bfloat16
    #: None = all rows. An int takes that many rows per class (dry runs).
    limit_per_class: int | None = None

    # --- step 3: probe training (the shared AdamW trainer, seeded) ---
    detector_arch: str = "linear-probe"
    normalize_input: str = "unit_norm"
    train_cfg: BaseTrainConfig = field(default_factory=_default_train_cfg)
    #: "band" = the fixed contiguous block `layer_band` chosen by cv_select.py;
    #: "loss_threshold" = layers whose final train loss is below
    #: layer_loss_threshold (all 32 when none passes); "all" = all 32
    layer_selection: str = "loss_threshold"
    layer_loss_threshold: float = 0.6
    #: (first layer, width) for layer_selection="band"
    layer_band: tuple[int, int] | None = None
    #: the one score per row the headline AUC is computed on, over the selected
    #: layers: "mean_sigmoid" (the RL pipeline's averaged score) or "mean_logit"
    combined_score: str = spec.COMBINED_SCORE
    #: probe name -> overrides applied on top of this config for that probe only
    #: (cv_select.py's chosen_overrides.json)
    per_probe_overrides: dict[str, dict] = field(default_factory=dict)
    eval_aggregator_method: str = "mean"
    probe_seed: int = spec.PROBE_SEED

    # --- cross-validation (cv_sweep.py / cv_select.py) ---
    cv_folds: int = spec.CV_FOLDS
    cv_seed: int = spec.CV_SEED
    cv_learning_rates: tuple[float, ...] = spec.CV_LEARNING_RATES
    cv_max_steps: tuple[int, ...] = spec.CV_MAX_STEPS
    cv_band_widths: tuple[int, ...] = (spec.CV_BAND_WIDTH,)

    # ----------------------------------------------------------------- #

    @property
    def arm_dir(self) -> Path:
        return self.arms_dir / self.arm

    @property
    def labels_path(self) -> Path:
        return self.arm_dir / "labels.jsonl"

    @property
    def probe_sets_dir(self) -> Path:
        return self.arm_dir / "probe_sets"

    @property
    def manifest_path(self) -> Path:
        return self.probe_sets_dir / "manifest.json"

    @property
    def act_dir(self) -> Path:
        return self.arm_dir / "activations"

    @property
    def cv_dir(self) -> Path:
        """cv/<arm>/<probe>/..., next to the arms dir."""
        return self.arms_dir.parent / "cv" / self.arm

    @property
    def run_dir(self) -> Path:
        """runs/<arm>/{config.json, probe_metrics.json, eval_scores.json, probe_weights/}"""
        return self.arms_dir.parent / "runs" / self.arm

    @property
    def weights_dir(self) -> Path:
        return self.run_dir / "probe_weights"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "probe_metrics.json"

    @property
    def eval_scores_path(self) -> Path:
        return self.run_dir / "eval_scores.json"

    def dataset_keys(self) -> list[tuple[str, str]]:
        """(family, definition) pairs -- one probe_sets/ file pair each."""
        return [(f, d) for f in self.datasets for d in self.definitions]

    def dataset_name(self, family: str, definition: str) -> str:
        return f"{family}_{definition}"

    def probe_specs(self) -> list[ProbeSpec]:
        """Every probe this config trains, in a stable order."""
        specs = []
        for family in self.datasets:
            for definition in self.definitions:
                for kind in PROBE_KINDS[family]:
                    if kind not in self.probe_kinds:
                        continue
                    position = POSITION_OF_FAMILY[family]
                    specs.append(
                        ProbeSpec(
                            name=f"{kind}_{definition}_{position}",
                            kind=kind,
                            definition=definition,
                            position=position,
                            dataset=self.dataset_name(family, definition),
                            family=family,
                            said=SAID_FILTER[kind],
                        )
                    )
        return specs

    def probe_names(self) -> list[str]:
        return [s.name for s in self.probe_specs()]

    def spec(self, name: str) -> ProbeSpec:
        for s in self.probe_specs():
            if s.name == name:
                return s
        raise KeyError(f"unknown probe {name!r}; have {self.probe_names()}")


def arm_config(arm: str, arms_dir: Path = spec.ARMS_DIR, **overrides) -> PipelineConfig:
    """The config for one arm: train sizes and probe kinds from arms/sizes.json,
    datasets restricted to the families that still have a probe."""
    assert arm in spec.ARMS, f"unknown arm {arm!r}; have {spec.ARMS}"
    n_train_per_class, probe_kinds = n_train_per_class_for(arm, arms_dir)
    datasets = tuple(family for family in ("bias", "deception", "bias_followed") if any(k in probe_kinds for k in PROBE_KINDS[family]))
    if "deception" in datasets:
        assert "deception_yes" in probe_kinds and "deception_no" in probe_kinds, (
            "the deception dataset is the union of the said-Yes and said-No draws; "
            f"a deception probe cannot be built with probe_kinds {probe_kinds}"
        )
    return PipelineConfig(
        arm=arm, arms_dir=arms_dir, datasets=datasets, probe_kinds=probe_kinds, n_train_per_class=n_train_per_class, **overrides
    )


def apply_overrides(cfg: PipelineConfig, overrides: dict) -> PipelineConfig:
    """`dataclasses.replace` from a JSON-style dict of PipelineConfig fields.

    `train_cfg` is merged into the existing BaseTrainConfig (a new object, so
    the shared default is never mutated); `layer_band` comes back from JSON as
    a list and is turned into a tuple.
    """
    overrides = dict(overrides)
    if "train_cfg" in overrides:
        cfg = dataclasses.replace(cfg, train_cfg=dataclasses.replace(cfg.train_cfg, **overrides.pop("train_cfg")))
    if overrides.get("layer_band") is not None:
        overrides["layer_band"] = tuple(overrides["layer_band"])
    return dataclasses.replace(cfg, **overrides)


def config_to_json(cfg: PipelineConfig) -> dict:
    """JSON-safe view of a config, for stamping into output files."""
    out: dict = {"spec_hash": spec.spec_hash()}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, Path):
            out[f.name] = str(v)
        elif isinstance(v, torch.dtype):
            out[f.name] = str(v)
        elif dataclasses.is_dataclass(v):
            out[f.name] = dataclasses.asdict(v)
        elif isinstance(v, tuple):
            out[f.name] = list(v)
        else:
            out[f.name] = v
    for name in ("probe_sets_dir", "act_dir", "cv_dir", "run_dir"):
        out[name] = str(getattr(cfg, name))
    return out
