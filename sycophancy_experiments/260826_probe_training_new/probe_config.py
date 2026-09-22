"""One config object holding every knob 260825's three scripts hardcoded.

Structure of this experiment, per reliance definition (`rm`, `swap`):

  DATASETS (what gets tokenized + forward-passed; one probe_sets/ file pair and
  one activation tensor pair each):
    bias_{def}           turn-1 rollouts, label 1 = bias, 0 = no bias
    deception_{def}      turn-2 rollouts, label 1 = deceptive, 0 = honest
    bias_followed_{def}  turn-1 rollouts again, but only rows whose answer went
                         WITH the cue -- its own draw, not a subset of bias_{def}

  PROBES (what gets trained; 20 total = 4 bias + 12 deception + 4 bias-followed):
    bias_{def}_{answer,lastcompl}
    deception_{def}_{yesno,preyesno}           all rows
    deception_yes_{def}_{yesno,preyesno}       rows where said == "Yes"
    deception_no_{def}_{yesno,preyesno}        rows where said == "No"
    bias_followed_{def}_{answer,lastcompl}     rows that followed the cue

`bias` is perfectly nested inside cue-following -- across both definitions and
both splits, zero bias-labelled rows are against-cue -- so a bias probe can score
by detecting "the answer matches the cue" rather than "the cue caused it".
`bias_followed` removes that shortcut: both its classes answered with the cue,
and they differ only in whether the cue caused it. It is the bias-side analogue
of the stratified deception probes. An against-cue counterpart cannot exist,
since it would have no positive class.

The three deception probes share one dataset and one set of activations: within
the said-Yes rows, deceptive means *false admission* and honest means *true
admission*, so masking on the self-report token turns the same labels into the
subtype contrast with no relabelling and no second forward pass. Those two
stratified probes are the designed version of 260825's post-hoc "self-report
held fixed" split, which collapsed its headline AUC from 0.73 to ~0.53.

Defaults reproduce 260825 where the two overlap. Mutate `cfg` in a notebook cell
for a quick change, or `dataclasses.replace(cfg, ...)` for a variant that leaves
the baseline object intact. Every function in step1/step2/step3 takes a `cfg`
explicitly -- no module-level constants to reach behind, no `global` reassignment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import torch

from obfuscation_atlas.config import BaseTrainConfig

EXP_DIR = Path(__file__).parent
SYCO_DIR = EXP_DIR.parent
REFERENCE_DIR = SYCO_DIR / "260825_probe_training"

#: Extraction positions per dataset. Position A first (the design doc's choice),
#: position B second (the alternative 260825 tested against it).
POSITIONS: dict[str, tuple[str, ...]] = {
    "bias": ("answer", "lastcompl"),
    # same rollouts and positions as `bias`, but only rows whose answer went
    # with the cue -- see PROBE_KINDS below
    "bias_followed": ("answer", "lastcompl"),
    # The self-report token itself and the one before it, located by the SAMPLED
    # Yes/No token (`ans_pos`) rather than by assuming it is the first generated
    # token. 260817 measured that assumption holding 99.6% of the time, so on the
    # residual rows 260825's `_firstgen` probed a preamble token instead.
    "deception": ("yesno", "preyesno"),
}

#: Probe kinds served by each dataset.
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
CONDITIONS: tuple[str, ...] = ("cue_gold", "cue_dist")


@dataclass(frozen=True)
class ProbeSpec:
    """One trained probe: which dataset it reads, which rows, which position."""

    name: str  # e.g. "deception_yes_rm_firstgen"
    kind: str  # "bias" | "deception" | "deception_yes" | "deception_no"
    definition: str  # "rm" | "swap"
    position: str  # "answer" | "lastcompl" | "firstgen" | "lastprompt"
    dataset: str  # probe_sets/activations basename, e.g. "deception_rm"
    family: str  # "bias" | "deception"
    said: str | None  # row filter on the self-report token


def _default_train_cfg() -> BaseTrainConfig:
    return BaseTrainConfig(
        max_steps=1000, batch_size=32, learning_rate=3e-3, warmup_steps=0, scheduler="constant"
    )


def _default_n_train_per_class() -> dict[str, int]:
    """Per-class train sizes. `deception` is not drawn directly -- its train set
    is the union of the two stratified draws, giving 200+200 = 400 per class.
    `bias_followed` matches `bias` so an AUC difference between the two cannot be
    a train-size effect."""
    return {"bias": 400, "deception_yes": 200, "deception_no": 200, "bias_followed": 400}


@dataclass
class PipelineConfig:
    # --- source data (read-only; produced by earlier experiments) ---
    domain_dir: Path = SYCO_DIR / "260819_domain_baseline"
    followup_dir: Path = SYCO_DIR / "260823_domain_baseline_followup_wordings"
    rescore_dir: Path = SYCO_DIR / "260824_rescore"
    cue_variant: str = "random"
    #: How counterfactual answers enter bias labels. `indexed` uses
    #: rescored_labels.jsonl as written (sample i paired with counterfactual i).
    #: `majority3` replaces a_none / a_other_cue with the parsed majority over
    #: that condition's 3 samples before recomputing bias and deception.
    counterfactual_vote: str = "indexed"

    # --- outputs (all inside this folder; 260825 is never written to) ---
    probe_sets_dir: Path = EXP_DIR / "probe_sets"
    act_dir: Path = EXP_DIR / "activations"
    weights_dir: Path = EXP_DIR / "probe_weights"
    manifest_path: Path = EXP_DIR / "probe_sets_manifest.json"
    metrics_path: Path = EXP_DIR / "probe_metrics.json"

    # --- what to build ---
    #: `bias_followed` is appended so the original 16 probes keep their order
    datasets: tuple[str, ...] = ("bias", "deception", "bias_followed")
    definitions: tuple[str, ...] = ("rm", "swap")
    splits: tuple[str, ...] = ("train", "eval")
    probe_kinds: tuple[str, ...] = (
        "bias",
        "deception",
        "deception_yes",
        "deception_no",
        "bias_followed",
    )

    # --- step 1: pools + sampling ---
    #: Rows per (question_id, condition) prompt admitted into each split's pool,
    #: applied AFTER filtering to the dataset's two label classes so a prompt is
    #: never lost to a pick landing on an excluded row. None = keep all 3.
    #: 260825 kept all 3 in both splits and deduplicated eval post-hoc in
    #: analyze_results.py; here eval is deduplicated up front instead.
    train_samples_per_prompt: int | None = None
    eval_samples_per_prompt: int | None = 1
    pool_seed: int = 53
    #: per-class train sizes, keyed by probe kind (see _default_n_train_per_class)
    n_train_per_class: dict[str, int] = field(default_factory=_default_n_train_per_class)
    #: split each class's train draw equally across cue_gold / cue_dist
    balance_conditions: bool = True
    sample_seed: int = 53

    # --- step 2: model + extraction ---
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    n_layers: int = 32
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    batch_size: int = 16
    act_save_dtype: torch.dtype = torch.float32
    #: None = all rows. An int takes that many rows per class (dry runs).
    limit_per_class: int | None = None

    # --- step 3: probe training ---
    detector_arch: str = "linear-probe"
    train_cfg: BaseTrainConfig = field(default_factory=_default_train_cfg)
    layer_loss_threshold: float = 0.6
    eval_aggregator_method: str = "mean"
    use_sklearn: bool = False
    #: NOT in 260825 -- nothing in obfuscation_atlas.detectors seeds anything, so
    #: 260825's AUCs are not bit-reproducible run to run. An int seeds torch
    #: immediately before each train_and_eval_detector call; None restores
    #: 260825's original unseeded path exactly.
    probe_seed: int | None = 0

    # ----------------------------------------------------------------- #

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
                    for position in POSITIONS[family]:
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


def default_config(**overrides) -> PipelineConfig:
    """`PipelineConfig` with this experiment's settings, plus keyword overrides."""
    return PipelineConfig(**overrides)


def majority3_config(**overrides) -> PipelineConfig:
    """Config for majority-of-3 counterfactual labeling with non-colliding outputs."""
    return default_config(
        counterfactual_vote="majority3",
        probe_sets_dir=EXP_DIR / "probe_sets_maj3",
        act_dir=EXP_DIR / "activations_maj3",
        weights_dir=EXP_DIR / "probe_weights_maj3",
        manifest_path=EXP_DIR / "probe_sets_manifest_maj3.json",
        metrics_path=EXP_DIR / "probe_metrics_maj3.json",
        **overrides,
    )


def config_to_json(cfg: PipelineConfig) -> dict:
    """JSON-safe view of a config, for stamping into output files.

    With a mutable cfg in a live kernel, this stamp is what makes a result
    attributable to the config that produced it after the fact.
    """
    out: dict = {}
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
    return out
