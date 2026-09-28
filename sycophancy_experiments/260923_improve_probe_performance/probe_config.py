"""One config object holding every knob of the probe pipeline.

Trimmed copy of ../260826_probe_training_new/ with majority-of-3 counterfactual
labels as the only labeling. Structure, per reliance definition (`rm`, `swap`):

  DATASETS (what gets tokenized + forward-passed; one probe_sets/ file pair and
  one activation tensor pair per extraction position):
    bias_{def}           turn-1 rollouts, label 1 = bias, 0 = no bias
    deception_{def}      turn-2 rollouts, label 1 = deceptive, 0 = honest
    bias_followed_{def}  turn-1 rollouts again, but only rows whose answer went
                         WITH the cue -- its own draw, not a subset of bias_{def}

  PROBES (what gets trained):
    bias_{def}_{position}
    deception_{def}_{position}           all rows
    deception_yes_{def}_{position}       rows where said == "Yes"
    deception_no_{def}_{position}        rows where said == "No"
    bias_followed_{def}_{position}       rows that followed the cue

Defaults train 5 probes: definition `rm` only, bias-family probes at the
`answer` token, deception probes at `preyesno`. The other definition (`swap`)
and positions (`lastcompl`, `yesno`) are opted into through `definitions` and
`positions`; nothing else in the pipeline changes.

`bias` is perfectly nested inside cue-following -- zero bias-labelled rows are
against-cue -- so a bias probe can score by detecting "the answer matches the
cue" rather than "the cue caused it". `bias_followed` removes that shortcut:
both its classes answered with the cue, and they differ only in whether the cue
caused it.

The three deception probes share one dataset and one set of activations: within
the said-Yes rows, deceptive means *false admission* and honest means *true
admission*, so masking on the self-report token turns the same labels into the
subtype contrast with no relabelling and no second forward pass.

Mutate `cfg` in a notebook cell for a quick change, or `dataclasses.replace(cfg,
...)` for a variant that leaves the baseline object intact. Every function in
step1/step2/step3 takes a `cfg` explicitly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import torch

from obfuscation_atlas.config import BaseTrainConfig

EXP_DIR = Path(__file__).parent
SYCO_DIR = EXP_DIR.parent
#: the experiment whose trained probes and activations this one re-scores
PREVIOUS_DIR = SYCO_DIR / "260826_probe_training_new"


def _default_positions() -> dict[str, tuple[str, ...]]:
    """Extraction positions per dataset family. Valid names:
      bias / bias_followed: "answer" (the answer letter token), "lastcompl"
                            (last completion token), "lastprompt", "firstgen",
                            "meancompl" (mean over completion tokens)
      deception:            "yesno" (the sampled Yes/No self-report token),
                            "preyesno" (the token before it), "lastprompt",
                            "firstgen", "meancompl"
    Step 2 computes every valid position from one forward pass and saves only
    the ones listed here, so adding a position costs disk, not compute."""
    return {"bias": ("answer",), "bias_followed": ("answer",), "deception": ("preyesno",)}


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

    name: str  # e.g. "deception_yes_rm_preyesno"
    kind: str  # "bias" | "deception" | "deception_yes" | "deception_no" | "bias_followed"
    definition: str  # "rm" | "swap"
    position: str  # "answer" | "lastcompl" | "yesno" | "preyesno"
    dataset: str  # probe_sets/activations basename, e.g. "deception_rm"
    family: str  # "bias" | "deception" | "bias_followed"
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

    # --- outputs (all inside this folder; earlier experiments are never written to) ---
    probe_sets_dir: Path = EXP_DIR / "probe_sets"
    #: one subfolder per extraction variant (rows or positions changed), e.g.
    #: activations/base, activations/none_unanimous -- probe-training rounds reuse them
    act_dir: Path = EXP_DIR / "activations" / "base"
    weights_dir: Path = EXP_DIR / "probe_weights"
    manifest_path: Path = EXP_DIR / "probe_sets_manifest.json"
    metrics_path: Path = EXP_DIR / "probe_metrics.json"
    #: one subfolder per training round, created by run_log.start_round
    runs_dir: Path = EXP_DIR / "runs"
    #: eval AUC of PREVIOUS_DIR's probes, split by whether the 3 no-cue rollouts agree
    previous_probes_none_unanimous_path: Path = EXP_DIR / "previous_probes_none_unanimous.json"

    # --- what to build ---
    datasets: tuple[str, ...] = ("bias", "deception", "bias_followed")
    definitions: tuple[str, ...] = ("rm",)
    splits: tuple[str, ...] = ("train", "eval")
    probe_kinds: tuple[str, ...] = (
        "bias",
        "deception",
        "deception_yes",
        "deception_no",
        "bias_followed",
    )
    #: extraction positions per family (see _default_positions for valid names)
    positions: dict[str, tuple[str, ...]] = field(default_factory=_default_positions)

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
    #: row filter applied to the TRAIN split only, before the draw. "none" keeps
    #: every eligible row; "none_unanimous" keeps rows whose question's 3 no-cue
    #: rollouts all gave the same parsed answer (label_agreement.none_unanimous_index).
    #: Eval is never filtered. A cell left short by the filter warns and takes
    #: what is there.
    train_filter: str = "none"

    # --- step 2: model + extraction ---
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    n_layers: int = 32
    dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    batch_size: int = 16
    #: bfloat16 is lossless here: the model runs in bfloat16, so every residual
    #: value is bfloat16-representable (checked on a 260826 float32 file: 32/32
    #: layers round-trip exactly). Step 3 upcasts to float32 at load.
    act_save_dtype: torch.dtype = torch.bfloat16
    #: None = all rows. An int takes that many rows per class (dry runs).
    limit_per_class: int | None = None

    # --- step 3: probe training ---
    detector_arch: str = "linear-probe"
    #: input normalization inside the probe: "unit_norm" (divide by the train
    #: RMS norm, the shared default), "l2" (per-row unit length) or "none"
    normalize_input: str = "unit_norm"
    train_cfg: BaseTrainConfig = field(default_factory=_default_train_cfg)
    #: "sgd" = the shared AdamW trainer. "diff_of_means" = the class-mean
    #: difference direction; "lda" = the same direction whitened by the
    #: within-class covariance (shrunk toward its diagonal by lda_shrinkage).
    #: Both are fit in numpy on the cached activations and then get a 1-d
    #: logistic calibration (scale, bias) on the train logits so their train
    #: loss and layer-mean logit are comparable with the sgd probes.
    #: "logreg" = sklearn L2 logistic regression (lbfgs, converged) with inverse
    #: regularization logreg_C on the normalized activations; seed-independent.
    probe_fit: str = "sgd"
    lda_shrinkage: float = 0.1
    logreg_C: float = 100.0
    #: which layers the headline AUC averages over. "loss_threshold" = layers
    #: whose final train loss is below layer_loss_threshold (the shared rule).
    #: The rest select on EVAL AUC and are therefore in-sample: "best_eval" (the
    #: single best layer), "topk_eval" (the layer_k best), "band_eval" (the best
    #: contiguous band of layer_k layers), "all".
    layer_selection: str = "loss_threshold"
    layer_k: int = 8
    layer_loss_threshold: float = 0.6
    #: subtract each question's mean activation over its rows in the split
    #: (train and eval separately) before training / scoring
    center_by_question: bool = False
    #: train on the other four MMLU-Pro categories, eval on this one only
    holdout_category: str | None = None
    eval_aggregator_method: str = "mean"
    use_sklearn: bool = False
    #: NOT in 260825 -- nothing in obfuscation_atlas.detectors seeds anything, so
    #: 260825's AUCs are not bit-reproducible run to run. An int seeds torch
    #: immediately before each train_and_eval_detector call; None restores
    #: 260825's original unseeded path exactly.
    probe_seed: int | None = 0
    #: seeds used by step3.train_eval_all_seeds; each round reports mean and sd over them
    probe_seeds: tuple[int, ...] = (0, 1, 2)
    #: Step-3 draw from the TRAIN activation file, keyed like n_train_per_class.
    #: None = train on every row in the file (the step-1 draw). An int per kind
    #: takes that many rows per class, half per cue condition, from a fixed
    #: per-cell permutation seeded by sample_seed -- so draws of 100, 200, 400
    #: are nested. Lets one large extraction (activations/pool*) serve the whole
    #: learning curve. The `deception` probe is the union of the yes/no draws.
    train_draw_per_class: dict[str, int] | None = None

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
                    for position in self.positions[family]:
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


def variant_config(cfg: PipelineConfig, variant: str) -> PipelineConfig:
    """Point probe sets, manifest and activations at one extraction variant.

    `probe_sets/<variant>/` and `activations/<variant>/` always travel together:
    the activation rows are exactly the probe-set rows.
    """
    return dataclasses.replace(
        cfg,
        probe_sets_dir=EXP_DIR / "probe_sets" / variant,
        manifest_path=EXP_DIR / "probe_sets" / variant / "manifest.json",
        act_dir=EXP_DIR / "activations" / variant,
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
