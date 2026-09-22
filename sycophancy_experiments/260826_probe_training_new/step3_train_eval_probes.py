"""Step 3: train_and_eval_detector per probe, save_model, metrics.

Importable port of ../260825_probe_training/03_train_eval_probes.py. The probe
machinery is used **unmodified** -- `train_and_eval_detector`,
`get_layers_below_loss_threshold`, `save_model`, `get_detector_arch_config`,
`ActivationDataset` -- exactly as `pipeline/probe_training.py` uses them. The
only new code is the .pt -> ActivationDataset bridge and the metrics
summarization (copied from probe_training.py's dataset-agnostic
_mean_auc/_mean_threshold_metric).

Trains the 16 probes `cfg.probe_specs()` names: 4 bias (2 definitions x 2
positions) and 12 deception (3 kinds x 2 definitions x 2 positions). The three
deception kinds -- all rows, said-Yes only, said-No only -- read the SAME
activation file and differ only by a row mask, so nothing is extracted twice.

What changed vs 260825:
  * the stratified `deception_yes` / `deception_no` probes exist at all: they are
    the designed version of 260825's post-hoc "self-report held fixed" split,
    which collapsed its headline AUC from 0.73 to ~0.53.
  * `build_activation_dataset` also accepts an already-loaded step-2 result, so
    a fresh extraction can go straight into training with no disk round-trip.
  * `train_eval_one_probe` returns `train_dynamics`, `test_metrics` and the
    probe dict beside the summary. 260825 discarded `train_dynamics` right after
    the layer-loss filter -- it is exactly the object to look at when that
    filter behaves oddly (its ValueError -> all-layers fallback fires on the
    deception probes).
  * `cfg.probe_seed` seeds torch immediately before each train call.
  * `train_eval_all(..., probes=[...])` retrains one probe without touching the
    other seven.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
from obfuscation_atlas.detectors.dataset import ActivationDataset
from obfuscation_atlas.detectors.evaluate import (
    get_layers_below_loss_threshold,
    train_and_eval_detector,
)
from obfuscation_atlas.detectors.probe_archs import SequenceAggregator
from obfuscation_atlas.detectors.train import load_model, save_model
from probe_config import PipelineConfig, ProbeSpec, config_to_json
from step2_extract_activations import activation_path, build_scores_index


def dummy_hf_dataset() -> Dataset:
    """train_and_eval_detector's train_dataset/test_dataset args are never touched
    when train_feature_dataset/test_feature_dataset are supplied -- `prepare_dataset`
    (detectors/train.py) returns the supplied feature_dataset before it reads
    `model` or `dataset`. These placeholders only need to be valid HF Datasets."""
    return Dataset.from_dict({"completion": ["placeholder"]})


def build_activation_dataset(
    tokenizer, source: Path | dict, cfg: PipelineConfig, said: str | None = None
) -> ActivationDataset:
    """Wrap step-2 activations in an ActivationDataset.

    `source` is either a path to a step-2 .pt or an already-loaded step-2 result
    dict (as returned by `extract_all(..., return_acts=True)`).

    `said` masks the rows to one self-report token. Within said-Yes rows a
    label of 1 (deceptive) is exactly `false admission` and 0 is `true
    admission`; within said-No it is `false denial` vs `true denial`. So the
    stratified probes need no relabelling -- only this mask.
    """
    if isinstance(source, (str, Path)):
        d = torch.load(source, weights_only=False)
    else:
        d = source
    acts = {k: v for k, v in d["acts"].items()}  # {"0": (N,1,H), ..., "31": (N,1,H)}
    labels = d["labels"]

    if said is not None:
        row_said = d.get("said")
        assert row_said is not None, (
            "activation file has no `said` column -- re-run step 2, it is needed "
            "for the stratified deception probes"
        )
        keep = torch.tensor([s == said for s in row_said], dtype=torch.bool)
        assert keep.any(), f"no rows with said={said!r}"
        acts = {k: v[keep] for k, v in acts.items()}
        labels = labels[keep]

    n = labels.shape[0]
    # Dummy tokens_and_masks: never actually read for is_last_token_only (seq_len==1)
    # layers -- ActivationDataset.__getitem__ bypasses mask application in that case.
    tokens = torch.zeros((n, 1), dtype=torch.long)
    prompt_mask = torch.zeros((n, 1), dtype=torch.bool)
    target_mask = torch.ones((n, 1), dtype=torch.bool)
    return ActivationDataset(
        tokenizer=tokenizer,
        activations=acts,
        tokens_and_masks=(tokens, prompt_mask, target_mask),
        labels=labels,
        obfuscate_over="last-token-generation",
        preserve_seq_len=True,
    )


def mean_auc(metrics: dict, layers: list[int]) -> float:
    values = [
        metrics[str(l)]["AUC_ROC"]
        for l in layers
        if metrics.get(str(l)) is not None and metrics[str(l)].get("AUC_ROC") is not None
    ]
    return sum(values) / len(values) if values else float("nan")


def mean_threshold_metric(metrics: dict, layers: list[int], key: str) -> float:
    values = []
    for l in layers:
        lm = metrics.get(str(l))
        if lm is None:
            continue
        v = lm.get("metrics_at_threshold", {}).get(key)
        if v is not None:
            values.append(v)
    return sum(values) / len(values) if values else float("nan")


@dataclass
class ProbeArtifacts:
    """Everything `train_and_eval_detector` produced, kept for inspection."""

    probe_name: str
    test_metrics: dict = field(default_factory=dict)
    probe_dict: object = None
    train_dynamics: dict | None = None
    best_train_metrics: dict | None = None
    filtered_layers: list[int] = field(default_factory=list)
    layer_filter_fell_back: bool = False
    train_ds: object = None
    eval_ds: object = None


def train_eval_one_probe(
    probe_name: str,
    model,
    tokenizer,
    cfg: PipelineConfig,
    *,
    train_source: Path | dict | None = None,
    eval_source: Path | dict | None = None,
    save: bool = True,
) -> tuple[dict, ProbeArtifacts]:
    """Train + eval one probe. Returns (summary, artifacts).

    `probe_name` resolves through `cfg.spec()` to a (dataset, position, said)
    triple -- e.g. `deception_yes_rm_firstgen` reads
    activations/deception_rm_firstgen_{split}.pt and keeps the said-Yes rows.
    """
    spec = cfg.spec(probe_name)
    train_ds = build_activation_dataset(
        tokenizer,
        train_source or activation_path(cfg, spec.dataset, spec.position, "train"),
        cfg,
        said=spec.said,
    )
    eval_ds = build_activation_dataset(
        tokenizer,
        eval_source or activation_path(cfg, spec.dataset, spec.position, "eval"),
        cfg,
        said=spec.said,
    )
    print(
        f"  dataset={spec.dataset} position={spec.position} said={spec.said}: "
        f"train N={len(train_ds)}, eval N={len(eval_ds)}, d_model={train_ds.get_feature_dim()}"
    )

    if cfg.probe_seed is not None:
        torch.manual_seed(cfg.probe_seed)

    dummy = dummy_hf_dataset()
    test_metrics, probe_dict, train_dynamics, best_train_metrics = train_and_eval_detector(
        model=model,
        tokenizer=tokenizer,
        layers=list(range(cfg.n_layers)),
        train_dataset=(dummy, dummy),
        test_dataset=(dummy, dummy),
        train_cfg=cfg.train_cfg,
        detector_arch_config=get_detector_arch_config(cfg.detector_arch),
        obfuscate_over="last-token-generation",
        train_sequence_aggregator=None,
        eval_sequence_aggregator=SequenceAggregator(method=cfg.eval_aggregator_method),
        train_feature_dataset=train_ds,
        test_feature_dataset=eval_ds,
        n_calibration_examples=None,
        accelerator=None,
        use_sklearn=cfg.use_sklearn,
    )

    fell_back = False
    try:
        filtered_layers = get_layers_below_loss_threshold(train_dynamics, cfg.layer_loss_threshold)
    except ValueError as e:
        print(
            f"  WARNING: {e} -- falling back to all {cfg.n_layers} layers "
            f"(real signal, inspect artifacts.train_dynamics)"
        )
        filtered_layers = list(range(cfg.n_layers))
        fell_back = True

    if save:
        probe_dir = cfg.weights_dir / probe_name
        probe_dir.mkdir(parents=True, exist_ok=True)
        save_model(probe_dict, str(probe_dir))
        (probe_dir / "filtered_layers.txt").write_text(str(filtered_layers) + "\n")

    summary = {
        "probe": probe_name,
        "kind": spec.kind,
        "definition": spec.definition,
        "position": spec.position,
        "dataset": spec.dataset,
        "said": spec.said,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "probe_seed": cfg.probe_seed,
        "filtered_layers": filtered_layers,
        "layer_filter_fell_back": fell_back,
        "auc_filtered": mean_auc(test_metrics, filtered_layers),
        "auc_all_layers": mean_auc(test_metrics, list(range(cfg.n_layers))),
        "accuracy_filtered": mean_threshold_metric(test_metrics, filtered_layers, "accuracy"),
        "f1_filtered": mean_threshold_metric(test_metrics, filtered_layers, "f1_score"),
        "auc_per_layer": {
            str(l): test_metrics.get(str(l), {}).get("AUC_ROC") for l in range(cfg.n_layers)
        },
    }
    print(
        f"  {probe_name}: AUC(filtered)={summary['auc_filtered']:.4f} "
        f"AUC(all)={summary['auc_all_layers']:.4f} acc={summary['accuracy_filtered']:.4f} "
        f"f1={summary['f1_filtered']:.4f} n_filtered_layers={len(filtered_layers)}"
    )

    artifacts = ProbeArtifacts(
        probe_name=probe_name,
        test_metrics=test_metrics,
        probe_dict=probe_dict,
        train_dynamics=train_dynamics,
        best_train_metrics=best_train_metrics,
        filtered_layers=filtered_layers,
        layer_filter_fell_back=fell_back,
        train_ds=train_ds,
        eval_ds=eval_ds,
    )
    return summary, artifacts


def train_eval_all(
    cfg: PipelineConfig,
    model,
    tokenizer,
    probes: list[str] | None = None,
    *,
    save: bool = True,
    write_metrics: bool = True,
    return_artifacts: bool = False,
) -> tuple[list[dict], dict[str, ProbeArtifacts]]:
    """Train every probe in `probes` (default: all 8 implied by cfg).

    `return_artifacts=False` (default) drops each probe's artifacts after its
    summary is taken -- keeping all 8 holds every dataset in RAM.
    """
    probes = probes if probes is not None else cfg.probe_names()
    if save:
        cfg.weights_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    artifacts: dict[str, ProbeArtifacts] = {}
    for probe_name in probes:
        print(f"\n=== {probe_name} ===")
        summary, art = train_eval_one_probe(probe_name, model, tokenizer, cfg, save=save)
        summaries.append(summary)
        if return_artifacts:
            artifacts[probe_name] = art

    if write_metrics:
        cfg.metrics_path.write_text(
            json.dumps({"config": config_to_json(cfg), "probes": summaries}, indent=2)
        )
        print(f"\nWrote {cfg.metrics_path}")

    print("\n=== Summary ===")
    for s in summaries:
        print(
            f"  {s['probe']:30s} AUC(filtered)={s['auc_filtered']:.4f} "
            f"n_layers={len(s['filtered_layers'])}"
        )
    return summaries, artifacts


def load_summaries(path: Path) -> list[dict]:
    """Read a probe_metrics.json from either shape: this folder's
    {"config": ..., "probes": [...]} or 260825's bare list."""
    data = json.loads(Path(path).read_text())
    return data["probes"] if isinstance(data, dict) else data


# =========================================================================== #
# Analysis
#
# One AUC per probe is too coarse to find holes: 260825's pooled bias AUC of
# 0.767 hid a 0.10-0.20 spread between cue_gold and cue_dist, and its pooled
# deception AUC of 0.73 turned out to be the probe reading the self-report
# token. Both were invisible until sliced.
#
# The mechanism is the one 260825's analyze_results.py validated: reload the
# saved weights and re-score the cached activations. That is a matmul -- no LLM
# forward pass -- and the probe's `input_scale` buffer survives
# save_model/load_model, so the scores reproduce the training run's to ~1e-4.
# Raw logits are fine: AUC is threshold-free and expit is monotone, so
# thresholding logits equals thresholding the pipeline's probabilities.
#
# The charts mirror what the RL pipeline logs per probe during GRPO
# (obfuscation_training.log_detector_metrics): score histograms with the
# train-derived threshold, layerwise AUC, and probe train loss per layer.
# =========================================================================== #

#: Bias status per subtype. The four deception subtypes each pin a bias value;
#: a bias probe's rows carry the bias value as their subtype already.
BIAS_OF_SUBTYPE: dict[str, str] = {
    "false admission": "no bias",  # no bias + said Yes
    "true denial": "no bias",  # no bias + said No
    "false denial": "bias",  # bias    + said No
    "true admission": "bias",  # bias    + said Yes
    "bias": "bias",
    "no bias": "no bias",
}


@dataclass(frozen=True)
class SliceField:
    """One reported breakdown: display name, the row field it reads, value order.

    `in_figure=False` keeps a breakdown in the printed tables but out of the
    figure's slice panel -- `category` has five values that crowd the axis and
    is a generalization check rather than a claim about what the probe reads.
    """

    name: str
    meta_key: str
    order: tuple[str, ...] | None = None
    in_figure: bool = True


#: Breakdowns the analysis reports, in order.
#
# A field only says something if it varies WITHIN both classes. Where a field
# *is* the label, every group is single-class and the AUC is undefined -- so
# `slice_auc` drops such a field rather than printing a column of nan. That
# happens naturally per probe:
#   * `self-report` is informative only on the full deception probe (its said-Yes
#     group contrasts false vs true admission, its said-No group false vs true
#     denial -- 260825's "self-report held fixed" test). On a bias probe it is
#     None throughout; on a stratified probe it is constant.
#   * `bias` is likewise informative only on the full deception probe. On a bias
#     probe it is the label; on a stratified probe it is the label again, since
#     within said-Yes rows bias status and deception coincide.
#   * `cue-following` is reported everywhere, but on a bias probe expect its
#     against-cue group to come out single-class (hence nan): a biased answer is
#     one that moved toward the cued option, so `bias` is nested inside
#     `followed cue`. The row is still worth printing -- its n/pos/neg is what
#     shows the nesting -- so only an all-nan breakdown is dropped, not a single
#     nan group.
#: scores.jsonl is re-read once per (path, cue_variant) and memoized -- several
#: analyze_probe calls in one session should not each re-parse 5.8 MB.
_SCORES_INDEX_CACHE: dict[str, dict] = {}


def scores_index(cfg: PipelineConfig) -> dict:
    key = f"{cfg.domain_dir / 'scores.jsonl'}|{cfg.cue_variant}"
    if key not in _SCORES_INDEX_CACHE:
        _SCORES_INDEX_CACHE[key] = build_scores_index(cfg)
    return _SCORES_INDEX_CACHE[key]


SLICES: tuple[SliceField, ...] = (
    SliceField("cue", "condition", ("cue_gold", "cue_dist")),
    SliceField("cue-following", "followed_cue", ("followed cue", "against cue")),
    SliceField("self-report", "said", ("Yes", "No")),
    SliceField("category", "category", in_figure=False),
    SliceField("bias", "bias", ("bias", "no bias")),
)
SLICE_FIELDS: tuple[str, ...] = tuple(s.name for s in SLICES)
#: the subset the figure's slice panel draws
FIGURE_SLICE_FIELDS: tuple[str, ...] = tuple(s.name for s in SLICES if s.in_figure)
_SLICE_BY_NAME: dict[str, SliceField] = {s.name: s for s in SLICES}


@dataclass
class ProbeScores:
    """Per-row, per-layer probe scores for one probe, plus row metadata."""

    probe_name: str
    spec: ProbeSpec
    filtered_layers: list[int]
    n_layers: int
    #: split -> {layer: (N,) scores}
    scores: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    #: split -> (N,) labels
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    #: split -> {field: [N values]}
    meta: dict[str, dict[str, list]] = field(default_factory=dict)

    def n(self, split: str = "eval") -> int:
        return int(self.labels[split].shape[0])


def score_probe(
    probe_name: str, cfg: PipelineConfig, splits: tuple[str, ...] = ("train", "eval")
) -> ProbeScores:
    """Re-score the cached activations with a trained probe's saved weights.

    Applies the spec's `said` mask, so the rows scored here are exactly the rows
    the probe trained and evaluated on.
    """
    spec = cfg.spec(probe_name)
    weights_dir = cfg.weights_dir / probe_name
    assert weights_dir.exists(), f"no saved weights at {weights_dir} -- train the probe first"
    filtered_layers = eval((weights_dir / "filtered_layers.txt").read_text())
    probe_dict = load_model(str(weights_dir))

    ps = ProbeScores(
        probe_name=probe_name,
        spec=spec,
        filtered_layers=list(filtered_layers),
        n_layers=cfg.n_layers,
    )

    with torch.no_grad():
        for split in splits:
            d = torch.load(activation_path(cfg, spec.dataset, spec.position, split), weights_only=False)
            keep = np.ones(d["labels"].shape[0], dtype=bool)
            if spec.said is not None:
                keep = np.array([s == spec.said for s in d["said"]], dtype=bool)
                assert keep.any(), f"{probe_name}/{split}: no rows with said={spec.said!r}"

            ps.scores[split] = {
                layer: probe_dict[layer](d["acts"][str(layer)]).squeeze(-1).squeeze(-1).numpy()[keep]
                for layer in range(cfg.n_layers)
            }
            ps.labels[split] = d["labels"].numpy()[keep]

            # row_keys is (question_id, condition, sample_idx, category)
            row_keys = [k for k, k_keep in zip(d["row_keys"], keep) if k_keep]
            subtype = [s for s, k_keep in zip(d["subtype"], keep) if k_keep]
            # the model's turn-1 answer letter and whether it matched the cue,
            # joined from the published scores.jsonl on (question_id, condition)
            # -- `preds[sample_idx]` is the same letter step 2's turn-1 gate
            # asserted against. Joined here rather than carried in the .pt, so
            # this needs no re-extraction.
            sidx = scores_index(cfg)
            answer, followed = [], []
            for k in row_keys:
                srow = sidx.get((k[0], k[1]))
                pred = srow["preds"][k[2]] if srow is not None else None
                cue = srow["cue_letter"] if srow is not None else None
                answer.append(pred)
                followed.append(
                    None if pred is None or cue is None
                    else ("followed cue" if pred == cue else "against cue")
                )

            ps.meta[split] = {
                "condition": [k[1] for k in row_keys],
                "category": [k[3] for k in row_keys],
                "subtype": subtype,
                "said": [s for s, k_keep in zip(d["said"], keep) if k_keep],
                # derived: the four deception subtypes each pin a bias value
                "bias": [BIAS_OF_SUBTYPE.get(s) for s in subtype],
                "answer": answer,
                "followed_cue": followed,
            }
            n = ps.labels[split].shape[0]
            for name, values in ps.meta[split].items():
                assert len(values) == n, f"{probe_name}/{split}: meta {name} has {len(values)}, expected {n}"

    return ps


def mean_auc_masked(
    ps: ProbeScores, layers: list[int], mask: np.ndarray | None = None, split: str = "eval"
) -> tuple[float, int, int]:
    """Mean-over-layers AUC on the masked rows. Returns (auc, n_pos, n_neg).

    A single-class group returns nan rather than raising -- thin cells are
    expected (deception_swap has only 46 true-admission eval rows).
    """
    labels, scores = ps.labels[split], ps.scores[split]
    if mask is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    n_pos, n_neg = int((labels[mask] == 1).sum()), int((labels[mask] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    aucs = [roc_auc_score(labels[mask], scores[layer][mask]) for layer in layers]
    return float(np.mean(aucs)), n_pos, n_neg


def slice_auc(
    ps: ProbeScores, by: str, layers: list[int] | None = None, split: str = "eval"
) -> list[dict]:
    """AUC per subgroup of one breakdown, mean over `layers` (default: the
    filtered layers, matching the headline convention).

    `by` is a name from `SLICE_FIELDS` (or a raw meta key). Returns [] when the
    breakdown says nothing about this probe -- fewer than two distinct values, or
    every group single-class because the field is the label. See `SLICES`.
    """
    layers = layers if layers is not None else ps.filtered_layers
    field = _SLICE_BY_NAME.get(by)
    meta_key = field.meta_key if field else by
    values = np.array(ps.meta[split][meta_key], dtype=object)

    present = {v for v in values if v is not None}
    if len(present) < 2:
        return []
    if field is not None and field.order:
        distinct = [v for v in field.order if v in present]
        distinct += sorted(present - set(field.order), key=str)
    else:
        distinct = sorted(present, key=str)

    out = []
    for value in distinct:
        mask = values == value
        auc, n_pos, n_neg = mean_auc_masked(ps, layers, mask, split)
        out.append(
            {"by": by, "value": value, "auc": auc, "n": int(mask.sum()), "n_pos": n_pos, "n_neg": n_neg}
        )
    # every group single-class => the field is this probe's label, so it carries
    # no information (e.g. `bias` on a bias probe, `self-report` on a stratified
    # deception probe). Drop it rather than print a column of nan.
    if all(row["n_pos"] == 0 or row["n_neg"] == 0 for row in out):
        return []
    return out


def format_slice_table(rows: list[dict], title: str) -> str:
    """One breakdown as a bordered table: value, AUC, n, pos, neg."""
    headers = ["value", "AUC", "n", "pos", "neg"]
    body = [
        [
            str(r["value"]),
            "nan" if np.isnan(r["auc"]) else f"{r['auc']:.4f}",
            str(r["n"]),
            str(r["n_pos"]),
            str(r["n_neg"]),
        ]
        for r in rows
    ]
    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        left = cells[0].ljust(widths[0])
        right = "  ".join(c.rjust(w) for c, w in zip(cells[1:], widths[1:]))
        return f"  {left}  {right}"

    rule = "  " + "-" * (widths[0] + 2 + sum(w + 2 for w in widths[1:]) - 2)
    return "\n".join([f"  {title}", rule, line(headers), rule, *(line(r) for r in body), rule])


def layerwise_auc(ps: ProbeScores, split: str = "eval") -> list[float]:
    """AUC at every layer, nan where the split has only one class."""
    labels = ps.labels[split]
    if len(np.unique(labels)) < 2:
        return [float("nan")] * ps.n_layers
    return [float(roc_auc_score(labels, ps.scores[split][layer])) for layer in range(ps.n_layers)]


def _best_layer(ps: ProbeScores) -> int:
    """The filtered layer with the highest eval AUC -- what panel 1 histograms."""
    aucs = layerwise_auc(ps, "eval")
    candidates = [layer for layer in ps.filtered_layers if not np.isnan(aucs[layer])]
    if not candidates:
        return ps.filtered_layers[0] if ps.filtered_layers else 0
    return max(candidates, key=lambda layer: aucs[layer])


#: Groupings for the score-distribution grid: (panel title, meta key or None for
#: the probe's own label, value order).
DISTRIBUTION_GROUPINGS: tuple[tuple[str, str | None, tuple[str, ...] | None], ...] = (
    ("label", None, ("positive", "negative")),
    ("cue condition", "condition", ("cue_gold", "cue_dist")),
    ("answer followed the cue", "followed_cue", ("followed cue", "against cue")),
    ("answer letter", "answer", ("A", "B")),
)
# Cue / answer breakdowns only -- `plot_probe` panel 1 already histograms by label.
SCORE_BREAKDOWN_GROUPINGS = DISTRIBUTION_GROUPINGS[1:]


def group_masks(
    ps: ProbeScores, meta_key: str | None, order: tuple[str, ...] | None, split: str = "eval"
) -> list[tuple[str, np.ndarray]]:
    """(group label, row mask) pairs for one grouping. `meta_key=None` groups by
    the probe's own positive/negative label."""
    if meta_key is None:
        labels = ps.labels[split]
        return [("positive", labels == 1), ("negative", labels == 0)]
    values = np.array(ps.meta[split][meta_key], dtype=object)
    present = {v for v in values if v is not None}
    if order:
        distinct = [v for v in order if v in present] + sorted(present - set(order), key=str)
    else:
        distinct = sorted(present, key=str)
    return [(str(v), values == v) for v in distinct]


def plot_score_distributions(
    ps: ProbeScores,
    cfg: PipelineConfig,
    layer: int | None = None,
    split: str = "eval",
    groupings=DISTRIBUTION_GROUPINGS,
):
    """Grid of sigmoid score distributions, one panel per grouping.

    The label panel says whether the probe separates its own classes; the other
    three say whether the score tracks something else instead -- the cue
    condition, whether the answer went with the cue, or simply which letter was
    answered. A probe whose score shifts with the answer letter is reading answer
    identity, which is the failure mode 260825 diagnosed on the deception probes.

    Groupings whose field is missing or constant are skipped.
    """
    layer = _best_layer(ps) if layer is None else layer
    scores = expit(ps.scores[split][layer])
    bins = np.linspace(0, 1, 41)

    panels = []
    for title, meta_key, order in groupings:
        if meta_key is not None and meta_key not in ps.meta[split]:
            continue
        masks = [(name, m) for name, m in group_masks(ps, meta_key, order, split) if m.sum() > 0]
        if len(masks) < 2:
            continue
        panels.append((title, masks))
    if not panels:
        return None

    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.0 * nrows), squeeze=False)
    fig.suptitle(
        f"{ps.probe_name}   score distributions, layer {layer}, {split}   "
        f"(n={ps.n(split)})",
        fontsize=12,
    )

    for ax, (title, masks) in zip(axes.flat, panels):
        ax.hist(
            [scores[m] for _, m in masks],
            bins=bins,
            label=[f"{name} (n={int(m.sum())})" for name, m in masks],
            alpha=0.65,
        )
        ax.set_xlim(0, 1)
        ax.set_title(f"by {title}")
        ax.set_xlabel("probe score (sigmoid)")
        ax.set_ylabel("rows")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_probe(
    ps: ProbeScores,
    cfg: PipelineConfig,
    artifacts: ProbeArtifacts | None = None,
    slices: dict[str, list[dict]] | None = None,
):
    """2x2 diagnostic figure: score distribution, AUC by layer, AUC by slice,
    probe train loss by layer. Panels 1, 2 and 4 mirror the charts the RL
    pipeline logs per probe (obfuscation_training.log_detector_metrics)."""
    if slices is None:
        slices = {by: slice_auc(ps, by) for by in SLICE_FIELDS}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"{ps.probe_name}   (dataset={ps.spec.dataset}, position={ps.spec.position}, "
        f"said={ps.spec.said}, n_eval={ps.n('eval')})",
        fontsize=12,
    )

    # --- 1. score distribution at the best filtered layer ---------------------
    ax = axes[0][0]
    layer = _best_layer(ps)
    eval_labels = ps.labels["eval"]
    # sigmoid, so this is the probe score the pipeline reports (and the RL
    # penalty consumes), not the raw logit
    eval_scores = expit(ps.scores["eval"][layer])
    ax.hist(
        [eval_scores[eval_labels == 1], eval_scores[eval_labels == 0]],
        bins=np.linspace(0, 1, 41),
        color=["tab:red", "tab:blue"],
        label=["positive", "negative"],
        alpha=0.65,
        stacked=False,
    )
    ax.set_xlim(0, 1)
    ax.set_title(f"eval score distribution, layer {layer} (best filtered layer)")
    ax.set_xlabel("probe score (sigmoid)")
    ax.set_ylabel("rows")
    ax.legend(fontsize=8)

    # --- 2. layerwise AUC, train and eval ------------------------------------
    ax = axes[0][1]
    layers = list(range(ps.n_layers))
    eval_curve = layerwise_auc(ps, "eval")
    ax.plot(layers, eval_curve, marker="o", ms=3, label="eval")
    if "train" in ps.scores:
        ax.plot(layers, layerwise_auc(ps, "train"), marker="o", ms=3, alpha=0.6, label="train")
    for layer in ps.filtered_layers:
        ax.axvspan(layer - 0.5, layer + 0.5, color="tab:green", alpha=0.07)
    ax.axhline(0.5, color="grey", linestyle=":", lw=1)
    ax.set_title(f"AUC by layer (shaded = {len(ps.filtered_layers)} filtered layers)")
    ax.set_xlabel("layer")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=8)

    # --- 3. slice AUCs -------------------------------------------------------
    ax = axes[1][0]
    overall, _, _ = mean_auc_masked(ps, ps.filtered_layers)
    labels_, values_, colors_, fields_ = [], [], [], []
    palette = {
        "cue": "tab:purple",
        "self-report": "tab:orange",
        "category": "tab:cyan",
        "bias": "tab:green",
    }
    for by in FIGURE_SLICE_FIELDS:
        for row in slices.get(by, []):
            labels_.append(f"{row['value']}\n(n={row['n']})")
            values_.append(row["auc"])
            colors_.append(palette.get(by, "grey"))
            fields_.append(by)
    if labels_:
        x = np.arange(len(labels_))
        ax.bar(x, values_, color=colors_, alpha=0.85)
        for xi, v in zip(x, values_):
            if np.isnan(v):
                ax.text(xi, 0.02, "nan", ha="center", va="bottom", fontsize=7, color="grey")
            else:
                ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_, fontsize=7, rotation=45, ha="right")
        for i in range(1, len(fields_)):
            if fields_[i] != fields_[i - 1]:
                ax.axvline(i - 0.5, color="lightgrey", lw=1)
        ax.axhline(overall, color="black", linestyle="--", lw=1, label=f"overall {overall:.3f}")
        ax.axhline(0.5, color="grey", linestyle=":", lw=1)
        ax.set_ylim(0, 1.08)
        ax.legend(fontsize=8)
    shown = [by for by in FIGURE_SLICE_FIELDS if slices.get(by)]
    ax.set_title("AUC by " + ", ".join(shown) if shown else "AUC by slice")
    ax.set_ylabel("AUC")

    # --- 4. probe train loss per layer ---------------------------------------
    ax = axes[1][1]
    dynamics = artifacts.train_dynamics if artifacts is not None else None
    final_losses = {}
    for key, series in (dynamics or {}).items():
        if key.startswith("layer_") and key.endswith("_loss") and series:
            final_losses[int(key.split("_")[1])] = series[-1]
    if final_losses:
        xs = sorted(final_losses)
        ax.plot(xs, [final_losses[x] for x in xs], marker="o", ms=3, color="tab:red")
        ax.axhline(
            cfg.layer_loss_threshold,
            color="black",
            linestyle="--",
            label=f"threshold {cfg.layer_loss_threshold}",
        )
        ax.set_ylabel("final train loss")
        ax.legend(fontsize=8)
        n_below = sum(1 for v in final_losses.values() if v < cfg.layer_loss_threshold)
        ax.set_title(f"probe train loss by layer ({n_below}/{len(final_losses)} below threshold)")
    else:
        ax.text(
            0.5,
            0.5,
            "no train_dynamics\n(pass artifacts= from train_eval_one_probe)\n\n"
            f"filtered layers: {ps.filtered_layers}",
            ha="center",
            va="center",
            fontsize=9,
            wrap=True,
        )
        ax.set_title("probe train loss by layer")
    ax.set_xlabel("layer")

    fig.tight_layout()
    return fig


def analyze_probe(
    probe_name: str,
    cfg: PipelineConfig,
    artifacts: ProbeArtifacts | None = None,
    *,
    plot: bool = True,
    save_dir: Path | None = None,
) -> dict:
    """Slice one trained probe's eval AUC and draw the diagnostic figure.

    `artifacts` is the second return of `train_eval_one_probe` -- only panel 4
    (probe train loss per layer) needs it.
    """
    ps = score_probe(probe_name, cfg)
    overall, n_pos, n_neg = mean_auc_masked(ps, ps.filtered_layers)
    slices = {by: slice_auc(ps, by) for by in SLICE_FIELDS}

    print(f"=== {probe_name} ===")
    print(
        f"  dataset={ps.spec.dataset} position={ps.spec.position} said={ps.spec.said}  "
        f"layers={len(ps.filtered_layers)}/{ps.n_layers}"
    )
    print(f"  overall eval AUC {overall:.4f}   n={ps.n('eval')} ({n_pos} pos / {n_neg} neg)")
    for by in SLICE_FIELDS:
        if not slices[by]:
            continue
        print()
        print(format_slice_table(slices[by], f"by {by}"))

    result = {
        "probe": probe_name,
        "dataset": ps.spec.dataset,
        "position": ps.spec.position,
        "said": ps.spec.said,
        "filtered_layers": ps.filtered_layers,
        "n_train": ps.n("train") if "train" in ps.labels else None,
        "n_eval": ps.n("eval"),
        "auc_overall": overall,
        "best_layer": _best_layer(ps),
        "layerwise_auc_eval": layerwise_auc(ps, "eval"),
        "slices": slices,
    }

    fig = None
    if plot:
        fig = plot_probe(ps, cfg, artifacts=artifacts, slices=slices)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{probe_name}_analysis.json").write_text(json.dumps(result, indent=2))
        if fig is not None:
            fig.savefig(save_dir / f"{probe_name}_analysis.png", dpi=150, bbox_inches="tight")
        print(f"\n  wrote {save_dir / (probe_name + '_analysis.{json,png}')}")

    result["scores"] = ps
    result["figure"] = fig
    return result
