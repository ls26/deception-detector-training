"""Step 3: train_and_eval_detector per probe, save_model, metrics, eval scores.

Lean copy of ../260924_probe_cv_selection/step3_train_eval_probes.py: the
shared AdamW trainer (`train_and_eval_detector`, seeded) on cached
activations, one linear probe per layer, the headline being `auc_combined` --
the AUC of one combined score per row over the selected layers
(`cfg.combined_score`, mean sigmoid = the RL pipeline's averaged score).
Dropped: the numpy fits, the multi-seed aggregation, every row filter beyond
`said`, the analysis and plots. Added: per-row eval scores, written to
eval_scores.json for compare.py's bootstrap.

The three deception kinds -- all rows, said-Yes only, said-No only -- read the
SAME activation file and differ only by the `said` row mask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from scipy.special import expit
from sklearn.metrics import roc_auc_score

import spec
from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
from obfuscation_atlas.detectors.dataset import ActivationDataset
from obfuscation_atlas.detectors.evaluate import train_and_eval_detector
from obfuscation_atlas.detectors.probe_archs import SequenceAggregator
from obfuscation_atlas.detectors.train import save_model
from probe_config import PipelineConfig, ProbeSpec, apply_overrides, config_to_json


def activation_path(cfg: PipelineConfig, dataset: str, position: str, split: str) -> Path:
    """One file per (dataset, position, split) -- NOT per probe."""
    return cfg.act_dir / f"{dataset}_{position}_{split}.pt"


def load_activation_file(path: Path) -> dict:
    d = torch.load(path, weights_only=False)
    assert d["spec_hash"] == spec.spec_hash(), f"{path} was extracted under another spec"
    return d


def dummy_hf_dataset() -> Dataset:
    """train_and_eval_detector's train_dataset/test_dataset args are never touched
    when train_feature_dataset/test_feature_dataset are supplied; these
    placeholders only need to be valid HF Datasets."""
    return Dataset.from_dict({"completion": ["placeholder"]})


def said_mask(d: dict, said: str | None) -> torch.Tensor:
    """Row mask for one self-report token over a loaded step-2 file (all rows if None)."""
    n = d["labels"].shape[0]
    if said is None:
        return torch.ones(n, dtype=torch.bool)
    keep = torch.tensor([s == said for s in d["said"]], dtype=torch.bool)
    assert keep.any(), f"no rows with said={said!r}"
    return keep


def row_mask(d: dict, probe: ProbeSpec) -> torch.Tensor:
    """Every row filter a probe applies to a loaded file: only the `said` mask."""
    return said_mask(d, probe.said)


def build_activation_dataset(tokenizer, source: Path | dict, cfg: PipelineConfig, said: str | None = None, extra_mask: torch.Tensor | None = None) -> ActivationDataset:
    """Wrap step-2 activations in an ActivationDataset.

    `said` masks the rows to one self-report token: within said-Yes rows a label
    of 1 (deceptive) is exactly `false admission` and 0 `true admission`; within
    said-No it is `false denial` vs `true denial`. `extra_mask` (a CV fold) is
    applied on top.
    """
    d = load_activation_file(source) if isinstance(source, (str, Path)) else source
    # files are stored in bfloat16 (lossless, see cfg.act_save_dtype); train in float32
    acts = {k: v.float() for k, v in d["acts"].items()}  # {"0": (N,1,H), ..., "31": (N,1,H)}
    labels = d["labels"]

    keep = said_mask(d, said)
    if extra_mask is not None:
        keep = keep & extra_mask
    if not keep.all():
        acts = {k: v[keep] for k, v in acts.items()}
        labels = labels[keep]

    n = labels.shape[0]
    # Dummy tokens_and_masks: never read for seq_len == 1 activations.
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
    return float(np.mean([metrics[str(l)]["AUC_ROC"] for l in layers]))


def probe_auc(probe_dict: dict, ds: ActivationDataset, layer: int) -> float:
    """AUC of one layer's probe on a dataset (used for the train-set AUC)."""
    with torch.no_grad():
        logits = probe_dict[layer](ds.activations[str(layer)]).squeeze(-1).squeeze(-1).numpy()
    return float(roc_auc_score(ds.labels.numpy(), logits))


def layer_logits(probe_dict: dict, ds: ActivationDataset, layers: list[int]) -> np.ndarray:
    """(len(layers), N) raw probe logits. `probe_dict` is `{layer: probe}` on CPU
    as `train_and_eval_detector` returns it."""
    with torch.no_grad():
        return np.stack([probe_dict[layer](ds.activations[str(layer)]).squeeze(-1).squeeze(-1).numpy() for layer in layers])


COMBINED_SCORE_METHODS: tuple[str, ...] = ("mean_sigmoid", "mean_logit")


def combined_score(logits: np.ndarray, method: str) -> np.ndarray:
    """One score per row from (L, N) per-layer logits. "mean_sigmoid" is exactly
    the RL pipeline's averaged score (sigmoid per layer, then mean over the
    retained layers); "mean_logit" is the mean raw logit."""
    assert method in COMBINED_SCORE_METHODS, method
    if method == "mean_sigmoid":
        return expit(logits).mean(axis=0)
    return logits.mean(axis=0)


def combined_auc(labels: np.ndarray, logits: np.ndarray, method: str) -> float:
    return float(roc_auc_score(labels, combined_score(logits, method)))


def band_layers(band: tuple[int, int], n_layers: int) -> list[int]:
    start, width = band
    assert 0 <= start and start + width <= n_layers, f"band {band} does not fit in {n_layers} layers"
    return list(range(start, start + width))


def final_train_loss_per_layer(train_dynamics: dict, n_layers: int) -> list[float]:
    return [train_dynamics[f"layer_{l}_loss"][-1] for l in range(n_layers)]


def layers_below_loss_threshold(train_dynamics: dict, threshold: float, n_layers: int) -> tuple[list[int], bool]:
    """The pipeline's layer rule: layers whose final train loss is below
    `threshold`; all layers when none passes (`fell_back`)."""
    losses = final_train_loss_per_layer(train_dynamics, n_layers)
    layers = [l for l in range(n_layers) if losses[l] < threshold]
    if layers:
        return layers, False
    return list(range(n_layers)), True


def select_layers(train_dynamics: dict, cfg: PipelineConfig) -> tuple[list[int], bool]:
    """(selected layers, loss rule fell back). None of the modes looks at eval:
    "band" is cv_select.py's choice, "loss_threshold" the pipeline's train-loss
    rule, "all" all layers."""
    if cfg.layer_selection == "all":
        return list(range(cfg.n_layers)), False
    if cfg.layer_selection == "band":
        assert cfg.layer_band is not None, "layer_selection='band' needs cfg.layer_band"
        return band_layers(cfg.layer_band, cfg.n_layers), False
    if cfg.layer_selection == "loss_threshold":
        return layers_below_loss_threshold(train_dynamics, cfg.layer_loss_threshold, cfg.n_layers)
    raise ValueError(f"unknown layer_selection {cfg.layer_selection!r}")


@dataclass
class ProbeArtifacts:
    """Everything `train_and_eval_detector` produced, kept for inspection."""

    probe_name: str
    test_metrics: dict = field(default_factory=dict)
    probe_dict: object = None
    train_dynamics: dict | None = None
    filtered_layers: list[int] = field(default_factory=list)
    train_ds: object = None
    eval_ds: object = None


def fit_layer_probes(train_ds: ActivationDataset, eval_ds: ActivationDataset, cfg: PipelineConfig, tokenizer) -> tuple[dict, dict, dict]:
    """Fit one linear probe per layer on `train_ds` with the shared AdamW
    trainer, scoring `eval_ds` per layer. Returns (test_metrics, probe_dict,
    train_dynamics); `probe_dict` is `{layer: probe}` on CPU. Seeds torch from
    `cfg.probe_seed` first. Shared by `train_eval_one_probe` (eval_ds = the
    eval split) and cv_sweep.py (eval_ds = the held-out fold)."""
    torch.manual_seed(cfg.probe_seed)
    dummy = dummy_hf_dataset()
    test_metrics, probe_dict, train_dynamics, _best_train_metrics = train_and_eval_detector(
        model=None,
        tokenizer=tokenizer,
        layers=list(range(cfg.n_layers)),
        train_dataset=(dummy, dummy),
        test_dataset=(dummy, dummy),
        train_cfg=cfg.train_cfg,
        detector_arch_config=get_detector_arch_config(cfg.detector_arch, normalize_input=cfg.normalize_input),
        obfuscate_over="last-token-generation",
        train_sequence_aggregator=None,
        eval_sequence_aggregator=SequenceAggregator(method=cfg.eval_aggregator_method),
        train_feature_dataset=train_ds,
        test_feature_dataset=eval_ds,
        n_calibration_examples=None,
        accelerator=None,
        use_sklearn=False,
    )
    return test_metrics, probe_dict, train_dynamics


def eval_row_records(eval_file: dict, keep: torch.Tensor) -> tuple[list[int], list[str]]:
    """(question_id, condition) of the eval rows a probe scored, in dataset order."""
    keys = [k for k, k_keep in zip(eval_file["row_keys"], keep) if k_keep]
    return [int(k[0]) for k in keys], [str(k[1]) for k in keys]


def train_eval_one_probe(probe_name: str, tokenizer, cfg: PipelineConfig, *, save: bool = True) -> tuple[dict, ProbeArtifacts, dict]:
    """Train + eval one probe. Returns (summary, artifacts, eval_scores).

    `cfg.per_probe_overrides[probe_name]`, if present, is applied first.
    `eval_scores` is SCHEMA.md's per-row record: question_id, condition, label
    and the combined score of every eval row the probe scored.
    """
    cfg = apply_overrides(cfg, cfg.per_probe_overrides.get(probe_name, {}))
    probe = cfg.spec(probe_name)
    train_file = load_activation_file(activation_path(cfg, probe.dataset, probe.position, "train"))
    eval_file = load_activation_file(activation_path(cfg, probe.dataset, probe.position, "eval"))
    train_ds = build_activation_dataset(tokenizer, train_file, cfg, said=probe.said)
    eval_ds = build_activation_dataset(tokenizer, eval_file, cfg, said=probe.said)
    print(f"  dataset={probe.dataset} position={probe.position} said={probe.said}: train N={len(train_ds)}, eval N={len(eval_ds)}, d_model={train_ds.get_feature_dim()}")

    test_metrics, probe_dict, train_dynamics = fit_layer_probes(train_ds, eval_ds, cfg, tokenizer)
    filtered_layers, fell_back = select_layers(train_dynamics, cfg)
    print(f"  layer_selection={cfg.layer_selection}: {filtered_layers}")

    if save:
        probe_dir = cfg.weights_dir / probe_name
        probe_dir.mkdir(parents=True, exist_ok=True)
        save_model(probe_dict, str(probe_dir))
        (probe_dir / "filtered_layers.txt").write_text(str(filtered_layers) + "\n")

    eval_labels = eval_ds.labels.numpy()
    eval_logits_selected = layer_logits(probe_dict, eval_ds, filtered_layers)
    auc_mean_sigmoid = combined_auc(eval_labels, eval_logits_selected, "mean_sigmoid")
    auc_mean_logit = combined_auc(eval_labels, eval_logits_selected, "mean_logit")
    eval_scores_combined = combined_score(eval_logits_selected, cfg.combined_score)

    train_losses = final_train_loss_per_layer(train_dynamics, cfg.n_layers)
    train_auc_per_layer = {str(l): probe_auc(probe_dict, train_ds, l) for l in range(cfg.n_layers)}
    train_labels = train_ds.labels.numpy()

    summary = {
        "probe": probe_name,
        "kind": probe.kind,
        "definition": probe.definition,
        "position": probe.position,
        "dataset": probe.dataset,
        "said": probe.said,
        "n_train": len(train_ds),
        "n_train_pos": int((train_labels == 1).sum()),
        "n_train_neg": int((train_labels == 0).sum()),
        "n_eval": len(eval_ds),
        "n_eval_pos": int((eval_labels == 1).sum()),
        "n_eval_neg": int((eval_labels == 0).sum()),
        "probe_seed": cfg.probe_seed,
        "learning_rate": cfg.train_cfg.learning_rate,
        "max_steps": cfg.train_cfg.max_steps,
        "batch_size": cfg.train_cfg.batch_size,
        "epochs": cfg.train_cfg.max_steps * cfg.train_cfg.batch_size / len(train_ds),
        "layer_selection": cfg.layer_selection,
        "layer_band": list(cfg.layer_band) if cfg.layer_band is not None else None,
        "filtered_layers": filtered_layers,
        "layer_filter_fell_back": fell_back,
        "combined_score": cfg.combined_score,
        "auc_combined": auc_mean_sigmoid if cfg.combined_score == "mean_sigmoid" else auc_mean_logit,
        "auc_mean_sigmoid": auc_mean_sigmoid,
        "auc_mean_logit": auc_mean_logit,
        "auc_filtered": mean_auc(test_metrics, filtered_layers),
        "auc_all_layers": mean_auc(test_metrics, list(range(cfg.n_layers))),
        "auc_per_layer": {str(l): test_metrics[str(l)]["AUC_ROC"] for l in range(cfg.n_layers)},
        "train_loss_final_per_layer": {str(l): train_losses[l] for l in range(cfg.n_layers)},
        "train_auc_per_layer": train_auc_per_layer,
        "train_auc_filtered": float(np.mean([train_auc_per_layer[str(l)] for l in filtered_layers])),
        "train_loss_min": min(train_losses),
    }
    print(
        f"  {probe_name}: AUC(combined {cfg.combined_score})={summary['auc_combined']:.4f} AUC(mean logit)={auc_mean_logit:.4f} "
        f"AUC(per-layer mean, filtered)={summary['auc_filtered']:.4f} n_layers={len(filtered_layers)}  "
        f"lr={cfg.train_cfg.learning_rate} steps={cfg.train_cfg.max_steps} epochs={summary['epochs']:.1f}  "
        f"train AUC(filtered)={summary['train_auc_filtered']:.4f} min train loss={summary['train_loss_min']:.3f}"
    )

    question_ids, conditions = eval_row_records(eval_file, said_mask(eval_file, probe.said))
    eval_scores = {"question_id": question_ids, "condition": conditions, "label": eval_labels.tolist(), "score": eval_scores_combined.tolist()}
    artifacts = ProbeArtifacts(
        probe_name=probe_name, test_metrics=test_metrics, probe_dict=probe_dict, train_dynamics=train_dynamics, filtered_layers=filtered_layers, train_ds=train_ds, eval_ds=eval_ds
    )
    return summary, artifacts, eval_scores


def train_eval_all(cfg: PipelineConfig, tokenizer, probes: list[str] | None = None, *, save: bool = True) -> list[dict]:
    """Train every probe in `probes` (default: all of cfg.probe_names()); write
    probe_metrics.json and eval_scores.json into cfg.run_dir."""
    probes = probes if probes is not None else cfg.probe_names()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    eval_scores: dict[str, dict] = {}
    for probe_name in probes:
        print(f"\n=== {probe_name} ===")
        summary, _artifacts, scores = train_eval_one_probe(probe_name, tokenizer, cfg, save=save)
        summaries.append(summary)
        eval_scores[probe_name] = scores

    cfg.metrics_path.write_text(json.dumps({"spec_hash": spec.spec_hash(), "arm": cfg.arm, "config": config_to_json(cfg), "probes": summaries}, indent=2))
    cfg.eval_scores_path.write_text(json.dumps({"spec_hash": spec.spec_hash(), "arm": cfg.arm, **eval_scores}))
    print(f"\nWrote {cfg.metrics_path} and {cfg.eval_scores_path}")

    print("\n=== Summary ===")
    for s in summaries:
        print(f"  {s['probe']:30s} AUC(combined)={s['auc_combined']:.4f} AUC(filtered)={s['auc_filtered']:.4f} n_layers={len(s['filtered_layers'])}")
    return summaries
