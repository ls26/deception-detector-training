"""Fit the three probes on fit B's rows with one of three target sources.

    python train_soft.py --targets int      # fit B's int labels, UNCHANGED library path
                                            # (261007's fit_layer_probes -> train_and_eval_detector)
    python train_soft.py --targets hard01   # fit B's labels as float 0/1, soft-label path: the hard arm
    python train_soft.py --targets soft     # labels.jsonl's 3-sample targets, soft-label path
    dry run: add --max-steps 200 --out-dir dry/runs [--labels dry/old3/labels.jsonl]

Writes <out-dir>/<targets>/eval_scores.json (per probe: question_id, condition,
sample_idx, fit_b_label, score), metrics.json and probe_weights/<probe>/.
Seed 0, fit B's recipe and band. Eval rows are fit B's, scored with 261007's
layer_logits and mean-sigmoid combined_score. Agreement with fit B's saved
scores is reported, never asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

import soft_spec as S
from domain_imports import REPRODUCTION_RECIPE, import_probe_training

probe_config, step3 = import_probe_training()
PipelineConfig, apply_overrides, config_to_json = probe_config.PipelineConfig, probe_config.apply_overrides, probe_config.config_to_json
activation_path, band_layers, build_activation_dataset = step3.activation_path, step3.band_layers, step3.build_activation_dataset
combined_score, dummy_hf_dataset, fit_layer_probes = step3.combined_score, step3.dummy_hf_dataset, step3.fit_layer_probes
layer_logits, load_activation_file, said_mask = step3.layer_logits, step3.load_activation_file, step3.said_mask
from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
from obfuscation_atlas.detectors.dataset import ActivationDataset
from obfuscation_atlas.detectors.train import save_model, to_cpu_and_cleanup_detector, train_detector


def fit_b_config() -> PipelineConfig:
    """Fit B's config, rebuilt the way 261007's ref_decomposition.py built it."""
    saved = json.loads((S.FIT_B_RUN_DIR / "config.json").read_text())
    cfg = PipelineConfig(
        arm=S.FIT_B_ARM,
        arms_dir=S.FIT_B_ARMS_DIR,
        train_samples_per_prompt=S.FIT_B_TRAIN_SAMPLES_PER_PROMPT,
        n_train_per_class=saved["n_train_per_class"],
    )
    recipe = {
        probe: {
            "train_cfg": {"learning_rate": r["learning_rate"], "max_steps": r["max_steps"]},
            "layer_selection": "band",
            "layer_band": r["layer_band"],
        }
        for probe, r in REPRODUCTION_RECIPE.items()
    }
    cfg = apply_overrides(cfg, {"per_probe_overrides": recipe})
    assert json.loads(json.dumps(config_to_json(cfg))) == saved, "rebuilt config differs from fit B's config.json"
    return cfg


def load_train_targets(labels_path: Path, probe_set: str) -> dict[tuple, float]:
    """(question_id, condition, sample_idx) -> 3-sample soft target, train rows of one probe set."""
    targets = {}
    for line in labels_path.read_text().splitlines():
        rec = json.loads(line)
        if rec["probe_set"] == probe_set and rec["split"] == "train":
            targets[(rec["question_id"], rec["condition"], rec["sample_idx"])] = rec["target_3"]
    return targets


def row_ids(activation_file: dict, keep: torch.Tensor) -> list[tuple]:
    """(question_id, condition, sample_idx) of the kept rows, in file order."""
    return [(int(k[0]), str(k[1]), int(k[2])) for k, k_keep in zip(activation_file["row_keys"], keep) if k_keep]


def soft_mode_dataset(tokenizer, train_file: dict, keep: torch.Tensor, targets: torch.Tensor) -> ActivationDataset:
    """The kept train rows with float targets, in the order the integer path uses:
    fit B's label-1 rows first, then label-0 rows, each in file order. The DataLoader
    then draws the same row sequence as the hard fit, so only the targets differ."""
    fit_b_labels = train_file["labels"][keep]
    order = torch.cat([torch.nonzero(fit_b_labels == 1).squeeze(1), torch.nonzero(fit_b_labels == 0).squeeze(1)])
    assert order.shape[0] == fit_b_labels.shape[0], "fit B train labels are not all 0/1"
    acts = {k: v.float()[keep][order] for k, v in train_file["acts"].items()}
    n = order.shape[0]
    # Dummy tokens_and_masks, as in 261007's build_activation_dataset (never read for seq_len 1).
    tokens = torch.zeros((n, 1), dtype=torch.long)
    prompt_mask = torch.zeros((n, 1), dtype=torch.bool)
    target_mask = torch.ones((n, 1), dtype=torch.bool)
    return ActivationDataset(
        tokenizer=tokenizer,
        activations=acts,
        tokens_and_masks=(tokens, prompt_mask, target_mask),
        labels=targets[order],
        obfuscate_over="last-token-generation",
        preserve_seq_len=True,
        soft_labels=True,
    )


def fit_soft_mode(train_ds: ActivationDataset, probe_cfg: PipelineConfig, tokenizer) -> tuple[dict, dict]:
    """train_detector with the arguments train_and_eval_detector passes it, plus
    balance_dataset=False. Seeded immediately before the call, as 261007's
    fit_layer_probes seeds before train_and_eval_detector."""
    torch.manual_seed(probe_cfg.probe_seed)
    dummy = dummy_hf_dataset()
    probe_dict, _, train_dynamics = train_detector(
        model=None,
        tokenizer=tokenizer,
        train_dataset=(dummy, dummy),
        train_cfg=probe_cfg.train_cfg,
        train_feature_dataset=train_ds,
        layers=list(range(probe_cfg.n_layers)),
        detector_arch_config=get_detector_arch_config(probe_cfg.detector_arch, normalize_input=probe_cfg.normalize_input),
        cache_activations_save_path=None,
        obfuscate_over="last-token-generation",
        max_completion_length=None,
        max_sequence_length=None,
        model_batch_size=32,
        train_sequence_aggregator=None,
        black_box_detector=False,
        black_box_features=-1,
        accelerator=None,
        use_sklearn=False,
        train_completion_columns=("completion", "completion"),
        congruent_follow_up_prompts=[],
        incongruent_follow_up_prompts=[],
        balance_dataset=False,
    )
    return to_cpu_and_cleanup_detector(probe_dict, None), train_dynamics


def run_probe(probe: str, targets_mode: str, cfg: PipelineConfig, tokenizer, max_steps: int | None, out_dir: Path, fit_b_scores: dict, labels_path: Path) -> tuple[dict, dict]:
    probe_cfg = apply_overrides(cfg, cfg.per_probe_overrides[probe])
    if max_steps is not None:
        probe_cfg = apply_overrides(probe_cfg, {"train_cfg": {"max_steps": max_steps}})
    probe_spec = probe_cfg.spec(probe)
    assert probe_spec.dataset == S.PROBE_SET_OF[probe], (probe_spec.dataset, S.PROBE_SET_OF[probe])
    train_file = load_activation_file(activation_path(probe_cfg, probe_spec.dataset, probe_spec.position, "train"))
    eval_file = load_activation_file(activation_path(probe_cfg, probe_spec.dataset, probe_spec.position, "eval"))
    keep_train = said_mask(train_file, probe_spec.said)
    keep_eval = said_mask(eval_file, probe_spec.said)
    eval_ds = build_activation_dataset(tokenizer, eval_file, probe_cfg, said=probe_spec.said)

    if targets_mode == "int":
        train_ds = build_activation_dataset(tokenizer, train_file, probe_cfg, said=probe_spec.said)
        _test_metrics, probe_dict, train_dynamics = fit_layer_probes(train_ds, eval_ds, probe_cfg, tokenizer)
    else:
        fit_b_labels = train_file["labels"][keep_train]
        if targets_mode == "hard01":
            targets = fit_b_labels.float()
        else:
            by_row = load_train_targets(labels_path, probe_spec.dataset)
            targets = torch.tensor([by_row[r] for r in row_ids(train_file, keep_train)], dtype=torch.float32)
            assert torch.equal(targets.round().long(), fit_b_labels.long()), "rounded soft targets differ from fit B's labels"
        train_ds = soft_mode_dataset(tokenizer, train_file, keep_train, targets)
        probe_dict, train_dynamics = fit_soft_mode(train_ds, probe_cfg, tokenizer)

    band = band_layers(tuple(probe_cfg.layer_band), probe_cfg.n_layers)
    scores = combined_score(layer_logits(probe_dict, eval_ds, band), "mean_sigmoid")
    eval_rows = row_ids(eval_file, keep_eval)
    eval_labels = eval_file["labels"][keep_eval].numpy()

    saved = fit_b_scores[probe]
    assert [(q, c) for q, c, _ in eval_rows] == list(zip(saved["question_id"], saved["condition"])), "eval rows differ from fit B's"
    assert eval_labels.tolist() == saved["label"], "eval labels differ from fit B's"

    probe_dir = out_dir / "probe_weights" / probe
    probe_dir.mkdir(parents=True)
    save_model(probe_dict, str(probe_dir))
    (probe_dir / "filtered_layers.txt").write_text(str(band) + "\n")

    eval_scores = {
        "question_id": [q for q, _, _ in eval_rows],
        "condition": [c for _, c, _ in eval_rows],
        "sample_idx": [s for _, _, s in eval_rows],
        "fit_b_label": eval_labels.tolist(),
        "score": scores.tolist(),
    }
    metrics = {
        "probe": probe,
        "targets": targets_mode,
        "said": probe_spec.said,
        "n_train": len(train_ds),
        "n_eval": len(eval_rows),
        "learning_rate": probe_cfg.train_cfg.learning_rate,
        "max_steps": probe_cfg.train_cfg.max_steps,
        "band": band,
        "final_train_loss_band": {str(l): train_dynamics[f"layer_{l}_loss"][-1] for l in band},
        "auc_on_fit_b_labels": float(roc_auc_score(eval_labels, scores)),
        "fit_b_saved_auc": float(roc_auc_score(saved["label"], saved["score"])),
        "max_abs_score_diff_vs_fit_b": float(np.max(np.abs(scores - np.array(saved["score"])))),
    }
    print(
        f"  {probe} [{targets_mode}]: AUC on fit B labels {metrics['auc_on_fit_b_labels']:.6f} "
        f"(fit B saved {metrics['fit_b_saved_auc']:.6f}), max |score diff| vs fit B {metrics['max_abs_score_diff_vs_fit_b']:.3e}"
    )
    return eval_scores, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, choices=S.TARGET_MODES)
    parser.add_argument("--max-steps", type=int, default=None, help="dry run only")
    parser.add_argument("--out-dir", type=Path, default=S.RUNS_DIR)
    parser.add_argument("--labels", type=Path, default=S.LABELS, help="soft mode's target file")
    args = parser.parse_args()
    if args.max_steps is not None:
        assert args.out_dir != S.RUNS_DIR, "a --max-steps dry run must write outside runs/"
    if args.targets == "soft":
        print(f"soft targets from {args.labels}")

    out_dir = args.out_dir / args.targets
    assert not out_dir.exists(), f"{out_dir} exists"
    cfg = fit_b_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit_b_scores = json.loads((S.FIT_B_RUN_DIR / "eval_scores.json").read_text())

    all_scores, all_metrics = {}, []
    for probe in S.PROBES:
        print(f"\n=== {probe} ===")
        all_scores[probe], metrics = run_probe(probe, args.targets, cfg, tokenizer, args.max_steps, out_dir, fit_b_scores, args.labels)
        all_metrics.append(metrics)
    (out_dir / "eval_scores.json").write_text(json.dumps(all_scores))
    (out_dir / "metrics.json").write_text(json.dumps({"targets": args.targets, "max_steps_override": args.max_steps, "probes": all_metrics}, indent=2))
    print(f"\nWrote {out_dir}/eval_scores.json and metrics.json")


if __name__ == "__main__":
    main()
