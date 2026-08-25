"""Phase 3: train_and_eval_detector per probe, save_model, metrics.

Loads Phase-2's precomputed per-layer activation tensors, wraps them in
ActivationDataset (bypassing the shared package's text pipeline entirely --
see this experiment's plan doc for why), and trains + evaluates a linear
probe for each of the 8 probes (4 definitions x 2 extraction positions).

Reuses the unmodified obfuscation_atlas probe-training/eval machinery
(train_and_eval_detector, get_layers_below_loss_threshold, save_model) --
the only new code here is the .pt -> ActivationDataset bridge and the
metrics summarization (copied from pipeline/probe_training.py's
_mean_auc/_mean_threshold_metric, which are dataset-agnostic).
"""
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from obfuscation_atlas.config import BaseTrainConfig
from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
from obfuscation_atlas.detectors.dataset import ActivationDataset
from obfuscation_atlas.detectors.evaluate import train_and_eval_detector, get_layers_below_loss_threshold
from obfuscation_atlas.detectors.probe_archs import SequenceAggregator
from obfuscation_atlas.detectors.train import save_model

EXP_DIR = Path(__file__).parent
ACT_DIR = EXP_DIR / "activations"
WEIGHTS_DIR = EXP_DIR / "probe_weights"

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
N_LAYERS = 32
LAYER_LOSS_THRESHOLD = 0.6
DTYPE = torch.bfloat16

TRAIN_CFG = BaseTrainConfig(
    max_steps=1000, batch_size=32, learning_rate=3e-3, warmup_steps=0, scheduler="constant"
)
EVAL_SEQUENCE_AGGREGATOR = SequenceAggregator(method="mean")

PROBES = [
    "bias_rm_answer", "bias_rm_lastcompl",
    "bias_swap_answer", "bias_swap_lastcompl",
    "deception_rm_firstgen", "deception_rm_lastprompt",
    "deception_swap_firstgen", "deception_swap_lastprompt",
]


def load_model_and_tokenizer():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()
    return model, tokenizer


def dummy_hf_dataset() -> Dataset:
    """train_and_eval_detector's train_dataset/test_dataset args are never touched
    when train_feature_dataset/test_feature_dataset are supplied (verified against
    prepare_dataset's branching during planning) -- these placeholders only need to
    exist as valid 2-tuples of HF Datasets."""
    return Dataset.from_dict({"completion": ["placeholder"]})


def build_activation_dataset(tokenizer, pt_path: Path) -> ActivationDataset:
    d = torch.load(pt_path, weights_only=False)
    acts = d["acts"]  # {"0": (N,1,H), ..., "31": (N,1,H)}
    labels = d["labels"]
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


def train_eval_one_probe(probe_name: str, model, tokenizer) -> dict:
    train_pt = ACT_DIR / f"{probe_name}_train.pt"
    eval_pt = ACT_DIR / f"{probe_name}_eval.pt"
    train_ds = build_activation_dataset(tokenizer, train_pt)
    eval_ds = build_activation_dataset(tokenizer, eval_pt)
    print(f"  train N={len(train_ds)}, eval N={len(eval_ds)}, d_model={train_ds.get_feature_dim()}")

    dummy = dummy_hf_dataset()
    test_metrics, probe_dict, train_dynamics, _best_train_metrics = train_and_eval_detector(
        model=model,
        tokenizer=tokenizer,
        layers=list(range(N_LAYERS)),
        train_dataset=(dummy, dummy),
        test_dataset=(dummy, dummy),
        train_cfg=TRAIN_CFG,
        detector_arch_config=get_detector_arch_config("linear-probe"),
        obfuscate_over="last-token-generation",
        train_sequence_aggregator=None,
        eval_sequence_aggregator=EVAL_SEQUENCE_AGGREGATOR,
        train_feature_dataset=train_ds,
        test_feature_dataset=eval_ds,
        n_calibration_examples=None,
        accelerator=None,
        use_sklearn=False,
    )

    try:
        filtered_layers = get_layers_below_loss_threshold(train_dynamics, LAYER_LOSS_THRESHOLD)
    except ValueError as e:
        print(f"  WARNING: {e} -- falling back to all {N_LAYERS} layers (real signal, inspect train_dynamics)")
        filtered_layers = list(range(N_LAYERS))

    probe_dir = WEIGHTS_DIR / probe_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    save_model(probe_dict, str(probe_dir))
    (probe_dir / "filtered_layers.txt").write_text(str(filtered_layers) + "\n")

    summary = {
        "probe": probe_name,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "filtered_layers": filtered_layers,
        "auc_filtered": mean_auc(test_metrics, filtered_layers),
        "auc_all_layers": mean_auc(test_metrics, list(range(N_LAYERS))),
        "accuracy_filtered": mean_threshold_metric(test_metrics, filtered_layers, "accuracy"),
        "f1_filtered": mean_threshold_metric(test_metrics, filtered_layers, "f1_score"),
        "auc_per_layer": {str(l): test_metrics.get(str(l), {}).get("AUC_ROC") for l in range(N_LAYERS)},
    }
    print(
        f"  {probe_name}: AUC(filtered)={summary['auc_filtered']:.4f} "
        f"AUC(all)={summary['auc_all_layers']:.4f} acc={summary['accuracy_filtered']:.4f} "
        f"f1={summary['f1_filtered']:.4f} n_filtered_layers={len(filtered_layers)}"
    )
    return summary


def main() -> None:
    global ACT_DIR, WEIGHTS_DIR
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="use activations/dry_run/, write to probe_weights/dry_run/")
    args = parser.parse_args()
    if args.dry_run:
        ACT_DIR = ACT_DIR / "dry_run"
        WEIGHTS_DIR = WEIGHTS_DIR / "dry_run"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()

    all_summaries = []
    for probe_name in PROBES:
        print(f"\n=== {probe_name} ===")
        all_summaries.append(train_eval_one_probe(probe_name, model, tokenizer))

    metrics_path = EXP_DIR / ("probe_metrics_dry_run.json" if args.dry_run else "probe_metrics.json")
    metrics_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nWrote {metrics_path}")
    print("\n=== Summary ===")
    for s in all_summaries:
        print(f"  {s['probe']:30s} AUC(filtered)={s['auc_filtered']:.4f} n_layers={len(s['filtered_layers'])}")


if __name__ == "__main__":
    main()
