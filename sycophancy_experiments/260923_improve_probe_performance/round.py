"""One recorded round: run_log.start_round + step3.train_eval_all_seeds.

    python round.py --slug baseline --variant base
    python round.py --slug lr1e-3 --variant base --set '{"train_cfg": {"learning_rate": 1e-3}}'

`--set` is a JSON dict of PipelineConfig fields; `train_cfg` is merged into the
default BaseTrainConfig. The round folder gets config.json (from start_round),
overrides.json (the --set dict, so the round is re-runnable) and
probe_metrics.json. The runs.md row is appended afterwards by append_row.py,
once the result has been read.

Training needs no model: it reads activations/<variant>/. Runs on the GPU
(train_cfg.device) alongside an extraction without trouble.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import run_log  # noqa: E402
import step3_train_eval_probes as s3  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from probe_config import config_to_json, default_config, variant_config  # noqa: E402


def apply_overrides(cfg, overrides: dict):
    if "train_cfg" in overrides:
        overrides = dict(overrides)
        cfg = replace(cfg, train_cfg=replace(cfg.train_cfg, **overrides.pop("train_cfg")))
    if "positions" in overrides:
        overrides = dict(overrides)
        cfg = replace(cfg, positions={k: tuple(v) for k, v in overrides.pop("positions").items()})
    return replace(cfg, **overrides)


def train_fit_table(metrics_path: Path) -> str:
    """Train-vs-eval AUC and train loss per reported probe (mean over seeds of the first seed's diagnostics)."""
    metrics = json.loads(metrics_path.read_text())
    lines = [f"{'probe':28s} {'train AUC':>10s} {'eval AUC':>9s} {'min loss':>9s} {'fell back':>10s} {'n_train':>8s}"]
    for p in metrics["probes"]:
        lines.append(
            f"{p['probe']:28s} {p['train_auc_filtered']:10.3f} {p['auc_filtered_mean']:9.3f} "
            f"{p['train_loss_min']:9.3f} {str(p['layer_filter_fell_back']):>10s} {p['n_train']:8d}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    parser.add_argument("--variant", required=True, help="activations/<variant>")
    parser.add_argument("--set", default="{}", help="JSON dict of PipelineConfig overrides")
    args = parser.parse_args()

    bootstrap_hf_env()
    from transformers import AutoTokenizer

    overrides = json.loads(args.set)
    cfg = apply_overrides(variant_config(default_config(), args.variant), overrides)
    assert cfg.act_dir.exists(), f"no activations at {cfg.act_dir}"

    round_cfg = run_log.start_round(cfg, args.slug)
    (round_cfg.metrics_path.parent / "overrides.json").write_text(
        json.dumps({"variant": args.variant, "set": overrides}, indent=2)
    )
    print(json.dumps(config_to_json(round_cfg), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    s3.train_eval_all_seeds(round_cfg, tokenizer)

    print()
    print(run_log.five_probe_table(round_cfg.metrics_path))
    print()
    print(train_fit_table(round_cfg.metrics_path))
    print(f"\nROUND DONE {round_cfg.metrics_path.parent}")


if __name__ == "__main__":
    main()
