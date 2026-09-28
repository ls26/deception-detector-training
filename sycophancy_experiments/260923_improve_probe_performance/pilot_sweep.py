"""Exploratory single-seed sweep on a few probes, WITHOUT creating a round.

    python pilot_sweep.py --variant base --probes deception_yes_rm_preyesno deception_no_rm_preyesno \
        --configs '{"lr1e-2": {"train_cfg": {"learning_rate": 1e-2}}, "logregC100": {"probe_fit": "logreg"}}'

Used to pick which settings deserve a three-seed round; every pilot's numbers
are recorded in runs.md under "Added" so the choice is traceable. Writes
logs/pilot_<name>.json and prints one table. Nothing is saved to runs/ and no
weights are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import step3_train_eval_probes as s3  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from probe_config import default_config, variant_config  # noqa: E402
from round import apply_overrides  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--probes", nargs="+", required=True)
    parser.add_argument("--configs", required=True, help="JSON {label: overrides}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bootstrap_hf_env()
    from transformers import AutoTokenizer

    base_cfg = variant_config(default_config(), args.variant)
    tokenizer = AutoTokenizer.from_pretrained(base_cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    configs = json.loads(args.configs)

    results = []
    for label, overrides in configs.items():
        cfg = apply_overrides(base_cfg, overrides)
        cfg.probe_seed = args.seed
        for probe in args.probes:
            start = time.time()
            summary, _ = s3.train_eval_one_probe(probe, None, tokenizer, cfg, save=False)
            results.append(
                {
                    "label": label,
                    "overrides": overrides,
                    "probe": probe,
                    "auc_filtered": summary["auc_filtered"],
                    "auc_layer_mean_logit": summary["auc_layer_mean_logit"],
                    "train_auc_filtered": summary["train_auc_filtered"],
                    "train_loss_min": summary["train_loss_min"],
                    "n_filtered_layers": len(summary["filtered_layers"]),
                    "fell_back": summary["layer_filter_fell_back"],
                    "seconds": time.time() - start,
                }
            )

    out = EXP_DIR / "logs" / f"pilot_{args.name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n=== pilot {args.name} (seed {args.seed}, variant {args.variant}) ===")
    print(f"{'config':18s} {'probe':28s} {'eval AUC':>8s} {'lml AUC':>8s} {'train AUC':>9s} {'min loss':>8s} {'layers':>6s} {'s':>4s}")
    for r in results:
        print(
            f"{r['label']:18s} {r['probe']:28s} {r['auc_filtered']:8.3f} {r['auc_layer_mean_logit']:8.3f} "
            f"{r['train_auc_filtered']:9.3f} {r['train_loss_min']:8.3f} {r['n_filtered_layers']:6d} {r['seconds']:4.0f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
