"""Rescale a chosen recipe's step counts to another train set so the number of
passes over the data (epochs) stays the same.

    python epochs_matched_overrides.py --from cv/pool/chosen_overrides.json --variant pool_nu \
        --out cv/pool/chosen_overrides_pool_nu_epochs_matched.json

The chosen (lr, steps, band) per probe comes from cross-validation on `pool`.
The high-confidence variant trains on `pool_nu`, which has ~40% of the rows, so
the same step count would mean 2.5x the epochs. This keeps lr and band and sets
steps = round(epochs_on_pool * n_rows_on_variant / batch_size) per probe. Row
counts are read from the variant's train activation files under the probe's
`said` mask (the same rows step 3 trains on with train_draw_per_class=None).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import step3_train_eval_probes as s3  # noqa: E402
from probe_config import default_config, variant_config  # noqa: E402


def train_rows(cfg, probe_name: str) -> int:
    spec = cfg.spec(probe_name)
    d = torch.load(s3.activation_path(cfg, spec.dataset, spec.position, "train"), weights_only=False)
    return int((s3.said_mask(d, spec.said) & s3.row_mask(d, spec, cfg, "train")).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", required=True, help="cv_select.py's chosen_overrides.json")
    parser.add_argument("--variant", required=True, help="the train set the steps are rescaled to")
    parser.add_argument("--source-variant", default="pool", help="the train set the recipe was chosen on")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    chosen = json.loads(Path(args.source).read_text())["per_probe_overrides"]
    source_cfg = variant_config(default_config(), args.source_variant)
    target_cfg = variant_config(default_config(), args.variant)
    batch_size = default_config().train_cfg.batch_size

    out = {"per_probe_overrides": {}, "note": {}}
    for probe_name, overrides in chosen.items():
        n_source = train_rows(source_cfg, probe_name)
        n_target = train_rows(target_cfg, probe_name)
        steps_source = overrides["train_cfg"]["max_steps"]
        epochs = steps_source * batch_size / n_source
        steps_target = max(1, round(epochs * n_target / batch_size))
        rescaled = json.loads(json.dumps(overrides))
        rescaled["train_cfg"]["max_steps"] = steps_target
        out["per_probe_overrides"][probe_name] = rescaled
        out["note"][probe_name] = {
            "source_variant": args.source_variant,
            "n_train_source": n_source,
            "steps_source": steps_source,
            "epochs": epochs,
            "target_variant": args.variant,
            "n_train_target": n_target,
            "steps_target": steps_target,
        }
        print(f"{probe_name}: {steps_source} steps on {n_source} rows = {epochs:.2f} epochs -> {steps_target} steps on {n_target} rows")
    # the overrides file must hold PipelineConfig fields only (round.py --set-file); the
    # per-probe arithmetic goes beside it
    Path(args.out).write_text(json.dumps({"per_probe_overrides": out["per_probe_overrides"]}, indent=2))
    note_path = Path(args.out).with_suffix(".note.json")
    note_path.write_text(json.dumps(out["note"], indent=2))
    print(f"wrote {args.out} and {note_path}")


if __name__ == "__main__":
    main()
