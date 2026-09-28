"""Cross-validation sweep over (learning rate, max_steps) per probe, on the GPU.

    python cv_sweep.py --variant pool
    python cv_sweep.py --variant pool --probes deception_no_rm_preyesno --lrs 1e-2 --steps 50   # test

For each probe: the rows of the train activation file that the probe uses
(`said` mask + `row_mask`) are split into `cfg.cv_folds` folds by question_id
(GroupKFold, shuffled with `cfg.cv_seed`), so no question's rows appear in two
folds. For each (lr, steps) in the grid and each fold, the 32 per-layer probes
are fit with the unchanged AdamW trainer (`step3.fit_layer_probes`, seed
`cfg.probe_seed`) on the other folds and the held-out rows' per-layer logits are
stored. Layer bands are scored afterwards by cv_select.py from these logits, so
the band grid costs nothing here.

Output, one JSON per cell (skipped when it already exists, so a killed run
resumes): cv/<variant>/<probe>/lr{lr}_steps{steps}/fold{k}.json with
`row_index` (indices into the train .pt), `labels`, `logits` (32 x n held-out),
`train_loss_final_per_layer`, `layers_below_threshold` (the pipeline's 0.6 rule
on this fold's fit; [] when none passes) and `seconds`. cv/<variant>/<probe>/
folds.json records the split.

The eval split is never read. Run inside tmux: about 35 s per 1000 steps, i.e.
~1 h per probe for the default 3 x 3 grid x 5 folds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import step3_train_eval_probes as s3  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from obfuscation_atlas.detectors.evaluate import get_layers_below_loss_threshold  # noqa: E402
from probe_config import PipelineConfig, apply_overrides, config_to_json, default_config, variant_config  # noqa: E402

#: sweep order: the probes the user cares most about first; the unstratified
#: deception probe last, because it is reported but not optimized
DEFAULT_PROBE_ORDER: tuple[str, ...] = (
    "bias_followed_rm_answer",
    "deception_yes_rm_preyesno",
    "deception_no_rm_preyesno",
    "bias_rm_answer",
    "deception_rm_preyesno",
)


def cell_dir(cfg: PipelineConfig, variant: str, probe_name: str, lr: float, steps: int) -> Path:
    return cfg.cv_dir / variant / probe_name / f"lr{lr:g}_steps{steps}"


def assign_folds(question_ids: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Fold number per row (0..cv_folds-1); rows of one question share a fold."""
    splitter = GroupKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.cv_seed)
    fold_of_row = np.full(question_ids.shape[0], -1, dtype=int)
    for fold, (_, heldout_index) in enumerate(splitter.split(question_ids, groups=question_ids)):
        fold_of_row[heldout_index] = fold
    assert (fold_of_row >= 0).all()
    return fold_of_row


def sweep_probe(probe_name: str, cfg: PipelineConfig, variant: str, tokenizer, lrs: list[float], steps_grid: list[int]) -> None:
    spec = cfg.spec(probe_name)
    train_file = torch.load(s3.activation_path(cfg, spec.dataset, spec.position, "train"), weights_only=False)
    used = s3.said_mask(train_file, spec.said) & s3.row_mask(train_file, spec, cfg, "train")
    used_index = np.flatnonzero(used.numpy())
    question_ids = np.array([train_file["row_keys"][i][0] for i in used_index])
    fold_of_used_row = assign_folds(question_ids, cfg)

    probe_dir = cfg.cv_dir / variant / probe_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "folds.json").write_text(
        json.dumps(
            {
                "probe": probe_name,
                "variant": variant,
                "n_rows": int(used_index.shape[0]),
                "n_questions": int(np.unique(question_ids).shape[0]),
                "cv_folds": cfg.cv_folds,
                "cv_seed": cfg.cv_seed,
                "rows_per_fold": [int((fold_of_used_row == k).sum()) for k in range(cfg.cv_folds)],
                "row_index": used_index.tolist(),
                "fold_of_row": fold_of_used_row.tolist(),
            }
        )
    )
    print(f"\n=== {probe_name}: {used_index.shape[0]} rows, {np.unique(question_ids).shape[0]} questions, folds {[int((fold_of_used_row == k).sum()) for k in range(cfg.cv_folds)]}")

    n_rows_in_file = train_file["labels"].shape[0]
    grid = [(lr, steps) for lr in lrs for steps in steps_grid]
    for fold in range(cfg.cv_folds):
        pending = [(lr, steps) for lr, steps in grid if not (cell_dir(cfg, variant, probe_name, lr, steps) / f"fold{fold}.json").exists()]
        if not pending:
            print(f"  fold {fold}: all {len(grid)} cells exist, skipping")
            continue
        # the fold's datasets do not depend on (lr, steps): build them once (~20 s)
        heldout = torch.zeros(n_rows_in_file, dtype=torch.bool)
        heldout[used_index[fold_of_used_row == fold]] = True
        fit_rows = used & ~heldout
        fit_ds = s3.build_activation_dataset(tokenizer, train_file, cfg, said=spec.said, extra_mask=fit_rows)
        heldout_ds = s3.build_activation_dataset(tokenizer, train_file, cfg, said=spec.said, extra_mask=heldout)
        for lr, steps in pending:
            cell_cfg = apply_overrides(cfg, {"train_cfg": {"learning_rate": lr, "max_steps": steps}})
            out_dir = cell_dir(cfg, variant, probe_name, lr, steps)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"fold{fold}.json"
            started = time.time()
            _, probe_dict, train_dynamics, _ = s3.fit_layer_probes(fit_ds, heldout_ds, cell_cfg, tokenizer)
            logits = s3.layer_logits(probe_dict, heldout_ds, list(range(cfg.n_layers)))
            try:
                layers_below = get_layers_below_loss_threshold(train_dynamics, cfg.layer_loss_threshold)
            except ValueError:
                layers_below = []
            seconds = time.time() - started
            out_path.write_text(
                json.dumps(
                    {
                        "probe": probe_name,
                        "learning_rate": lr,
                        "max_steps": steps,
                        "batch_size": cell_cfg.train_cfg.batch_size,
                        "epochs": steps * cell_cfg.train_cfg.batch_size / len(fit_ds),
                        "fold": fold,
                        "n_fit": len(fit_ds),
                        "n_heldout": len(heldout_ds),
                        "row_index": used_index[fold_of_used_row == fold].tolist(),
                        "labels": heldout_ds.labels.numpy().tolist(),
                        "logits": logits.tolist(),
                        "train_loss_final_per_layer": [
                            (train_dynamics[f"layer_{l}_loss"] or [float("nan")])[-1] for l in range(cfg.n_layers)
                        ],
                        "layers_below_threshold": layers_below,
                        "seconds": seconds,
                    }
                )
            )
            oof_auc = s3.combined_auc(heldout_ds.labels.numpy(), logits, cfg.combined_score)
            print(
                f"  lr={lr:g} steps={steps} fold={fold}: fit {len(fit_ds)} / held-out {len(heldout_ds)}, "
                f"OOF AUC(all 32, {cfg.combined_score})={oof_auc:.4f}, "
                f"{len(layers_below)} layers below {cfg.layer_loss_threshold}, {seconds:.0f}s"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="260923's activations/<variant>")
    parser.add_argument("--probes", nargs="*", default=list(DEFAULT_PROBE_ORDER))
    parser.add_argument("--lrs", nargs="*", type=float, default=None, help="default: cfg.cv_learning_rates")
    parser.add_argument("--steps", nargs="*", type=int, default=None, help="default: cfg.cv_max_steps")
    args = parser.parse_args()

    bootstrap_hf_env()
    from transformers import AutoTokenizer

    cfg = variant_config(default_config(), args.variant)
    assert cfg.act_dir.exists(), f"no activations at {cfg.act_dir}"
    lrs = args.lrs if args.lrs is not None else list(cfg.cv_learning_rates)
    steps_grid = args.steps if args.steps is not None else list(cfg.cv_max_steps)

    (cfg.cv_dir / args.variant).mkdir(parents=True, exist_ok=True)
    (cfg.cv_dir / args.variant / "sweep_config.json").write_text(
        json.dumps({"config": config_to_json(cfg), "probes": args.probes, "lrs": lrs, "steps": steps_grid}, indent=2)
    )
    print(f"variant={args.variant} probes={args.probes} lrs={lrs} steps={steps_grid} folds={cfg.cv_folds}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for probe_name in args.probes:
        sweep_probe(probe_name, cfg, args.variant, tokenizer, lrs, steps_grid)
    print("\nSWEEP DONE")


if __name__ == "__main__":
    main()
