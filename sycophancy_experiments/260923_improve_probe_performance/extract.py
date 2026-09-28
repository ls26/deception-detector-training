"""Steps 1 + 2 for one extraction variant: build probe sets, extract activations.

    python extract.py --variant base --positions all
    python extract.py --variant pool --n-train-per-class '{"bias":1454,...}'
    python extract.py --variant pool_nu --train-filter none_unanimous ...

Writes probe_sets/<variant>/ (rows + manifest), activations/<variant>/ (one .pt
per dataset x position x split) and activations/<variant>/extraction.json (the
config, the step-1 gate results and the step-2 row counts). Loads the 8B model
on the GPU: run it inside tmux, one at a time.

`--positions all` saves every valid position of each family from the same
forward pass (bias: answer + lastcompl, deception: preyesno + yesno), so a
later positions round needs no re-extraction.

Every variant's eval rows must equal activations/base's (eval is never filtered
or redrawn); that is asserted after extraction when base exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import step1_build_probe_sets as s1  # noqa: E402
import step2_extract_activations as s2  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from probe_config import config_to_json, default_config, variant_config  # noqa: E402

ALL_POSITIONS = {"bias": ("answer", "lastcompl"), "bias_followed": ("answer", "lastcompl"), "deception": ("preyesno", "yesno")}


def step1_gates(cfg, indexed, sources, probe_sets, allow_unbalanced: bool = False) -> tuple[list[str], list[str]]:
    """The gates of run_pipeline.py cell 4. Balance problems are hard failures
    without a train filter and expected warnings with one (or with --allow-unbalanced)."""
    failures = list(s1.check_majority_relabeling_gates(indexed, sources.rescored))
    balance_problems: list[str] = []
    for ps in probe_sets:
        tr, ev = ps.manifest["train"], ps.manifest["eval"]
        if ev["n_total"] != ev["n_prompts_in_pool"]:
            failures.append(f"{ps.name}/eval: {ev['n_total']} rows over {ev['n_prompts_in_pool']} prompts")
        if tr["n_pos"] != tr["n_neg"]:
            balance_problems.append(f"{ps.name}/train: {tr['n_pos']} pos vs {tr['n_neg']} neg")
        for label, cells in tr["by_condition"].items():
            if cells.get("cue_gold") != cells.get("cue_dist"):
                balance_problems.append(f"{ps.name}/train: {label} {cells} not cue-balanced")
        if ps.family == "deception":
            strata = ps.manifest["train_strata"]
            n_yes, n_no = tr["n_said_yes"], tr["n_said_no"]
            if (strata["deception_yes"], strata["deception_no"]) != (n_yes, n_no):
                failures.append(f"{ps.name}: strata {strata} != said split ({n_yes}, {n_no})")
            if n_yes + n_no != tr["n_total"]:
                failures.append(f"{ps.name}: said-Yes + said-No != train total")
    if cfg.train_filter == "none" and not allow_unbalanced:
        failures.extend(balance_problems)
        balance_problems = []
    return failures, balance_problems


def eval_row_keys(cfg, dataset: str, position: str) -> list:
    return torch.load(s2.activation_path(cfg, dataset, position, "eval"), weights_only=False)["row_keys"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--train-filter", default="none", choices=["none", "none_unanimous"])
    parser.add_argument(
        "--positions", default="default", help='"default", "all" (answer+lastcompl / preyesno+yesno) or a JSON dict per family'
    )
    parser.add_argument("--n-train-per-class", default=None, help="JSON dict keyed like cfg.n_train_per_class")
    parser.add_argument("--limit-per-class", type=int, default=None, help="dry run: rows per class")
    parser.add_argument(
        "--allow-unbalanced",
        action="store_true",
        help="demote the step-1 class/cue balance gates to warnings (for a draw of every eligible row)",
    )
    args = parser.parse_args()

    bootstrap_hf_env()
    cfg = variant_config(default_config(), args.variant)
    cfg = replace(cfg, train_filter=args.train_filter, limit_per_class=args.limit_per_class)
    if args.positions == "all":
        cfg = replace(cfg, positions=dict(ALL_POSITIONS))
    elif args.positions != "default":
        cfg = replace(cfg, positions={family: tuple(p) for family, p in json.loads(args.positions).items()})
    if args.n_train_per_class is not None:
        cfg = replace(cfg, n_train_per_class=json.loads(args.n_train_per_class))
    assert not cfg.act_dir.exists(), f"{cfg.act_dir} exists -- pick a new variant name"
    print(json.dumps(config_to_json(cfg), indent=2))

    # step 1
    indexed = s1.load_jsonl(cfg.rescore_dir / "rescored_labels.jsonl")
    sources = s1.load_sources(cfg)
    probe_sets = s1.build_all(cfg, sources=sources)
    failures, warnings = step1_gates(cfg, indexed, sources, probe_sets, allow_unbalanced=args.allow_unbalanced)
    for w in warnings:
        print("  WARN (expected under train_filter)", w)
    for f in failures:
        print("  FAIL", f)
    assert not failures, failures
    print(f"\nstep 1: 0 failures, {len(warnings)} expected warnings")

    # step 2
    torch.cuda.empty_cache()
    model, tokenizer = s2.load_model_and_tokenizer(cfg)
    scores_idx = s2.build_scores_index(cfg)
    results: dict = {}
    for family in cfg.datasets:
        results.update(
            s2.extract_all(
                replace(cfg, datasets=(family,)),
                model,
                tokenizer,
                scores_idx=scores_idx if family != "deception" else None,
            )
        )
    print(s2.summarize(results))

    # eval rows must be identical across variants
    base_cfg = variant_config(cfg, "base")
    eval_check = {}
    if base_cfg.act_dir.exists() and args.variant != "base":
        for family, definition in cfg.dataset_keys():
            dataset = cfg.dataset_name(family, definition)
            same = eval_row_keys(cfg, dataset, cfg.positions[family][0]) == eval_row_keys(
                base_cfg, dataset, default_config().positions[family][0]
            )
            eval_check[dataset] = same
            assert same, f"{dataset}: eval rows differ from activations/base"
        print(f"eval rows identical to base: {eval_check}")

    (cfg.act_dir / "extraction.json").write_text(
        json.dumps(
            {
                "config": config_to_json(cfg),
                "step1_warnings": warnings,
                "datasets": [ps.manifest for ps in probe_sets],
                "extraction": results,
                "eval_rows_match_base": eval_check,
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {cfg.act_dir / 'extraction.json'}")


if __name__ == "__main__":
    main()
