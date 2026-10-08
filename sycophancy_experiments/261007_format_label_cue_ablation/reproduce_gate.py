"""The two gates on the new probe code, against 260923 / 260924's numbers.

    python reproduce_gate.py --part reproduction --probe bias_rm_answer   # one probe, ~10 min; run the five in parallel
    python reproduce_gate.py --part extraction                            # loads the 8B model
    python reproduce_gate.py --part report                                # assembles review/reproduction.md

(a) reproduction: the new step 3 trains each probe on 260923's
    activations/pool (full train rows, the `said` mask only -- 260924's row
    masks were said-only too) with spec.REPRODUCTION_RECIPE, seed
    spec.PROBE_SEED, and its eval AUC (mean sigmoid over the band) must be
    within spec.REPRODUCTION_TOL of spec.REPRODUCTION_TARGETS.
(b) extraction: spec.EXTRACTION_GATE_ROWS rows of 260923's pool eval probe
    sets go through the new step 2 (position finding + forward pass +
    lm_head check); their activations must have cosine >=
    spec.EXTRACTION_GATE_MIN_COSINE with the cached rows at every layer.

Each part writes review/parts/<name>.json; `report` writes
review/reproduction.md with PASS / FAIL per gate and asserts both pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import spec  # noqa: E402
import step1_build_probe_sets as s1  # noqa: E402
import step2_extract_activations as s2  # noqa: E402
import step3_train_eval_probes as s3  # noqa: E402
from domain_imports import extract_answer  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from probe_config import PipelineConfig, apply_overrides  # noqa: E402

POOL_ACTS = spec.PROBE_SOURCE_DIR / "activations" / "pool"
POOL_SETS = spec.PROBE_SOURCE_DIR / "probe_sets" / "pool"
PARTS_DIR = spec.REVIEW_DIR / "parts"
#: 260924 probe name -> (dataset, position, said)
SOURCE_OF_PROBE: dict[str, tuple[str, str, str | None]] = {
    "bias_rm_answer": ("bias_rm", "answer", None),
    "bias_followed_rm_answer": ("bias_followed_rm", "answer", None),
    "deception_rm_preyesno": ("deception_rm", "preyesno", None),
    "deception_yes_rm_preyesno": ("deception_rm", "preyesno", "Yes"),
    "deception_no_rm_preyesno": ("deception_rm", "preyesno", "No"),
}


def base_config() -> PipelineConfig:
    """Model / trainer settings only; no arm data is read."""
    return PipelineConfig(arm=spec.REFERENCE_ARM)


def load_tokenizer(cfg: PipelineConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def reproduction_part(probe_name: str) -> dict:
    recipe = spec.REPRODUCTION_RECIPE[probe_name]
    dataset, position, said = SOURCE_OF_PROBE[probe_name]
    cfg = apply_overrides(base_config(), {"train_cfg": {"learning_rate": recipe["learning_rate"], "max_steps": recipe["max_steps"]}, "layer_selection": "band", "layer_band": recipe["layer_band"]})
    tokenizer = load_tokenizer(cfg)
    # 260923's files predate the spec stamp: load them directly
    train_file = torch.load(POOL_ACTS / f"{dataset}_{position}_train.pt", weights_only=False)
    eval_file = torch.load(POOL_ACTS / f"{dataset}_{position}_eval.pt", weights_only=False)
    train_ds = s3.build_activation_dataset(tokenizer, train_file, cfg, said=said)
    eval_ds = s3.build_activation_dataset(tokenizer, eval_file, cfg, said=said)
    print(f"{probe_name}: train N={len(train_ds)} eval N={len(eval_ds)} lr={recipe['learning_rate']} steps={recipe['max_steps']} band={recipe['layer_band']}")

    started = time.time()
    _, probe_dict, train_dynamics = s3.fit_layer_probes(train_ds, eval_ds, cfg, tokenizer)
    seconds = time.time() - started
    layers, _ = s3.select_layers(train_dynamics, cfg)
    auc = s3.combined_auc(eval_ds.labels.numpy(), s3.layer_logits(probe_dict, eval_ds, layers), "mean_sigmoid")
    target = spec.REPRODUCTION_TARGETS[probe_name]
    result = {
        "probe": probe_name,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "recipe": recipe,
        "seed": cfg.probe_seed,
        "auc_mean_sigmoid": auc,
        "target": target,
        "tolerance": spec.REPRODUCTION_TOL,
        "pass": abs(auc - target) <= spec.REPRODUCTION_TOL,
        "fit_seconds": seconds,
        "seconds_per_1000_steps": seconds / recipe["max_steps"] * 1000,
    }
    print(f"{probe_name}: AUC {auc:.4f} vs target {target:.3f} (tol {spec.REPRODUCTION_TOL}) -> {'PASS' if result['pass'] else 'FAIL'}; fit {seconds:.0f}s")
    return result


def extraction_part() -> dict:
    cfg = base_config()
    model, tokenizer = s2.load_model_and_tokenizer(cfg)
    lm_head_check: dict = {}
    out: dict = {"n_rows": spec.EXTRACTION_GATE_ROWS, "min_cosine_required": spec.EXTRACTION_GATE_MIN_COSINE, "datasets": {}}
    for family, dataset, position in (("bias", "bias_rm", "answer"), ("deception", "deception_rm", "preyesno")):
        rows = s1.load_jsonl(POOL_SETS / f"{dataset}_eval.jsonl")[: spec.EXTRACTION_GATE_ROWS]
        if family == "bias":
            for r in rows:
                r["answer_letter"] = extract_answer(r["completion"])[0]
            result = s2.extract_bias_positions(rows, model, tokenizer, cfg, lm_head_check)
        else:
            result = s2.extract_deception_positions(rows, model, tokenizer, cfg, lm_head_check)
        cached = torch.load(POOL_ACTS / f"{dataset}_{position}_eval.pt", weights_only=False)
        cached_index = {tuple(k): i for i, k in enumerate(cached["row_keys"])}
        row_index = [cached_index[(r["question_id"], r["condition"], r["sample_idx"], r["category"])] for r in rows]
        assert [cached["positions"][i] for i in row_index] == result["positions"], f"{dataset}: token positions differ from the cached extraction"
        per_layer_min = []
        for layer in range(cfg.n_layers):
            new = result["acts"][str(layer)].squeeze(1).float()
            old = cached["acts"][str(layer)][row_index].squeeze(1).float()
            per_layer_min.append(float(torch.nn.functional.cosine_similarity(new, old, dim=1).min()))
        out["datasets"][dataset] = {"min_cosine_per_layer": per_layer_min, "min_cosine": min(per_layer_min), "pass": min(per_layer_min) >= spec.EXTRACTION_GATE_MIN_COSINE}
        print(f"{dataset}: min cosine over {len(rows)} rows x {cfg.n_layers} layers = {min(per_layer_min):.6f} -> {'PASS' if out['datasets'][dataset]['pass'] else 'FAIL'}")
    out["lm_head_check_max_abs_diff"] = lm_head_check["max_abs_diff"]
    out["pass"] = all(d["pass"] for d in out["datasets"].values())
    return out


def report() -> None:
    reproduction = {probe: json.loads((PARTS_DIR / f"reproduction_{probe}.json").read_text()) for probe in spec.REPRODUCTION_RECIPE}
    extraction = json.loads((PARTS_DIR / "extraction.json").read_text())
    for part in list(reproduction.values()) + [extraction]:
        assert part["spec_hash"] == spec.spec_hash(), "a gate part was run under another spec"
    reproduction_pass = all(r["pass"] for r in reproduction.values())
    lines = [
        "# Reproduction and extraction gates",
        "",
        f"spec_hash {spec.spec_hash()}",
        "",
        f"## Reproduction gate: {'PASS' if reproduction_pass else 'FAIL'}",
        "",
        f"New step 3 on 260923's activations/pool, full train rows, seed {spec.PROBE_SEED}, 260924's recipe per probe; "
        f"eval AUC of the mean sigmoid over the band must be within {spec.REPRODUCTION_TOL} of 260924's number.",
        "",
        "| probe | lr | steps | band | n_train | AUC | target | diff | pass | fit seconds | s / 1000 steps |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for probe, r in reproduction.items():
        band = r["recipe"]["layer_band"]
        lines.append(
            f"| {probe} | {r['recipe']['learning_rate']:g} | {r['recipe']['max_steps']} | {band[0]}-{band[0] + band[1] - 1} | {r['n_train']} | "
            f"{r['auc_mean_sigmoid']:.4f} | {r['target']:.3f} | {r['auc_mean_sigmoid'] - r['target']:+.4f} | {'PASS' if r['pass'] else 'FAIL'} | "
            f"{r['fit_seconds']:.0f} | {r['seconds_per_1000_steps']:.0f} |"
        )
    lines += [
        "",
        f"## Extraction gate: {'PASS' if extraction['pass'] else 'FAIL'}",
        "",
        f"{extraction['n_rows']} rows of 260923's pool eval probe sets through the new step 2; cosine with the cached "
        f"activations at every layer must be >= {extraction['min_cosine_required']}. lm_head on the layer-31 residual vs "
        f"model(...).logits: max abs diff {extraction['lm_head_check_max_abs_diff']:.3g} (< {s2.LM_HEAD_CHECK_MAX_ABS_DIFF}).",
        "",
        "| dataset | min cosine (all layers) | pass |",
        "|---|---|---|",
    ]
    for dataset, d in extraction["datasets"].items():
        lines.append(f"| {dataset} | {d['min_cosine']:.6f} | {'PASS' if d['pass'] else 'FAIL'} |")
    lines.append("")
    spec.REVIEW_DIR.mkdir(exist_ok=True)
    (spec.REVIEW_DIR / "reproduction.md").write_text("\n".join(lines))
    print("\n".join(lines))
    assert reproduction_pass and extraction["pass"], "a gate failed; see review/reproduction.md"
    print("GATES PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True, choices=["reproduction", "extraction", "report"])
    parser.add_argument("--probe", default=None, choices=list(spec.REPRODUCTION_RECIPE))
    args = parser.parse_args()
    bootstrap_hf_env()
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.part == "reproduction":
        assert args.probe is not None, "--part reproduction needs --probe"
        result = reproduction_part(args.probe)
        (PARTS_DIR / f"reproduction_{args.probe}.json").write_text(json.dumps({"spec_hash": spec.spec_hash(), **result}, indent=2))
    elif args.part == "extraction":
        result = extraction_part()
        (PARTS_DIR / "extraction.json").write_text(json.dumps({"spec_hash": spec.spec_hash(), **result}, indent=2))
    else:
        report()


if __name__ == "__main__":
    main()
