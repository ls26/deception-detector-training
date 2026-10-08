"""Steps 1 + 2 for one arm: build probe sets, extract activations, gate.

    python extract.py --arm ref_cot
    python extract.py --arm ref_cot --arms-dir tests_fixture/arms --limit-per-class 40   # fixture

Writes arms/<arm>/probe_sets/ (rows + manifest), arms/<arm>/activations/ (one
.pt per dataset x split) and arms/<arm>/activations/extraction.json (config,
gates, counts, spec_hash). Loads the 8B model on the GPU: run inside tmux.

Gates (every one an assert):
  before:  activations/ does not exist; free disk >= spec.DISK_MIN_FREE_GB
  step 1:  eval = one row per prompt; train classes equal in size and
           cue-balanced, at the size arms/sizes.json prescribes; the deception
           train set is exactly its two strata
  step 2:  (arms whose labels carry letter logprobs: the no-CoT arms) on
           bias-family rows with |lp_A - lp_B| > spec.HF_VLLM_MARGIN_NAT the
           HF top letter equals vLLM's on >= spec.HF_VLLM_AGREEMENT_MIN of rows
Also reported: the share of prompts in labels.jsonl with two rows of the same
answer letter (byte-identical inputs, hence duplicate activations, which the
one-row-per-prompt pools remove).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import spec  # noqa: E402
import step1_build_probe_sets as s1  # noqa: E402
import step2_extract_activations as s2  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from probe_config import CONDITIONS, POSITION_OF_FAMILY, PipelineConfig, arm_config, config_to_json  # noqa: E402


def step1_gates(cfg: PipelineConfig, probe_sets: list[s1.ProbeSet]) -> list[str]:
    failures: list[str] = []
    for ps in probe_sets:
        tr, ev = ps.manifest["train"], ps.manifest["eval"]
        if ev["n_total"] != ev["n_prompts"] or ev["n_total"] != ev["n_prompts_in_pool"]:
            failures.append(f"{ps.name}/eval: {ev['n_total']} rows over {ev['n_prompts']} prompts (pool {ev['n_prompts_in_pool']})")
        if tr["n_total"] != tr["n_prompts"]:
            failures.append(f"{ps.name}/train: {tr['n_total']} rows over {tr['n_prompts']} prompts")
        if tr["n_pos"] != tr["n_neg"]:
            failures.append(f"{ps.name}/train: {tr['n_pos']} pos vs {tr['n_neg']} neg")
        for label, cells in tr["by_condition"].items():
            if any(cells.get(c) != cells.get(CONDITIONS[0]) for c in CONDITIONS):
                failures.append(f"{ps.name}/train: {label} {cells} not cue-balanced")
        if ps.family in ("bias", "bias_followed"):
            n_per_class = cfg.n_train_per_class[ps.family]
            if tr["n_pos"] != n_per_class:
                failures.append(f"{ps.name}/train: {tr['n_pos']} per class, sizes.json prescribes {n_per_class}")
        else:
            strata = ps.manifest["train_strata"]
            n_yes, n_no = tr["n_said_yes"], tr["n_said_no"]
            if (strata["deception_yes"], strata["deception_no"]) != (n_yes, n_no):
                failures.append(f"{ps.name}: strata {strata} != said split ({n_yes}, {n_no})")
            if n_yes + n_no != tr["n_total"]:
                failures.append(f"{ps.name}: said-Yes + said-No != train total")
            for kind in ("deception_yes", "deception_no"):
                if strata[kind] != 2 * cfg.n_train_per_class[kind]:
                    failures.append(f"{ps.name}: stratum {kind} has {strata[kind]} rows, sizes.json prescribes {2 * cfg.n_train_per_class[kind]}")
    return failures


def vllm_letter_and_margin(row: dict) -> tuple[str, float] | None:
    """The letter vLLM's logprobs favour and |lp_A - lp_B| in nats, from
    `p_a_cued` (logprob arm: P(A) renormalized over {A, B}, so its logit is
    exactly lp_A - lp_B) or from `lp_a` / `lp_b` (a letter missing from the
    top-20 counts as infinitely far behind). None when the row carries neither."""
    if row.get("p_a_cued") is not None:
        p_a = row["p_a_cued"]
        if p_a in (0.0, 1.0):
            return ("A" if p_a == 1.0 else "B"), math.inf
        return ("A" if p_a > 0.5 else "B"), abs(math.log(p_a) - math.log(1 - p_a))
    if "lp_a" not in row and "lp_b" not in row:
        return None
    lp_a, lp_b = row.get("lp_a"), row.get("lp_b")
    if lp_a is None and lp_b is None:
        return None
    if lp_a is None or lp_b is None:
        return ("A" if lp_b is None else "B"), math.inf
    return ("A" if lp_a > lp_b else "B"), abs(lp_a - lp_b)


def hf_vllm_agreement(cfg: PipelineConfig) -> dict | None:
    """Over every extracted bias-family row that carries vLLM letter logprobs:
    agreement of the HF top letter (hf_letter_logprobs) with vLLM's on the rows
    whose vLLM margin exceeds spec.HF_VLLM_MARGIN_NAT. None when no row
    carries logprobs (the CoT arms)."""
    n_gated, n_agree, n_with_logprobs, n_rows = 0, 0, 0, 0
    for family, definition in cfg.dataset_keys():
        if family not in ("bias", "bias_followed"):
            continue
        dataset = cfg.dataset_name(family, definition)
        for split in cfg.splits:
            rows_by_key = {(r["question_id"], r["condition"], r["sample_idx"], r["category"]): r for r in s1.load_jsonl(s2.probe_set_path(cfg, family, definition, split))}
            saved = torch.load(s2.activation_path(cfg, dataset, POSITION_OF_FAMILY[family], split), weights_only=False)
            for key, (lp_a, lp_b) in zip(saved["row_keys"], saved["hf_letter_logprobs"]):
                n_rows += 1
                vllm = vllm_letter_and_margin(rows_by_key[tuple(key)])
                if vllm is None:
                    continue
                n_with_logprobs += 1
                vllm_letter, margin = vllm
                if margin <= spec.HF_VLLM_MARGIN_NAT:
                    continue
                n_gated += 1
                n_agree += int(("A" if lp_a > lp_b else "B") == vllm_letter)
    if n_with_logprobs == 0:
        return None
    assert n_with_logprobs == n_rows, f"only {n_with_logprobs}/{n_rows} bias-family rows carry vLLM letter logprobs"
    share = n_agree / n_gated
    return {"n_rows": n_rows, "n_gated": n_gated, "n_agree": n_agree, "agreement": share, "margin_nat": spec.HF_VLLM_MARGIN_NAT, "min": spec.HF_VLLM_AGREEMENT_MIN}


def same_letter_duplicate_share(labels: list[dict]) -> dict:
    """Share of (question, condition) prompts in labels.jsonl with at least two
    rows carrying the same answer letter: identical text, identical activations."""
    letters_by_prompt: dict[tuple, list] = {}
    for r in labels:
        letters_by_prompt.setdefault((r["question_id"], r["condition"]), []).append(r["answer_letter"])
    n_duplicate = sum(1 for letters in letters_by_prompt.values() if any(n >= 2 for letter, n in Counter(letters).items() if letter is not None))
    return {"n_prompts": len(letters_by_prompt), "n_with_same_letter_duplicates": n_duplicate, "share": n_duplicate / len(letters_by_prompt)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=spec.ARMS)
    parser.add_argument("--arms-dir", type=Path, default=spec.ARMS_DIR)
    parser.add_argument("--limit-per-class", type=int, default=None, help="dry run: rows per class in step 2")
    args = parser.parse_args()

    bootstrap_hf_env()
    cfg = arm_config(args.arm, args.arms_dir.resolve(), limit_per_class=args.limit_per_class)
    assert not cfg.act_dir.exists(), f"{cfg.act_dir} exists -- this arm was already extracted"
    free_gb = shutil.disk_usage("/workspace").free / 1e9
    assert free_gb >= spec.DISK_MIN_FREE_GB, f"only {free_gb:.1f} GB free on /workspace, need {spec.DISK_MIN_FREE_GB}"
    print(json.dumps(config_to_json(cfg), indent=2))

    # step 1
    labels = s1.load_labels(cfg)
    probe_sets = s1.build_all(cfg, labels)
    failures = step1_gates(cfg, probe_sets)
    for f in failures:
        print("  FAIL", f)
    assert not failures, failures
    duplicates = same_letter_duplicate_share(labels)
    print(f"\nstep 1: 0 failures; prompts with same-letter duplicate rows: {duplicates['n_with_same_letter_duplicates']}/{duplicates['n_prompts']} ({duplicates['share']:.4f})")

    # step 2
    torch.cuda.empty_cache()
    model, tokenizer = s2.load_model_and_tokenizer(cfg)
    lm_head_check: dict = {}
    results = s2.extract_all(cfg, model, tokenizer, lm_head_check)
    print(json.dumps(results, indent=2))

    agreement = hf_vllm_agreement(cfg)
    if agreement is None:
        assert spec.GEN_OF_ARM[cfg.arm] != "nocot", f"{cfg.arm} is a no-CoT arm but its labels carry no letter logprobs (lp_a/lp_b or p_a_cued)"
        print("HF-vs-vLLM agreement gate: no letter logprobs in this arm's labels (CoT arm), not applicable")
    else:
        print(f"HF-vs-vLLM agreement gate: {agreement['n_agree']}/{agreement['n_gated']} = {agreement['agreement']:.4f} on rows with margin > {spec.HF_VLLM_MARGIN_NAT} nat")
        assert agreement["agreement"] >= spec.HF_VLLM_AGREEMENT_MIN, f"HF/vLLM top-letter agreement {agreement['agreement']:.4f} < {spec.HF_VLLM_AGREEMENT_MIN}"

    (cfg.act_dir / "extraction.json").write_text(
        json.dumps(
            {
                "spec_hash": spec.spec_hash(),
                "arm": cfg.arm,
                "config": config_to_json(cfg),
                "gates": {
                    "disk_free_gb": free_gb,
                    "step1_failures": failures,
                    "lm_head_check_max_abs_diff": lm_head_check["max_abs_diff"],
                    "hf_vllm_agreement": agreement,
                    "same_letter_duplicate_prompts": duplicates,
                },
                "datasets": [ps.manifest for ps in probe_sets],
                "extraction": results,
            },
            indent=2,
        )
    )
    print(f"wrote {cfg.act_dir / 'extraction.json'}")
    print(f"EXTRACT DONE {cfg.arm}")


if __name__ == "__main__":
    main()
