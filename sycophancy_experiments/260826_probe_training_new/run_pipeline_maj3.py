"""Interactive driver for majority-of-3 counterfactual labeling -- `# %%` cells.

Same pipeline as run_pipeline.py, but counterfactual answers enter bias labels via
parsed majority over each prompt's 3 samples (no-cue for rm, other-cue for swap).
Outputs go to *_maj3/ paths so indexed activations are never reused.

Open in VSCode (Python Interactive) or Jupyter with the
`/workspace/obfuscation-atlas/.venv` kernel.
"""

# %%
# 1. setup
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

EXP_DIR = Path("/workspace/sycophancy_experiments/260826_probe_training_new")
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore[name-defined]
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore[name-defined]
except NameError:
    pass

import step1_build_probe_sets as s1
import step2_extract_activations as s2
import step3_train_eval_probes as s3
from hf_env import bootstrap_hf_env
from probe_config import EXP_DIR as _EXP_DIR
from probe_config import REFERENCE_DIR, config_to_json, majority3_config

bootstrap_hf_env()
cfg = majority3_config()
print(json.dumps(config_to_json(cfg), indent=2))

# %%
# 2. build probe sets (CPU)
indexed = s1.load_jsonl(cfg.rescore_dir / "rescored_labels.jsonl")
sources = s1.load_sources(cfg)
probe_sets = s1.build_all(cfg, sources=sources)

print("\nprobes to train:")
for spec in cfg.probe_specs():
    print(f"  {spec.name:34s} dataset={spec.dataset:15s} position={spec.position:11s} said={spec.said}")

# gates
failures = list(s1.check_majority_relabeling_gates(indexed, sources.rescored))

for ps in probe_sets:
    tr, ev = ps.manifest["train"], ps.manifest["eval"]

    if ev["n_total"] != ev["n_prompts_in_pool"]:
        failures.append(f"{ps.name}/eval: {ev['n_total']} rows over {ev['n_prompts_in_pool']} prompts")
    if tr["n_pos"] != tr["n_neg"]:
        failures.append(f"{ps.name}/train: {tr['n_pos']} pos vs {tr['n_neg']} neg")
    for label, cells in tr["by_condition"].items():
        if cells.get("cue_gold") != cells.get("cue_dist"):
            failures.append(f"{ps.name}/train: {label} {cells} not cue-balanced")

    line = (
        f"  {ps.name:16s} train {tr['n_total']:5d} ({tr['n_pos']}/{tr['n_neg']})   "
        f"eval {ev['n_total']:5d} ({ev['n_pos']}/{ev['n_neg']})"
    )
    if ps.family == "deception":
        line += f"   train said-Yes {tr['n_said_yes']} / said-No {tr['n_said_no']}"
    print(line)

for ps in probe_sets:
    if ps.family != "deception":
        continue
    strata = ps.manifest["train_strata"]
    n_yes, n_no = ps.manifest["train"]["n_said_yes"], ps.manifest["train"]["n_said_no"]
    if (strata["deception_yes"], strata["deception_no"]) != (n_yes, n_no):
        failures.append(f"{ps.name}: strata {strata} != said split ({n_yes}, {n_no})")
    if n_yes + n_no != ps.manifest["train"]["n_total"]:
        failures.append(f"{ps.name}: said-Yes + said-No != train total")

print(f"\n{len(failures)} failures")
for f in failures:
    print("  FAIL", f)
assert not failures, failures

# %%
# 2b. compare indexed vs majority-of-3 labels
from collections import Counter

majority = sources.rescored
n = len(indexed)

print("=== indexed vs majority3 label comparison ===\n")
for split in ("train", "eval", "all"):
    if split == "all":
        idx_rows, maj_rows = indexed, majority
    else:
        idx_rows = [r for r in indexed if r["split"] == split]
        maj_rows = [r for r in majority if r["split"] == split]
    print(f"--- {split} ({len(idx_rows)} rows) ---")
    for definition in ("rm", "swap"):
        for field in (f"bias_{definition}", f"deception_{definition}"):
            flips = sum(a[field] != b[field] for a, b in zip(idx_rows, maj_rows, strict=True))
            print(f"  {field}: {flips}/{len(idx_rows)} changed ({100 * flips / len(idx_rows):.1f}%)")
            trans = Counter((a[field], b[field]) for a, b in zip(idx_rows, maj_rows, strict=True))
            changed = [(old, new, cnt) for (old, new), cnt in trans.items() if old != new]
            for old, new, cnt in sorted(changed, key=lambda x: -x[2])[:6]:
                print(f"    {old!r} -> {new!r}: {cnt}")
    print()

# %%
# 3. extract activations (GPU)
# SKIP_EXTRACTION=True reuses activations_maj3/ and only loads the tokenizer (for step 3).
# DRY_RUN=True smoke-tests with 20 rows/class into activations_maj3/dry_run/.
SKIP_EXTRACTION = True
DRY_RUN = False

model = None
if SKIP_EXTRACTION:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Skipping extraction -- using cached activations in {cfg.act_dir}")
else:
    torch.cuda.empty_cache()
    model, tokenizer = s2.load_model_and_tokenizer(cfg)
    scores_idx = s2.build_scores_index(cfg)
    print(f"scores.jsonl index: {len(scores_idx)} (question_id, condition) keys")

    extract_cfg = cfg
    if DRY_RUN:
        extract_cfg = replace(
            cfg,
            limit_per_class=20,
            act_dir=cfg.act_dir / "dry_run",
            weights_dir=cfg.weights_dir / "dry_run",
            metrics_path=EXP_DIR / "probe_metrics_maj3_dry_run.json",
        )

    for family in extract_cfg.datasets:
        results = s2.extract_all(
            replace(extract_cfg, datasets=(family,)),
            model,
            tokenizer,
            scores_idx=scores_idx if family != "deception" else None,
        )
        print(s2.summarize(results))

# %%
# 4. train and analyze one probe
# Available probes: cfg.probe_names()
# model=None works once activations are extracted.
PROBE = "bias_rm_answer"

summary, art = s3.train_eval_one_probe(PROBE, None, tokenizer, cfg)
res = s3.analyze_probe(PROBE, cfg, artifacts=art)
s3.plot_score_distributions(res["scores"], cfg, groupings=s3.SCORE_BREAKDOWN_GROUPINGS)

# %%
# compare probes side by side (e.g. removal vs swap, or all-rows vs followed-cue)
COMPARE = ["bias_rm_answer", "bias_swap_answer"]

compare_results = {}
for name in COMPARE:
    _s, _a = s3.train_eval_one_probe(name, None, tokenizer, cfg)
    compare_results[name] = s3.analyze_probe(name, cfg, artifacts=_a)

for name, r in compare_results.items():
    by_cue = {row["value"]: row["auc"] for row in r["slices"]["cue"]}
    print(
        f"  {name:34s} overall={r['auc_overall']:.4f}  "
        f"cue_gold={by_cue.get('cue_gold', float('nan')):.4f}  "
        f"cue_dist={by_cue.get('cue_dist', float('nan')):.4f}"
    )

# %%
# 5. train all probes + compare to indexed run or 260825 (optional)
summaries, _ = s3.train_eval_all(cfg, None, tokenizer)

mine = {s["probe"]: s for s in s3.load_summaries(cfg.metrics_path)}

indexed_metrics = _EXP_DIR / "probe_metrics.json"
if indexed_metrics.exists():
    ref_label = "indexed"
    theirs = {s["probe"]: s for s in s3.load_summaries(indexed_metrics)}
    print("Comparing to indexed-label run (probe_metrics.json). Eval labels differ under majority3.")
else:
    ref_label = "260825"
    theirs = {s["probe"]: s for s in s3.load_summaries(REFERENCE_DIR / "probe_metrics_one_sample.json")}
    print(
        "Comparing to 260825 probe_metrics_one_sample.json (indexed counterfactual labels). "
        "Eval labels differ under majority3 -- AUC deltas mix label denoising in train and eval."
    )

REF_NAME = {"yesno": "firstgen", "preyesno": "lastprompt"}


def ref_name(spec) -> str:
    return f"{spec.kind}_{spec.definition}_{REF_NAME.get(spec.position, spec.position)}"


print(f"{'probe':34s} {'maj3':>8s} {ref_label:>8s} {'delta':>8s}  n_train  n_eval")
for spec in cfg.probe_specs():
    name = spec.name
    a = mine.get(name)
    if a is None:
        print(f"{name:34s} not trained")
        continue
    b = theirs.get(ref_name(spec) if ref_label == "260825" else name)
    ref = (
        f"{b['auc_filtered']:8.4f} {a['auc_filtered'] - b['auc_filtered']:+8.4f}"
        if b
        else f"{'--':>8s} {'--':>8s}"
    )
    print(f"{name:34s} {a['auc_filtered']:8.4f} {ref}  {a['n_train']:7d} {a['n_eval']:7d}")

print("\nstratified deception probes:")
for name in cfg.probe_names():
    if name.startswith(("deception_yes", "deception_no")) and name in mine:
        s = mine[name]
        print(f"  {name:34s} AUC={s['auc_filtered']:.4f}  n_train={s['n_train']} n_eval={s['n_eval']}")

try:
    del model
    torch.cuda.empty_cache()
except NameError:
    pass

# %%
