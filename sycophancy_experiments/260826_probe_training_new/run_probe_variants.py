"""Probe hyperparameter experiments -- `# %%` cells.

Copy of run_pipeline.py for testing training variants (sklearn LBFGS, weight_decay)
without cluttering the main workflow. Requires cached activations from run_pipeline.py
cell 3 (or set SKIP_EXTRACTION=False here to extract first).

Open in VSCode (Python Interactive) with `/workspace/obfuscation-atlas/.venv`.
"""

# %%
# 1. setup
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from obfuscation_atlas.config import BaseTrainConfig

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
from probe_config import config_to_json, default_config

bootstrap_hf_env()
cfg = default_config()
print(json.dumps(config_to_json(cfg), indent=2))

# %%
# 2. build probe sets (CPU)
sources = s1.load_sources(cfg)
probe_sets = s1.build_all(cfg, sources=sources)

print("\nprobes to train:")
for spec in cfg.probe_specs():
    print(f"  {spec.name:34s} dataset={spec.dataset:15s} position={spec.position:11s} said={spec.said}")

# gates
REF_N_EVAL = {"bias_rm": 1355, "bias_swap": 1364, "deception_rm": 1355, "deception_swap": 1364}
failures = []

for ps in probe_sets:
    tr, ev = ps.manifest["train"], ps.manifest["eval"]

    if ev["n_total"] != ev["n_prompts_in_pool"]:
        failures.append(f"{ps.name}/eval: {ev['n_total']} rows over {ev['n_prompts_in_pool']} prompts")
    if ps.name in REF_N_EVAL and ev["n_total"] != REF_N_EVAL[ps.name]:
        failures.append(f"{ps.name}/eval: n={ev['n_total']}, 260825 reported {REF_N_EVAL[ps.name]}")
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
# 3. extract activations (GPU) -- default skip; run run_pipeline.py cell 3 first
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
            metrics_path=EXP_DIR / "probe_metrics_dry_run.json",
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
# 4. training variants -- each writes to its own weights_dir / metrics_path
_ADAMW = dict(max_steps=1000, batch_size=32, learning_rate=3e-3, warmup_steps=0, scheduler="constant")

VARIANTS = {
    "baseline": replace(
        cfg,
        weights_dir=cfg.weights_dir.parent / "probe_weights_baseline",
        metrics_path=EXP_DIR / "probe_metrics_baseline.json",
    ),
    "sklearn": replace(
        cfg,
        use_sklearn=True,
        train_cfg=BaseTrainConfig(**_ADAMW, weight_decay=0.01),
        weights_dir=cfg.weights_dir.parent / "probe_weights_sklearn",
        metrics_path=EXP_DIR / "probe_metrics_sklearn.json",
    ),
    "wd_0.01": replace(
        cfg,
        train_cfg=BaseTrainConfig(**_ADAMW, weight_decay=0.01),
        weights_dir=cfg.weights_dir.parent / "probe_weights_wd001",
        metrics_path=EXP_DIR / "probe_metrics_wd001.json",
    ),
    "wd_0.1": replace(
        cfg,
        train_cfg=BaseTrainConfig(**_ADAMW, weight_decay=0.1),
        weights_dir=cfg.weights_dir.parent / "probe_weights_wd01",
        metrics_path=EXP_DIR / "probe_metrics_wd01.json",
    ),
}

VARIANT = "wd_0.1"  # baseline | sklearn | wd_0.01 | wd_0.1
vcfg = VARIANTS[VARIANT]
print(f"Selected variant: {VARIANT}")
print(f"  use_sklearn={vcfg.use_sklearn}  weight_decay={vcfg.train_cfg.weight_decay}")
print(f"  weights_dir={vcfg.weights_dir}")

# %%
# 5. train and analyze one probe under the selected variant
PROBE = "deception_yes_rm_yesno"

summary, art = s3.train_eval_one_probe(PROBE, None, tokenizer, vcfg)
res = s3.analyze_probe(PROBE, vcfg, artifacts=art)
s3.plot_score_distributions(res["scores"], vcfg, groupings=s3.SCORE_BREAKDOWN_GROUPINGS)

# %%
# 6. compare all variants on the same probe
COMPARE_PROBE = "deception_rm_yesno"
variant_rows = []

for vname, vcfg_i in VARIANTS.items():
    summary_i, art_i = s3.train_eval_one_probe(COMPARE_PROBE, None, tokenizer, vcfg_i, save=True)
    variant_rows.append((vname, summary_i, art_i, vcfg_i))

print(f"\nVariant comparison for {COMPARE_PROBE}:")
print(f"  {'variant':12s} {'AUC':>8s}  {'AUC all':>8s}  n_layers  fell_back")
for vname, summary_i, _, _ in variant_rows:
    print(
        f"  {vname:12s} {summary_i['auc_filtered']:8.4f}  {summary_i['auc_all_layers']:8.4f}  "
        f"{len(summary_i['filtered_layers']):>8d}  {summary_i['layer_filter_fell_back']}"
    )

for vname, summary_i, art_i, vcfg_i in variant_rows:
    print(f"\n=== {vname}  AUC={summary_i['auc_filtered']:.4f} ===")
    res_i = s3.analyze_probe(COMPARE_PROBE, vcfg_i, artifacts=art_i)
    s3.plot_score_distributions(res_i["scores"], vcfg_i, groupings=s3.SCORE_BREAKDOWN_GROUPINGS)

# optional: train all probes under one variant and write metrics
# summaries, _ = s3.train_eval_all(VARIANTS["sklearn"], None, tokenizer)

try:
    del model
    torch.cuda.empty_cache()
except NameError:
    pass

# %%
