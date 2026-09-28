"""Interactive driver -- `# %%` cells.

Cells 2-3 are CPU-only and read 260826's trained probes and activations
without modifying them. Cells 4-7 are the probe pipeline (build sets, extract,
train) for this folder's own runs.

Open in VSCode (Python Interactive) or Jupyter with the
`/workspace/obfuscation-atlas/.venv` kernel. Use a fresh kernel: the module
names here shadow 260826's.
"""

# %%
# 1. setup
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

EXP_DIR = Path("/workspace/sycophancy_experiments/260923_improve_probe_performance")
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore[name-defined]
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore[name-defined]
except NameError:
    pass

import label_agreement as la
import run_log
import step1_build_probe_sets as s1
import step2_extract_activations as s2
import step3_train_eval_probes as s3
from hf_env import bootstrap_hf_env
from probe_config import PREVIOUS_DIR, config_to_json, default_config

bootstrap_hf_env()
cfg = default_config()
print(json.dumps(config_to_json(cfg), indent=2))
print("\nprobes to train:")
for spec in cfg.probe_specs():
    print(f"  {spec.name:34s} dataset={spec.dataset:17s} position={spec.position:9s} said={spec.said}")

# %%
# 2. how often do the 3 samples of a cell agree, per condition (CPU)
agreement = la.cell_agreement(cfg)
la.print_cell_agreement(agreement)

# %%
# 3. eval AUC of 260826's probes, split by whether the 3 no-cue rollouts agree (CPU)
# Re-scores 260826's cached activations with its saved weights; nothing is
# retrained or re-extracted. Other probes: replace(cfg, positions=..., definitions=...).
prev_cfg = replace(
    cfg,
    weights_dir=PREVIOUS_DIR / "probe_weights_maj3",
    act_dir=PREVIOUS_DIR / "activations_maj3",
)
prev_metrics = {
    s["probe"]: s for s in json.loads((PREVIOUS_DIR / "probe_metrics_maj3.json").read_text())["probes"]
}

by_none_unanimous = []
for name in cfg.probe_names():
    ps = s3.score_probe(name, prev_cfg)
    overall, n_pos, n_neg = s3.mean_auc_masked(ps, ps.filtered_layers)
    recorded = prev_metrics[name]["auc_filtered"]
    assert abs(overall - recorded) < 1e-3, f"{name}: re-scored {overall:.4f} vs recorded {recorded:.4f}"
    slices = s3.slice_auc(ps, "none-unanimous")
    print(f"\n=== {name} ===  overall eval AUC {overall:.4f}   n={ps.n('eval')} ({n_pos} pos / {n_neg} neg)")
    print(s3.format_slice_table(slices, "by none-unanimous"))
    by_none_unanimous.append(
        {
            "probe": name,
            "auc_overall": overall,
            "n_eval": ps.n("eval"),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "filtered_layers": ps.filtered_layers,
            "by_none_unanimous": slices,
        }
    )

cfg.previous_probes_none_unanimous_path.write_text(
    json.dumps(
        {
            "config": config_to_json(cfg),
            "previous_weights_dir": str(prev_cfg.weights_dir),
            "previous_act_dir": str(prev_cfg.act_dir),
            "cell_agreement": agreement,
            "probes": by_none_unanimous,
        },
        indent=2,
    )
)
print(f"\nwrote {cfg.previous_probes_none_unanimous_path}")

# %%
# 3b. summary table of the split AUCs (reads cell 3's output file)
saved = json.loads(cfg.previous_probes_none_unanimous_path.read_text())
# bias-family probes first, then deception, each in cell 3's order
probes_in_order = sorted(saved["probes"], key=lambda p: 0 if p["probe"].startswith("bias") else 1)

col_width = 24
header = (
    f"{'probe':28s} {'overall':>8s}   "
    f"{'unanimous':^{col_width}s}   {'not unanimous':^{col_width}s}"
)
subheader = (
    f"{'':28s} {'AUC':>8s}   "
    f"{'AUC  (n, pos/neg)':^{col_width}s}   {'AUC  (n, pos/neg)':^{col_width}s}"
)
rule = "-" * len(header)
print(header)
print(subheader)
print(rule)
for p in probes_in_order:
    by_value = {row["value"]: row for row in p["by_none_unanimous"]}
    line = f"{p['probe']:28s} {p['auc_overall']:8.3f}"
    for row in (by_value[True], by_value[False]):
        counts = f"({row['n']}, {row['n_pos']}/{row['n_neg']})"
        line += f"   {row['auc']:6.3f}  {counts:>{col_width - 8}s}"
    print(line)
print(rule)

# %%
# 4. build probe sets (CPU)
indexed = s1.load_jsonl(cfg.rescore_dir / "rescored_labels.jsonl")
sources = s1.load_sources(cfg)
probe_sets = s1.build_all(cfg, sources=sources)

# gates. Balance checks are hard failures without a train filter; with one, a
# short cell is expected (the builder warned and took what it had), so they print.
failures = list(s1.check_majority_relabeling_gates(indexed, sources.rescored))
balance_problems = failures if cfg.train_filter == "none" else []

for ps in probe_sets:
    tr, ev = ps.manifest["train"], ps.manifest["eval"]

    if ev["n_total"] != ev["n_prompts_in_pool"]:
        failures.append(f"{ps.name}/eval: {ev['n_total']} rows over {ev['n_prompts_in_pool']} prompts")
    if tr["n_pos"] != tr["n_neg"]:
        balance_problems.append(f"{ps.name}/train: {tr['n_pos']} pos vs {tr['n_neg']} neg")
    for label, cells in tr["by_condition"].items():
        if cells.get("cue_gold") != cells.get("cue_dist"):
            balance_problems.append(f"{ps.name}/train: {label} {cells} not cue-balanced")

    line = (
        f"  {ps.name:18s} train {tr['n_total']:5d} ({tr['n_pos']}/{tr['n_neg']})   "
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

if cfg.train_filter != "none":
    for w in balance_problems:
        print("  WARN (expected under train_filter)", w)
print(f"\n{len(failures)} failures")
for f in failures:
    print("  FAIL", f)
assert not failures, failures

# %%
# 5. extract activations (GPU)
# SKIP_EXTRACTION=True reuses activations/ and only loads the tokenizer (for step 3).
# DRY_RUN=True smoke-tests with 20 rows/class into activations/dry_run/.
SKIP_EXTRACTION = False
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
# 6. train and analyze one probe
# Available probes: cfg.probe_names(). model=None works once activations are extracted.
PROBE = "bias_rm_answer"

summary, art = s3.train_eval_one_probe(PROBE, None, tokenizer, cfg)
res = s3.analyze_probe(PROBE, cfg, artifacts=art)
s3.plot_score_distributions(res["scores"], cfg, groupings=s3.SCORE_BREAKDOWN_GROUPINGS)

# %%
# 7. train all probes once (single seed, cfg.weights_dir / cfg.metrics_path)
summaries, _ = s3.train_eval_all(cfg, None, tokenizer)
print(run_log.five_probe_table(cfg.metrics_path))

# %%
# 8. one recorded round: all probes x cfg.probe_seeds into runs/{NN}_{SLUG}/, then a runs.md row.
# SLUG names the one thing this round changes; CHANGE and DECISION are the runs.md text.
SLUG = "baseline"
CHANGE = "baseline config, activations/base"
DECISION = ""

round_cfg = run_log.start_round(cfg, SLUG)
s3.train_eval_all_seeds(round_cfg, tokenizer)
print(run_log.five_probe_table(round_cfg.metrics_path))
run_log.append_runs_table(round_cfg, CHANGE, DECISION)

# %%
try:
    del model
    torch.cuda.empty_cache()
except NameError:
    pass

# %%
