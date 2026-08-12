# experiments/

Configs, one-off launcher scripts, and per-trial outputs for the multi-round
probe + GRPO pipeline. The reusable pipeline code itself (generation, probe
training, GRPO launch/verify, trial orchestration) lives in
[`/workspace/pipeline/`](../pipeline/), not here — see that package's modules
for the actual implementation. This directory is intentionally lightweight:
YAML configs, small wrapper scripts, and output data.

## Layout

- **`trials/trial{N}/`** — Per-trial state (`trial_state.json`, `trial_config.yaml`),
  `buffer/` (on-policy generations), `rounds/round{R}/` (probe weights,
  activations, GRPO run metrics), and `results/` (exported
  `round_analysis.py` CSVs) for `pipeline/run_trial.py`'s automated
  multi-round loop. Never move or rename — trial state/configs reference
  these paths directly, and some contain absolute paths.
- **`artifacts/`** — Shared on-policy JSONLs (base + det0 × train/eval) reused
  across trials/configs as fixed anchor sources.
- **`supplemented_probe_eval/`** — Post-hoc supplemented on-policy probe eval
  for completed trial lineages. See [docs/SUPPLEMENTED_PROBE_EVAL.md](../docs/SUPPLEMENTED_PROBE_EVAL.md).
- **`run_round_manual.py`** — Active cell-by-cell (`# %%`) scratch driver for
  manually walking through a round when the automated loop needs a one-off
  deviation.
- **`2507_*/`** — Past experiment groups (see below). Each may contain a
  `single_runs/` subfolder for that group's standalone `run_ablation.py`
  outputs.

## Past experiment groups (`2507_*/`)

Each folder groups the YAML config(s) and any one-off launcher script(s) for
a past batch of ablations/sweeps, named `YYMM_description/`:

| Folder | Contents |
|---|---|
| `2507_det0_baselines/` | No-detector-penalty baseline/ceiling configs + seed sweep |
| `2507_nofu_probe/` | Nofu (no-follow-up) probe extraction sweep |
| `2507_prompt_mix/` | Prompt-mixing (`shown_prompt_ratio`) sweep |
| `2507_mixed_reward/` | Mixed-reward (`mixed_reward_pass_all_prob`) sweep |
| `2507_round1_seed_sweeps/` | Round-1 code-length-penalty / seed sweeps (~24 configs) |
| `2507_round2_variants/` | Round-2 baseline, calibrated-penalty, and KL-from-base variants |

These are purely organizational — moving a config here never changes its
behavior: `pipeline/run_ablation.py` takes the config path as a CLI arg, and
`pipeline.grpo_launcher.resolve_path()` always resolves relative paths
(`manifest`, `checkpoint`, `output_dir`) against `experiments/` itself,
regardless of which subfolder the YAML lives in. See `docs/EXPERIMENT_LOG.md`
for the methodology/results behind each group.

## Convention for new experiments

Create one `YYMMDD_description/` folder per new experiment, holding its
configs, wrapper scripts, logs, and notebooks — numbered (`1_generate.sh`,
`2_analyze.ipynb`) when order matters. Prefer duplicating a config/script over
building shared abstractions across experiment folders; core, genuinely
reusable logic belongs in `pipeline/` instead. The data directories listed
above (`trials/`, `artifacts/`) must never move.

## Running the pipeline

```bash
# Multi-round automated trial
python -m pipeline.run_trial --trial 6 --num-rounds 4

# Single run or sweep from a YAML config
python -m pipeline.run_ablation experiments/2507_det0_baselines/det0_baseline.yaml

# Round-1 on-dataset probe train/eval
python -m pipeline.run_round1_probe --trial 6
```

Run from `/workspace` after `source atlas_setup.sh` (sets `PYTHONPATH` so
`pipeline` is importable) and activating the `obfuscation-atlas` venv.
