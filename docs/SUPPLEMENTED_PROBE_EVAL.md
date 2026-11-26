# Supplemented on-policy probe eval

> **Paper-era MBPP code.** This file describes the `obfuscation-atlas/` + `pipeline/` code used for the MBPP workshop paper's Llama-3-8B-Instruct runs. The MBPP rerun in [paper_plan.md](./paper_plan.md) uses `probe-rl/` instead; see [CONTEXT.md](./CONTEXT.md).

Post-hoc analysis for multi-round trial lineages where **native on-policy split eval** has too few honest or deceptive examples (often &lt;20) for stable probe score/AUC metrics.

**Related:** [PROBE_EVAL_PIPELINE.md](./PROBE_EVAL_PIPELINE.md) (on-policy split definition), [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) methodology #27.

**Code:** [`experiments/supplemented_probe_eval/`](../experiments/supplemented_probe_eval/)

---

## What it does

For each model checkpoint `model_k` (base = 0, round 1 = 1, …):

1. **Build a supplemented eval set** — 20 deceptive + 20 honest rows (`MIN_CLASS_COUNT=20`). Uses all native rows from the current model's eval JSONL when enough exist; otherwise **walks back** through earlier rounds → base → (deceptive only) det0 anchor JSONL.
2. **Score with round probes** — same on-policy split logic as pipeline probe eval (nofu / last-token-generation, layer-averaged scores, AUC at FPR=0.01).
3. **Record start/end metrics per round** — for round *R*, **start** = probe trained at round *R* scored on model_{R−1}; **end** = probe trained at round *R* scored on model_R.

Outputs JSON + shift plots under `experiments/trials/trial{N}/results/`.

This is **not** part of `run_trial.py` or GRPO post-eval stages. Run it manually after a lineage completes.

---

## Prerequisites

Per `(trial, variant)` lineage:

| Artifact | Path pattern |
|----------|--------------|
| Trial state | `experiments/trials/trial{N}/trial_state_{variant}.json` (or `trial_state.json` if variant is empty) |
| Eval generations | `experiments/trials/trial{N}/buffer_{variant}/trial{N}_round{R}_eval_generations.jsonl` (or `_wandb.jsonl`) |
| Probe weights | `experiments/trials/trial{N}/rounds_{variant}/round{R}/probe_weights/` |
| GRPO checkpoints | Resolved via `round_grpo_ids` in trial state → `results/run_results/...` |

Shared anchors (always present):

- `experiments/artifacts/base/base_eval_generations.jsonl` — base model eval set
- `experiments/artifacts/det0/det0_eval_generations.jsonl` — deceptive top-up fallback

**GPU:** One GPU with enough free VRAM for Llama-3-8B + LoRA (~16 GB+). Pin training/eval to GPU 0:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Activations are extracted to a **temp directory** and not saved to disk.

**Environment:**

```bash
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
```

Long runs: launch inside **tmux** (see [Long-running processes](../.cursor/rules/long-running-processes.mdc)).

---

## Quick start

### 1. Fetch missing eval JSONLs (if needed)

Local buffers often have rounds 1–4 but not round 5. Add `(trial, variant, round, run_id)` to `MISSING_WANDB_RUNS` in `common.py`, then:

```bash
python /workspace/experiments/supplemented_probe_eval/fetch_missing_gens.py
```

Writes `trial{N}_round{R}_eval_generations_wandb.jsonl` into the variant's buffer dir.

### 2. Run eval

```bash
python /workspace/experiments/supplemented_probe_eval/run_eval.py \
  --trial 4 --variant early_retraining_auc_v2
```

Any `(trial, variant)` pair works if the files above exist — **no need to edit `LINEAGES`**.

Omit both flags to run all entries in `LINEAGES`. Filter with just `--trial` or just `--variant` to subset `LINEAGES`.

Round count comes from `last_completed_round` in trial state (not hardcoded).

### 3. Plot

**Run one lineage per invocation** (multiple `--trial`/`--variant` flags cartesian-product and will fail):

```bash
python /workspace/experiments/supplemented_probe_eval/plot_shifts.py \
  --trial 4 --variant early_retraining_auc_v2
```

---

## Outputs

| Output | Path |
|--------|------|
| Metrics JSON | `experiments/trials/trial{N}/results/supplemented_probe_eval_{variant}.json` |
| Score shift plot | `.../supplemented_on_policy_score_shift_trial{N}_{variant}.png` |
| AUC shift plot | `.../supplemented_on_policy_auc_shift_trial{N}_{variant}.png` |

JSON structure: `rounds["R"]["start"]` and `rounds["R"]["end"]` each have `deceptive`, `honest`, `auc`, plus `supplement` provenance metadata.

---

## Script map

| Script | Role |
|--------|------|
| `common.py` | Paths, supplementation logic, `LINEAGES`, `MISSING_WANDB_RUNS`, `resolve_targets()` |
| `run_eval.py` | Main eval loop: load checkpoints, score supplemented rows, write JSON |
| `plot_shifts.py` | Generate start/end shift plots from JSON |
| `fetch_missing_gens.py` | Pull stage-F `on_policy_responses` from W&B into buffer JSONLs |
| `run_all.sh` / `run_remaining_lineages.sh` | Batch wrappers (hardcoded lineage lists) |

---

## Variant-specific behavior

- **`variant=""`** (original): uses `buffer/`, `rounds/`, `trial_state.json`.
- **`variant="frozen"`**: probe weights always loaded from **round 1** (`probe_round = 1`), regardless of scoring round.
- **Empty variant on CLI**: pass `--variant ""` in bash.

Labels for plots: `variant_label()` in `common.py` (e.g. `early_retraining_auc_v3` → "early retraining (AUC)").

---

## Validation

When native class counts are already ≥20 and supplementation uses only the current model's rows, `run_eval.py` compares **start** metrics against `rounds_{variant}/round{R}/probe_metrics.json` (±0.02 tolerance). Skip with `--no-validate`.

---

## Adding a new lineage

1. Confirm trial state + buffer eval JSONLs + probe weights exist for all rounds.
2. If round-5 (or other) eval JSONL is missing locally, add W&B run ID to `MISSING_WANDB_RUNS` and run `fetch_missing_gens.py`.
3. Run eval + plot (one `--trial`/`--variant` pair per plot command).
4. Optionally append to `LINEAGES` in `common.py` for batch `run_all.sh` convenience only.

---

## Common pitfalls

| Issue | Fix |
|-------|-----|
| `No eval jsonl for trial…` | Run `fetch_missing_gens.py` or regenerate missing round eval JSONL |
| Plot `FileNotFoundError` for wrong variant | Don't pass multiple `--trial`/`--variant` to `plot_shifts.py` in one call |
| OOM on multi-GPU pod | Set `CUDA_VISIBLE_DEVICES=0` so eval doesn't spill onto vLLM's GPU |
| Supplementation `RuntimeError` | Not enough deceptive/honest rows even after walk-back + det0; check eval JSONL labels |
