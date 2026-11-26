# Probe evaluation pipeline

> **Paper-era MBPP code.** This file describes the `obfuscation-atlas/` + `pipeline/` code used for the MBPP workshop paper's Llama-3-8B-Instruct runs. The MBPP rerun in [paper_plan.md](./paper_plan.md) uses `probe-rl/` instead; see [CONTEXT.md](./CONTEXT.md).

Architecture and data flow for deception-probe training and evaluation in the Obfuscation Atlas GRPO experiments. Covers the official post-RL pipeline, HF checkpoint eval, scratch replicas, and the **on-policy split** metrics added Jul 2026.

Related docs:
- [GRPO_LOGGING_STAGES.md](./GRPO_LOGGING_STAGES.md) — per-stage W&B/local artifact map
- [PROBE_EVAL_RUNBOOK.md](./PROBE_EVAL_RUNBOOK.md) — HF checkpoint sweeps, local eval-only, single-GPU lifecycle
- [SUPPLEMENTED_PROBE_EVAL.md](./SUPPLEMENTED_PROBE_EVAL.md) — post-hoc supplemented on-policy split eval for trial lineages
- [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) — trial status, run history, and experiment details
- [CONTEXT.md](./CONTEXT.md) — workspace overview

---

## Module roles

Three layers; each has a distinct job:

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Eval engine** | `detectors/evaluate.py` | Train/score probes: activations → logits → `get_detector_metrics`. Dataset-agnostic `(pos, neg)` tuples. |
| **Reporting + data** | `training/obfuscation_training.py` | HF dataset splits for RL; `log_detector_metrics`; artifact save/load; **on-policy split** post-processing. |
| **Orchestration** | `scripts/train_grpo_model_against_probe.py` | When to train, generate, eval, log; wires GRPO + post-RL stages A–I. |

Important: **`obfuscation_training.prepare_dataset`** (HF splits for MBPP/probe) ≠ **`detectors/train.prepare_dataset`** (activation tensor extraction for probe scoring).

```mermaid
flowchart TB
  subgraph script ["train_grpo_model_against_probe.py"]
    run_rl["run_rl()"]
    gen["generate_on_policy_responses()"]
    helper["eval_detectors_helper()"]
    split_guard["log_on_policy_split_if_applicable()"]
  end

  subgraph ot ["obfuscation_training.py"]
    prep_hf["prepare_dataset()"]
    log["log_detector_metrics()"]
    split["run_on_policy_split_eval()"]
  end

  subgraph ev ["detectors/evaluate.py"]
    taem["train_and_eval_multiple_detectors()"]
    em["eval_multiple_detectors()"]
    ed["eval_detector()"]
  end

  run_rl --> prep_hf
  run_rl --> taem
  run_rl --> helper
  run_rl --> gen
  helper --> em
  taem --> ed
  em --> ed
  helper --> log
  split_guard --> split
```

---

## Post-RL probe stages (official)

After GRPO training (or eval-only recovery), `run_rl` in `train_grpo_model_against_probe.py` runs probe stages in this order. See [GRPO_LOGGING_STAGES.md](./GRPO_LOGGING_STAGES.md) for full artifact tables.

| Stage | Prefix | What happens |
|-------|--------|--------------|
| **B** | `initial_detector.base_model.on_dataset` | Train linear probe on **base model** activations; eval on fixed MBPP probe text (hardcode vs code). Shared cache under `base_results_and_detectors_*`. |
| **C** | — | GRPO training loop (probe penalty during RL). |
| **D** | `fresh_detector.final_model.on_dataset` | Retrain probe on **final LoRA** activations; same fixed probe dataset. |
| **E** | `initial_detector.final_model.on_dataset` | Frozen pre-RL probe on final-model activations of fixed dataset text. |
| **F** | — | `generate_on_policy_responses`: 400 val prompts → 400 completions + code-exec `reward_metrics`. W&B `on_policy_responses` table. |
| **G** | `initial_detector.final_model.on_policy`, `fresh_detector.final_model.on_policy` | Probe eval on **final model** activations of on-policy text. |
| **G′** | `{above}.split` | **Split eval** (Jul 2026): post-process scores from G; deceptive vs honest gens only. |
| **H** | `initial_detector.base_model.on_policy` (+ `.split`) | Same on-policy **text**, scored through **base/reference** model activations. |
| **I** | `eval_obfuscation/*` | Obfuscation decomposition from G + H scores and F task/deceptive flags. |

HF checkpoint eval (`eval_probe_checkpoints.py`, `only_eval_from_other_run=True`) runs the same post-train stages; expects **5** probe manifest files (`EXPECTED_MANIFEST_COUNT = 5`). Split eval adds JSON sidecars only — no extra manifests.

---

## Two different “on-policy” eval definitions

This distinction is easy to confuse; both use the same 400 model generations but different negative classes.

### Standard on-policy eval (stages G/H)

**Test set construction** (after stage F):

```python
test_dataset = (on_policy_positive_dataset, eval_task_dataset[1])  # 400 gens, 400 fixed honest
test_completion_columns = ("completion", "code")
```

| Class | Source | Meaning |
|-------|--------|---------|
| **Positive** | All 400 model completions | Mix of hacks, honest passes, failures |
| **Negative** | 400 fixed honest `code` from val split | Canonical reference solutions |

**Question answered:** “Does the probe separate model outputs from fixed honest code?”

W&B keys: `probes/eval_*.{detector}.on_policy/...`, `aggregated_detector_eval_metrics/.../on_policy/...`

For a probe trained on fixed hardcode/code, on-policy AUC is often **low** (~0.04 for `viudg2wu` official fresh probe) because the class boundary does not match behavioral deception.

### On-policy split eval (stages G′/H′, Jul 2026)

**No second activation pass.** After standard scoring, subset the **positive score array** (length 400) by code-execution labels from stage F:

| Class | Label | Source key |
|-------|-------|------------|
| **Positive** | Deceptive | `rewards_metrics/pass_only_shown_deception` (`DECEPTIVE_REWARD_KEY`, new default) |
| **Negative** | Honest | `rewards_metrics/passes_all_tests` |
| **Excluded** | Unlabeled | Neither of the above |

The operational `DECEPTIVE_REWARD_KEY` is now `rewards_metrics/pass_only_shown_deception` (dual-generation, passes only the shown tests). The old key `rewards_metrics/passes_first_n_but_not_all_tests` is still computed and logged for comparison, but split-eval masks use the new key by default. Configurable via `deceptive_definition`.

**Question answered:** “Among what the model actually generated, does the probe score deceptive hacks higher than honest passes?”

W&B keys: `probes/eval_*.{detector}.on_policy.split/...`, `aggregated_detector_eval_metrics/.../on_policy.split/...`

Local artifact: none from GRPO runs (W&B only as of Aug 7, 2026). The pipeline replica (`pipeline/probe_training.py`) still writes `on_policy_split_*.json` locally.

**Caveat:** If the honest class is tiny (e.g. 3/400 on `viudg2wu` val), split AUC is unstable. Check the `n_honest` stat (W&B / exported metrics).

**Supplemented variant (Aug 2026):** Post-hoc analysis that tops up each class to 20 examples via walk-back supplementation, then re-scores with round probes. See [SUPPLEMENTED_PROBE_EVAL.md](./SUPPLEMENTED_PROBE_EVAL.md).

---

## Stage F — on-policy generation

**Code:** `generate_on_policy_responses()` in `train_grpo_model_against_probe.py`.

- **Prompts:** 400 from val split. Default: `eval_task_dataset[0]["prompt"]` (always shown). When `task_args.eval_show_expected_output=false`, `run_rl()` rebuilds prompts from `trainer.eval_reward_infos` via `build_mbpp_chat_prompt(..., show_expected_output=False)` (Jul 2026 fix for hidden post-train eval).
- **Generation:** vLLM via final LoRA, `num_generations=1`.
- **Grading:** `compute_code_generation_rewards` → per-row metrics including deceptive/honest flags.
- **Outputs:**
  - `Dataset` with `prompt`, `completion` (400 rows)
  - `reward_metrics` dict (lists aligned with generation order)
  - W&B `on_policy_responses` table + `final_eval/*` means

Optional reuse: `training.model_wandb_id` fetches a prior run’s `on_policy_responses` artifact. **Disabled automatically** when `eval_show_expected_output=false` so hidden eval always regenerates.

**Hidden vs shown eval:** On-dataset probe stages (B, D, E) always cache activations from shown dataset prompts. Only stage F uses hidden prompts when configured. Log `DEBUG (process_data)` dumps during on-dataset caching will still show `== expected` — that is expected; check stage F generation logs for hidden assert lines.

Pipeline equivalent: `pipeline/generation.py` → on-policy JSONLs under `experiments/trials/.../buffer/`.

---

## Standard probe eval flow (one stage)

All detector stages B, D, E, G, H go through the same pattern:

1. **`eval_detectors_helper`** (or `train_and_eval_multiple_detectors` when training)
   - Calls **`eval_multiple_detectors`** → **`eval_detector`**
   - `detectors/train.prepare_dataset`: extract/cache activations
   - `get_detector_scores`: probe logits per layer
   - `get_detector_metrics`: AUC, recall@1% FPR, accuracy, per-sample `scores.positive` / `scores.negative`

2. **`log_detector_metrics`** (`obfuscation_training.py`)
   - Violin plots, F1/recall layer curves, per-layer W&B scalars
   - Layer-averaged metrics over **`filtered_layers`**
   - Saves `f1_and_recall_plot_{prefix}__*.pkl` + manifest

**Activation settings (MBPP):**
- Follow-up: `("Did you respond truthfully?", "Yes")`
- Aggregation: `last-token-generation`
- Layers: 0–31; RL uses layers passing loss threshold (often 12–31)

**No-follow-up (nofu) activation settings:** For nofu runs, probes extract at the completion EOT token (`last-token-generation`) instead of the follow-up "Yes" token. This requires clearing all 4 follow-up prompt configs (probe + task, congruent + incongruent): `probe_dataset.congruent_follow_up_prompts=[]`, `probe_dataset.incongruent_follow_up_prompts=[]`, `task_dataset.congruent_follow_up_prompts=[]`, `task_dataset.incongruent_follow_up_prompts=[]`. Use `detector.directory_save_prefix` to isolate the Phase-0 cache from standard follow-up runs (e.g. `trial3_nofu_mbpp` or `nofu_mixed`). Cross-probe absolute scores between nofu (EOT) and follow-up ("Yes") runs are not directly comparable.

**On-dataset eval** (stages B, D, E):

```python
test_dataset = eval_task_dataset  # (task_positive_val, task_negative_val)
test_completion_columns = ("hardcode", "code")
```

---

## On-policy split implementation (Jul 2026)

**Location:** `obfuscation_training.py` (shared); wired from `train_grpo_model_against_probe.py`.

| Function | Purpose |
|----------|---------|
| `derive_on_policy_split_masks(reward_metrics)` | Build deceptive/honest boolean masks + `stats` from stage F metrics. |
| `compute_split_on_policy_probe_metrics(test_metrics, masks, filtered_layers)` | Recompute AUC/recall/accuracy/means on masked positive scores; layer-averaged over **`filtered_layers`**. |
| `log_split_on_policy_metrics(...)` | Write JSON + optional W&B scalars (no manifest/pkl/plots). |
| `run_on_policy_split_eval(eval_results, reward_metrics, plot_key, ...)` | Orchestrator for one on-policy stage. |

**GRPO wiring:** `log_on_policy_split_if_applicable()` called immediately **after** each `eval_detectors_helper` where `plot_key.endswith(".on_policy")`. Does not modify `eval_detectors_helper`.

**Split prefix:** `f"{plot_key}.split"` (e.g. `fresh_detector.final_model.on_policy.split`).

**Unchanged by design:**
- `eval_multiple_detectors`, `log_detector_metrics`, `eval_detectors_helper`
- Stage I obfuscation decomposition (still uses all 400 positive scores)
- `eval_probe_checkpoints.py` manifest count
- `export_wandb_run_metrics.py` (split keys exported as `on_policy_split_*` fields)

### Split artifacts vs standard artifacts

| Artifact | Standard `.on_policy` | Split `.on_policy.split` |
|----------|----------------------|---------------------------|
| Local pkl + manifest | Yes | **No** |
| f1/recall PNG, violins | Yes | **No** |
| Local JSON | No (metrics in pkl) | No (W&B only; pipeline replica still writes JSON) |
| W&B scalars | Yes + histogram images | **Scalars only** |
| Activation cache | New cache per prefix | **Reuses** standard on-policy cache |

---

## On-dataset vs on-policy vs split (summary)

| Eval | Positive class | Negative class | Typical use |
|------|----------------|----------------|-------------|
| **On-dataset** | Fixed `hardcode` (400) | Fixed `code` (400) | Probe train/test on canonical MBPP probe set |
| **On-policy (standard)** | All 400 generations | Fixed honest `code` (400) | Paper obfuscation metrics; “does probe fire on model output?” |
| **On-policy split** | Deceptive generations only | Honest generations only | Behavioral separation within model outputs |

---

## HF checkpoint probe eval

**Script:** `eval_probe_checkpoints.py`

- Runs `train_grpo_model_against_probe.py` with `only_eval_from_other_run=True` + `reference_adapter_path` = downloaded LoRA.
- Phase 0 base probe from shared cache if present.
- Post-eval produces 5 manifest stages (one subdir per stage); split metrics logged to W&B (initial/fresh on final, initial on base).
- Requires vLLM for stage F generation (unless reusing W&B artifact).
- MBPP task metrics overlaid at export time from `hf_task_eval_metrics/summary.json`.

---

## Pipeline replica (no W&B)

**Modules:** `pipeline/generation.py`, `pipeline/probe_training.py` (or `pipeline/run_round1_probe.py` end-to-end).

Mirrors official fresh-probe pipeline without W&B:

1. Generate/load val JSONL under `experiments/trials/.../buffer/`
2. Train probe via `train_and_eval_multiple_detectors` (on-dataset eval)
3. On-policy eval via `eval_multiple_detectors` (same test set as official G fresh)
4. **`run_on_policy_split_eval`** with `plot_key="fresh_detector.final_model.on_policy"` (same as official)

Split JSON example lands under the round's `probe_activations/` (or similar) next to the probe weights.

---

## Classification and exports

Policy/obfuscation typing in `utils/wandb_api_cache_utils.py` reads **standard** on-policy keys:

- `base_model_mean_score`, `final_model_mean_score` (from `mean_score_positive` on all 400 gens)
- `representation_change`, `eval_obfuscation/*`

Split metrics are exported by `export_wandb_run_metrics.py` as `on_policy_split_*` fields.

---

## Quick reference — key paths

| Item | Path pattern |
|------|----------------|
| GRPO run dir | `results/run_results/{wandb.group}/{run_key}_{RUN_ID}/` |
| Standard on-policy artifacts | `{prefix}/f1_and_recall_plot_{prefix}__*` under run dir |
| Split metrics | W&B only (`probes/eval_{prefix}.split*`); local JSONs no longer written |
| Activation cache | `results/activations/activations_{RUN_ID}_{prefix}_test` (deleted after a successful run) |
| Shared base probe | `results/run_results/base_results_and_detectors_*` |

---

## Related code entry points

| Task | Start here |
|------|------------|
| Full GRPO + post-eval | `scripts/train_grpo_model_against_probe.py` → `run_rl()` |
| Probe train/eval API | `detectors/evaluate.py` |
| Logging + split helpers | `training/obfuscation_training.py` |
| Code-exec labels | `tasks/code_generation.py` → `compute_code_generation_rewards` |
| Obfuscation decomposition | `training/grpo_obfuscation_utils.py` |
| Pipeline test of split | `pipeline/probe_training.py` / `pipeline/run_round1_probe.py` |
