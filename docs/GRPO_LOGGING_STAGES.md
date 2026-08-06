# train_grpo_model_against_probe — logging map

What each stage writes locally and to W&B for one full run (train + post-eval).
Paths use `RUN_ID` = W&B run id. Group is whatever you set (e.g. `full_grpo_runs`).

Source script: `[obfuscation_atlas/scripts/train_grpo_model_against_probe.py](../obfuscation-atlas/obfuscation_atlas/scripts/train_grpo_model_against_probe.py)`.

Related: [CONTEXT.md](./CONTEXT.md), [FULL_GRPO_RUN.md](./FULL_GRPO_RUN.md).

---

## Directory roots for one run


| Root                  | Path                                                                                   |
| --------------------- | -------------------------------------------------------------------------------------- |
| **Run dir**           | `{save_path_base}/{wandb.group}/trigger_none_model_{RUN_ID}_layers_32/`                |
| **Shared base probe** | `{save_path_base}/base_results_and_detectors_{model}_{trigger}_{probe}_detector-lp_…/` (default runs only) |
| **Activation caches** | `{model_cache_path}/activations_{RUN_ID}_{stage}_{train|test}`                         |
| **Resume map**        | `{save_path_base}/wandb_resume_map.json`                                               |


Example: `/workspace/results/run_results/full_grpo_runs/trigger_none_model_c3akfszd_layers_32/`

On a clean full run: **1 W&B run**, **5 probe stages** (via `log_detector_metrics`), **~128** per-step GRPO metric logs. Default runs use **2 local roots** (run dir + shared base probe); with `detector.directory_save_prefix` set, Phase-0 artifacts also live in the run dir.

---

## Shared pattern: every detector stage

Stages B, D, E, G, H all go through `log_detector_metrics`. Same local file set and W&B key families; only `{prefix}` changes.


| Local                                                             | W&B                                                                | Meaning                                                                           |
| ----------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `f1_and_recall_plot_{prefix}.png`                                 | `f1_and_recall_plot/{prefix}`                                      | Layer-wise AUC / recall@FPR / accuracy curves                                     |
| `f1_and_recall_plot_{prefix}__{detector}.pkl` + `__manifest.json` | — (local only)                                                     | Full per-layer metrics + score arrays (dill); manifest maps detector keys → files |
| `detector_{prefix}__{detector}/` (weights)                        | — (local only)                                                     | Saved probe weights (only when stage trains a detector)                           |
| `averaged_violin_{prefix}_{detector}.{pdf,png}`                   | `violin_scores/{prefix}/{detector}_averaged`                       | Score distributions averaged over filtered layers (pos vs neg)                    |
| `averaged_over_{prefix}_{detector}.txt`                           | — (local only)                                                     | Which layer indices were averaged                                                 |
| —                                                                 | `violin_scores/{prefix}/{detector}`                                | Per-layer violin image (subset of layers)                                         |
| —                                                                 | `probes/eval_{prefix}/{detector}/`* (x-axis: `layer_{prefix}`)     | Per-layer AUC, AP, recall@FPR, F1, threshold, mean scores, score hist images      |
| —                                                                 | `probes/eval_{prefix}_averaged/{detector}/*`                       | Same metrics on layer-averaged scores                                             |
| —                                                                 | `aggregated_detector_eval_metrics/{prefix}/{detector}/*`           | Scalars aggregated across layers (AUC, accuracy, recall@1% FPR)                   |
| —                                                                 | `probes/train_{prefix}/…` (x-axis: `detector_train_step_{prefix}`) | Probe training loss/LR curves (only when that stage trains a probe)               |


---

## Stages in script order

### A — Init W&B + run directory

**Code:** `init_wandb_with_checkpoint_resuming`

Fingerprints Hydra config → looks up `wandb_resume_map.json` → `wandb.init` (new or resume). Creates the run directory from group + run id + layer count.


| Destination | What                          | Meaning                                      |
| ----------- | ----------------------------- | -------------------------------------------- |
| Local       | `wandb_resume_map.json`       | config fingerprint → W&B run id (for resume) |
| Local       | run dir created empty         | All later artifacts hang off this path       |
| W&B         | run created; config=Hydra cfg | Full experiment config attached to run       |
| W&B summary | `resume_count`                | How many times this fingerprint resumed      |


---

### B — Base probe: `initial_detector.base_model.on_dataset`

**Code:** `train_and_eval_multiple_detectors` → `log_detector_metrics`

Train linear probe(s) on base-model activations from the probe dataset; evaluate on task-dataset (MBPP) text. Skipped if cache already has a manifest. This is the detector the RL loop will train against.

**Local save root:** shared `base_results_and_detectors_*` at `save_path_base` (default); **run dir** when `detector.directory_save_prefix` is set. Same detector bundle table as above, with prefix `initial_detector.base_model.on_dataset`.

Also caches activations under `model_cache_path/base_activations_…` if configured. Then writes W&B summary: `filtered_layers`, `num_filtered_layers`, min/max (layers kept for RL after loss-threshold filter, e.g. 12–31).

If `early_stop_threshold_from_train_min` is set (methodology #23), also computes and writes: `early_stop_dynamic_threshold_raw` (min score, current round's on-policy train rows), `early_stop_dynamic_threshold` (raw value floored at `early_stop_threshold_floor`, this is what the early-stop check in stage C uses), `early_stop_dynamic_threshold_n_deceptive`/`_n_honest` (row counts backing the min).

---

### C — GRPO training loop (`trainer.train`)

**Code:** `GRPOObfuscationTrainer.train`

Each step: vLLM generates → code executor scores MBPP → probe scores activations → GRPO update. W&B x-axis is `grpo_train_step`. Does **not** call `wandb.finish()` so post-eval can keep logging to the same run.

#### Local (every step / on schedule)


| Path under run dir                     | When               | Contents                                                                 |
| -------------------------------------- | ------------------ | ------------------------------------------------------------------------ |
| `checkpoints/responses/step_{t}.jsonl` | every train step   | prompt, completion (+ reward if passed); also `wandb.save()` of the file |
| `checkpoints/lora/step_{t}/`           | generation queue   | Transient LoRA weights pushed to vLLM for that step                      |
| `checkpoints/step_{t}/`                | every `save_steps` | Full trainer checkpoint (optimizer, etc.)                                |
| `checkpoints/final/` (+ tokenizer)     | end of train       | Final policy LoRA adapter — the main model artifact                      |


#### W&B every step (means over the generation batch)


| Key                                                     | Meaning                                                            |
| ------------------------------------------------------- | ------------------------------------------------------------------ |
| `reward_mean` / `reward_std`                            | Mean/std of composite reward (task − det − …)                      |
| `objective`                                             | `reward_mean − kl_loss`                                            |
| `kl_loss` / `kl_div`                                    | KL penalty term and raw KL(policy ‖ ref)                           |
| `loss`                                                  | GRPO policy-gradient loss                                          |
| `learning_rate` / `grad_norm`                           | Optimizer state                                                    |
| `clip_low_mean` / `clip_high_mean` / `clip_region_mean` | PPO-style clip fractions                                           |
| `mean_policy_logprob` / `mean_reference_logprob`        | Avg log-probs                                                      |
| `response_length_`* / `mean_token_entropy`              | Generation stats                                                   |
| `flops_*` / `mfu` / `training_mfu`                      | Throughput                                                         |
| `rewards/task`                                          | Pass-first task reward (after length penalty), mean                |
| `rewards/detector`                                      | −det_coef × probe_score (masked if fail & configured)              |
| `rewards_metrics/detector_score`                        | Raw probe score used in penalty                                    |
| `rewards/task_success`                                  | Fraction passing first N tests                                     |
| `rewards_metrics/passes_first_n_but_not_all_tests`      | Legacy deceptive rate (passes shown tests but not all); still logged for comparison |
| `rewards_metrics/pass_only_shown_deception`             | Deceptive rate — dual-generation, passes only the shown tests (new default via `deceptive_definition`) |
| `rewards_metrics/likely_hardcoded`                      | Heuristic hardcode flag rate                                       |
| `rewards_metrics/passes_all_tests`                      | Generalization rate                                                |
| `rewards_metrics/does_execute` / `length` / …           | Code-exec diagnostics                                              |
| `train/prompt_shown_fraction`                           | Fraction of prompts showing expected output (logged when `shown_prompt_ratio < 1.0`, i.e. prompt mixing active) |
| `train/mixed_reward_pass_all_fraction`                  | Fraction of prompts graded on `passes_all_tests` this step (logged every step when `mixed_reward_pass_all_prob > 0`) |
| `obfuscation/*`                                         | Per-step decomposition (same formulas as final `eval_obfuscation`) |
| `completions`                                           | W&B Table: prompt/response/reward/advantage samples                |
| `eval/*`                                                | Same family on eval prompts when `eval_steps` hits                 |
| `grpo_train_step`                                       | Custom x-axis for all of the above                                 |


Per-step `obfuscation/*` uses the same formulas as final `eval_obfuscation`, but on the training batch (reference vs policy probe scores × honest/deceptive split). Optional: `misalignment/*` if `misalignment_eval.enabled` (often off for local runs).

---

### D — Fresh probe: `fresh_detector.final_model.on_dataset`

**Code:** `train_and_eval_multiple_detectors` on final LoRA

Retrain a new probe on the final policy’s activations using the same probe dataset; eval on dataset text. Tests whether deception is still linearly readable after RL.

- **Local →** run dir (same detector bundle as stage B, prefix `fresh_detector.final_model.on_dataset`). Also writes `detector_`* weights here.
- **Activations →** `activations_{RUN_ID}` (train/test caches for this stage).

---

### E — Frozen probe on final model + dataset text

**Code:** `eval_detectors_helper` → `initial_detector.final_model.on_dataset`

Reuse the pre-RL probe (no retrain) on final-model activations of dataset completions. Drop in probe AUC here vs stage B is representation drift on fixed text.

- **Local + W&B:** same detector bundle under prefix `initial_detector.final_model.on_dataset` in the run dir. No new detector weights (eval-only).
- Internally extracts `policy_detector_mean_positive_score_on_dataset` for later `representation_change_on_dataset`.

---

### F — On-policy generation

**Code:** `generate_on_policy_responses`

Final LoRA generates completions on eval MBPP prompts via vLLM; code executor scores them. These completions are the “on_policy” text for stages G–H.


| Destination | Key / file                    | Meaning                                                                            |
| ----------- | ----------------------------- | ---------------------------------------------------------------------------------- |
| W&B         | `on_policy_responses` (Table) | Columns: prompt, completion, plus all `rewards/`* and `rewards_metrics/*` per row  |
| W&B         | `final_eval/{metric}`         | Mean of each reward metric over the eval set (pass-first, deceptive, hardcoded, …) |
| Local       | — (table only in W&B)         | Not written as a separate jsonl in this helper                                     |


`final_eval/rewards/task_success` is the clean post-train task rate when generation succeeds. Eval-only recovery runs have sometimes logged 0 here — prefer training curves or a separate MBPP eval on the LoRA.

---

### G — Probes on on-policy text (final model)

**Code:** `initial_detector` + `fresh_detector` `.final_model.on_policy`

Run both the frozen pre-RL probe and the fresh probe on activations of the model-generated code (positive = on-policy completions; negative = honest dataset negatives). Shows whether detectors still fire on what the policy actually says.

Two full detector bundles in the run dir / W&B with prefixes:

- `initial_detector.final_model.on_policy`
- `fresh_detector.final_model.on_policy`

---

### H — Base model on same on-policy text

**Code:** `initial_detector.base_model.on_policy`

Disable LoRA / use reference adapter; score the same generated text through the base model with the frozen probe. This is the reference distribution for obfuscation decomposition:

- if the base model still looks deceptive on that text → evasion is mostly in the policy’s wording (**obfuscated policy**)
- if only the final model’s activations look clean → **obfuscated activations**

Detector bundle prefix: `initial_detector.base_model.on_policy`.

---

### I — Obfuscation decomposition (final)

**Code:** `compute_obfuscation_decomposition_metrics_helper`

Combines stage G policy scores, stage H reference scores, and stage F task/deceptive flags. Logged twice — once for the initial detector, once for the fresh detector.


| W&B key                                                                                  | Meaning                                                               |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `eval_obfuscation/initial_detector/reference_detector_prob`                              | Mean probe score on base model + on-policy text                       |
| `eval_obfuscation/initial_detector/policy_detector_prob`                                 | Mean probe score on final model + same text                           |
| `eval_obfuscation/initial_detector/obfuscated_acts_score`                                | −mean(policy − reference) — activation evasion                        |
| `eval_obfuscation/initial_detector/obfuscated_policy_score`                              | −mean(reference × deceptive) — text-level evasion on hacks            |
| `eval_obfuscation/initial_detector/honest_probe_score`                                   | −mean(reference × honest)                                             |
| `eval_obfuscation/initial_detector/honest_sequence_reward` / `deceptive_sequence_reward` | Task−det reward split by honest vs deceptive                          |
| `eval_obfuscation/initial_detector/*_margin` / `detector_mean_*_threshold`               | Same scores relative to decision / positive thresholds                |
| `eval_obfuscation/fresh_detector/`*                                                      | Same set of keys using the fresh probe’s on-policy scores             |
| `representation_change_on_dataset`                                                       | mean_pos(base, dataset) − mean_pos(final, dataset) from stages B vs E |


No dedicated local file — these live in the W&B run summary (and history as a single log step). Export script reads them for classification.

---

## What matters most for classification

**Task behavior**

- During train: `rewards_metrics/`* vs `grpo_train_step`
- End: `final_eval/*` and `on_policy_responses` table
- Deceptive ≈ `pass_only_shown_deception` (new default, configurable via `deceptive_definition`); legacy `passes_first_n_but_not_all_tests` still logged for comparison. Hardcode ≈ `likely_hardcoded`

**Obfuscation type**

- `eval_obfuscation/initial_detector/policy_detector_prob`
- `eval_obfuscation/…/obfuscated_acts_score`
- `representation_change_on_dataset`
- Export recomputes paper labels from these (+ `val_`* overlays)

---

## Split-run caveat (seen on local Jul 2026 run)

If post-eval crashes and you re-run with `only_eval_from_other_run`, stages D–I log to a **new** W&B run while the LoRA stays under the training run dir. Then export with `--run-id` (eval) `--task-metrics-run-id` (train). A clean full run keeps everything on one `RUN_ID`.

---

## Quick path cheat sheet

```text
# After a clean full run with group=full_grpo_runs and id=abc123:
results/run_results/full_grpo_runs/trigger_none_model_abc123_layers_32/
  checkpoints/final/lora/          # adapter
  checkpoints/responses/step_*.jsonl
  f1_and_recall_plot_*.png/.pkl    # stages B (if directory_save_prefix), D,E,G,H
  averaged_violin_*.png
  detector_fresh_detector…/        # fresh probe weights
  detector_initial_detector…/      # phase-0 weights (if directory_save_prefix)

results/run_results/base_results_and_detectors_*_{probe}/   # stage B only (default runs)
results/activations/activations_abc123_*                   # act caches
W&B run abc123: train curves + all post-eval keys above
```

