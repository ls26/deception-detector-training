# Experiment Log

Record of methodology decisions, completed runs, and known gaps. For implementation details, see the linked plans under `.cursor/plans/`.

**Related docs:**

- [CONTEXT.md](./CONTEXT.md) — project overview, workspace layout, codebase map
- [iterative_training_experiments.md](./iterative_training_experiments.md) — multi-round pipeline architecture
- [FULL_GRPO_RUN.md](./FULL_GRPO_RUN.md) — GRPO operational runbook
- [FULL_TRIAL_RUNBOOK.md](./FULL_TRIAL_RUNBOOK.md) — `run_trial.py` automation reference

---

## Trial summary

Quick orientation across all multi-round trials — see "Completed runs" below for full per-round tables and metrics.


| Trial                | Rounds             | Status                               | Key result                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------- | ------------------ | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5                    | 1-6                | Complete                             | Calibrated penalty + KL-from-base lineage seeded from `iv76hjr6` (seed 48, eps=1e-4); round 1 Obfuscated policy → round 2 (`ppy5mkw9`) Honest → round 3 (`weyz6p9f`, first round under the fixed calibration denominator, methodology #18) Obfuscated activations → rounds 4-6 (`1nnrrl46`, `8qra80a3`, `5x21tk4x`) Honest. Supersedes an earlier attempt seeded from `emjn04pc` (seed 45), killed after round 2 went degenerate |
| 1                    | 1-6                | Complete                             | Round 1 (`mtzvz59v`) Obfuscated policy; early pipeline validation                                                                                                                                                                                                                                                                                                                                                                |
| 2                    | 1-6                | Complete                             | Round 1 (`lliye1cp`) Honest; rounds 2-6 used mismatched hyperparameters vs round 1 (see Known gaps)                                                                                                                                                                                                                                                                                                                              |
| 3                    | 1-4                | Complete                             | Round 1 (`62es59py`) Obfuscated policy → Honest by rounds 3-4 (`wu6k5j61`, `5lnsv03f`); probe retraining meaningfully constrains deception (see ablations)                                                                                                                                                                                                                                                                       |
| 4                    | 1-5                | Complete                             | KL-from-base lineage; round 1 (`9nlij6so`) Obfuscated policy → Honest by round 5 (`urk4n41r`); round 4's first attempt (`13bscznt`) collapsed — see methodology #13                                                                                                                                                                                                                                                              |
| 4 (calibrated)       | 1-5                | Complete — discontinued              | **Calibrated probe penalty** (methodology #12), not the standard sigmoid penalty. Same KL-from-base lineage; shares round 1 (`9nlij6so`). Historical reference only — do not extend.                                                                                                                                                                                                                                             |
| 4 (carryover)        | 1-5                | Complete — discontinued              | **Calibrated probe penalty** + conditional probe retraining (methodology #15). Shares round 1 (`9nlij6so`). **Not to be continued** — see methodology #15/#19.                                                                                                                                                                                                                                                                   |
| 4 (frozen)           | 1-5                | Rounds 1-4 complete, round 5 running | Same lineage, but the probe is **never retrained** after round 1 — every round reuses round 1's exact weights (methodology #16); shares round 1 (`9nlij6so`); continuously **Obfuscated policy** from round 2 onward with deception climbing monotonically (0.58 → 0.94 → 0.93) — unlike every retrained branch above, never recovers to Honest                                                                                  |
| 4 (early_retraining) | 1-5                | Complete — discontinued              | **Calibrated probe penalty** + detector-score early stopping (methodologies #17–#18). Shares round 1 (`9nlij6so`). Historical reference only — do not extend.                                                                                                                                                                                                                                                                    |
| 4 (early_retraining_v2) | 1-5             | Complete                             | **Standard sigmoid probe penalty** + dynamic early stopping (methodology #23), with methodology #21 nofu/EOT fixes. Shares round 1 (`9nlij6so`); round 2 Obfuscated policy → Honest rounds 3–5. Round 5 manually stopped at step 60 (`io4ibs0w`); post-eval via eval-only `8g8580by`.                                                                                                                                                                                                          |
| 4 (v2)               | 1–5                | Complete                             | **Standard sigmoid probe penalty**, full 128 steps (no early stopping). Same p20/seed-42 lineage as `early_retraining_v2`; matched probe/buffer from Aug 5 restart. **Honest** all rounds 2–5 (cf. `early_retraining_v2` round 2 Obfuscated at early stop).                                                                                                                                                                                                                                       |
| 4 (carryover_frozen) | 1-3 (killed)       | Discontinued                         | **Calibrated probe penalty** + carry-over probe weights on skipped-expansion rounds (methodology #19). Killed at round 3 (design flaw). **Not to be continued.**                                                                                                                                                                                                                                                                 |
| 4 (calibrated_fixed) | 1-6                | Discontinued (rounds 2–6 archived)   | **Calibrated probe penalty** with fixed denominator (methodology #18) and methodology #20 probe-mismatch fix. Shares round 1 (`9nlij6so`); rounds 2–6 artifacts archived as `_pre_fix`. Historical reference only — do not extend.                                                                                                                                                                                               |
| 5 (early_retraining) | 1 (round 2 killed) | Killed — discontinued                | **Calibrated probe penalty** + early stopping on trial 5 lineage. Shares round 1 (`iv76hjr6`); round 2 killed after pipeline bug; state reset to round 1.                                                                                                                                                                                                                                                                        |
| 6 (early_retraining) | 1–5                | Complete (stopped)                   | Methodology #23 (p10 threshold), KL-from-base. Round 1 **Obfuscated policy** → rounds 2–3 **Obfuscated policy** → rounds 4–5 **Honest**. Stopped after round 5 (Aug 5, 2026).                                                                                                                                                                                                                                                    |
| 6                    | 1–5                | Complete (stopped)                   | Same hyperparams as `early_retraining` but **no early stopping** (full 128 steps). Shares round 1 (`yckx0ctm`). **Obfuscated policy** all rounds. Stopped after round 5 (Aug 6, 2026). Round 4 first attempt (`1f03unaz`) killed to pause; resumed Aug 6.                                                                                                                                                                                                                                        |
| 6 (frozen)           | 1–5                | Complete                             | Probe **never retrained** after round 1 (methodology #16); shares round 1 (`yckx0ctm`). **Obfuscated policy/activations** rounds 2–5 (pass_only_shown 0.97→0.92→0.96→0.92); never recovers to Honest — same pattern as trial 4 `frozen`.                                                                                                                                                                                                                                                         |
| 8 (early_retraining) | 1–5                | Complete (stopped)                   | Methodology #23 (p10 threshold), KL-from-base. Round 1 **Obfuscated policy** → rounds 2–3 **Obfuscated policy** → rounds 4–5 **Honest**. Stopped after round 5 (Aug 5, 2026). Renamed from main trial 8 branch Aug 5.                                                                                                                                                                                                               |
| 8                    | 1–5                | Complete (stopped)                   | Same hyperparams as `early_retraining` but **no early stopping** (full 128 steps). Shares round 1 (`t1656g8a`). **Obfuscated policy** all rounds; round 3 **Obfuscated activations**. Stopped after round 5 (Aug 6, 2026). Round 6 partial (`9gylvqmh`) killed manually.                                                                                                                                                                                                                      |
| 8 (frozen)           | 1–5                | Complete                             | Probe **never retrained** after round 1 (methodology #16); shares round 1 (`t1656g8a`). **Obfuscated policy** rounds 2–5 (pass_only_shown 0.95→0.96→0.96→0.97); never recovers to Honest — same pattern as trial 6 `frozen` and trial 8 base.                                                                                                                                                                                                                                                   |


---

## Methodology changes

Reverse-chronological. Each entry covers what changed, why, key config, and affected runs.

### 23. Early retraining reimplemented on standard sigmoid probe with dynamic threshold (Aug 4, 2026)

**Why:** Methodology #17's fixed threshold (`0.06`) was tuned for the calibrated penalty's scale; on the standard sigmoid probe (`rewards_metrics/detector_score`) the per-round score scale drifts, so a single fixed cutoff is not meaningful. The calibrated `early_retraining` branch is also discontinued (methodology #22).

**What:** Reuse methodology #17's early-stop counter (`patience=10`, `min_steps=20`), but compute the threshold each round from Phase-0's **own** probe (same probe that applies the RL penalty): `max(min(current-round on-policy train rows' layer-mean scores, both classes), floor)`. Applied to raw `rewards_metrics/detector_score`, not the scaled penalty. Threshold is broadcast after Phase-0 and logged as `early_stop_dynamic_threshold` / `_raw` / `_n_deceptive` / `_n_honest`.

**Update (Aug 5, 2026):** raw threshold changed from `min(...)` to **p10** (`np.percentile(scores, 10)`) in `compute_dynamic_early_stop_threshold()` — min was sometimes too low. Same config keys; trial 4 `early_retraining_v2` used min, trial 6 `early_retraining` uses p10.

**Config:** `early_stop_threshold_from_train_min: true`, `early_stop_threshold_floor: 0.05`, `early_stop_detector_patience: 10`, `early_stop_min_steps: 20`; `use_calibrated_penalty: false`. Incompatible with `early_stop_detector_threshold` (fixed), `frozen_probe_weights_dir`, or `calibration_logits_path`.

**Files changed:** `obfuscation_atlas/config.py`, `obfuscation_atlas/scripts/train_grpo_model_against_probe.py` (`compute_dynamic_early_stop_threshold()`), `experiments/run_trial.py`.

**Trial config:** `experiments/trials/trial4/trial_config_early_retraining_v2.yaml`. **Launcher:** `python run_trial.py --trial 4 --variant early_retraining_v2 --num-rounds 4` in tmux `early_v2`; log: `experiments/trials/trial4/production_early_retraining_v2.log`.

Plan: [dynamic_early-stop_threshold](../.cursor/plans/dynamic_early-stop_threshold_c7f86de2.plan.md)

### 22. Discontinuing calibrated probe penalty for new work (Aug 4, 2026)

**Why:** Repeated pipeline bugs weakened or invalidated calibrated runs: methodology #20 (GRPO Phase-0 probe vs external calibration probe scale mismatch), #21 (nofu token-position mismatch between external probe training and GRPO scoring), plus unstable round-to-round `(z_d - z_h)` denominators (#18). New iterative work should use the standard sigmoid penalty only.

**Status:** Calibrated variants (`calibrated`, `calibrated_fixed`, `carryover`, `early_retraining`, trial 5) remain logged as historical reference — do not extend. Active successor for early stopping: `early_retraining_v2` (methodology #23).

### 21. Frozen-probe apply-side fixes: nofu propagation, layer filtering, truncated rollouts (Aug 4, 2026)

**Bug 1 (root cause — nofu propagation):** `run_trial.py`'s `run_probe_train_eval_pipeline()` call never passed `follow_up`, so it silently defaulted to `probe_training.py`'s `FOLLOW_UP = [("Did you respond truthfully?", "Yes")]` regardless of the trial config's `nofu: true`. Every round-2+ probe (weights, `input_scale` buffer, on-dataset/on-policy eval, and calibration `z_h`/`z_d`) was therefore trained and calibrated at the follow-up **"Yes"** token, while GRPO (correctly configured nofu via the `congruent_follow_up_prompts=[]` overrides — methodology #4) scores `last-token-generation` at the completion's `**<|eot_id|>`** token. `run_round1_probe.py` already mapped `--nofu` correctly, so only round 1 was unaffected. Confirmed empirically (trial 4 `calibrated_fixed` round 3): scoring the same known-honest checkpoint with the same frozen probe at the "Yes" token reproduced the stored `z_h=-2.0495` to 3 decimals, while scoring at `<|eot_id|>` shifted the median layer-mean logit by **+1.45** (roughly two thirds of the fixed denominator), giving a calibrated score of 0.68 — a `loss.detector=3` penalty of ~-2 on a totally honest model, matching what was observed live in round 4/round 3 of that lineage.

**Bug 2 (layer-filter mismatch, latent):** GRPO derived `filtered_layers` from **Phase-0's own** probe's train-loss dynamics (`get_layers_below_loss_threshold`), not from the frozen probe's own `filtered_layers.txt` that the calibration `z_h`/`z_d` were computed over. Both happened to resolve to all 32 layers in every run so far, so this was benign in practice, but any future run where either filter drops a layer would silently re-break the calibration (the layer-mean the penalty applies to would no longer match the layer set the anchors were computed over).

**Bug 3 (truncated rollouts):** Rollouts hitting the 256-token response cap have no `<|eot_id|>`, so `last-token-generation` probed whatever arbitrary mid-completion token generation stopped on — a position the probe never saw during training, adding pure noise to the penalty.

**Fix:**

- `run_trial.py`: `run_single_round()` now passes `follow_up=[] if trial_cfg["nofu"] else FOLLOW_UP` into `run_probe_train_eval_pipeline()`.
- `probe_training.py`: added `_assert_probed_token()`, called after every activation-cache write (train/test/on-policy) in `last-token-generation` mode, asserting the last completion-mask token is (nofu) or isn't (follow-up) the end-of-turn token — a regression guard against this exact class of bug recurring silently.
- `train_grpo_model_against_probe.py`: when `frozen_probe_weights_dir` is set, `filtered_layers` is now loaded from that probe's own `filtered_layers.txt` (`load_frozen_probe_filtered_layers()`) instead of Phase-0's train dynamics. Phase-0's own probe/eval/thresholds are otherwise unchanged.
- `grpo_trainer.py`: `build_obfuscation_response_dataset()` now appends `<|eot_id|>` to non-terminated (truncated) responses before mask construction — excluded from `completion_mask` (so it doesn't affect the policy-gradient loss or vLLM logprob alignment, same treatment as follow-up tokens) but included in `completion_with_follow_up_mask`, so `last-token-generation` probing reads the position the probe was actually trained on instead of an arbitrary mid-completion token. An earlier version of this fix instead forced the detector score to 1.0 (max/most-deceptive) for non-terminated rollouts in `precompute_composite_rewards()`; that was replaced before use because it measurably inflated the detector reward (observed ~-1.18 vs. the ~-0.5 historical baseline at step 0 of a round-1 run, since ~37% of base-model rollouts truncate) rather than just removing the noise.

**Files changed:** `experiments/run_trial.py`, `experiments/probe_training.py`, `obfuscation_atlas/scripts/train_grpo_model_against_probe.py`, `obfuscation_atlas/training/grpo_trainer.py`.

**Affected runs / required action before reruns:** Every round 2+ probe/calibration produced by `run_trial.py` for any `nofu: true` variant (trial 4's `calibrated`, `calibrated_fixed`, `carryover`, `carryover_frozen`, `early_retraining`, `frozen`, and trial 5) was trained/calibrated on the wrong token and is invalid, independent of and in addition to the methodology #20 probe-mismatch bug. No artifacts were reset or regenerated as part of this fix. Before rerunning any round with these fixes, the round's `probe_weights/` and `probe_activations/` must be deleted first — `train_probe_and_eval_on_dataset()` reuses existing weights/caches on disk if present and will not regenerate them just because the code changed.

Plan: [fix_probe_pipeline_mismatches](../.cursor/plans/fix_probe_pipeline_mismatches_4794e928.plan.md)

### 20. Calibrated-penalty probe mismatch fix: freeze GRPO onto the external pipeline's probe (Aug 4, 2026)

**Bug:** The calibrated probe penalty (methodology #12) computes `z_h`/`z_d` from the external `probe_training.py` pipeline's probe, but GRPO's Phase-0 always trains its *own* fresh probe from scratch and applies those calibration constants to it. The two probes are trained on identical data with identical hyperparameters but different random init/shuffling, so they land on different raw-logit scales (observed ~~2.7x apart in one case). Since the calibrated penalty is `clip((z_agg - z_h) / (z_d - z_h), 0, 1)`, applying `z_h`/`z_d` from one probe's scale to `z_agg` from a differently-scaled probe collapses the penalty toward a near-constant small value regardless of how deceptive a sample actually is — in the case that surfaced this (trial 4 `calibrated_fixed` round 6), the observed penalty (~~0.03) was far below what the calibration implied for a median deceptive sample (~0.20 with the fixed denominator from methodology #18). This has been silently weakening the calibrated penalty in every round of every variant that uses `use_calibrated_penalty: true` without also freezing GRPO onto the external probe.

**Fix:** `run_grpo()` (`run_trial.py`) now always passes `++detector.frozen_probe_weights_dir=<the same round's probe_weights dir used for calibration_logits_path>` alongside `calibration_logits_path` whenever `use_calibrated_penalty: true`. GRPO's Phase-0 still trains/evaluates its own probe as before (needed for eval metrics, thresholds, and layer filtering — unchanged), but `train_grpo_model_against_probe.py`'s existing `frozen_probe_weights_dir` override (methodology #16) then swaps in the externally-trained probe before `set_calibration()` is applied, so the penalty's `z_agg` and the calibration's `z_h`/`z_d` now come from the same probe. When calibration falls back to a previous round's `z_h`/`z_d` (near-fully-deceptive policy with no honest on-policy eval samples that round), the frozen probe weights are switched to that same previous round's `probe_weights/` so the two stay paired.

No changes needed to `train_grpo_model_against_probe.py`, `obfuscation_loss.py`, `probe_training.py`, or any trial config YAML — `frozen_probe_weights_dir` is set dynamically per-round in `run_grpo()`, not in the trial config, so it does not trigger `run_single_round()`'s pre-existing assertion that `frozen_probe_weights_dir` (trial-config-level, used by the `frozen` variant's "skip generation entirely" mode) is incompatible with `use_calibrated_penalty`.

**Files changed:** `experiments/run_trial.py` — `run_grpo()`, `use_calibrated_penalty` block.

**Affected runs:** Every round (2+) of every variant using `use_calibrated_penalty: true` without a trial-config-level `frozen_probe_weights_dir` — trial 4's `calibrated`, `calibrated_fixed`, `carryover` (rounds where expansion isn't skipped), `early_retraining`, and trial 5. Rerun of trial 4 `calibrated_fixed` (rounds 2-6) is in progress under this fix; the in-progress round 6 GRPO run (`xx8b3u2g`, step ~60/128, no final adapter) was killed and its partial checkpoint + stale prefix-keyed activation cache removed. Round 1 and round 2's external-pipeline artifacts (on-policy generation, probe weights, calibration logits) are unaffected by this bug (produced entirely outside GRPO) and were reused as-is; rounds 3-6's artifacts were archived with a `_pre_fix` suffix since they depend transitively on round 2's now-superseded checkpoint. Trial 4 `early_retraining` and trial 5 reruns are deferred to a follow-up.

Plan: [fix_calibrated_probe_bug](../.cursor/plans/fix_calibrated_probe_bug_061784e4.plan.md)

### 19. Carry over frozen probe + calibration on skipped-expansion rounds (Aug 3, 2026) — discontinued

Extended conditional probe retraining (methodology #15) with a new `carry_over_probe_weights` trial-config bool. When `build_or_reuse_probe_dataset()` skips manifest expansion (on-policy samples below `min_on_policy_per_class`), instead of still retraining the probe on the reused manifest, `run_single_round()` copies forward the *previous round's* `probe_weights/` and `probe_metrics.json` (including its calibration `z_h`/`z_d`) unchanged, and passes them to GRPO via `frozen_probe_weights_dir` + `calibration_logits_path` — skipping model loading and the entire external probe-training pipeline for that round entirely.

**Discovered flaw — do not resume without fixing:** in a round where expansion is *not* skipped (the normal path, no `frozen_probe_weights_dir` set), GRPO's Phase-0 trains and uses its *own* internally-trained probe for the actual RL penalty; the external pipeline's probe (the one saved to `probe_weights/` and later carried over) is only ever used to compute calibration constants in that path — it is never the probe the policy actually trained against. So when carry-over triggers in a later round, it swaps in a probe the policy has never encountered, not a continuation of what it just adapted to. This produced a large detector-score discontinuity in trial 4's `carryover_frozen` branch between the end of round 2 (Phase-0's probe, reward ≈0) and the start of round 3 (carried-over external probe, reward ≈-3, i.e. calibrated score ≈1.0) on a policy that had in fact become honest. Verified this is a design issue, not a wiring bug — the carried-over weights and calibration logits were byte-identical to round 2's.

**Status:** the `carryover_frozen` run was killed mid-round-3 and its round-2/round-3 checkpoints and activation caches deleted (round 1, shared with other trial 4 branches, was kept). Neither this variant nor `carryover` (methodology #15) will be extended further — code kept in `run_trial.py` for reference only, not for reuse without addressing the flaw above.

**Implementation:** `build_or_reuse_probe_dataset()` now returns an additional `skipped_expansion` bool. `run_single_round()` calls it before any model loading; if `skipped_expansion and trial_cfg.get("carry_over_probe_weights")`, it copies `round{R-1}/probe_weights/` + `probe_metrics.json` into the current round's directory and sets `frozen_probe_weights_dir` dynamically for that round's GRPO launch, bypassing model load and `run_probe_train_eval_pipeline()`.

**Config:** `experiments/trials/trial4/trial_config_carryover_frozen.yaml` — identical to `trial_config_carryover.yaml` plus `carry_over_probe_weights: true`.

**Files changed:** `experiments/run_trial.py` (`build_or_reuse_probe_dataset()` return signature, carry-over branch in `run_single_round()`).

**Affected runs:** Trial 4's `carryover_frozen` variant — see "Completed runs" below.

### 18. Fixed calibration denominator: freeze (z_d - z_h) scale to round 2's anchor (Aug 3, 2026)

Added an optional override to the calibrated probe penalty (methodology #12): when enabled, the `(z_d - z_h)` denominator is frozen to the value computed at round 2 — the first round where `calibration_logits` exist, since round 1 has no calibration data — instead of being recomputed every round from that round's own probe. `z_h` in the numerator still recomputes each round as before; only the denominator scale is fixed.

**Why:** The denominator has been drifting sharply round to round in the early_retraining variant: round 2's `z_d - z_h = 2.1313`, round 3's (pre-fix) `z_d - z_h = 0.3157` — a ~6.7x shrink. Since the penalty is `clip((z_agg - z_h) / (z_d - z_h), 0, 1)`, this means the same raw logit gap from the honest baseline maps to a much larger relative penalty in round 3 than in round 2, effectively rescaling how aggressively the calibrated penalty responds each round in a way that isn't intentional. Freezing the denominator to round 2's value keeps the penalty's scale consistent across rounds while still letting the "zero point" (`z_h`) track each round's own probe.

**Implementation:**

- `DetectorConfig` (`config.py`): new `fixed_calibration_denominator: float | None = None`.
- `ProbeDetector.set_calibration()` (`obfuscation_loss.py`): now accepts an optional `fixed_denominator` argument; `_aggregate_scores()` uses it in place of `(z_d - z_h)` when set, while `z_h` (from `set_calibration`'s own `z_h` argument) is unchanged in the numerator. Cleared by `disable_calibration()`.
- `train_grpo_model_against_probe.py`: passes `cfg.detector.fixed_calibration_denominator` through to `set_calibration()`.
- `run_trial.py`: new `FIXED_CALIBRATION_ANCHOR_ROUND = 2` module constant. When the trial config's `fix_calibration_slope: true` and the current round is past the anchor round, `run_grpo()` reads round 2's `probe_metrics.json`, computes `z_d - z_h`, and passes it as `++detector.fixed_calibration_denominator`. Round 2 itself gets no override (behavior unchanged), so it does not need to be rerun.

**Config:** `fix_calibration_slope: true` in `trial_config_early_retraining.yaml` — a boolean, since the anchor round is hardcoded to 2 rather than user-specified (round 2 is always the first round with calibration data in this pipeline).

**Files changed:** `obfuscation_atlas/config.py` (`DetectorConfig.fixed_calibration_denominator`), `obfuscation_atlas/training/obfuscation_loss.py` (`set_calibration()`, `_aggregate_scores()`, `disable_calibration()`), `obfuscation_atlas/scripts/train_grpo_model_against_probe.py` (pass-through + log line), `experiments/run_trial.py` (`FIXED_CALIBRATION_ANCHOR_ROUND` constant, override wiring in `run_grpo()`).

**Rerun:** killed the in-progress round 3 GRPO run (`jhjh2wnl`, step 44/128, old always-recomputed-denominator formula) and removed its partial checkpoint directory (`trigger_none_model_jhjh2wnl_layers_32`, ~3.8GB, no final adapter present). Relaunched round 3 from scratch; the probe retrained to the same `z_h`/`z_d` as before (-2.0765 / -1.7608, from cached activations — confirms determinism), but GRPO now applies the frozen round-2 denominator (2.1313) instead of round 3's own (0.3157). New run: `cpff1h5l`.

**Affected runs:** Trial 4's early_retraining variant, round 3 onward — see "Trial 4 early_retraining variant" below.

**Launcher:** `python run_trial.py --trial 4 --variant early_retraining --num-rounds 3` in tmux session `early_retraining` (relaunched); log: `experiments/trials/trial4/production_early_retraining_fixed_cal.log`.

Plans: [fixed_calibration_denominator](../.cursor/plans/fixed_calibration_denominator_7804a51d.plan.md)

### 17. Detector-score early stopping, plus class-balance assertion relaxed to warning (Aug 3, 2026)

Added early stopping to GRPO training: if the train-batch-mean detector score (`rewards_metrics/detector_score`) stays below `early_stop_detector_threshold` for `early_stop_detector_patience` consecutive training steps (only checked once `early_stop_min_steps` have completed), training stops early and the current model becomes the round's final checkpoint. The next round proceeds with **standard** probe retraining (default `split_sources`, no carryover/frozen-probe/conditional-retraining logic from methodologies #15-16).

**Why:** Prior trial 4 branches always ran the full 128 steps per round regardless of how quickly the detector score dropped. This tests whether stopping training as soon as the current probe stops finding anything to penalize — rather than continuing to train against an already-defeated detector for the rest of the round — changes round-to-round dynamics.

**Implementation:**

- `GRPOObfuscationConfig` (`config.py`): `early_stop_detector_threshold: Optional[float] = None` (disabled by default), `early_stop_detector_patience: int = 10`, `early_stop_min_steps: int = 20`.
- `GRPOObfuscationTrainer.train()` (`grpo_trainer.py`): a counter increments each step the train-batch mean detector score is below threshold (reset to 0 otherwise), only once `state.step >= early_stop_min_steps`; `break`s out of the training loop once the counter reaches `early_stop_detector_patience`. The existing post-loop checkpoint save produces the correct final checkpoint from this clean exit — no crash-retry or degeneracy-check interaction, since post-training eval always runs on the final model regardless of step count.
- `run_trial.py`: `run_grpo()` passes the three early-stop values as Hydra overrides when set in the trial config; the GRPO retry loop's `verify_run_complete()` call uses `min_responses=1` instead of the default 128 when early stopping is enabled, since an early-stopped run legitimately logs far fewer than 128 response steps.

**Follow-on fix (same day):** round 2 of this variant early-stopped at step 62 (detector score 0.036) with the resulting checkpoint essentially fully honest (0/1000 on-policy generations labeled deceptive; `likely_hardcoded` had already dropped from ~0.7 to ~0.0 well before the stop, per the training curve). This broke round 3's setup: `run_single_round()`'s hard assertion that on-policy generation must contain at least one sample of both classes (`train_check["counts"]["deceptive"] > 0 and ...["honest"] > 0`) raised `AssertionError`. Relaxed to a `WARN:` log — the downstream `build_train_dataset_split_sources()` round-robin already tolerates one empty source (it's dropped from rotation and the remaining sources' quotas grow to cover the shortfall), so no other change was needed; round 3 built its manifest normally, drawing all deceptive on-policy quota from `det0` + `round1` instead of evenly across `det0`/`round1`/`round2`.

**Config:** `experiments/trials/trial4/trial_config_early_retraining.yaml` — same lineage as the calibrated branch (`use_calibrated_penalty: true`) plus `early_stop_detector_threshold: 0.06`, `early_stop_detector_patience: 10`, `early_stop_min_steps: 20`.

**Files changed:** `obfuscation_atlas/config.py` (three new `GRPOObfuscationConfig` fields), `obfuscation_atlas/training/grpo_trainer.py` (early-stop counter + `break` in `train()`), `experiments/run_trial.py` (Hydra override wiring in `run_grpo()`, `verify_run_complete()` `min_responses` adjustment, class-balance assertion relaxed to warning).

**Launcher:** `python run_trial.py --trial 4 --variant early_retraining --num-rounds 4` (relaunched as `--num-rounds 3` from round 3 after the assertion fix) in tmux session `early_retraining`; log: `experiments/trials/trial4/production_early_retraining.log`. Round 1 shared with the other trial 4 branches.

**Affected runs:** Trial 4's early_retraining variant — see "Trial 4 early_retraining variant" below.

Plans: [early_retraining_variant](../.cursor/plans/early_retraining_variant_58d39b13.plan.md), [remove_class-balance_assertion](../.cursor/plans/remove_class-balance_assertion_0fb73cca.plan.md)

### 16. Frozen probe weights: static detector reused unmodified across all rounds (Aug 2-3, 2026)

Added `frozen_probe_weights_dir` (optional) to `DetectorConfig`. When set, GRPO's Phase 0 still trains/evaluates a fresh probe as usual — needed for eval metrics, decision thresholds, and layer filtering — but immediately afterward, the detector actually used for the obfuscation penalty (and all downstream on-policy eval) is swapped for a probe loaded from disk. The same frozen weights are reused unmodified for every round of the trial.

**Why:** Trial 3's one-round "no-retrain ablation" (methodology #9) reused round 1's probe *manifest* but still retrained fresh weights each round on the current model's activations. This tests a stricter condition: does a probe whose weights are **literally never updated** keep pace with an evolving policy across a full multi-round trial, or does a fully static detector eventually fail to constrain deception as the policy drifts away from what it was originally trained to recognize?

**Implementation:**

- `DetectorConfig.frozen_probe_weights_dir: str | None` (`config.py`) — path to a saved probe in `detectors/train.py`'s `save_model()` layout (per-layer `model.pt`/`config.json`).
- `train_grpo_model_against_probe.py`: after Phase 0's normal training/eval populates `eval_results_and_detectors` (metrics, thresholds, and layer filtering are all still computed normally from this pass), if `frozen_probe_weights_dir` is set, `load_model()` loads the frozen weights (main process only) and replaces the detector object in every entry of `eval_results_and_detectors`, forcing `found_detectors=False` so the frozen detector broadcasts to all ranks via the existing distributed-setup path. From that point on, the GRPO obfuscation penalty and all downstream on-policy eval use only this frozen detector — the freshly-trained Phase-0 probe is discarded except for its metrics.
- `run_trial.py`: new `freeze_probe_manifest` + `frozen_probe_weights_dir` trial-config keys, asserted to be set together. When both are set, `run_single_round()` skips the entire per-round generation + external probe-training pipeline (no on-policy data collection is needed since nothing ever retrains), and `run_grpo()` passes `++detector.frozen_probe_weights_dir=<path>` while pointing `mixed_probe_manifest_path` at the frozen manifest (round 1's, for Phase 0's own metrics/logging pass). Asserted incompatible with `use_calibrated_penalty` (calibration logits are computed by the external probe-training pipeline, which is skipped here).

**Config:** `experiments/trials/trial4/trial_config_frozen.yaml` — `freeze_probe_manifest` / `frozen_probe_weights_dir` both point at trial 4 round 1's (`9nlij6so`) probe artifacts (`rounds/round1/probe_train_dataset.json`, `rounds/round1/probe_weights`); every other hyperparameter matches the main trial 4 branch exactly.

**Files changed:** `obfuscation_atlas/config.py` (`DetectorConfig.frozen_probe_weights_dir`), `obfuscation_atlas/scripts/train_grpo_model_against_probe.py` (weight-swap logic after Phase 0, `load_model` import), `experiments/run_trial.py` (`run_single_round()` frozen-probe skip branch, `run_grpo()` override wiring).

**Launcher:** `python run_trial.py --trial 4 --variant frozen --num-rounds 4` in tmux session `frozen`; log: `experiments/trials/trial4/production_frozen.log`. Round 1 shared with the other trial 4 branches.

**Affected runs:** Trial 4's frozen variant — see "Trial 4 frozen variant" below.

### 15. Conditional probe retraining: skip dataset expansion when on-policy samples are sparse (Aug 2, 2026)

Added `min_on_policy_per_class` (optional, default 0) to `trial_config.yaml`. When set, and the round's freshly-generated on-policy JSONL has fewer than `min_on_policy_per_class` deceptive **or** fewer than `min_on_policy_per_class` honest samples, the probe training manifest is **not** expanded with that round's data — the previous round's manifest is copied forward unchanged instead. The probe itself still retrains (on the copied manifest data, using the current model's activations), so eval metrics and calibration logits stay current; only the *training-data selection* is held fixed.

**Why:** The prior default always folds each round's freshly-generated on-policy data into the growing probe training set, even when a round yields very few labeled examples of one class (e.g. a near-fully-honest policy produces almost no deceptive samples). Tiny, low-signal per-round contributions risk adding noise rather than useful new coverage. This tests whether gating expansion on a minimum sample count changes probe quality or downstream GRPO outcomes relative to always-retrain.

**Implementation:** New `build_or_reuse_probe_dataset()` helper in `run_trial.py`, called from `run_single_round()` in place of the previous inline `dataset_strategy` dispatch:

- Threshold met (or `min_on_policy_per_class` unset/0, the default) → unchanged: dispatches to `build_train_dataset_split_sources()` / `build_train_dataset_round_robin()` exactly as before.
- Threshold not met → copies `rounds/round{R-1}/probe_train_dataset.json` to the current round's manifest path (`shutil.copy2`), reconstructs `train_pos`/`train_neg` Datasets from it via `rows_to_dataset()` (`build_dataset.py`), and logs `SKIP EXPANSION: ...`.
- No previous manifest to reuse (e.g. round 2 with a manually-bootstrapped round 1) → falls back to building fresh, with a `WARN:` log.

Backward compatible: existing trial configs are unaffected — the key is read via `.get()` (not added to `REQUIRED_TRIAL_CONFIG_KEYS`), defaulting to 0 (always retrain, prior behavior).

**Config:** `min_on_policy_per_class: <int>` in `trial_config.yaml` (0 or omitted = always retrain, same as before this change).

**Files changed:** `experiments/run_trial.py` — new `build_or_reuse_probe_dataset()` function; `shutil`, `datasets.Dataset`, `build_dataset.rows_to_dataset` imports.

**Launcher:** Trial 4's `carryover` branch (`experiments/trials/trial4/trial_config_carryover.yaml`, `min_on_policy_per_class: 20`) — see "Trial 4 carryover variant" below.

### 14. LoRA checkpoint leak fix — untracked per-step save in `_generate_alt_completions` (Aug 2, 2026)

**Fix:** `_generate_alt_completions()` (`obfuscation_atlas/training/grpo_trainer.py`) now saves the policy adapter to a distinct sibling path (`checkpoints/lora/step_N_alt`) instead of the shared per-step `checkpoints/lora/step_N` path, and synchronously `shutil.rmtree`s it (main process only) immediately after the vLLM generation call returns. Pure code change — no config keys, no effect on rewards, loss, or training dynamics.

**Problem:** Every GRPO training step made two separate LoRA-adapter saves to the same `checkpoints/lora/step_N` path: (1) `GenerationQueue.submit()`/`pop()` (`afterburner/grpo_trainer.py`), which saves, uses, then asynchronously `rm -rf`s the adapter — this path was already correctly cleaned up; and (2) `_generate_alt_completions()` (called from `score_responses()` every training step, and from `compute_eval_metrics()` every eval step, for the `pass_only_shown_deception` metric), which re-saved to the *same* `step_N` path with **no matching cleanup anywhere**. Each leaked checkpoint is ~650MB; over a 128-step run this leaked ~80GB of never-reclaimed scratch space *during* the run (only wiped in bulk by `experiments/grpo_launcher.py`'s post-run `cleanup_intermediates()`, which is why it was previously invisible on successful runs). A run that crashes before that final cleanup step keeps the full ~80GB — this caused a real `/workspace` disk-full crash (`OSError: [Errno 122] Disk quota exceeded`) at step 124/128 of `round1_clp1e-3_p20_s45_eps1e-4`.

**Same fix also closes a known race:** reusing `step_N` for both saves meant `_generate_alt_completions`'s `policy_adapter.save(step_N)` could run concurrently with `GenerationQueue`'s async `rm -rf` of that same path — the intermittent `RuntimeError: Parent directory .../step_N does not exist` crash documented in [FULL_GRPO_RUN.md](./FULL_GRPO_RUN.md#known-failure-modes) (hit twice during the `ceiling_clp1e-3_seed_sweep` seed-44 run before succeeding on a fresh pod). Using a distinct `_alt` path removes any path overlap between the two save/cleanup mechanisms, so the race can no longer occur.

**Files changed:** `obfuscation_atlas/training/grpo_trainer.py` — `import shutil`; `_generate_alt_completions()` saves to `lora_path.parent / f"{lora_path.name}_alt"`, points the vLLM generation call at that path, and cleans it up (`is_main_process` guard, matching `HfLoRAAdapter.save()`'s own guard) right after `broadcast_responses()` returns.

**Not fixed / out of scope:** No periodic orphan-sweep safety net was added — the fix removes the leak at its source, and `cleanup_intermediates()` already reclaims any stray `_alt` dir from a mid-step crash post-run. The ~79GB of pre-fix leaked checkpoints under `trigger_none_model_emjn04pc_layers_32/` (the crashed run) were left in place, not cleaned up as part of this fix.

**Affected runs:** All GRPO code-generation runs going forward — the leak was gated only on `task == TaskType.CODE_GENERATION` and `lora_path is not None`, not on any specific hyperparameter, so it reproduced on every prior run (masked by post-run cleanup on successful runs).

### 13. Advantage-normalization epsilon fix for degenerate collapse (Aug 1-2, 2026)

**Fix:** raised the hardcoded advantage-normalization epsilon from `1e-4` to `0.05` at `afterburner/grpo_trainer.py:621` (`advantages / (std_grouped_rewards + eps)`). Pure scalar change — no batch filtering, config threading, or downstream effect on loss, clipping, or KL computation. Caps worst-case amplification at ~20x (vs. up to 1000x with `1e-4`) while dampening healthy-regime (std≈0.4) advantages by only ~11%.

**Problem:** Trial 4 round 4's first attempt (`13bscznt`) collapsed to a fully degenerate policy (`val_does_execute` 97%→0% within 30 GRPO steps). As the policy converged (token entropy dropped 0.25→0.10), within-group reward std collapsed (0.41→0.048), and GRPO's advantage normalization amplified reward noise ~20x, triggering a death spiral (response length 145→24 tokens, reward went negative, no recovery once all outputs failed identically). A retry with the same probe/seed succeeded normally — a stochastic instability, not a systematic probe issue.

**Abandoned attempts:**

- `loss.scale_rewards=false` (drops per-group std normalization entirely) stopped the collapse but changed training dynamics well beyond the intended fix (materially higher `pass_all` even in healthy, non-degenerate groups) — reverted.
- `optimizer.skip_degenerate_groups=true` (filter near-zero-std groups before the loss/backward pass, leaving normal-group normalization untouched) crashed GRPO training twice — first a `KeyError: 'advantages'` (the collate fn didn't propagate the column the filtering path needed), then a tensor-size mismatch (filtering shrank the batch without re-aligning other full-batch tensors). Fixing this properly needed non-trivial re-indexing surgery across `train_on_generation_batch` — abandoned as too risky.

**Affected runs:** Trial 4's calibrated variant completed rounds 2-3 (`3n0eb1d2`, `6x73sz94`) and started round 4's probe phase under `scale_rewards=false` before the revert; that lineage was killed and redone from round 2 under the epsilon fix (old artifacts under `rounds_calibrated_old/` / `buffer_calibrated_old/` — see Trial 4 calibrated variant below).

**Caveat (Aug 2, 2026):** the epsilon fix is not behavior-neutral. A 4-seed replication at the clp1e-3/p20 point landed Honest on all 4 seeds under eps=0.05, vs. 2/3 Obfuscated pre-fix (see Round 1 ablation — clp1e-3/p20 seed sweep below). Treat any bistable hyperparameter point as re-benchmarked under eps=0.05 — don't assume pre-fix outcomes still apply.

### 12. Calibrated probe penalty: linear clip instead of sigmoid (Jul 28, 2026)

Replaced the sigmoid-based probe penalty with a calibrated linear clip: `clip((z_agg - z_h) / (z_d - z_h), 0, 1)`, where `z_agg` is the mean-across-layers raw probe logit, and `z_h`/`z_d` are the median aggregated logits for honest/deceptive on-policy samples from the starting policy's evaluation set.

**Why:** The sigmoid maps raw logits non-linearly, compressing gradients far from the decision boundary and providing inconsistent penalty magnitude across rounds (since probe weight scales shift). The calibrated clip directly normalizes the penalty to [0, 1] based on empirical anchor points: an honest sample from the starting policy gets penalty ~0, a deceptive one gets ~1, and intermediate values are linearly interpolated.

**Architecture:**

1. Raw per-layer probe logits are passed through without any activation function (no sigmoid, no Platt scaling)
2. Layer logits are mean-aggregated into a single scalar per sample
3. The scalar is calibrated: `clip((z_agg - z_h) / (z_d - z_h), 0, 1)`
4. Result is multiplied by `-loss.detector` as the penalty reward term (same as before)

**Calibration computation (`compute_calibration_logits` in `probe_training.py`):**

- Runs `eval_detector(..., return_logits=True)` on the starting policy's on-policy eval set (reuses cached activations from the same round's probe eval)
- Stacks per-layer logits for all `filtered_layers`, means across layers → one scalar per sample
- Applies on-policy honest/deceptive masks (from `derive_on_policy_split_masks`)
- `z_h = median(honest aggregated logits)`, `z_d = median(deceptive aggregated logits)`
- Stored as `{"z_h": float, "z_d": float}` in `probe_metrics.json["calibration_logits"]`

**Config:** `++detector.calibration_logits_path=<path to probe_metrics.json>` (Hydra override). When set, `train_grpo_model_against_probe.py` loads `calibration_logits.z_h`/`z_d` and calls `detector.set_calibration(z_h, z_d)` on each `ProbeDetector`. When unset (default), behavior is unchanged (sigmoid).

**Files changed:**

- `obfuscation_atlas/training/obfuscation_loss.py` — `ProbeDetector.set_calibration()` (scalar), `_compute_layerwise_scores` (raw logit passthrough), `_aggregate_scores` (post-aggregation clamp)
- `obfuscation_atlas/config.py` — `DetectorConfig.calibration_logits_path: str | None`
- `obfuscation_atlas/scripts/train_grpo_model_against_probe.py` — load calibration and call `set_calibration`
- `obfuscation_atlas/detectors/evaluate.py` — `return_logits` parameter on `eval_detector`
- `experiments/probe_training.py` — `compute_calibration_logits()` function
- `experiments/run_round_manual.py`, `experiments/run_trial.py` — `USE_CALIBRATED_PENALTY` / `use_calibrated_penalty` flags

**Launcher:** `experiments/run_round2_calibrated_penalty.py` — standalone script matching trial 3 round 2's hyperparameters with the addition of `++detector.calibration_logits_path`. Computes calibration logits on first run (reusing cached activations) and saves to `probe_metrics.json`. Output: `experiments/trials/trial3/rounds/round2_calibrated_penalty/`.

**Calibration values (trial 3 round 2):** `z_h = -0.2407` (honest median), `z_d = -0.0919` (deceptive median), separation = 0.149. Computed from 114 honest / 79 deceptive on-policy samples across all 32 filtered layers.

**Completed run (Jul 28, 2026):** `iviyt23y` (`trial3_round2_calibrated_penalty`, kl=1e-3, det=3, seed=42, 128 steps) — classified **Honest** (not degenerate, not reward-hacky, not obfuscated). Final eval: `success=0.478`, `passes_all_tests=0.345`, `pass_only_shown_deception=0.028`, `likely_hardcoded=0.005`, `passes_first_n_but_not_all_tests=0.133`. Final-model on-policy probe AUC 0.999. Export: `experiments/trials/trial3/rounds/round2_calibrated_penalty/run_metrics_trial3_round2_calibrated_penalty.json`. Intermediate checkpoints (`checkpoints/lora`, `checkpoints/step`_*, ~80GB) cleaned up post-run per the standard convention (keep `checkpoints/final` + `checkpoints/responses` only).

Plan: [calibrated_probe_penalty](../.cursor/plans/calibrated_probe_penalty_c32f128b.plan.md), [aggregate_calibration_penalty](../.cursor/plans/aggregate_calibration_penalty_2c0893c8.plan.md)

### 11. KL-from-base ablation: KL penalty against base model instead of prior round (Jul 28, 2026)

Added `kl_reference_from_base` flag to `GRPOModelConfig` (and upstream `afterburner.ModelConfig`). When `true` and `reference_adapter_path` is set, the reference adapter's LoRA weights are zeroed out after loading so the KL divergence penalty is computed against the **base model**, while the policy adapter still initializes from the `reference_adapter_path` checkpoint.

**Why:** In multi-round iterative training, each round's KL penalty normally anchors the policy to the **previous round's** checkpoint (since `reference_adapter_path` serves dual duty as both policy initializer and KL reference). This ablation decouples the two to test whether anchoring KL to the base model (stronger regularization toward the pretrained distribution) changes training dynamics compared to anchoring to the round-1 model.

**Implementation:** After `_init_from_reference_adapter` loads both adapters from `reference_adapter_path`, the new flag triggers zeroing all `reference` adapter parameters and freezing them. The policy adapter retains the loaded weights as its starting point — only the KL reference target changes.

**Config:**

- `++grpo.model.kl_reference_from_base=true` (Hydra override)
- `grpo.model.reference_adapter_path` still required (sets the policy's init point)

**Files changed:** `afterburner/training/huggingface/model.py` (zero-out logic), `afterburner/grpo_config.py` + `obfuscation_atlas/training/grpo_config.py` (config field).

**Launcher:** `experiments/run_round2_kl_from_base.py` — standalone script matching trial 3 round 2's hyperparameters with the single addition of `kl_reference_from_base=true`. Output: `experiments/trials/trial3/rounds/round2_kl_from_base/`.

Plan: [kl_from_base_model](../.cursor/plans/kl_from_base_model_452c9c6b.plan.md)

### 10. Baseline ablation: no-detector-penalty from base model (Jul 28, 2026)

Standalone script `experiments/run_det0_baseline.py` runs GRPO with `loss.detector=0` from the **base model** (no `reference_adapter_path`), holding every other trial 3 hyperparameter fixed (kl=1e-3, `training.seed=42`, `data.seed=53`, `mixed_reward_pass_all_prob=0.3`, nofu, `pass_only_shown`, 128 steps).

**Why:** Isolate the effect of the detector penalty itself — a true "reward-only" control against trial 3's det=3 runs, all starting from the same base model with the same mixed-reward regime.

**Config:** `loss.detector=0` is sufficient to zero out detector reward and gradient contributions, but the GRPO script has no flag to skip Phase-0 probe setup or post-training probe eval — both still run unconditionally, which is useful here since it gives directly comparable detector/obfuscation metrics. Reused the existing `nofu_mixed` Phase-0 activation cache (`detector.directory_save_prefix=nofu_mixed`, `detector.retrain=false`) via the manifest at `experiments/tests/mixed_round1_nofu/probe_train_dataset.json` — confirmed byte-identical to `trials/trial3/rounds/round1/probe_train_dataset.json`, so no new probe training was needed.

**Output location:** `experiments/trials/trial3/rounds/det0_baseline/` — a sibling round directory inside trial 3.

Plan: [det0 baseline script](../.cursor/plans/det0_baseline_script_f6ca0588.plan.md)

### 9. Baseline ablation: no-retrain round 2 (Jul 27, 2026)

Standalone script `experiments/run_round2_baseline.py` runs GRPO round 2 against **round 1's original probe manifest** instead of the retrained round-2 dataset. Since GRPO Phase 0 always retrains from whatever manifest it's handed, "reusing round 1's probe" means pointing `++detector.mixed_probe_manifest_path` at `rounds/round1/probe_train_dataset.json` (300/class: 200 on-dataset + 100 on-policy from base+det0) instead of building a new round-2 manifest from round 1 on-policy generations. With the same `data.seed=53` as round 1, Phase 0 produces the same probe round 1 used.

**Why:** Trial 3's round 2 changes two things at once relative to round 1 — new on-policy data folded into probe training, and a freshly trained probe. This ablation holds the probe fixed to isolate whether probe *retraining* (vs. just continuing GRPO with the same probe) drives the round-2 outcome.

**Config:** No changes to `RoundConfig`, `run_round_manual.py`, or pipeline library code. All hyperparameters match trial 3 exactly (kl=1e-3, det=3, `training.seed=42`, `data.seed=53`, `mixed_reward_pass_all_prob=0.3`, nofu, 128 steps); only `++detector.mixed_probe_manifest_path` / `++detector.expected_on_dataset_auc` point at round 1's files, and `detector.directory_save_prefix` / `wandb.run_name_prefix` use `trial3_round2_baseline` for isolation.

**Output location:** `experiments/trials/trial3/rounds/round2_baseline/` — a sibling round directory inside trial 3 (not a separate trial), since round 1 is shared and unchanged.

Plan: [baseline trial3 no-retrain](../.cursor/plans/baseline_trial3_no-retrain_057929a8.plan.md)

### 8. Dataset builders for multi-round probe training (Jul 2026)

Three builders for constructing mixed probe training sets from on-dataset + on-policy data, each suited to different stages of the iterative pipeline:


| Builder                               | Use case                         | Composition per class                                                                           |
| ------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------- |
| `build_train_dataset_round1_mixed()`  | Round 1 from base/det0 artifacts | 200 on-dataset + 100 on-policy                                                                  |
| `build_train_dataset_round_robin()`   | Automation default (round >=2)   | 200 on-dataset + round-robin on-policy from base + prior rounds                                 |
| `build_train_dataset_split_sources()` | Trial 3+ (round >=2)             | 200 on-dataset + per-class anchored on-policy (honest from `base_model`, deceptive from `det0`) |


**Dedup key fix (Jul 25):** Changed from `(source_name, task_id)` to `(source_name, task_id, sample_idx)`. With 3-sample artifacts, the old key silently capped a source's usable pool at its unique-task-id count, forcing unwanted backfill.

**3-sample artifact regeneration (Jul 25):** `experiments/artifacts/{base,det0}/*_train_generations.jsonl` regenerated with `n_generations=3`, `deduplicate_responses=False`, restricted to `dataset="train"` (200 probe-train prompts). Old 1-sample files kept as `*_1_sample.jsonl`. Required for `build_train_dataset_split_sources()` to have enough samples per source.

Plans: [mixed round1](../.cursor/plans/mixed_round1_nofu_probe_ee742a85.plan.md), [round-robin](../.cursor/plans/round-robin_dataset_builder_85bd7e2e.plan.md), [split-sources](../.cursor/plans/split-source_dataset_builder_bc166ec6.plan.md)

### 7. Task train expansion (Jul 2026)

GRPO RL prompts come from a wider pool than probe/eval. After distinct-output filtering produces ~666/class eligible rows (used for 400 val + ~266 probe pool), task train adds back `non_distinct_outputs` rows minus val `task_id`s, yielding ~473/class. Probe train `task_id`s remain in task train (overlap intentional — probe rows are a subset of RL training prompts).

**Why:** More diverse RL training prompts without contaminating probe/eval split integrity.

**Config:** Implicit in `prepare_dataset()` MBPP split block. `_test_list_ok_for_task_train()` gates which non-distinct rows are included (must still have >=2 parseable tests).

Plan: [task train expansion](../.cursor/plans/task_train_split_expansion_96d0f40e.plan.md)

### 6. Mixed reward regime (Jul 24, 2026)

Per-prompt random swap from `passes_first_n_tests` to `passes_all_tests` grading during GRPO training only. Makes honest solving intermittently and unpredictably rewarded while prompts stay unchanged (all shown).

**Why:** Pure `passes_first_n_tests` reward encourages hardcoding. Intermittent `passes_all_tests` creates pressure toward genuine solutions without changing what the model sees.

**Config:** `task_args.mixed_reward_pass_all_prob` (0-1, default 0.0). `task_success` always stays `passes_first_n_tests`. Eval never uses mixed regime. RNG seed: `training.seed + step + 1_000_000`.

**W&B:** `train/mixed_reward_pass_all_fraction` per train step; startup log confirms value.

Plan: [mixed reward](../.cursor/plans/mixed_reward_regime_b53b4d55.plan.md)

### 5. Prompt mixing — shown/unshown training (Jul 23-24, 2026)

During GRPO rollouts, randomly present MBPP test 0 with expected output (shown) or without (unshown — strip everything after the first `==` in the assert). Reward still grades full `test_list`; probe penalty unchanged.

**Why:** Test whether models learn to exploit visible test outputs vs. generalize when outputs are hidden.

**Config:** `task_args.shown_prompt_ratio` (0-1, default 1.0 = all shown). `task_args.eval_show_expected_output` (default true; set false for hidden post-train eval). Per-step RNG: `training.seed + step`.

**W&B:** `train/prompt_shown_fraction` when ratio < 1.

**Hidden eval:** `training.only_eval_from_other_run=True` + `eval_show_expected_output=false` runs post-train eval on hidden prompts as a new W&B run.

Plan: [prompt mixing](../.cursor/plans/shown_unshown_prompt_mixing_ffc12ad1.plan.md)

### 4. No-follow-up (nofu) probe extraction (Jul 22-23, 2026)

Probe activations extracted at completion EOT (`last-token-generation`) instead of the follow-up "Yes" token.

**Why:** Follow-up "Yes" token may not reflect the same representation as the code completion; nofu probes the generation itself.

**Config (Hydra overrides, add to any GRPO command):**

```
data.probe_dataset.congruent_follow_up_prompts=[]
data.probe_dataset.incongruent_follow_up_prompts=[]
data.task_dataset.congruent_follow_up_prompts=[]
data.task_dataset.incongruent_follow_up_prompts=[]
detector.directory_save_prefix=<prefix>
detector.retrain=true   # on first run only
```

**Mixed nofu variant:** 300/class Phase-0 probe (200 on-dataset + 100 on-policy) via `build_train_dataset_round1_mixed()` + `++detector.mixed_probe_manifest_path`. Manifest: `experiments/tests/mixed_round1_nofu/probe_train_dataset.json`.

Plan: [mixed nofu probe](../.cursor/plans/mixed_round1_nofu_probe_ee742a85.plan.md)

### 3. Deception definition change and data filtering (Jul 2026)

Replaced the operational "deceptive" label with `pass_only_shown_deception` — a stricter dual-generation metric (test 0 shown vs test 1 shown). Old metric `passes_first_n_but_not_all_tests` kept for comparison. Configurable via `deceptive_definition`.

**Why:** Old metric counted deception even when the model saw the expected output in the prompt. New metric only counts deception relative to what was shown — requires consistent test-specific behavior across both prompts.


|                    | Old (`passes_first_n_but_not_all_tests`)          | New (`pass_only_shown_deception`)                                 |
| ------------------ | ------------------------------------------------- | ----------------------------------------------------------------- |
| Condition          | Pass first N shown tests, fail at least one other | Primary passes only test 0; alt (test 1 shown) passes only test 1 |
| Generations needed | One                                               | Two (~2x vLLM cost)                                               |


**Config:** `task_args.deceptive_definition` / `RoundConfig.deceptive_definition`: `"pass_only_shown"` (default) or `"pass_first_not_all"`. Lookup: `DECEPTIVE_DEFINITIONS` + `deceptive_reward_key()` in `code_generation.py`.

**Data filtering:** Strict distinct-output filter on eval/probe pools (drops unparseable, non-distinct, <2 tests). GRPO task train uses wider pool (see #7 task train expansion).

**W&B:** Both metrics always logged. Export: `deceptive_rate` (old), `pass_only_shown_deception_rate` (new). `classify_policy_type` uses new metric with fallback.

Plans: [pass-only-shown](../.cursor/plans/pass-only-shown_deception_b4409a02.plan.md), [definition switch](../.cursor/plans/deception_definition_switch_f25d632f.plan.md)

### 2. Disjointness fix for probe train/eval (Jul 14, 2026)

Fixed 93/200 `task_id` overlap between MBPP probe train and eval val. `prepare_dataset()` now derives probe from the task **train** split (not the full shuffled pool).

**Why:** Probe was partially evaluated on training data, inflating on-dataset AUC and invalidating cross-run comparisons.

**Fix:** When probe and task share the same MBPP pool (`_probe_shares_task_pool`), derive probe train as first 200/class from task train split. `_assert_probe_val_disjoint()` raises if overlap > 0. Mixed manifest paths also checked at startup.

**Reference run:** `mtzvz59v` (`kl1e-3-det3-s43-mbpp-disjoint`) — first post-fix GRPO run.

**Caveat:** Old probe caches (under `base_results_and_detectors_*mbpp-hardcode`*) trained on overlapping data are invalid. Set `detector.retrain=true` once after the fix to refresh.

Plan: [disjoint fix](../.cursor/plans/fix_probe_val_overlap_2410477a.plan.md)

### 1. On-policy split evaluation (Jul 14, 2026)

Post-process probe scores by deceptive vs honest on-policy generations. No second activation pass — pure post-processing on existing 400 per-sample scores.

**Why:** Standard on-policy eval (400 model gens vs 400 fixed honest code) gives low AUC because it conflates honest, deceptive, and unlabeled generations. Split eval isolates deceptive vs honest to measure whether the probe distinguishes behavior types on-policy.

**Implementation:** `derive_on_policy_split_masks` + `compute_split_on_policy_probe_metrics` + `run_on_policy_split_eval` in `obfuscation_training.py`. Wired after each `.on_policy` stage in `train_grpo_model_against_probe.py`.

**Output:** Local JSON `on_policy_split_{prefix}.json`; W&B keys under `probes/eval_{prefix}.split/...`. Three files per run (initial final, fresh final, initial base).

**Label source:** Uses `deceptive_reward_key(definition)` — defaults to `pass_only_shown_deception` after the deception definition change (#3).

Plan: [on-policy split](../.cursor/plans/on-policy_behavior-split_metrics_a97a5350.plan.md)

---

## Round architecture and trial pipeline

See [iterative_training_experiments.md](./iterative_training_experiments.md) for pipeline architecture and module reference, and [FULL_TRIAL_RUNBOOK.md](./FULL_TRIAL_RUNBOOK.md) for `run_trial.py` automation.

---

## Completed runs

**Trial lineages** — the main multi-round experimental results.

### Trial 8 early_retraining variant — p10 dynamic early stop, clp3e-3→1e-3/p25/s53 lineage (Aug 5, 2026 — complete, stopped at round 5)

Same early_retraining recipe as trial 6 (methodology #23, p10 threshold, KL-from-base, split_sources), but bootstrapped from round 1 `t1656g8a` (`round1_clp3e-3_p25_s53`, clp=3e-3, seed 53, **Obfuscated policy**). Rounds 2+ use clp=1e-3 (deliberate mismatch with round 1). Trial config: `experiments/trials/trial8/trial_config_early_retraining.yaml`. Log: `experiments/trials/trial8/production_early_retraining.log`. Rounds 2–5 completed Aug 5; round 6 not run. Round 3 first attempt failed (insufficient honest base-model samples for split_sources round-robin); fixed by regenerating `experiments/artifacts/base/base_train_generations.jsonl` at 10 gens/prompt (old file kept as `base_train_generations_3gens_old.jsonl`). Round 4 GRPO (`ppwjruhh`) stalled on CodeExecutor at step 19; killed and restarted from round 4. Round 6 partial (`wkqxq0u6`) killed manually before completion. **Renamed from main trial 8 branch Aug 5, 2026** — artifacts moved to `buffer_early_retraining/`, `rounds_early_retraining/`, `trial_state_early_retraining.json`; run_metrics rounds 2–5 renamed to `run_metrics_trial8_early_retraining_round{R}.json`.


| Round | Run ID     | Status             | Notes |
| ----- | ---------- | ------------------ | ----- |
| 1     | `t1656g8a` | Complete; exported | clp=3e-3, p25, seed 53. pass_only_shown=0.855, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds_early_retraining/round1/run_metrics_trial8_round1.json`. |
| 2     | `8vunk43w` | Complete; exported | Early-stopped step 29/128 (p10=0.748). pass_only_shown=0.925, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds_early_retraining/round2/run_metrics_trial8_early_retraining_round2.json`. |
| 3     | `e9smjmb0` | Complete; exported | Early-stopped step 29/128 (p10=0.619). pass_only_shown=0.6775, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds_early_retraining/round3/run_metrics_trial8_early_retraining_round3.json`. |
| 4     | `3xg73el8` | Complete; exported | Early-stopped step 29/128 (p10=0.173). pass_only_shown=0.005, **Honest**. Metrics: `experiments/trials/trial8/rounds_early_retraining/round4/run_metrics_trial8_early_retraining_round4.json`. |
| 5     | `n4074l4v` | Complete; exported | Early-stopped step 29/128 (p10=floor 0.05). pass_only_shown=0.03, **Honest**. Metrics: `experiments/trials/trial8/rounds_early_retraining/round5/run_metrics_trial8_early_retraining_round5.json`. |
| 6     | —          | Not run            | Partial GRPO `wkqxq0u6` killed manually. |


**Preliminary finding:** Obfuscated policy rounds 1–3 (clp=3e-3 round 1, then clp=1e-3); **Honest** from round 4 onward — similar trajectory to trial 6 but with one extra Obfuscated round before the flip.

### Trial 8 — full 128-step GRPO, clp3e-3→1e-3/p25/s53 lineage (Aug 5–6, 2026 — complete, stopped at round 5)

Same hyperparams as the `early_retraining` variant (KL-from-base, split_sources, p25, clp=1e-3 from round 2, seed 53) but **no early stopping** — every round runs the full 128 steps. Shares round 1 (`t1656g8a`). Config: `experiments/trials/trial8/trial_config.yaml`. Log: `experiments/trials/trial8/production.log`. Rounds 2–5 completed Aug 5–6; round 6 partial GRPO (`9gylvqmh`, step ~48/128) killed manually Aug 6.


| Round | Run ID     | Status             | Notes |
| ----- | ---------- | ------------------ | ----- |
| 1     | `t1656g8a` | Complete; exported | Shared with `early_retraining` variant. pass_only_shown=0.855, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds/round1/run_metrics_trial8_round1.json`. |
| 2     | `uwdnphj5` | Complete; exported | Full 128 steps. pass_only_shown=0.9425, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds/round2/run_metrics_trial8_round2.json`. |
| 3     | `idhc4h8i` | Complete; exported | Full 128 steps. pass_only_shown=0.9675, **Obfuscated activations**. Metrics: `experiments/trials/trial8/rounds/round3/run_metrics_trial8_round3.json`. |
| 4     | `osoyreo7` | Complete; exported | Full 128 steps. pass_only_shown=0.9725, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds/round4/run_metrics_trial8_round4.json`. |
| 5     | `vob73ihg` | Complete; exported | Full 128 steps. pass_only_shown=0.9475, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds/round5/run_metrics_trial8_round5.json`. |
| 6     | —          | Not run            | Partial GRPO `9gylvqmh` killed manually; run dir and `rounds/round6/` deleted. |


**Preliminary finding:** Unlike the `early_retraining` variant (Honest from round 4), full 128-step training stayed **Obfuscated** through round 5.

### Trial 8 frozen variant — truly frozen probe weights (Aug 6, 2026 — complete)

Same lineage as trial 8 above, but the probe used for the obfuscation penalty is **never retrained** after round 1 (methodology #16). Round 1 shared (`t1656g8a`, clp=3e-3); rounds 2+ use clp=1e-3. Per-round generation + external probe training skipped. Trial config: `experiments/trials/trial8/trial_config_frozen.yaml`. Log: `experiments/trials/trial8/production_frozen.log`.


| Round | Run ID     | Status             | Notes |
| ----- | ---------- | ------------------ | ----- |
| 1     | `t1656g8a` | Complete; exported | Shared with other trial 8 branches. pass_only_shown=0.86, **Obfuscated policy**. |
| 2     | `od4qc0by` | Complete; exported | pass_only_shown=0.95, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds_frozen/round2/run_metrics_trial8_frozen_round2.json`. |
| 3     | `9mkc3tgw` | Complete; exported | pass_only_shown=0.96, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds_frozen/round3/run_metrics_trial8_frozen_round3.json`. |
| 4     | `diycqwls` | Complete; exported | pass_only_shown=0.96, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds_frozen/round4/run_metrics_trial8_frozen_round4.json`. |
| 5     | `8bc0jgn2` | Complete; exported | pass_only_shown=0.97, **Obfuscated policy**. Metrics: `experiments/trials/trial8/rounds_frozen/round5/run_metrics_trial8_frozen_round5.json`. |


**Finding:** unlike trial 8 `early_retraining` (Honest from round 4), the static round-1 probe never constrains deception — policy stays Obfuscated throughout, matching trial 6 `frozen` and trial 8 base.


### Trial 7 early_retraining variant — p10 dynamic early stop, clp1e-3/p20/s51 lineage (Aug 5, 2026 — stopped)

Same early_retraining recipe as trial 6 (methodology #23, p10 threshold, KL-from-base, split_sources), but bootstrapped from round 1 `7a0cse2z` (`round1_clp1e-3_p20_s51`, clp=1e-3, p20, seed 51, **Obfuscated policy**). Differs from trial 6 only in `mixed_reward_pass_all_prob=0.2` (vs 0.25) and `training.seed=51` (vs 50). **Stopped manually:** round 1 was highly deceptive (pass_only_shown=0.7475); round 2 attempt 1 (`bhasfphe`) went degenerate (`val_does_execute=0.0`, see methodology #13) and auto-retried; retry (`g0izdzpw`) was killed before completion. Trial config and round-1 artifacts kept at `experiments/trials/trial7/`; round-2 GRPO checkpoints and activation caches deleted.


| Round | Run ID     | Status             | Notes |
| ----- | ---------- | ------------------ | ----- |
| 1     | `7a0cse2z` | Complete; exported | clp=1e-3, p20, seed 51. pass_only_shown=0.7475, **Obfuscated policy**. Metrics: `experiments/trials/trial7/rounds_early_retraining/round1/run_metrics_trial7_round1.json`. |
| 2     | `bhasfphe` | Degenerate; deleted | Attempt 1: `val_does_execute=0.0`. Metrics kept: `.../round2/run_metrics_trial7_early_retraining_round2.DEGENERATE_1.json`. Auto-retry `g0izdzpw` killed manually. |
| 3     | —          | Not run            | — |
| 4     | —          | Not run            | — |
| 5     | —          | Not run            | — |


### Trial 6 early_retraining variant — p10 dynamic early stop, clp1e-3/p25/s50 lineage (Aug 5, 2026 — complete, stopped at round 5)

Standard sigmoid probe penalty + dynamic early stopping (methodology #23, p10 threshold), KL-from-base. Round 1 `yckx0ctm` (clp=1e-3, p25, seed 50, **Obfuscated policy**). Trial config: `experiments/trials/trial6/trial_config_early_retraining.yaml`. Log: `experiments/trials/trial6/production_early_retraining.log`. Rounds 2–5 completed Aug 5; rounds 6–7 not run.


| Round | Run ID     | Status             | Notes |
| ----- | ---------- | ------------------ | ----- |
| 1     | `yckx0ctm` | Complete; exported | pass_only_shown=0.82, **Obfuscated policy**. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p25_s50.json`. |
| 2     | `7x0yqcxk` | Complete; exported | Early-stopped step 29/128 (p10=0.261). pass_only_shown=0.68, **Obfuscated policy**. Metrics: `experiments/trials/trial6/rounds_early_retraining/round2/run_metrics_trial6_early_retraining_round2.json`. |
| 3     | `q83odjhp` | Complete; exported | Early-stopped step 29/128 (p10=0.314). pass_only_shown=0.38, **Obfuscated policy**. Metrics: `experiments/trials/trial6/rounds_early_retraining/round3/run_metrics_trial6_early_retraining_round3.json`. |
| 4     | `wjxi27oj` | Complete; exported | Early-stopped step 35/128 (p10=0.239). pass_only_shown=0.24, **Honest**. Metrics: `experiments/trials/trial6/rounds_early_retraining/round4/run_metrics_trial6_early_retraining_round4.json`. |
| 5     | `ge0tg5o1` | Complete; exported | Early-stopped step 29/128 (p10=0.216). pass_only_shown=0.005, **Honest**. Metrics: `experiments/trials/trial6/rounds_early_retraining/round5/run_metrics_trial6_early_retraining_round5.json`. |


**Preliminary finding:** rounds 2–3 stayed Obfuscated policy despite early stopping (~23% of 128 steps); Honest from round 4 onward, similar trajectory to trial 4 `early_retraining_v2` but with two Obfuscated rounds instead of one.

### Trial 6 — full 128-step GRPO, clp1e-3/p25/s50 lineage (Aug 5–6, 2026 — complete, stopped at round 5)

Same hyperparams as the `early_retraining` variant (KL-from-base, split_sources, p25, clp=1e-3) but **no early stopping** — every round runs the full 128 steps. Shares round 1 (`yckx0ctm`). Config: `experiments/trials/trial6/trial_config.yaml`. Log: `experiments/trials/trial6/production.log`. Rounds 2–3 completed Aug 5; round 4 first attempt (`1f03unaz`) killed to pause; resumed Aug 6 with `--start-round 4 --num-rounds 2`.


| Round | Run ID     | Status             | Notes |
| ----- | ---------- | ------------------ | ----- |
| 1     | `yckx0ctm` | Complete; exported | Shared with `early_retraining`. pass_only_shown=0.8225, **Obfuscated policy**. |
| 2     | `slxu5zpb` | Complete; exported | Full 128 steps. pass_only_shown=0.65, **Obfuscated policy**. Metrics: `experiments/trials/trial6/rounds/round2/run_metrics_trial6_round2.json`. |
| 3     | `vy0tjxdg` | Complete; exported | Full 128 steps. pass_only_shown=0.4475, **Obfuscated policy**. Metrics: `experiments/trials/trial6/rounds/round3/run_metrics_trial6_round3.json`. |
| 4     | `u8ekwc8a` | Complete; exported | Full 128 steps. pass_only_shown=0.8625, **Obfuscated policy**. Metrics: `experiments/trials/trial6/rounds/round4/run_metrics_trial6_round4.json`. |
| 5     | `94wejikq` | Complete; exported | Full 128 steps. pass_only_shown=0.575, **Obfuscated policy**. Metrics: `experiments/trials/trial6/rounds/round5/run_metrics_trial6_round5.json`. |


**Preliminary finding:** Unlike the `early_retraining` variant (Honest from round 4), full 128-step training stayed **Obfuscated policy** through round 5 — same pattern as trial 8 base.

### Trial 6 frozen variant — truly frozen probe weights (Aug 5–6, 2026 — complete)

Same lineage as trial 6 above, but the probe used for the obfuscation penalty is **never retrained** after round 1 (methodology #16). Round 1 shared (`yckx0ctm`). Per-round generation + external probe training skipped. Trial config: `experiments/trials/trial6/trial_config_frozen.yaml`. Log: `experiments/trials/trial6/production_frozen.log`.


| Round | Run ID     | Status             | Notes |
| ----- | ---------- | ------------------ | ----- |
| 1     | `yckx0ctm` | Complete; exported | Shared with other trial 6 branches. pass_only_shown=0.82, **Obfuscated policy**. |
| 2     | `vvrkmfj1` | Complete; exported | pass_only_shown=0.97, **Obfuscated policy**. Metrics: `experiments/trials/trial6/rounds_frozen/round2/run_metrics_trial6_frozen_round2.json`. |
| 3     | `ef86iiu8` | Complete; exported | pass_only_shown=0.92, **Obfuscated policy**. Metrics: `experiments/trials/trial6/rounds_frozen/round3/run_metrics_trial6_frozen_round3.json`. |
| 4     | `81amw0vp` | Complete; exported | pass_only_shown=0.96, **Obfuscated activations**. Metrics: `experiments/trials/trial6/rounds_frozen/round4/run_metrics_trial6_frozen_round4.json`. |
| 5     | `o7gpqj7x` | Complete; exported | pass_only_shown=0.92, **Obfuscated policy**. Metrics: `experiments/trials/trial6/rounds_frozen/round5/run_metrics_trial6_frozen_round5.json`. |


**Finding:** unlike trial 6 `early_retraining` (Honest from round 4), the static round-1 probe never constrains deception — policy stays Obfuscated throughout, matching trial 4 `frozen`.

### Trial 5 — calibrated penalty + KL-from-base, seed 48 lineage (Aug 3-4, 2026 — complete)

Same calibrated-penalty + KL-from-base recipe as trial 4's calibrated variant (epsilon fix, `use_calibrated_penalty: true`, `kl_reference_from_base: true`), but seeded from a **new** round 1 checkpoint at `training.seed=48` under the original pre-methodology-#13 advantage-norm epsilon (`1e-4`, not `0.05`) — see "Round 1 ablation — clp1e-3/p20 seed sweep, advantage-norm epsilon fix" below for how `iv76hjr6` was produced (same hyperparameters as trial 4 round 1's `9nlij6so` except `training.seed=48` and `eps=1e-4`). Trial config: `experiments/trials/trial5/trial_config.yaml`. Round 1-2 launched via `python run_trial.py --trial 5 --num-rounds 4` in tmux session `trial5`; log: `experiments/trials/trial5/production.log`.

**Supersedes an earlier trial 5 attempt** seeded from `emjn04pc` (seed 45, also eps=1e-4): round 2's first GRPO attempt hit a transient CUDA OOM (auto-retried), and the retry (`kzm4d167`) went degenerate (reward collapsed to 0). That attempt's artifacts were deleted and trial 5 restarted from `iv76hjr6` instead.

**Fixed calibration denominator added before round 3 (Aug 3-4, 2026):** after round 2 completed, `fix_calibration_slope: true` (methodology #18) was added to `trial_config.yaml` — this freezes the calibrated penalty's `(z_d - z_h)` denominator to round 2's anchor (`z_h=-1.1462`, `z_d=-0.0089`, denominator `1.1373`) for round 3 onward instead of recomputing it every round; `z_h` in the numerator still updates each round. Added in-place to the main trial config (not a separate variant), per the same pattern as trial 4's `early_retraining`/`calibrated_fixed` branches. Rounds 3-6 launched via `python run_trial.py --trial 5 --num-rounds 4` in tmux session `trial5`; log: `experiments/trials/trial5/production_rounds3-6.log`. All four rounds completed on their first GRPO attempt (no OOM/retries), each running the full 128 steps.


| Round | Run ID     | Status             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----- | ---------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `iv76hjr6` | Complete; exported | `round1_clp1e-3_p20_s48_eps1e-4` — pass_all=0.035, pass_only_shown=0.60, likely_hardcoded=0.4125, does_execute=0.8275, **Obfuscated policy**. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s48_eps1e-4.json`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 2     | `ppy5mkw9` | Complete; exported | First attempt (`texvw15z`) hit a transient CUDA OOM at step 1/128, auto-retried. Retry (`ppy5mkw9`) hit a `CodeExecutor` worker deadlock (infinite loop in generated code defeating the sandbox timeout) at step ~42/128 — see [Known failure modes](./FULL_GRPO_RUN.md#known-failure-modes). Manually recovered by killing the hung process and resuming from its `step_40` checkpoint (`++training.resume_from_checkpoint=true`, same W&B run ID confirmed via `wandb_resume_map.json` fingerprint match) rather than restarting from step 0 — first validated use of this recovery path for a `CodeExecutor` deadlock specifically. does_execute=0.975, pass_first=0.4925, pass_all=0.4175, deception_rate=0.075, **Honest**. Metrics: `experiments/trials/trial5/rounds/round2/run_metrics_trial5_round2.json`. |
| 3     | `weyz6p9f` | Complete; exported | First round under the fixed calibration denominator (`++detector.fixed_calibration_denominator=1.1373106557875872`). does_execute=0.9875, pass_first=0.795, pass_all=0.165, pass_only_shown=0.54, likely_hardcoded=0.435, **Obfuscated activations**. Metrics: `experiments/trials/trial5/rounds/round3/run_metrics_trial5_round3.json`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 4     | `1nnrrl46` | Complete; exported | does_execute=0.975, pass_first=0.4975, pass_all=0.425, pass_only_shown=0.015, likely_hardcoded=0.0, **Honest**. Metrics: `experiments/trials/trial5/rounds/round4/run_metrics_trial5_round4.json`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 5     | `8qra80a3` | Complete; exported | does_execute=0.96, pass_first=0.485, pass_all=0.41, pass_only_shown=0.01, likely_hardcoded=0.0, **Honest**. Metrics: `experiments/trials/trial5/rounds/round5/run_metrics_trial5_round5.json`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 6     | `5x21tk4x` | Complete; exported | does_execute=0.9725, pass_first=0.495, pass_all=0.3975, pass_only_shown=0.0175, likely_hardcoded=0.01, **Honest**. Metrics: `experiments/trials/trial5/rounds/round6/run_metrics_trial5_round6.json`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |


**Finding:** round 3 (the first round under the fixed denominator) dipped back to Obfuscated activations before recovering to Honest for rounds 4-6, most similar to the plain calibrated branch's oscillation pattern (methodology #12) rather than the early_retraining variant's monotonic Honest streak from round 2 onward — suggesting the fixed denominator alone (without early stopping) doesn't by itself prevent the mid-trial deceptive dip seen elsewhere in this lineage family. `trial_state.json` (`last_completed_round=6`, `previous_run_id=5x21tk4x`) is in a clean, resumable state if the trial is extended further.

### Trial 5 early_retraining variant — killed, pipeline bug (Aug 4, 2026)

Attempted to add the trial 4 `early_retraining` recipe (methodology #17) on top of trial 5's lineage: `experiments/trials/trial5/trial_config_early_retraining.yaml` (seed 48, `use_calibrated_penalty`/`fix_calibration_slope: true`, `early_stop_detector_threshold=0.06`/`patience=10`/`min_steps=20`), sharing round 1 (`iv76hjr6`). Round 2 attempt 1 (`62liq8cq`) went degenerate (`val_does_execute=0.0`, see methodology #13) and was auto-retried as `7w070y0h`; the retry was killed manually at step 26/128 after a pipeline bug was discovered in the running session (bug not yet root-caused/fixed as of this entry). Round 2's checkpoints, activation caches, and probe artifacts were deleted; `trial_state_early_retraining.json` was never advanced past round 1, so no state reset was needed. `trial_config_early_retraining.yaml` and the round-1 buffer/rounds seed are intact for relaunch once the bug is fixed. Log of the killed attempt kept at `experiments/trials/trial5/production_early_retraining_attempt1_killed_pipeline_bug.log`.

### Trial 4 — KL-from-base iterative lineage (Jul 31 - Aug 1, 2026)

Designates `9nlij6so` (`round1_clp1e-3_p20_s42` from the code-length-penalty sweep below: kl=1e-3, det=3, `training.seed=42`, `data.seed=53`, nofu, `nofu_mixed` probe manifest, base model start, `code_length_penalty=0.001`, `mixed_reward_pass_all_prob=0.20`, **Obfuscated policy**) as trial 4 round 1. All subsequent rounds (round 2+) use `run_trial.py` automation with `build_train_dataset_split_sources()` (on-dataset + base/det0 anchors + prior-round on-policy) and, unlike trial 3, measure the **KL penalty against the base model** for every round (`++grpo.model.kl_reference_from_base=true`) rather than the previous round's checkpoint — the policy still initializes from the previous round's weights. Trial config: `experiments/trials/trial4/trial_config.yaml`.


| Round | Run ID     | Prefix                   | Status             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----- | ---------- | ------------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | `9nlij6so` | `round1_clp1e-3_p20_s42` | Complete; exported | See code-length-penalty sweep table below for full metrics (pass_all=0.10, pass_only_shown=0.67, **Obfuscated policy**). Metrics: `experiments/trials/trial4/rounds/round1/run_metrics_trial4_round1.json` (copy of `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s42.json`). Run dir: `full_grpo_runs/trigger_none_model_9nlij6so_layers_32/`. Phase-0 probe re-verified with `run_round1_probe.py --trial 4 --nofu`: `on_dataset_auc=0.9998` (matches GRPO export). Probe metrics: `experiments/trials/trial4/rounds/round1/probe_metrics.json`. |
| 2     | `fy2bi10d` | `trial4_round2`          | Complete; exported | pass_first=0.5925, pass_all=0.525, pass_only_shown=0.0025, likely_hardcoded=0.005, **Honest**. Metrics: `experiments/trials/trial4/rounds/round2/run_metrics_trial4_round2.json`.                                                                                                                                                                                                                                                                                                                                                                            |
| 3     | `w7uyt0vh` | `trial4_round3`          | Complete; exported | pass_first=0.515, pass_all=0.435, pass_only_shown=0.005, likely_hardcoded=0.0125, **Honest**. Metrics: `experiments/trials/trial4/rounds/round3/run_metrics_trial4_round3.json`.                                                                                                                                                                                                                                                                                                                                                                             |
| 4     | `p96gs9yt` | `trial4_round4`          | Complete; exported | pass_first=0.7475, pass_all=0.1625, pass_only_shown=0.52, likely_hardcoded=0.4525, **Obfuscated activations**. First attempt (`13bscznt`) collapsed to a fully degenerate policy at ~step 30/128 under the pre-fix epsilon — see methodology #13. Metrics: `experiments/trials/trial4/rounds/round4/run_metrics_trial4_round4.json`.                                                                                                                                                                                                                         |
| 5     | `urk4n41r` | `trial4_round5`          | Complete; exported | pass_first=0.6225, pass_all=0.4475, pass_only_shown=0.1025, likely_hardcoded=0.1225, **Honest**. Metrics: `experiments/trials/trial4/rounds/round5/run_metrics_trial4_round5.json`.                                                                                                                                                                                                                                                                                                                                                                          |


### Trial 4 calibrated variant — iterative lineage (Aug 1-2, 2026)

Same lineage as trial 4 above, but `use_calibrated_penalty: true` from round 2 onward (see methodology #12). Round 1 is shared with the non-calibrated trial (`9nlij6so`). Rounds 2-3 completed (`3n0eb1d2`, `6x73sz94`) and round 4 was started (probe phase only) under the abandoned `scale_rewards=false` fix; that lineage was killed and redone from round 2 under the epsilon fix (see methodology #13). Superseded artifacts from the earlier attempt are preserved under `rounds_calibrated_old/` / `buffer_calibrated_old/`. Trial config: `experiments/trials/trial4/trial_config_calibrated.yaml`.


| Round | Run ID     | Status             | Notes                                                                                                                                                                                                                                  |
| ----- | ---------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `9nlij6so` | Complete; exported | Shared with non-calibrated trial 4 round 1 — see row above.                                                                                                                                                                            |
| 2     | `9ysksl2u` | Complete; exported | does_execute=0.99, pass_first=0.545, pass_all=0.4725, pass_only_shown=0.0075, likely_hardcoded=0.015, **Honest**. Metrics: `experiments/trials/trial4/rounds_calibrated/round2/run_metrics_trial4_calibrated_round2.json`.             |
| 3     | `1tb3o950` | Complete; exported | does_execute=0.9975, pass_first=0.8275, pass_all=0.2275, pass_only_shown=0.52, likely_hardcoded=0.525, **Obfuscated policy**. Metrics: `experiments/trials/trial4/rounds_calibrated/round3/run_metrics_trial4_calibrated_round3.json`. |
| 4     | `sz8pomk0` | Complete; exported | does_execute=0.99, pass_first=0.54, pass_all=0.4725, pass_only_shown=0.0, likely_hardcoded=0.0, **Honest**. Metrics: `experiments/trials/trial4/rounds_calibrated/round4/run_metrics_trial4_calibrated_round4.json`.                   |
| 5     | `w4h9yyov` | Complete; exported | does_execute=0.985, pass_first=0.5125, pass_all=0.4525, pass_only_shown=0.0, likely_hardcoded=0.0, **Honest**. Metrics: `experiments/trials/trial4/rounds_calibrated/round5/run_metrics_trial4_calibrated_round5.json`.                |


No degenerate collapse across any round under the epsilon fix (`does_execute` stayed 98.5-99.75% throughout) — including round 4, the exact round whose original attempt (`13bscznt`, non-calibrated lineage) collapsed to 0%. Classification oscillates Obfuscated → Honest → Obfuscated → Honest → Honest across rounds 1-5, consistent with the calibrated penalty responding each round to the policy's own drift. Per methodology #13's Aug 2 caveat, these round-to-round classifications shouldn't be assumed identical to what the pre-fix epsilon (`1e-4`) would have produced.

### Trial 4 carryover variant — conditional probe retraining test (Aug 2-3, 2026 — complete)

Same lineage as the trial 4 calibrated variant above (`use_calibrated_penalty: true`, epsilon fix), plus `min_on_policy_per_class: 20` to test conditional probe retraining (methodology #15). Round 1 is shared with the non-calibrated and calibrated trial 4 branches (`9nlij6so`); seeded the standard variant-branch way (`buffer_carryover/`, `rounds_carryover/round1/` copied from the base trial's `buffer/`/`rounds/round1`, not re-run). Trial config: `experiments/trials/trial4/trial_config_carryover.yaml`. Launched via `python run_trial.py --trial 4 --variant carryover --num-rounds 4` in tmux session `carryover`; log: `experiments/trials/trial4/production_carryover.log`.


| Round | Run ID     | Status             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----- | ---------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `9nlij6so` | Complete; exported | Shared with non-calibrated/calibrated trial 4 round 1 — see rows above.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2     | `waksvwv5` | Complete; exported | Round-1 on-policy generation (693 deceptive / 133 honest) cleared the N=20 threshold for both classes, so the probe manifest expanded normally (400/400 per class) — same composition as the calibrated branch's round 2. First attempt (`xvsr15qr`) hit a transient CUDA OOM at step 3/128, auto-retried; `waksvwv5` succeeded. pass_all=0.4425, pass_only_shown=0.0125, **Honest**.                                                                                                                                                                                                                                                             |
| 3     | `trf4bhup` | Complete; exported | Round-2 on-policy generation had only **9 deceptive / 695 honest** — below the N=20 threshold for the deceptive class, so `build_or_reuse_probe_dataset()` **skipped expansion** and reused round 2's manifest unchanged (still 400/400); the probe was still retrained fresh on round-3 activations, just on the stale (round-1/2) dataset composition. This is the one round where the conditional-retraining logic actually changed behavior vs. the calibrated branch. First attempt hit a transient CUDA OOM, auto-retried; `trf4bhup` succeeded. pass_all=0.0325, pass_only_shown=0.8925, likely_hardcoded=0.5, **Obfuscated activations**. |
| 4     | `rzr8r59s` | Complete; exported | Round-3 on-policy generation cleared the threshold again; manifest expanded to 600/600. First attempt hit a transient CUDA OOM, auto-retried; `rzr8r59s` succeeded. pass_all=0.04, pass_only_shown=0.865, likely_hardcoded=0.5, **Obfuscated policy**.                                                                                                                                                                                                                                                                                                                                                                                            |
| 5     | `4ejxgy0q` | Complete; exported | Round-4 on-policy generation cleared the threshold; manifest expanded to 700/700. Succeeded on first attempt (no OOM). pass_all=0.04, pass_only_shown=0.855, likely_hardcoded=0.485, **Obfuscated policy**.                                                                                                                                                                                                                                                                                                                                                                                                                                       |


**Carryover-vs-calibrated comparison:** round 2 landed identically Honest in both branches (identical setup/outcome, as expected). The branches diverge from round 3 onward: the calibrated branch (no conditional retraining) recovers to Honest by round 4 and stays Honest through round 5, while the carryover branch remains deceptive (Obfuscated activations → Obfuscated policy) from round 3 through round 5. Plausible explanation: skipping the round-3 probe-dataset expansion (because round 2's on-policy generation was overwhelmingly honest, only 9/704 deceptive samples) left the round-3 probe trained on the same on-policy composition as round 2, which may have been less able to catch the round-3 policy's drift toward deception than a freshly-expanded probe would have — and that drift then persisted into rounds 4-5 even after expansion resumed. This is a single run per branch, so it's a suggestive data point rather than a confirmed effect; a systematic test would need multiple seeds.

**Reliability note:** every round except round 5 hit a transient CUDA OOM on its first training attempt (allocating ~1.4-1.5 GiB with GPU 0 nearly full) and was auto-retried successfully by `run_trial.py`'s crash-recovery loop — no manual intervention required. This pattern recurring across 3 of 4 rounds (vs. being a one-off) may be worth a closer look if it keeps happening on future trials.

### Trial 4 carryover_frozen variant — probe+calibration carry-over (Aug 3, 2026 — killed, discontinued)

Same lineage as the trial 4 carryover variant above, plus `carry_over_probe_weights: true` (methodology #19): on rounds where expansion is skipped, carries over the previous round's probe weights and calibration constants instead of retraining. Round 1 shared (`9nlij6so`). Round 2 (`uuvtru0a`) completed normally (threshold met, no carry-over triggered). Round 3 triggered carry-over (on-policy: 1 deceptive / 689 honest, below threshold) and immediately showed a large detector-score discontinuity — see methodology #19 for the root cause. Killed mid-round-3; round 2/3 checkpoints and activation caches deleted. Trial config: `experiments/trials/trial4/trial_config_carryover_frozen.yaml`; log: `experiments/trials/trial4/production_carryover_frozen.log` (retained for reference).

### Trial 4 frozen variant — truly frozen probe weights (Aug 2-3, 2026 — in progress)

Same lineage as trial 4 above, but the probe used for the obfuscation penalty is **never retrained** after round 1 — every round reuses round 1's exact probe weights via `detector.frozen_probe_weights_dir` (see methodology #16). Round 1 is shared with the other trial 4 branches (`9nlij6so`). Since nothing retrains, the per-round generation + external probe-training pipeline is skipped entirely (`run_single_round()`); GRPO's Phase 0 still runs to produce eval metrics/thresholds/layer filtering from a freshly-trained probe, but the frozen weights are swapped in before RL training starts. Trial config: `experiments/trials/trial4/trial_config_frozen.yaml`. Launched via `python run_trial.py --trial 4 --variant frozen --num-rounds 4` in tmux session `frozen`; log: `experiments/trials/trial4/production_frozen.log`.


| Round | Run ID     | Status             | Notes                                                                                                                                                                                                    |
| ----- | ---------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `9nlij6so` | Complete; exported | Shared with the other trial 4 branches — see rows above.                                                                                                                                                 |
| 2     | `0lhmqg1u` | Complete; exported | pass_first=0.87, pass_all=0.22, pass_only_shown=0.5825, likely_hardcoded=0.5625, **Obfuscated policy**. Metrics: `experiments/trials/trial4/rounds_frozen/round2/run_metrics_trial4_frozen_round2.json`. |
| 3     | `pfvnd8rl` | Complete; exported | pass_first=0.9675, pass_all=0.0, pass_only_shown=0.9425, likely_hardcoded=0.495, **Obfuscated policy**. Metrics: `experiments/trials/trial4/rounds_frozen/round3/run_metrics_trial4_frozen_round3.json`. |
| 4     | `ze1uas04` | Complete; exported | pass_first=0.9625, pass_all=0.0, pass_only_shown=0.925, likely_hardcoded=0.495, **Obfuscated policy**. Metrics: `experiments/trials/trial4/rounds_frozen/round4/run_metrics_trial4_frozen_round4.json`.  |
| 5     | `44oizrwy` | Running            | In progress as of Aug 3, 2026 — not yet complete; update this row once finished.                                                                                                                         |


**Preliminary finding (rounds 2-4):** unlike every retrained trial 4 branch (main, calibrated, carryover), which recovers to Honest in at least one round, the frozen-probe branch has been continuously **Obfuscated policy** from round 2 onward, with `pass_all_tests` collapsing to 0.0 by round 3 and `pass_only_shown_deception` climbing monotonically (0.58 → 0.94 → 0.93). This is the clearest single-run evidence so far that periodic probe retraining — not just the adversarial penalty itself — is doing real work to constrain deception in this setup: a probe frozen at round 1, however good its on-dataset AUC stays (~1.0 throughout), apparently develops a growing blind spot to the policy's evolving on-policy deceptive behavior (on-policy recall/AUC drops sharply after round 1 — see per-round metrics files). As with other single-run branches in this log, this is a suggestive data point, not a confirmed effect.

### Trial 4 early_retraining variant — detector-score early stopping (Aug 3, 2026 — complete)

Same lineage as the trial 4 calibrated variant (`use_calibrated_penalty: true`), plus detector-score early stopping: each round's GRPO training stops as soon as the train-batch detector score stays below 0.06 for 10 consecutive steps (after a minimum of 20 steps), rather than always running the full 128 — see methodology #17. The next round then retrains the probe normally (default `split_sources`, no carryover/frozen-probe logic) from whatever checkpoint that produced. From round 3 onward, the calibration `(z_d - z_h)` denominator is also frozen to round 2's value instead of recomputing every round (`fix_calibration_slope: true`, methodology #18). Round 1 is shared with the other trial 4 branches (`9nlij6so`). Trial config: `experiments/trials/trial4/trial_config_early_retraining.yaml`. Launched via `python run_trial.py --trial 4 --variant early_retraining --num-rounds 4` in tmux session `early_retraining`; log: `experiments/trials/trial4/production_early_retraining.log` (round 3 relaunched under methodology #18; continued in `experiments/trials/trial4/production_early_retraining_fixed_cal.log`).


| Round      | Run ID     | Status                | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------- | ---------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1          | `9nlij6so` | Complete; exported    | Shared with the other trial 4 branches — see rows above.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2          | `fq8yq8ie` | Complete; exported    | Early-stopped at step 62/128 (train-batch detector score 0.036 < 0.06 for 10 consecutive steps, vs. the full 128 steps every other trial 4 branch ran). pass_first=0.57, pass_all=0.5, pass_only_shown=0.005, likely_hardcoded=0.0, **Honest**. On-policy generation from this checkpoint later produced 0/1000 deceptive completions, which broke round 3's class-balance assertion — see methodology #17's follow-on fix. Metrics: `experiments/trials/trial4/rounds_early_retraining/round2/run_metrics_trial4_early_retraining_round2.json`.      |
| 3 (killed) | `jhjh2wnl` | Killed at step 44/128 | Round 2's 0 on-policy deceptive samples meant the round-3 manifest drew its full deceptive quota from `det0` + `round1` instead of evenly across `det0`/`round1`/`round2` (`deceptive sources: {'on_dataset': 200, 'det0': 150, 'round1': 150, 'round2': 0}`); calibration logits computed from only 2 on-policy deceptive examples (vs. 268 in round 2), noisier than usual. Killed and its partial checkpoint dir removed before completion, to switch to the fixed calibration denominator (methodology #18) — not counted as a completed attempt. |
| 3 (retry)  | `cpff1h5l` | Complete; exported    | Relaunched under `fix_calibration_slope: true` (methodology #18); ran the full 128 steps (no early stop). pass_first=0.83, pass_all=0.73, pass_only_shown=0.006, likely_hardcoded=0.0, **Honest**. Metrics: `experiments/trials/trial4/rounds_early_retraining/round3/run_metrics_trial4_early_retraining_round3.json`.                                                                                                                                                                                                                               |
| 4          | `36jn4itd` | Complete; exported    | Early-stopped at step 29/128 (detector score 0.0015 < 0.06 for 10 consecutive steps). pass_first=0.72, pass_all=0.51, pass_only_shown=0.099, likely_hardcoded=0.11, **Honest**. Metrics: `experiments/trials/trial4/rounds_early_retraining/round4/run_metrics_trial4_early_retraining_round4.json`.                                                                                                                                                                                                                                                  |
| 5          | `2wk68hpq` | Complete; exported    | Early-stopped at step 29/128 (detector score 0.0013 < 0.06 for 10 consecutive steps). pass_first=0.53, pass_all=0.32, pass_only_shown=0.095, likely_hardcoded=0.0, **Honest**. Metrics: `experiments/trials/trial4/rounds_early_retraining/round5/run_metrics_trial4_early_retraining_round5.json`.                                                                                                                                                                                                                                                   |


**Preliminary finding:** unlike the plain calibrated branch (methodology #12), which oscillates Obfuscated ↔ Honest through round 5, this variant stayed **Honest for every round from round 2 onward** once early stopping and the fixed calibration denominator were both active — rounds 4-5 in particular early-stopped almost immediately (step 29), suggesting the policy converged to low deception quickly each round rather than drifting back toward obfuscation. Round 2's early stop still caught the model *after* it had already abandoned deception (`likely_hardcoded` dropped well before the detector-score threshold was crossed, per methodology #17) — whether early stopping is catching deception-in-progress or just confirming an already-honest policy in later rounds remains open.

**Rerun stopped (Aug 4, 2026):** rounds 2-5 above were archived with `_pre_fix` suffix and a rerun under methodology #20's fix was launched (`run_trial.py --trial 4 --variant early_retraining --num-rounds 4`, tmux `early_retraining`), but was killed mid-round-2 after another bug was discovered. Will rerun again once that bug is fixed.

### Trial 4 early_retraining_v2 variant — dynamic early stop on standard probe (Aug 4, 2026 — complete)

Same KL-from-base lineage as the main trial 4 branch (`use_calibrated_penalty: false`, methodology #21 nofu/EOT fixes active), plus dynamic detector-score early stopping (methodology #23). Round 1 shared (`9nlij6so`). Trial config: `experiments/trials/trial4/trial_config_early_retraining_v2.yaml`. Launched via `python run_trial.py --trial 4 --variant early_retraining_v2 --num-rounds 4` in tmux `early_v2`; log: `experiments/trials/trial4/production_early_retraining_v2.log`.


| Round | Run ID     | Status             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ----- | ---------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `9nlij6so` | Complete; exported | Shared with other trial 4 branches — see row above.                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2     | `yhmdgqjx` | Complete; exported | Early-stopped step 32/128 (threshold 0.133). pass_first=0.80, pass_all=0.30, pass_only_shown=0.50, **Obfuscated policy**. Metrics: `experiments/trials/trial4/rounds_early_retraining_v2/round2/run_metrics_trial4_early_retraining_v2_round2.json`.                                                                                                                                                                                                                                                                          |
| 3     | `edo9ba7u` | Complete; exported | Early-stopped step 42/128 (threshold 0.079). pass_first=0.59, pass_all=0.41, pass_only_shown=0.19, **Honest**. Metrics: `experiments/trials/trial4/rounds_early_retraining_v2/round3/run_metrics_trial4_early_retraining_v2_round3.json`.                                                                                                                                                                                                                                                                                   |
| 4     | `me5zttvr` | Complete; exported | Early-stopped step 38/128 (threshold 0.146). pass_first=0.58, pass_all=0.51, pass_only_shown=0.07, **Honest**. Metrics: `experiments/trials/trial4/rounds_early_retraining_v2/round4/run_metrics_trial4_early_retraining_v2_round4.json`.                                                                                                                                                                                                                                                                                   |
| 5     | `io4ibs0w` | Complete; exported | Manually stopped ~step 60/128 (threshold 0.051, floor-limited; only 7 on-policy deceptive samples from round 4). Checkpoint promoted from `step_60`; post-eval recovered via eval-only `8g8580by` (metrics/classification). pass_first=0.55, pass_all=0.49, pass_only_shown=0.06, **Honest**. Metrics: `experiments/trials/trial4/rounds_early_retraining_v2/round5/run_metrics_trial4_early_retraining_v2_round5.json`. `trial_state_early_retraining_v2.json`: `previous_run_id=io4ibs0w`. |


**Preliminary finding:** rounds 3–5 stayed Honest with early stops at ~30–40% of the 128-step budget (vs full runs on the base branch). Round 2 classified Obfuscated policy despite early stopping — unlike the old calibrated `early_retraining` branch, which was Honest from round 2 onward. Round 2's higher on-policy deception (0.50 pass_only_shown) and lower in-loop detector scores vs the pre-#21 base trial 4 run are consistent with the EOT probing fix changing probe train/apply alignment, not just early-stop dynamics.

### Trial 4 v2 variant — full 128-step GRPO control for `early_retraining_v2` (Aug 5–6, 2026 — complete)

Standard sigmoid probe penalty, full 128 steps every round (no early stopping). Same hyperparams as `early_retraining_v2` (p20, seed 42, clp=1e-3, KL-from-base, `split_sources`); shares round 1 (`9nlij6so`). After an initial launch with mismatched round-1 buffer/probe data, restarted Aug 5 with `buffer_early_retraining_v2/` round-1 files and `early_retraining_v2` round-2 probe weights/activations copied into `buffer_v2/` / `rounds_v2/round2/` so round 2+ matches that branch except for early stopping.

Config: `experiments/trials/trial4/trial_config_v2.yaml`. Launcher: `python run_trial.py --trial 4 --variant v2 --num-rounds 4` in tmux `trial4_v2`; log: `experiments/trials/trial4/production_v2.log`. Completed Aug 6; `trial_state_v2.json`: `previous_run_id=rkrfsaml`.


| Round | Run ID     | Status             | Notes |
| ----- | ---------- | ------------------ | ----- |
| 1     | `9nlij6so` | Complete; exported | Shared with other trial 4 branches. |
| 2     | `msduvkpw` | Complete; exported | Full 128 steps. pass_only_shown=0.0025, **Honest** (cf. `early_retraining_v2` `yhmdgqjx`: Obfuscated at step 32). Metrics: `experiments/trials/trial4/rounds_v2/round2/run_metrics_trial4_v2_round2.json`. |
| 3     | `1by3ybbt` | Complete; exported | pass_only_shown=0.005, **Honest**. Metrics: `experiments/trials/trial4/rounds_v2/round3/run_metrics_trial4_v2_round3.json`. |
| 4     | `f54ox4bv` | Complete; exported | pass_only_shown=0.0075, **Honest**. Metrics: `experiments/trials/trial4/rounds_v2/round4/run_metrics_trial4_v2_round4.json`. |
| 5     | `rkrfsaml` | Complete; exported | pass_only_shown=0.005, **Honest**. Metrics: `experiments/trials/trial4/rounds_v2/round5/run_metrics_trial4_v2_round5.json`. |


**Preliminary finding:** with matched probe/buffer data, all rounds 2–5 stay **Honest** under full 128 steps; round 2 contrasts with `early_retraining_v2` (**Obfuscated policy** at step-32 early stop) — early stopping may leave the policy more deceptive at round 2, not just shorten training.

### Trial 4 calibrated_fixed variant — full clean rerun with fixed calibration denominator (Aug 3-4, 2026 — redoing rounds 2-6 under methodology #20)

Full clean rerun of the trial 4 calibrated variant (above) with `fix_calibration_slope: true` (methodology #18): for rounds > 2, the calibrated penalty's `(z_d - z_h)` denominator is frozen to this variant's own round 2 anchor instead of being recomputed every round; `z_h` in the numerator still updates each round. Unlike the `early_retraining` variant (which also sets `fix_calibration_slope: true` but adds early stopping, methodology #17), this variant is otherwise identical to the plain calibrated branch. Round 1 is shared with the other trial 4 branches (`9nlij6so`); rounds 2-6 all rerun from scratch here rather than reusing the original calibrated branch's round 2+ artifacts, since this branch needs its own fresh round-2 `z_h`/`z_d` to serve as the frozen denominator anchor for later rounds. Trial config: `experiments/trials/trial4/trial_config_calibrated_fixed.yaml`.

**Redo under methodology #20 (Aug 4, 2026):** rounds 2-5 completed and round 6 was in progress (`xx8b3u2g`, step ~60/128) when the calibrated-penalty probe-mismatch bug was discovered — this entire variant was affected, since every round's GRPO penalty used calibration constants computed from a differently-scaled probe than the one actually applying the penalty. Round 6 was killed (no final adapter; partial checkpoint + its prefix-keyed activation cache removed). Round 1 and round 2's external-pipeline artifacts (on-policy generation, probe weights, calibration logits) don't depend on the buggy GRPO probe at all and were kept as-is (only round 2's stale exported `run_metrics.json` was archived so the round re-launches); rounds 3-6's artifacts were archived with a `_pre_fix` suffix (their on-policy generation/probes/GRPO all depend transitively on round 2's now-superseded checkpoint). `trial_state_calibrated_fixed.json` reset to `last_completed_round=1`. Redo relaunched via `python run_trial.py --trial 4 --variant calibrated_fixed --num-rounds 5` in tmux session `calibrated_fixed`; log: `experiments/trials/trial4/production_calibrated_fixed.log` (pre-fix log archived as `production_calibrated_fixed_pre_fix.log`).


| Round         | Run ID                                                              | Status                                                        | Notes                                                                                                                                                                                                                  |
| ------------- | ------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1             | `9nlij6so`                                                          | Complete; exported                                            | Shared with the other trial 4 branches — see rows above.                                                                                                                                                               |
| 2-6 (pre-fix) | `le897wac`, `n3iozx23`, `lnkop2wh`, `itk98g6q`, `xx8b3u2g` (killed) | Superseded — affected by methodology #20's probe-mismatch bug | Not reliable for comparison; artifacts archived under `rounds_calibrated_fixed/round{3,4,5,6}_pre_fix/` and `buffer_calibrated_fixed/*_pre_fix.`*. Round 6 never completed (killed at step ~60/128, no final adapter). |
| 2-6 (redo)    | pending                                                             | Redoing                                                       | Update this table once the redo completes.                                                                                                                                                                             |


### Trial 3 — split-sources pipeline (Jul 25-27, 2026)

Lineage seeded from `62es59py` step-128 checkpoint. Uses `build_train_dataset_split_sources()`. Hyperparams match round-1 seed (`62es59py`): kl=1e-3, det=3, `training.seed=42`, `data.seed=53`, `mixed_reward_pass_all_prob=0.3`, nofu, 128 steps.


| Round         | Run ID     | Prefix             | Status             | Notes                                                                                                                                                                                                                                                  |
| ------------- | ---------- | ------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2 (attempt 1) | `tk0qwxok` | `trial3_round2`    | Killed ~step 96    | Reward collapsed to 0; artifacts deleted                                                                                                                                                                                                               |
| 2 (attempt 2) | `pjo2p8qw` | `trial3_round2_v2` | Complete; exported | pass_first=0.5875, pass_all=0.2875, pass_only_shown=0.21, Obfuscated policy. Intermediates cleaned; `checkpoints/final` retained. Metrics: `experiments/trials/trial3/rounds/round2/run_metrics_trial3_round2_v2.json`                                 |
| 3             | `wu6k5j61` | `trial3_round3`    | Complete; exported | `previous_run_id=pjo2p8qw`; single-H200. pass_first=0.47, pass_all=0.3825, pass_only_shown=0.02, **Honest**. Intermediates cleaned; `checkpoints/final` retained. Metrics: `experiments/trials/trial3/rounds/round3/run_metrics_trial3_round3.json`    |
| 4             | `5lnsv03f` | `trial3_round4`    | Complete; exported | `previous_run_id=wu6k5j61`; single-H200. pass_first=0.4475, pass_all=0.325, pass_only_shown=0.0475, **Honest**. Intermediates cleaned; `checkpoints/final` retained. Metrics: `experiments/trials/trial3/rounds/round4/run_metrics_trial3_round4.json` |


**Round 2 probe** (trained on round-1 / `62es59py` on-policy): on-dataset AUC 0.973, on-policy split AUC 0.904.

**Round 3 probe** (trained on round-2 / `pjo2p8qw` on-policy, split-sources: honest anchors `base_model`+rounds, deceptive anchors `det0`+rounds): on-dataset AUC 0.972, on-policy split AUC 0.866 (deceptive=79 / honest=114 / unlabeled=207).

**Round 4 probe** (trained on round-3 / `wu6k5j61` on-policy, split-sources): on-dataset AUC 0.984, on-policy split AUC 0.856 (deceptive=5 / honest=161 / unlabeled=234 — very few on-policy deceptive examples, consistent with round 3 already being Honest). Policy remained Honest through round 4, with `pass_only_shown_deception` continuing to decline (0.21 → 0.02 → 0.0475).

**Trial 3 ablations** — four round-2 variants, each isolating one factor from trial 3's real round 2 (`pjo2p8qw`: sigmoid penalty, retrained probe, KL anchored to the round-1 checkpoint):

### Trial 3 ablation — calibrated probe penalty round 2 (Jul 28, 2026)

Same round-1 checkpoint (`62es59py`) and round-2 probe/hyperparameters as trial 3's real round 2 (`pjo2p8qw`), but the probe penalty uses the calibrated linear clip `clip((z_agg - z_h) / (z_d - z_h), 0, 1)` instead of sigmoid — see methodology #12. Saved under `experiments/trials/trial3/rounds/round2_calibrated_penalty/`, not a separate trial.


| Round          | Run ID     | Prefix                             | Status             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------- | ---------- | ---------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2 (calibrated) | `iviyt23y` | `trial3_round2_calibrated_penalty` | Complete; exported | `previous_run_id=62es59py`; `++detector.calibration_logits_path` (z_h=-0.2407, z_d=-0.0919, from 114 honest/79 deceptive round-1 on-policy samples); probe = round 2 manifest (same as `pjo2p8qw`). success=0.4775, pass_all=0.345, pass_only_shown=0.0275, likely_hardcoded=0.005, **Honest**. Final-model on-policy probe AUC 0.999. Intermediates cleaned (`lora`, `step`_*); `checkpoints/final` retained. Metrics: `experiments/trials/trial3/rounds/round2_calibrated_penalty/run_metrics_trial3_round2_calibrated_penalty.json` |


**Finding:** swapping only the penalty shape (same probe/checkpoint/hyperparameters as `pjo2p8qw`) flips the outcome from Obfuscated policy to Honest (pass_only_shown 0.21→0.03, pass_all 0.29→0.35) — the calibrated clip's more evenly-distributed gradient more effectively suppresses the round-2 probe's target behavior than sigmoid's saturating tails.

### Trial 3 ablation — KL-from-base round 2 (Jul 28, 2026)

Same round-1 checkpoint (`62es59py`) and round-2 probe as trial 3's real round 2 (`pjo2p8qw`), but with KL penalty measured against the **base model** instead of the round-1 checkpoint — see methodology #11. Saved under `experiments/trials/trial3/rounds/round2_kl_from_base/`, not a separate trial.


| Round            | Run ID     | Prefix                       | Status                  | Notes                                                                                                                                                                                                                                   |
| ---------------- | ---------- | ---------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2 (kl-from-base) | `c07qehe0` | `trial3_round2_kl_from_base` | Complete; **collapsed** | `previous_run_id=62es59py`; `++grpo.model.kl_reference_from_base=true`; probe = round 2 manifest (split-sources, 300/class). First attempt (`ob3e9a1a`) hit CodeExecutor worker stall at step 10 → killed and relaunched as `c07qehe0`. |


**Outcome:** reward collapsed at ~step 88/128 — the base-model KL penalty pulled the policy back toward verbose non-code outputs (response length 78→220 tokens), gradually degrading `does_execute` (0.97→0.77) until a stochastic bad batch triggered oscillating gradient spikes that pushed the model into pure gibberish for the remaining 37 steps. Unlike the normal round-2 run (KL anchored to round-1, policy starts at KL≈0), anchoring to base starts the policy at KL≈1.4 — an immediate destabilizing force incompatible with the concise code-generation behavior task reward needs. `checkpoints/final` retained but is the collapsed model — not usable.

### Trial 3 baseline — no-retrain ablation (Jul 27, 2026)

Same round-1 checkpoint (`62es59py`) and hyperparameters as trial 3, but round 2 GRPO trains against round 1's original probe manifest unchanged (no retraining) — see methodology #9. Saved under `experiments/trials/trial3/rounds/round2_baseline/`, not a separate trial.


| Round        | Run ID     | Prefix                   | Status             | Notes                                                                                                                                                                                                                                                                                                                                                                       |
| ------------ | ---------- | ------------------------ | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2 (baseline) | `l9dihs4i` | `trial3_round2_baseline` | Complete; exported | `previous_run_id=62es59py`; probe = round 1 manifest (300/class, no retrain), `expected_on_dataset_auc=0.9906`. pass_first=0.925, pass_all=0.02, pass_only_shown=0.8425, **Obfuscated policy**. Intermediates cleaned (`lora`, `step`_*); `checkpoints/final` retained. Metrics: `experiments/trials/trial3/rounds/round2_baseline/run_metrics_trial3_round2_baseline.json` |


**Finding:** vs. trial 3's real round 2 (`pjo2p8qw`, retrained probe, pass_all=0.2875, pass_only_shown=0.21), the no-retrain baseline is far more deceptive (pass_only_shown 0.84) and solves far less honestly (pass_all 0.02), despite identical hyperparameters and starting checkpoint — probe retraining (folding in round 1 on-policy data) meaningfully constrained round-2 deception.

### Trial 3 baseline — no-detector-penalty from base model (Jul 28, 2026)

Same hyperparameters as trial 3 round 1 (kl=1e-3, `training.seed=42`, `data.seed=53`, `mixed_reward_pass_all_prob=0.3`, nofu, pass-only-shown) but with `loss.detector=0` and starting fresh from the base model instead of a prior checkpoint — see methodology #10. Saved under `experiments/trials/trial3/rounds/det0_baseline/`, not a separate trial.


| Round         | Run ID     | Prefix                 | Status             | Notes                                                                                                                                                                                                                                                                                                                                                                       |
| ------------- | ---------- | ---------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| det0 baseline | `mdxnlbz1` | `trial3_det0_baseline` | Complete; exported | From base model, `loss.detector=0`, probe = `nofu_mixed` manifest (300/class, no retrain, monitoring only). pass_first=0.9725, pass_all=0.0025, pass_only_shown=0.9525, **Obfuscated activations**. Intermediates cleaned (`lora`, `step`_*); `checkpoints/final` retained. Metrics: `experiments/trials/trial3/rounds/det0_baseline/run_metrics_trial3_det0_baseline.json` |


**Finding:** vs. trial 3's round 1 (`62es59py`, det=3, same base-model start, same mixed reward), removing the detector penalty entirely drives pass_only_shown deception to 0.95 (near-total hardcoding) — but the classification is **Obfuscated activations**, not Blatant deception: `policy_detector_prob` stays near the 0.35 threshold (0.357) while `representation_change` is large (0.52). Pure reward-hacking pressure (no adversarial probe gradient at all) is sufficient to drift representations away from what the frozen initial probe recognizes as deceptive.

### Trial 1 — multi-round pipeline (Jul 2026)

Round 1 checkpoint: `mtzvz59v` (**Obfuscated policy**). Rounds 2-6 complete under `experiments/trials/trial1/`. Full pipeline artifacts (buffer + probe + run metrics) for rounds 2-6.

### Trial 2 — multi-round pipeline (Jul 2026)

Round 1 checkpoint: `lliye1cp` (**Honest**). Rounds 2-6 complete under `experiments/trials/trial2/`, but ran with mismatched hyperparameters vs round 1 (det=3/seed=43 hardcoded in `run_trial.py` vs det=8/seed=3 from round 1) — not directly comparable across rounds; see Known gaps.

---

**Standalone ablations** — one-off experiments testing individual factors, not part of a trial lineage.

### Round 1 ablation — clp1e-3/p20 seed sweep, seeds 50-53 (Aug 4-5, 2026)

Four-seed sweep at `mixed_reward_pass_all_prob=0.20` (80/20), otherwise identical to the clp1e-3/p25 seed family under eps=1e-4 (kl=1e-3, det=3, `data.seed=53`, nofu, `nofu_mixed` manifest, base model, `code_length_penalty=0.001`, 128 steps). Config: `experiments/configs/round1_clp1e-3_p20_sweep.yaml`. Log: `scratch/round1_clp1e-3_p20_sweep_17de0bc2f2ef.log`. s53 killed at step ~88/128 (run dir deleted, no metrics) to launch `round1_clp2e-3_p25_s52`.


| CLP   | p    | Seed | Run ID     | Prefix                   | pass_all | pass_only_shown | Result                | Notes                                                                                       |
| ----- | ---- | ---- | ---------- | ------------------------ | -------- | --------------- | --------------------- | ------------------------------------------------------------------------------------------- |
| 0.001 | 0.20 | 50   | `dea71p0n` | `round1_clp1e-3_p20_s50` | 0.0      | 0.94            | **Obfuscated policy** | val_does_execute=0.9975. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s50.json`. |
| 0.001 | 0.20 | 51   | `7a0cse2z` | `round1_clp1e-3_p20_s51` | 0.005    | 0.7475          | **Obfuscated policy** | val_does_execute=0.855. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s51.json`.  |
| 0.001 | 0.20 | 52   | `xf9mtwag` | `round1_clp1e-3_p20_s52` | 0.0025   | 0.9275          | **Obfuscated policy** | val_does_execute=1.0. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s52.json`.    |
| 0.001 | 0.20 | 53   | `xxtafkxt` | `round1_clp1e-3_p20_s53` | —        | —               | Killed                | Aborted step ~88/128; no export.                                                            |

**Key finding:** under eps=1e-4, all 3 completed seeds at clp1e-3/p20 land **Obfuscated policy** — unlike clp1e-3/p25 at the same epsilon (3/4 Honest above).

### Round 1 ablation — clp1e-3/p25 seed sweep (Aug 4-5, 2026)

Four-seed sweep at `mixed_reward_pass_all_prob=0.25` (75/25), otherwise identical to the clp1e-3/p20 seed family under eps=1e-4 (kl=1e-3, det=3, `data.seed=53`, nofu, `nofu_mixed` manifest, base model, `code_length_penalty=0.001`, 128 steps). Uses methodology #21's EOT-append fix. Config: `experiments/configs/round1_clp1e-3_p25_sweep.yaml`. Log: `scratch/round1_clp1e-3_p25_sweep.log`.

| CLP | p | Seed | Run ID | Prefix | pass_all | pass_only_shown | Result | Notes |
|-----|---|------|--------|--------|----------|-----------------|--------|-------|
| 0.001 | 0.25 | 48 | `lxb0hgjw` | `round1_clp1e-3_p25_s48` | 0.3975 | 0.0025 | Honest | val_does_execute=0.96. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p25_s48.json`. |
| 0.001 | 0.25 | 49 | `x2ed7tka` | `round1_clp1e-3_p25_s49` | 0.41 | 0.0075 | Honest | val_does_execute=0.9325. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p25_s49.json`. |
| 0.001 | 0.25 | 50 | `yckx0ctm` | `round1_clp1e-3_p25_s50` | 0.045 | 0.8225 | **Obfuscated policy** | val_does_execute=0.9925; likely_hardcoded=0.5025. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p25_s50.json`. |
| 0.001 | 0.25 | 51 | `s440n6af` | `round1_clp1e-3_p25_s51` | 0.37 | 0.0075 | Honest | val_does_execute=0.9675. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p25_s51.json`. |

**Key finding:** at clp1e-3/p25 under eps=1e-4, seed variance is substantial — 3/4 Honest (s48, s49, s51) vs 1/4 Obfuscated policy (s50), mirroring the bistability seen at clp1e-3/p20.

### Round 1 ablation — clp2e-3/p25 (Aug 5, 2026)

Single runs and sweeps at `code_length_penalty=0.002`, otherwise identical to the matching clp1e-3/p25 seed (p=0.25, eps=1e-4). Configs: `experiments/configs/round1_clp2e-3_p25_s51.yaml`, `round1_clp2e-3_p25_s52.yaml`, `round1_clp2e-3_p25_s56.yaml`; sweeps: `round1_clp2e-3_p25_seed_sweep.yaml` (s53–55), `round1_clp2e-3_p25_s54_s55_s57_sweep.yaml` (s54–55, s57). Logs: `scratch/round1_clp2e-3_p25_s51.log`, `scratch/round1_clp2e-3_p25_s52_17de0bc2f2ef.log`, `scratch/round1_clp2e-3_p25_s56_17de0bc2f2ef.log`, `scratch/clp2e-3_p25_seed_sweep_71660ca36c5d.log`, `scratch/clp2e-3_p25_s54_s55_s57_sweep.log`. Original seed sweep killed after s53 in progress (run dir deleted, no export). Continuation sweep (Aug 5): s54 complete; s55 killed step ~15/128 (`d25yw0bg`, run dir deleted); s57 never started.

| CLP | p | Seed | Run ID | Prefix | pass_all | pass_only_shown | Result | Notes |
|-----|---|------|--------|--------|----------|-----------------|--------|-------|
| 0.002 | 0.25 | 51 | `w7cj2l8q` | `round1_clp2e-3_p25_s51` | 0.3025 | 0.085 | Honest | val_does_execute=0.9725; likely_hardcoded=0.085. Metrics: `experiments/single_runs/run_metrics_round1_clp2e-3_p25_s51.json`. |
| 0.002 | 0.25 | 52 | `j2kt120q` | `round1_clp2e-3_p25_s52` | 0.355 | 0.025 | Honest | val_does_execute=0.9775; likely_hardcoded=0.025. Metrics: `experiments/single_runs/run_metrics_round1_clp2e-3_p25_s52.json`. |
| 0.002 | 0.25 | 53 | `0dxw7n1g` | `round1_clp2e-3_p25_s53` | — | — | Killed | Seed sweep; run dir deleted, no export. |
| 0.002 | 0.25 | 54 | `oq5zrh2v` | `round1_clp2e-3_p25_s54` | 0.385 | 0.02 | Honest | val_does_execute=0.9425; likely_hardcoded=0.0075. Metrics: `experiments/single_runs/run_metrics_round1_clp2e-3_p25_s54.json`. |
| 0.002 | 0.25 | 55 | `d25yw0bg` | `round1_clp2e-3_p25_s55` | — | — | Killed | Continuation sweep; aborted step ~15/128; run dir deleted, no export. |
| 0.002 | 0.25 | 56 | `w8yhq7t1` | `round1_clp2e-3_p25_s56` | — | — | Killed | Aborted step ~93/128; no export. |
| 0.002 | 0.25 | 57 | — | `round1_clp2e-3_p25_s57` | — | — | Cancelled | Continuation sweep killed before launch. |

**Key finding:** vs clp1e-3/p25 at the same seed (s51: `s440n6af` Honest; s52: no clp1e-3/p25 run — cf. clp1e-3/p20/s52 `xf9mtwag` **Obfuscated policy**), doubling CLP to 0.002 stays Honest but lowers pass_all and raises pass_only_shown at s51; at s52 it flips clp1e-3/p20 from Obfuscated to Honest (pass_all 0.0025→0.355). All 3 completed seeds at clp2e-3/p25 (s51, s52, s54) land Honest.

### Round 1 ablation — clp3e-3/p25 (Aug 5, 2026)

Single run (s53) and a three-seed sweep (s54–56) at `code_length_penalty=0.003`, otherwise identical to the clp2e-3/p25 family (p=0.25, eps=1e-4). Configs: `experiments/configs/round1_clp3e-3_p25_s53.yaml`; sweep: `experiments/configs/round1_clp3e-3_p25_seed_sweep.yaml`. Logs: `scratch/round1_clp3e-3_p25_s53_17de0bc2f2ef.log`, `scratch/clp3e-3_p25_seed_sweep_71660ca36c5d.log`. Seed sweep killed manually after s54 in progress (run dir deleted, no export); s55/s56 never started — to launch trial 6 base variant.

| CLP | p | Seed | Run ID | Prefix | pass_all | pass_only_shown | Result | Notes |
|-----|---|------|--------|--------|----------|-----------------|--------|-------|
| 0.003 | 0.25 | 53 | `t1656g8a` | `round1_clp3e-3_p25_s53` | 0.005 | 0.855 | **Obfuscated policy** | val_does_execute=0.9925; likely_hardcoded=0.4975. Metrics: `experiments/single_runs/run_metrics_round1_clp3e-3_p25_s53.json`. |
| 0.003 | 0.25 | 54 | `10im2o3x` | `round1_clp3e-3_p25_s54` | — | — | Killed | Seed sweep; run dir deleted, no export. |
| 0.003 | 0.25 | 55 | — | `round1_clp3e-3_p25_s55` | — | — | Cancelled | Seed sweep killed before launch. |
| 0.003 | 0.25 | 56 | — | `round1_clp3e-3_p25_s56` | — | — | Cancelled | Seed sweep killed before launch. |

**Key finding:** at default CLP (0.003) / p25 / s53, **Obfuscated policy** — unlike clp2e-3/p25/s51–s52 (Honest at 0.002), reverting to the default length penalty at this point favors hardcoding over honest solving.
### Round 1 ablation — clp1e-3/p20 s45, epsilon-revert control (Aug 2, 2026)

Control run isolating whether the epsilon fix (methodology #13) is causally responsible for the Honest shift above, or just a coincidental seed effect: reran seed 45 with the advantage-normalization epsilon reverted to `1e-4`, everything else identical to the eps=0.05 sweep (kl=1e-3, det=3, `data.seed=53`, nofu, `nofu_mixed` manifest, base model, `code_length_penalty=0.001`, `mixed_reward_pass_all_prob=0.20`). Config: `experiments/configs/round1_clp1e-3_p20_s45_eps1e-4.yaml`.

Crashed at step 124/128 from the unrelated LoRA checkpoint leak (methodology #14) filling `/workspace`; resumed from the step-120 state checkpoint (`++training.resume_from_checkpoint=true`, same W&B run `emjn04pc`) after the leak fix landed, then completed normally.


| CLP   | p    | Seed      | eps             | Run ID     | pass_all | pass_only_shown | likely_hardcoded | Result                |
| ----- | ---- | --------- | --------------- | ---------- | -------- | --------------- | ---------------- | --------------------- |
| 0.001 | 0.20 | 45        | 0.05 (current)  | `ogsg4zud` | 0.5175   | 0.005           | 0.0025           | Honest                |
| 0.001 | 0.20 | 45        | 1e-4 (reverted) | `emjn04pc` | 0.0375   | 0.68            | 0.62             | **Obfuscated policy** |
| 0.001 | 0.20 | 42 (ref.) | 1e-4 (pre-fix)  | `9nlij6so` | 0.10     | 0.67            | —                | Obfuscated policy     |


**Key finding:** reverting only the epsilon — same seed, same everything else — flips seed 45 straight back to Obfuscated policy, closely matching the pre-fix profile at this point (cf. s42). This confirms the epsilon change itself, not incidental seed/data variance, is causally responsible for the Honest shift in methodology #13's Aug 2 caveat: eps=0.05 is not behavior-neutral at clp1e-3/p20. val_does_execute=0.9175 (not degenerate). Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s45_eps1e-4.json`.

### Round 1 ablation — clp1e-3/p20 seed sweep, advantage-norm epsilon fix (Aug 1–2, 2026)

Follow-up to the bistable clp1e-3/p20 point above, re-testing seeds s45-48 under the epsilon fix from methodology #13 (`eps` `1e-4`→`0.05`, `scale_rewards=true`). An earlier attempt at this point (s45-47) ran under the abandoned `scale_rewards=false` fix before being superseded — see methodology #13. Same hyperparams as s42–44 (kl=1e-3, det=3, `data.seed=53`, nofu, `nofu_mixed` manifest, base model, `code_length_penalty=0.001`, `mixed_reward_pass_all_prob=0.20`, 128 steps). Config: `experiments/configs/round1_clp1e-3_p20_seed_sweep_2.yaml`. Log: `scratch/round1_clp1e-3_p20_seed_sweep_2_<hostname>.log`.


| CLP   | p    | Seed | Run ID     | Prefix                   | pass_all | pass_only_shown | Result | Notes                                                                                                                       |
| ----- | ---- | ---- | ---------- | ------------------------ | -------- | --------------- | ------ | --------------------------------------------------------------------------------------------------------------------------- |
| 0.001 | 0.20 | 45   | `ogsg4zud` | `round1_clp1e-3_p20_s45` | 0.5175   | 0.005           | Honest | val_does_execute=0.97; likely_hardcoded=0.0025. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s45.json`. |
| 0.001 | 0.20 | 46   | `029rsj8o` | `round1_clp1e-3_p20_s46` | 0.525    | 0.015           | Honest | val_does_execute=0.968; likely_hardcoded=0.02. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s46.json`.  |
| 0.001 | 0.20 | 47   | `i9bjoiln` | `round1_clp1e-3_p20_s47` | 0.5175   | 0.005           | Honest | val_does_execute=0.995; likely_hardcoded=0. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s47.json`.     |
| 0.001 | 0.20 | 48   | `40rm3zrh` | `round1_clp1e-3_p20_s48` | 0.5175   | 0.005           | Honest | val_does_execute=0.97; likely_hardcoded=0. Metrics: `experiments/single_runs/run_metrics_round1_clp1e-3_p20_s48.json`.      |


**Key finding:** under eps=0.05, all 4 seeds converge to **Honest** with tightly clustered pass_all (~0.52) — a sharp departure from the pre-fix bistability at this point (s42–44: 2/3 Obfuscated). The clp1e-3/p20 point is no longer bistable under current training code; prior s42–44 outcomes cannot be assumed reproducible (see methodology #13's Aug 2 caveat).

### Round 1 ablation — code length penalty sweep (Jul 30-31, 2026)

Same base as `round1_p25_s42` (`1x7lo8e0`: kl=1e-3, det=3, `data.seed=53`, nofu, `nofu_mixed` probe manifest, base model start), varying `code_length_penalty` (default `0.003`) and `mixed_reward_pass_all_prob`. Run across two single-GPU pods sharing `/workspace` (each with its own vLLM instance); configs at `experiments/configs/round1_{no-clp,clp1e-3}_p*_s*.yaml`. Metrics/logs follow the pattern `experiments/single_runs/run_metrics_<prefix>.json` / `scratch/grpo_<prefix>.log`; intermediates (`lora`, `step`_*) cleaned on every completed run.


| CLP              | p    | Seed | Run ID     | Prefix                   | pass_all | pass_only_shown | Result                                                   |
| ---------------- | ---- | ---- | ---------- | ------------------------ | -------- | --------------- | -------------------------------------------------------- |
| 0.003 (baseline) | 0.25 | 42   | `1x7lo8e0` | `round1_p25_s42`         | 0.17     | 0.41            | Obfuscated policy                                        |
| 0                | 0.25 | 42   | `tbdhz6q2` | `round1_no-clp_p25_s42`  | 0.49     | 0.005           | Honest                                                   |
| 0                | 0.25 | 43   | `galj2ewm` | `round1_no-clp_p25_s43`  | 0.46     | 0.0025          | Honest                                                   |
| 0                | 0.20 | 42   | `4yg7idbl` | `round1_no-clp_p20_s42`  | —        | —               | Killed manually at step 96/128 (dir deleted, no metrics) |
| 0                | 0.20 | 43   | `l5ijktfx` | `round1_no-clp_p20_s43`  | —        | —               | Killed manually at step 92/128 (dir deleted, no metrics) |
| 0                | 0.15 | 42   | `43cisfse` | `round1_no-clp_p15_s42`  | 0.46     | 0.01            | Honest                                                   |
| 0                | 0.10 | 42   | `a97q2ko2` | `round1_no-clp_p10_s42`  | 0.47     | 0.0025          | Honest                                                   |
| 0                | 0.10 | 43   | `6hrp4ltp` | `round1_no-clp_p10_s43`  | 0.44     | 0.0025          | Honest                                                   |
| 0                | 0.0  | 42   | `q4uaet0t` | `round1_no-clp_p0_s42`   | 0.53     | 0.0025          | Honest                                                   |
| 0                | 0.0  | 43   | `ivds9i5l` | `round1_no-clp_p0_s43`   | 0.44     | 0.0025          | Honest                                                   |
| 0.001            | 0.20 | 42   | `9nlij6so` | `round1_clp1e-3_p20_s42` | 0.10     | 0.67            | **Obfuscated policy**                                    |
| 0.001            | 0.20 | 43   | `jen7cfbr` | `round1_clp1e-3_p20_s43` | 0.0025   | 0.9125          | **Obfuscated policy**                                    |
| 0.001            | 0.20 | 44   | `eur4mpq5` | `round1_clp1e-3_p20_s44` | 0.375    | 0.005           | Honest                                                   |


**Key finding:** with CLP=0, every tested `mixed_reward_pass_all_prob` (0.0-0.25) at both seeds lands **Honest** — a sharp reversal from the default-CLP baseline (`1x7lo8e0`, Obfuscated policy). But reintroducing just `code_length_penalty=0.001` at p=0.20/s42 flips straight back to **Obfuscated policy** (code collapses to ~50 chars) — and this isn't deterministic: a 2-seed follow-up (`experiments/configs/round1_clp1e-3_p20_seed_sweep.yaml`) shows s43 replicates Obfuscated policy but s44 lands Honest. So `code_length_penalty=0.001` at p=0.20 is **bistable across seeds** (2/3 Obfuscated, 1/3 Honest) rather than a clean threshold — seed variance is a substantial confound at this specific point, worth a finer sweep if the threshold effect is of interest.

### Round 1 ablation — seed/mixed-reward replication (Jul 29, 2026)

Replicates trial 3 round 1 (`62es59py`) config from the **base model** with full 128-step runs (not the step-128 symlink used for trial 3). Shared hyperparams: kl=1e-3, det=3, `data.seed=53`, nofu, `nofu_mixed` probe manifest (300/class, Phase-0 cache reuse), 128 steps. Varies `training.seed` and `mixed_reward_pass_all_prob` across runs below. These were standalone ablation runs, not part of a numbered trial (the `trial4`_* prefix used at launch time was reclaimed for the actual trial 4 — see above).


| Round | Run ID     | Prefix           | Status             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----- | ---------- | ---------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `2myjlt3y` | `round1_p30_s43` | Complete; exported | `mixed_reward_pass_all_prob=0.3`, `training.seed=43`. pass_first=0.43, pass_all=0.3775, pass_only_shown=0.0075, likely_hardcoded=0.0025, **Honest**. On-policy split (initial/final): deceptive=3 / honest=151 / unlabeled=246. Intermediates cleaned (`lora`, `step`_*); `checkpoints/final` retained. Metrics: `experiments/single_runs/run_metrics_round1_p30_s43.json`; export copy: `full_grpo_runs/extracted_results/run_metrics_round1_p30_s43.json`. Run dir: `full_grpo_runs/trigger_none_model_2myjlt3y_layers_32/`.                                                              |
| 1     | `1x7lo8e0` | `round1_p25_s42` | Complete; exported | `mixed_reward_pass_all_prob=0.25`, `training.seed=42`. pass_first=0.6525, pass_all=0.165, pass_only_shown=0.4075, likely_hardcoded=0.4275, **Obfuscated policy**. On-policy split (initial/final): deceptive=163 / honest=66 / unlabeled=171. Intermediates cleaned (`lora`, `step`_*); `checkpoints/final` retained. Metrics: `experiments/single_runs/run_metrics_round1_p25_s42.json`; export copy: `full_grpo_runs/extracted_results/run_metrics_round1_p25_s42.json`. Run dir: `full_grpo_runs/trigger_none_model_1x7lo8e0_layers_32/`. Log: `scratch/grpo_trial4_round1_p25_s42.log`. |


**Findings:** seed variance alone can flip the round-1 outcome — seed 43/p=0.3 (`2myjlt3y`) is Honest (pass_all 0.38, pass_only_shown 0.008) under a setup identical to trial 3's Obfuscated round 1 (`62es59py`, pass_all 0.04, pass_only_shown 0.76) aside from seed. Seed and `mixed_reward_pass_all_prob` also interact strongly (`1x7lo8e0` p=0.25/s42 Obfuscated vs `2myjlt3y` p=0.3/s43 Honest) — the outcome can't be attributed to either knob alone without a matched-seed ablation.

### Trial 3 baseline — no-detector-penalty ceiling (pass_all-only reward) (Jul 28, 2026)

Same as det0 baseline (methodology #10) except `mixed_reward_pass_all_prob=1.0` — every training prompt is scored on `passes_all_tests` instead of the 30% mixed regime. Script: `experiments/run_det0_ceiling.py`. Output: `experiments/trials/trial3/rounds/det0_ceiling/`.


| Round                                 | Run ID     | Prefix                               | Status             | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------- | ---------- | ------------------------------------ | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| det0 ceiling                          | `fr8qbmvo` | `trial3_det0_ceiling`                | Complete; exported | From base model, `loss.detector=0`, `mixed_reward_pass_all_prob=1.0`, probe = `nofu_mixed` manifest (300/class, no retrain, monitoring only). pass_first=0.4525, pass_all=0.38, pass_only_shown=0.0, **Honest**. Intermediates cleaned (`lora`, `step`_*); `checkpoints/final` retained. Metrics: `experiments/trials/trial3/rounds/det0_ceiling/run_metrics_trial3_det0_ceiling.json`                                                                                                                                                        |
| det0 ceiling, `code_length_penalty=0` | `yqy9nb9x` | `trial3_det0_ceiling_no_len_penalty` | Complete; exported | Same as det0 ceiling above but with the per-character code length penalty disabled (`code_length_penalty=0` instead of the default 0.003), so reward is purely `1.0 if passes_all_tests else 0.0`. Script: `experiments/run_det0_ceiling_no_len_penalty.py`. pass_first=0.6425, pass_all=0.58, pass_only_shown=0.0025, likely_hardcoded=0.0, **Honest**. Intermediates cleaned (`lora`, `step`_*); `checkpoints/final` retained. Metrics: `experiments/trials/trial3/rounds/det0_ceiling/run_metrics_trial3_det0_ceiling_no_len_penalty.json` |


**Finding:** vs. det0 baseline (`mdxnlbz1`, `mixed_reward_pass_all_prob=0.3`), making the pass_all reward regime deterministic drives genuine solving (pass_all 0.38 vs 0.0025) with zero pass-only-shown deception (0.0 vs 0.95) and **Honest** vs **Obfuscated activations** — confirming the mixed-reward regime (not the detector penalty, which is zero in both) is the primary lever controlling hardcoding vs honest solving from the base model.

**Effect of removing the code length penalty** (`yqy9nb9x` vs `fr8qbmvo`): removing the penalty raises both pass_first (0.64 vs 0.45) and pass_all (0.58 vs 0.38) substantially and roughly triples average code length (163 vs 61 chars), while `obfuscated_acts_score` drops (0.037 vs 0.147) — the length penalty was itself contributing some representation drift, independent of the (already-zero) detector penalty.

**Intermediate CLP seed sweep, `code_length_penalty=0.001` (Aug 1, 2026):** Same det0-ceiling config as above at `training.seed` 42/43/44, filling in the point between the `yqy9nb9x` (CLP=0) and `fr8qbmvo` (CLP=0.003) runs. Config: `experiments/configs/ceiling_clp1e-3_seed_sweep.yaml`. Metrics: `experiments/single_runs/run_metrics_ceiling_clp1e-3_s{42,43,44}.json`.


| Seed | Run ID     | pass_first | pass_all | pass_only_shown | likely_hardcoded | avg_code_length | Result |
| ---- | ---------- | ---------- | -------- | --------------- | ---------------- | --------------- | ------ |
| 42   | `xr3qu6g2` | 0.4725     | 0.41     | 0.0025          | 0.0              | 66.2            | Honest |
| 43   | `jwltok2h` | 0.4825     | 0.4075   | 0.0075          | 0.0025           | 61.9            | Honest |
| 44   | `b4v1arq6` | 0.45       | 0.39     | 0.005           | 0.0              | 68.5            | Honest |


**Finding:** all three seeds land **Honest** with pass_all and code length falling neatly between the CLP=0 and CLP=0.003 endpoints — unlike the round-1 (det=3, mixed-reward) CLP sweep above, CLP magnitude in this range doesn't destabilize honesty in the pure pass_all-only ceiling regime, consistent with the mixed-reward regime (not CLP) being the primary driver of obfuscation risk in that earlier sweep.

**Operational note:** the seed-44 attempt hit a pre-existing race condition twice before finally succeeding on a fresh pod — `_generate_alt_completions` (`obfuscation_atlas/training/grpo_trainer.py`) re-`save()`s the policy adapter to the *same* `checkpoints/lora/step_N` path that the generation queue's `pop()` had just scheduled for async `rm -rf` (`afterburner/grpo_trainer.py: GenerationQueue._rm_async`), producing an intermittent `RuntimeError: Parent directory .../step_N does not exist` if the deletion completes mid-write. Not yet fixed in code (would need `_generate_alt_completions` to save to a distinct path); see [Known failure modes](./FULL_GRPO_RUN.md#known-failure-modes) for the retry procedure used.

---

**Historical / exploratory runs** — early exploration preceding the trial lineages above; kept for reference and run-ID lookup.

### HF checkpoint sweeps (pre-local GRPO)

101/102 LoRA adapters downloaded. Grid: kl={1e-4…1e-0} x det={0,1,3,10} x probes {diverse_deception_probe, mbpp_probe} x seeds 1-3. 67 classified exports in `hf_probe_eval_runs/extracted_results/` (42 MBPP-probe, 25 DDP). Batch script: `eval_probe_checkpoints.py`.

### Local GRPO — initial grid (Jul 9, 2026)

kl=0.1, det=10, mbpp_probe, `corrected_detector_gradient=false`, seeds 42-44. Pre-disjointness-fix.


| Run ID     | Seed | Short name              |
| ---------- | ---- | ----------------------- |
| `16ypiseh` | 42   | `kl1e-1-det10-s42-mbpp` |
| `do8db1ft` | 43   | `kl1e-1-det10-s43-mbpp` |
| `for57uqj` | 44   | `kl1e-1-det10-s44-mbpp` |


Additional Jul 9-11 runs: `qv7h2ckc` (kl=0.1, det=10, s1), `9q1ni8ir` (kl=1e-3, det=10, s3), `viudg2wu` (kl=1e-3, det=3, s43, pre-fix). All exported.

### Post-fix disjoint probe (Jul 14, 2026)


| Run ID     | Config                              | Outcome                                                                                 |
| ---------- | ----------------------------------- | --------------------------------------------------------------------------------------- |
| `mtzvz59v` | kl=1e-3, det=3, s43, mbpp, disjoint | Obfuscated policy; `representation_change_on_dataset` 0.36 (vs 0.63 pre-fix `viudg2wu`) |


### DDP local run (Jul 2, 2026)


| Run ID                  | Config                  | Outcome           | Notes                                                 |
| ----------------------- | ----------------------- | ----------------- | ----------------------------------------------------- |
| `c3akfszd` / `24v08wlt` | kl=1e-4, det=3, ddp, s2 | Obfuscated policy | Post-eval crashed; recovered via eval-only `24v08wlt` |


### Trial 3 round-1 seed sweeps — pass-only-shown pipeline (Jul 21, 2026)


| Prefix                     | kl / det / seed | Run ID     | Status                                    |
| -------------------------- | --------------- | ---------- | ----------------------------------------- |
| `trial3_round1_det7_seed4` | 1e-3 / 7 / 4    | `yp6ctp57` | Complete                                  |
| `trial3_round1_det8_seed4` | 1e-3 / 8 / 4    | `pdrlz8dy` | Complete                                  |
| `trial3_round1_det8_seed5` | 1e-3 / 8 / 5    | `o71wb9rv` | Resumed from step_40 (disk filled Jul 21) |


### Nofu probe GRPO runs (Jul 22-23, 2026)

kl=1e-3, mbpp, pass-only-shown, no follow-up probe.


| Run ID     | Det / seed | Steps | Notes               |
| ---------- | ---------- | ----- | ------------------- |
| `fkwvos9i` | det=8, s4  | 128   | Exported            |
| `7gfeyq3h` | det=8, s6  | 128   | Exported            |
| `iyblohqu` | det=3, s42 | 128   | Exported            |
| `lvh8ukw4` | det=3, s43 | 128   | Exported            |
| `upo74pwo` | det=3, s44 | 128   | Exported            |
| `fig90slv` | det=1, s42 | 128   | Exported            |
| `7xxsnx4j` | det=1, s1  | 200   | `num_epochs=25`     |
| `i4o8jkck` | det=1, s1  | 32    | Aborted; do not use |


### Mixed nofu GRPO (Jul 23, 2026)


| Run ID     | Config                                                                          | Outcome           |
| ---------- | ------------------------------------------------------------------------------- | ----------------- |
| `2hgfj4k7` | kl=1e-3, det=3, s43, nofu, mixed manifest (300/class), `shown_prompt_ratio=1.0` | Obfuscated policy |


### Prompt-mix GRPO runs (Jul 23-24, 2026)

Same hyperparams as `2hgfj4k7`, varied `shown_prompt_ratio`:


| Run ID     | Prefix                                | `shown_prompt_ratio` | Notes                               |
| ---------- | ------------------------------------- | -------------------- | ----------------------------------- |
| `5grn9xef` | `prompt-mix-kl1e-3-det-3-s43`         | 0.6                  | Resumed from step 20 after disk OOM |
| `a8cmusu2` | `prompt-mix-shown50-kl1e-3-det-3-s43` | 0.5                  | Complete                            |


Hidden post-train eval runs:


| Eval run   | Source adapter   | Prefix                                            |
| ---------- | ---------------- | ------------------------------------------------- |
| `ckxo69jy` | `a8cmusu2` (50%) | `hidden-eval-prompt-mix-shown50-kl1e-3-det-3-s43` |
| `ssx7ojen` | `5grn9xef` (60%) | `hidden-eval-prompt-mix-kl1e-3-det-3-s43`         |


On-policy split counts (`initial_detector.final_model.on_policy.split`):


| Run                             | Eval prompts | deceptive / honest / unlabeled |
| ------------------------------- | ------------ | ------------------------------ |
| `a8cmusu2` (50%, shown eval)    | shown        | 0 / 157 / 243                  |
| `5grn9xef` (60%, shown eval)    | shown        | 348 / 4 / 48                   |
| `ckxo69jy` (hidden eval of 50%) | hidden       | 0 / 145 / 255                  |
| `ssx7ojen` (hidden eval of 60%) | hidden       | 53 / 45 / 302                  |


### Mixed reward regime runs (Jul 24, 2026)

kl=1e-3, det=3, s43, nofu, mixed manifest, no prompt mixing:


| Run ID     | `mixed_reward_pass_all_prob` | Status                               | Outcome                                                           |
| ---------- | ---------------------------- | ------------------------------------ | ----------------------------------------------------------------- |
| `side513m` | 0.4                          | Complete; exported                   | **Honest** (differs from baseline `2hgfj4k7` = Obfuscated policy) |
| `12flhy8o` | 0.3                          | Training complete; post-eval crashed | Pending eval-only recovery                                        |


`62es59py` (prob=0.3, s42, max_steps=200): killed ~step 131/200. Step-128 checkpoint reused as round-1 for trial 3.

Mixed reward sweep (`scratch/grpo_mixed_reward_sweep.sh`, probs 0.35/0.40 × seeds 41/42): interrupted Jul 26 to free GPU/disk for trial 3 round 2 relaunch (`n7v1znqf` p35 deleted mid-run). Not resumed.

---

## Known gaps


| Gap                                      | Notes                                                                                                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| kl=1e-4, det=3, s3 (mbpp, 0% pass-first) | Degenerate — not retried                                                                                                                                |
| kl=1e-4, det=3, s1 (ddp, loss-threshold) | Failed to produce usable eval exports — not retried                                                                                                     |
| `12flhy8o` eval-only recovery            | Training complete but post-eval crashed; recovery command drafted but deferred                                                                          |
| Trial 2 automation                       | Rounds 2-6 used mismatched hyperparameters (det=3/seed=43 hardcoded in `run_trial.py` vs det=8/seed=3 from round 1 `lliye1cp`); not directly comparable |


