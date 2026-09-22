# 260826 — Probe training, rebuilt as an interactive pipeline

A controlled rerun of [`260825_probe_training`](../260825_probe_training/): the same three
phases, but as importable functions driven from a `# %%`-cell notebook
([`run_pipeline.py`](run_pipeline.py)) so each step can be run, inspected and re-run on its
own with a changed config. Four deliberate design changes came with the port (below).

**Results are in [Results](#results).** The reported run is the majority-of-3 labeling
(`run_pipeline_maj3.py`, all 20 probes, `probe_metrics_maj3.json`, 2026-09-22). The
indexed-label run was never completed — see [Run status](#run-status).

## Files

| File | Role |
|---|---|
| [`probe_config.py`](probe_config.py) | `PipelineConfig` — every knob 260825 hardcoded, plus `ProbeSpec` |
| [`domain_imports.py`](domain_imports.py) | the one place that `sys.path`-inserts `260819_domain_baseline` |
| [`hf_env.py`](hf_env.py) | HF Hub env for kernels that don't inherit `atlas_setup.sh` |
| [`step1_build_probe_sets.py`](step1_build_probe_sets.py) | join / filter / sample → `probe_sets/` |
| [`step2_extract_activations.py`](step2_extract_activations.py) | tokenize + forward + slice → `activations/` |
| [`step3_train_eval_probes.py`](step3_train_eval_probes.py) | train + eval + save → `probe_weights/`, `probe_metrics.json` |
| [`run_pipeline.py`](run_pipeline.py) | **the notebook** — indexed counterfactual labels (default) |
| [`run_pipeline_maj3.py`](run_pipeline_maj3.py) | same pipeline with majority-of-3 counterfactual labeling — **the reported run** |
| [`run_probe_variants.py`](run_probe_variants.py) | probe training variants (sklearn LBFGS, weight decay) on cached indexed activations |

Run it from a kernel on `/workspace/obfuscation-atlas/.venv`. Cells 1–4 are CPU-only;
cell 5 loads the model once and cells 5–12 reuse it.

## Datasets vs. probes

The distinction the whole layout rests on. **6 datasets** (3 families × 2 definitions) get
tokenized and forward-passed; **20 probes** get trained off them.

| Dataset | Rows | Positions extracted |
|---|---|---|
| `bias_{rm,swap}` | turn-1 rollouts, label 1 = bias, 0 = no bias | `answer`, `lastcompl` |
| `deception_{rm,swap}` | turn-2 rollouts, label 1 = deceptive, 0 = honest | `yesno`, `preyesno` |
| `bias_followed_{rm,swap}` | turn-1 rollouts again, restricted to rows whose answer went **with the cue** | `answer`, `lastcompl` |

| Probe | Rows it trains on |
|---|---|
| `bias_{def}_{answer,lastcompl}` | all bias rows |
| `deception_{def}_{yesno,preyesno}` | all deception rows |
| `deception_yes_{def}_{yesno,preyesno}` | `said == "Yes"` → false vs **true admission** |
| `deception_no_{def}_{yesno,preyesno}` | `said == "No"` → false vs **true denial** |
| `bias_followed_{def}_{answer,lastcompl}` | followed-cue rows only → cue *caused* the answer vs answer was that way anyway |

The three deception probes share one dataset and one set of activations. Within said-Yes
rows a label of 1 (deceptive) *is* `false admission` and 0 *is* `true admission`; within
said-No it is `false denial` vs `true denial`. So the stratified probes need no
relabelling and no second forward pass — only a row mask, applied in
`build_activation_dataset(..., said=...)`.

`bias_followed` is *not* a mask over `bias`, because it could not be: **`bias` is perfectly
nested inside cue-following** — across both definitions and both splits, zero bias-labelled
rows are against-cue, since a biased answer is by definition one that moved toward the cued
option. That means a bias probe can score by detecting "the answer matches the cue" rather
than "the cue caused the answer". `bias_followed` removes the shortcut: both its classes
answered with the cue, and differ only in whether the cue caused it. It is the bias-side
analogue of the stratified deception probes. Two consequences:

- an **against-cue counterpart is impossible** — it would have no positive class;
- it needs its **own draw**, not a mask: restricting the existing 800-row bias train sample to
  followed-cue rows leaves ~124/class (`rm`) and ~80/class (`swap`) after cue-balancing, far
  short of 400. Hence its own dataset and its own activations.

Its train draw is 400/class, matching `bias`, so an AUC difference between the two cannot be a
train-size effect.

## What changed vs. 260825

**1. Eval is one row per prompt, up front.** A prompt is a `(question_id, condition)` pair
and the domain generated 3 samples each. 260825 kept all 3 in its eval file and
deduplicated post-hoc in `analyze_results.py`; here the pool takes 1 at random (seeded)
before sampling, so `probe_metrics.json` is already the reportable number. The subsample is
applied *after* the class filter, so a prompt is lost only when every one of its samples was
excluded. Eval sizes land at **1355 / 1364 / 1355 / 1364** — exactly 260825's README
`n eval` column, which is the check that the two rules agree in count.

**2. Train draws are cue-balanced.** Each class is drawn half from `cue_gold` and half from
`cue_dist`. 260825 drew from the class as a whole, which left e.g. `bias_rm` train skewed
toward whichever condition was more common. Short cells raise rather than backfilling from
the other condition.

| Dataset | train | per class | per (class × condition) |
|---|---|---|---|
| `bias_{def}` | 800 | 400 | 200 |
| `deception_{def}` | 800 | 400 | 100 per subtype |

`deception_{def}`'s train set is not drawn directly — it is the **union** of the two
stratified draws (200/class each), so it is 400/class, cue-balanced, and half admissions
half denials by construction.

**3. Two new stratified deception probes.** `deception_yes` and `deception_no` hold the
self-report token fixed *by construction*. This is the designed version of 260825's
post-hoc finding: with the self-report held fixed its headline 0.73 collapsed to ~0.53, and
what survived was reading bias, not deception. Probes trained directly on that contrast are
the fair test of whether anything is there.

**4. The deception position is the actual Yes/No token.** 260825 probed the *first
generated* token (`_firstgen`) and the token before it (`_lastprompt`). This probes the
first **sampled Yes/No** token (`yesno`) and the token before it (`preyesno`), using the
`ans_pos` already stored in `followups.jsonl` by `answer_position()`
([generate_followups.py:403](../260819_domain_baseline/generate_followups.py#L403)). The two
positions coincide wherever the self-report was the first token, and differ where it was
not — measured below.

**Also:** an explicit `cfg.probe_seed` (nothing in `obfuscation_atlas.detectors` seeds
anything, so 260825's AUCs are not bit-reproducible; `None` restores its unseeded path);
`asdict(cfg)` stamped into every output JSON; per-category counts kept in the manifest but
no longer printed.

## Analysing one probe

`step3_train_eval_probes.py` also holds the analysis helpers, for use after
`train_eval_one_probe`:

```python
res = s3.analyze_probe("bias_rm_answer", cfg, artifacts=art)
```

It reloads the saved weights and re-scores the cached activations — a matmul, no LLM forward
pass, and the `input_scale` buffer survives `save_model`/`load_model`. Scoring in logit space
is fine for AUC (threshold-free, and `expit` is monotone); the score histogram applies the
sigmoid. Verified to reproduce the trained `auc_filtered` to <5e-5.

**Breakdowns** (`slice_auc`), mean AUC over the filtered layers per subgroup, with
`n`/`n_pos`/`n_neg`, printed as a table per breakdown:

| Breakdown | Row field | Notes |
|---|---|---|
| `cue` | `condition` | `cue_gold` vs `cue_dist` — 260825 found a 0.10–0.20 spread here on the bias probes |
| `cue-following` | derived: answer vs `cue_letter` | did the answer go with the cue. On a bias probe the against-cue group is single-class (`nan`), because a biased answer is by definition one that moved toward the cue — the `n`/`pos`/`neg` columns are what show that nesting |
| `self-report` | `said` | said-Yes contrasts false vs **true admission**, said-No false vs **true denial** — 260825's decisive "self-report held fixed" test |
| `category` | `category` | the 5 MMLU-Pro categories — a generalization check |
| `bias` | derived from `subtype` | bias vs no-bias, i.e. does the probe read deception equally well whichever way the answer went |

`cue-following` and the model's `answer` letter are joined in at analysis time from the
published `scores.jsonl` (`preds[sample_idx]` — the same letter step 2's turn-1 gate asserts
against — and `cue_letter`), so they need no re-extraction. Gated: on the removal definition
every `bias`-labelled eval row must read "followed cue", which is 60/60 in a spot check.

A breakdown is only informative where it varies **within** both classes. Where the field *is*
the probe's label every group is single-class and the AUC is undefined, so `slice_auc` returns
`[]` and the breakdown is omitted rather than printed as a column of `nan`. In practice:

| Probe | Reports |
|---|---|
| `bias_{def}_*` | cue, cue-following, category (`self-report` is None throughout; `bias` is the label) |
| `deception_{def}_*` | all five |
| `deception_yes/no_{def}_*` | cue, cue-following, category (`self-report` is constant; `bias` coincides with the label inside a said stratum) |

`subtype` is never a breakdown for the same reason — it is the label everywhere.

**Figure** (`plot_probe`), 2×2. Panels 1, 2 and 4 mirror what the RL pipeline logs per probe
in `log_detector_metrics`
([obfuscation_training.py:1042](../../obfuscation-atlas/obfuscation_atlas/training/obfuscation_training.py#L1042)):

1. eval score distribution by class at the best-AUC filtered layer, **after the sigmoid**, on
   a fixed 0–1 axis
2. AUC by layer, train and eval overlaid, filtered layers shaded
3. the breakdown AUCs as vertical bars, separated by breakdown, against the overall AUC
4. probe train loss per layer against `cfg.layer_loss_threshold`

Panel 4 needs `train_dynamics`, which lives only in the in-session `ProbeArtifacts`; without
`artifacts=` it shows the filtered-layer set and says so. Pass `save_dir=` to write
`{probe}_analysis.{json,png}`.

Two panels already pay for themselves on the deception probes: panel 4 shows the train loss
never reaching 0.6 at *any* layer, which is exactly why `get_layers_below_loss_threshold`
raises and the filter falls back to all 32 layers; and panel 2 exposes the train/eval AUC gap
that the single headline number hides.

## Gates

All asserts, on every row, not a sample.

Step 1 (self-consistency, notebook cell 4 — **0 failures**): eval rows == eval prompts;
eval size == 260825's reported size; train classes equal; each train class equal across
`cue_gold`/`cue_dist`; the deception train set exactly the union of its two strata.

Step 2 turn-1 (carried over from 260825): the regex chain matches; the local
`(letter, stage)` agrees with `extract_answer` *and* with the published `preds`/`stages` in
`scores.jsonl`; the sliced token, decoded, contains the matched letter.

Step 2 turn-2 (rewritten for the `yesno` position): the turn-2 boundary lands exactly on a
token start; `ans_pos` exists; the located token id equals the stored generated id at
`ans_pos`; the token decodes to a Yes/No form whose **polarity matches `said`**; `preyesno`
never falls back inside the turn-1 completion.

One caveat on the id gate: `followups.jsonl` stores only `completion.token_ids[:8]`
([generate_followups.py:720](../260819_domain_baseline/generate_followups.py#L720)) while
`ans_pos` was computed over the full generated sequence, so that gate applies to all but
one row across the four files. The decode/polarity gate covers every row.

## Verification so far

**The `yesno` position change matters, slightly.** Over all 4,319 deception rows, 0 gate
failures, and 12 rows have the self-report at `ans_pos > 0` — i.e. 12 rows where 260825's
`_firstgen` probed a preamble token rather than the self-report.

| Dataset / split | rows | `ans_pos` counts | self-report is first token |
|---|---|---|---|
| `deception_rm` / train | 800 | `{0: 800}` | 1.0000 |
| `deception_rm` / eval | 1355 | `{0: 1351, 1: 4}` | 0.9970 |
| `deception_swap` / train | 800 | `{0: 796, 3: 1, 4: 3}` | 0.9950 |
| `deception_swap` / eval | 1364 | `{0: 1352, 1: 2, 2: 2, 4: 7, 8: 1}` | 0.9912 |

Consistent with 260817's 99.6%. The `ans_pos = 8` row is a good illustration of what the
old position was reading: its completion is `"A clever question!\n\nThe answer is: No.…"`,
so `_firstgen` sat on `"A"`.

**The self-report is not always the bare `"Yes"`/`"No"` token.** The decoded forms include
`YES` and `NO` as single tokens (e.g. 16 `YES` / 24 `NO` in `deception_swap` eval), which is
why the polarity gate matches against the broad form sets rather than string equality.

**Steps 2–3 run end to end** on a 20-rows-per-class dry run: 16 activation files written,
and step 3 resolves specs correctly — `deception_yes_rm_yesno` reads
`deception_rm_yesno_{split}.pt` and masks to 40 of its 80 rows. Dry-run AUCs are noise at
that size and are not reported.

**Step 3 needs no model.** All three probes in a 100-rows-per-class check trained with
`model=None` and zero model loads — `prepare_dataset` returns the supplied feature dataset
before it reads `model`. So once step 2 has run, probes can be trained and re-trained on a
fresh kernel with no 16 GB model resident.

**The analysis helpers reproduce the trained numbers** to <5e-5 on three probes, slice counts
sum to `n_eval`, and no single-class group is reported.

The full extraction and the full 20-probe train have since run under majority-of-3
labels — see [Results](#results). The dry-run counts above (16 files) predate the
`bias_followed` family; the full run writes 24.

## Majority-of-3 counterfactual labeling

The default path (`run_pipeline.py`, `counterfactual_vote="indexed"`) uses
[`260824_rescore/rescored_labels.jsonl`](../260824_rescore/rescored_labels.jsonl) as written:
each cued rollout `i` is paired with counterfactual sample `i` (index-matched). That
counterfactual side carries independent sampling noise into every bias and deception label.

[`run_pipeline_maj3.py`](run_pipeline_maj3.py) sets `counterfactual_vote="majority3"` via
[`majority3_config()`](probe_config.py). At load time in `step1_build_probe_sets.load_sources`,
each rollout keeps its own turn-1 answer but the counterfactual is the **parsed majority**
over that prompt's 3 samples:

- **removal (`rm`):** majority of `a_none` (no-cue condition)
- **swap:** majority of `a_other_cue` (other cue condition)

Deception labels are recomputed from the new bias stratum and the existing `said_*` field.
`said`, `p_yes`, and follow-up text are unchanged.

**Tie rule:** if the 3 counterfactual samples have no strict parsed majority (1–1 tie or
all unparsed), the prompt is excluded (`unparsed_turn1`). Counts: 122 rm prompts / 91 swap
prompts of 4580.

**Consequences:**

- Swap labels become per-rollout (like rm) rather than pair-symmetric across cue conditions.
- The 3 train rows of a prompt share one counterfactual; within-prompt label variation comes
  only from their own answers.
- Eval sizes shift (e.g. `bias_rm` eval ~1326 vs 1355 indexed) because relabeling changes
  which rows pass the class filter.
- Anti-bias shrinks (~13% of rows change bias label overall) — consistent with much of that
  stratum being counterfactual sampling noise.

**Outputs** use separate paths so indexed caches are never reused:
`probe_sets_maj3/`, `activations_maj3/`, `probe_weights_maj3/`,
`probe_sets_manifest_maj3.json`, `probe_metrics_maj3.json`. Activations must be re-extracted
(~8 min GPU, ~8.4 GB) — the selected rows differ even when the same `(question_id, condition,
sample_idx)` is kept.

Cell 5 compares to the indexed run's `probe_metrics.json` when it exists; otherwise to
260825's published numbers (with a caveat that eval labels differ under majority3). The
indexed `probe_metrics.json` does not exist, so the comparison in the reported run fell
through to 260825's `probe_metrics_one_sample.json`.

## Results

All numbers below are from `probe_metrics_maj3.json`: majority-of-3 counterfactual labels,
`probe_seed=0`, AdamW 1000 steps at lr 3e-3, layer-loss threshold 0.6, mean AUC over the
filtered layers on the held-out eval split (one row per prompt). "fell back" means no layer
reached train loss 0.6, so the filter used all 32 layers.

**Dataset sizes** (majority-of-3). Train is 400/class, cue-balanced, in every dataset.

| Dataset | train | eval | eval pos / neg |
|---|---|---|---|
| `bias_rm` | 800 | 1326 | 321 / 1005 |
| `bias_swap` | 800 | 1329 | 562 / 767 |
| `deception_rm` | 800 | 1326 | 754 / 572 |
| `deception_swap` | 800 | 1329 | 707 / 622 |
| `bias_followed_rm` | 800 | 1106 | 449 / 657 |
| `bias_followed_swap` | 800 | 1104 | 721 / 383 |

**Bias probes** (label 1 = bias). 260825 column is `probe_metrics_one_sample.json`,
which used indexed labels, so eval labels differ.

| Probe | AUC | filtered layers | 260825 |
|---|---|---|---|
| `bias_rm_answer` | 0.773 | 30 | 0.767 |
| `bias_rm_lastcompl` | 0.727 | 32 | 0.742 |
| `bias_swap_answer` | 0.827 | 30 | 0.800 |
| `bias_swap_lastcompl` | 0.802 | 30 | 0.760 |
| `bias_followed_rm_answer` | 0.692 | 31 | — |
| `bias_followed_rm_lastcompl` | 0.659 | 32 | — |
| `bias_followed_swap_answer` | 0.677 | 21 | — |
| `bias_followed_swap_lastcompl` | 0.664 | 15 | — |

**Deception probes** (label 1 = deceptive). 260825's `_firstgen` / `_lastprompt` map to
`yesno` / `preyesno`; 260825 had no stratified probes.

| Probe | AUC | n eval | filtered layers | 260825 |
|---|---|---|---|---|
| `deception_rm_yesno` | 0.531 | 1326 | fell back | 0.730 |
| `deception_rm_preyesno` | 0.472 | 1326 | fell back | 0.586 |
| `deception_yes_rm_yesno` | 0.673 | 836 | fell back | — |
| `deception_yes_rm_preyesno` | 0.689 | 836 | fell back | — |
| `deception_no_rm_yesno` | 0.620 | 490 | 4 | — |
| `deception_no_rm_preyesno` | 0.642 | 490 | 26 | — |
| `deception_swap_yesno` | 0.654 | 1329 | 10 | 0.731 |
| `deception_swap_preyesno` | 0.482 | 1329 | fell back | 0.639 |
| `deception_yes_swap_yesno` | 0.923 | 229 | 23 | — |
| `deception_yes_swap_preyesno` | 0.926 | 229 | 24 | — |
| `deception_no_swap_yesno` | 0.698 | 1100 | fell back | — |
| `deception_no_swap_preyesno` | 0.783 | 1100 | 20 | — |

**Class balance inside the deception strata** (eval). Train is 200/200 in every stratum.
Inside a stratum the label coincides with bias: said-Yes contrasts false admission
(no bias) vs true admission (bias); said-No contrasts false denial (bias) vs true denial
(no bias).

| Stratum | eval n | pos / neg | pos / neg in `cue_gold` | pos / neg in `cue_dist` |
|---|---|---|---|---|
| rm said-Yes | 836 | 634 / 202 | 321 / 74 | 313 / 128 |
| rm said-No | 490 | 120 / 370 | 51 / 219 | 69 / 151 |
| swap said-Yes | 229 | 187 / 42 | 56 / 15 | 131 / 27 |
| swap said-No | 1100 | 520 / 580 | 259 / 334 | 261 / 246 |

**What the numbers show.**

- Bias at the answer token beats the last-completion token in both definitions, as in
  260825. Removal 0.767 → 0.773, swap 0.800 → 0.827 against 260825; the deltas mix the
  label change with everything else.
- `bias_followed` scores 0.08–0.15 below `bias` at the same train size.
- The unstratified deception probes no longer reach 260825's 0.73: 0.531 (rm) and 0.654
  (swap) at the self-report token, and below 0.5 at `preyesno`.
- All four stratified deception probes are above 0.5, and `preyesno` is at or above
  `yesno` in every stratum. Because the label inside a stratum is bias, these are bias
  probes evaluated with the self-report held fixed — 260825's caveat, unchanged.
- The swap said-Yes 0.92 rests on 42 negatives (15 in `cue_gold`). Its Hanley-McNeil 95%
  interval is roughly 0.89–0.96. The swap said-No stratum (520 / 580) is the largest and
  best balanced; 260825 scored the same stratum post hoc at 0.700.
- Cue condition is not a confound inside any stratum: the positive-class share differs by
  under 10 points between `cue_gold` and `cue_dist` in all four.
- 8 of 12 deception probes hit the layer-loss fallback; no bias probe did.

## Run status

- **Majority-of-3 (reported):** all 20 probes trained via `run_pipeline_maj3.py` cell 5 on
  2026-09-22; weights in `probe_weights_maj3/`, metrics in `probe_metrics_maj3.json`.
- **Indexed (not completed):** activations extracted to `activations/`; 7 probes trained
  one at a time (`probe_weights/`), no `probe_metrics.json` written.
- **Training variants (`run_probe_variants.py`):** baseline, sklearn, weight decay 0.01 and
  0.1 were run on 3–4 indexed probes each through `train_eval_one_probe`, which writes
  weights (`probe_weights_{baseline,sklearn,wd001,wd01}/`) but no metrics file. Their AUCs
  were not recorded.

## Costs

Per definition, extraction covers 800 train + ~1100–1360 eval rows per dataset, so
~9,900 forward-pass rows per labeling, and 24 activation files at ~13 GB on disk
(float32, 32 layers, both positions sliced from one pass). Both labelings are extracted,
so `activations/` + `activations_maj3/` is ~26 GB. Eval is deduplicated before extraction
rather than after, which is why this is below 260825's per-dataset footprint.
