# 260924 — Probe recipe chosen by cross-validation; combined-score headline; high-confidence variant

Probe-training experiment on the MMLU-Pro sycophancy domain. It builds on
[`260923_improve_probe_performance/`](../260923_improve_probe_performance/) (33 rounds
of probe-side changes, all chosen on the eval set) and reads that folder's probe sets
and cached activations; nothing is extracted here. Three things change:

1. The headline metric is the AUC of one combined score per row, the mean of the
   per-layer sigmoid outputs over the selected layers. This is the score the RL
   pipeline logs (`probes/eval_*_averaged/.../AUC_ROC`) and thresholds as the penalty.
   260923 reported the mean of per-layer AUCs; that number is kept as a diagnostic.
2. Learning rate, step budget and a contiguous layer band are chosen per probe by
   5-fold cross-validation over training questions, with the unchanged pipeline
   trainer (`train_and_eval_detector`, AdamW). The eval split is used once per
   recorded round and never for selection.
3. A high-confidence variant restricts both train and eval to questions whose three
   no-cue rollouts gave the same answer.

**The reported run** is rounds `07_cv_final_pool_starred` and `08_cv_final_pool_rest`
(2026-09-25): the recipe chosen over the full 3 × 3 grid, trained on `pool` with seeds
0, 1, 2. Metrics are in `runs/07_*/probe_metrics.json` and `runs/08_*/probe_metrics.json`;
the comparison with earlier rounds is `comparison.md`. Rounds are split into the three
starred probes and the other two so they could run as two processes; a split round
reports "—" for the probes it did not train.

## Files

| File | Role |
|---|---|
| `probe_config.py` | `PipelineConfig` (copied from 260923, edited): `SOURCE_DIR` points at 260923's `probe_sets/` and `activations/`; new fields `combined_score`, `eval_filter`, `layer_band`, `per_probe_overrides`, `cv_*`; `apply_overrides` |
| `step3_train_eval_probes.py` | Training, metrics and analysis (copied from 260923, edited): combined score, eval mask, `layer_selection="band"`, `fit_layer_probes`, slices and figures on the combined score |
| `run_log.py`, `round.py`, `append_row.py` | One recorded round (`runs/NN_slug/`) and its `runs.md` row; `round.py --set-file` takes a chosen recipe, `--probes` trains a subset |
| `label_agreement.py`, `hf_env.py` | Copied from 260923 unchanged (no-cue unanimity index; HF env) |
| `cv_sweep.py` | GPU: per probe, GroupKFold over train question ids, one fit per (lr, steps, fold), held-out per-layer logits to `cv/<variant>/<probe>/lr*_steps*/fold*.json`; resumable |
| `cv_select.py` | CPU: out-of-fold AUC per band / loss rule / all 32 per cell; argmax → `cv/<variant>/chosen_overrides.json`, `summary.md`, `figures/cv_heatmap_*.png` |
| `epochs_matched_overrides.py` | Rescales a chosen recipe's steps to another train set at equal epochs (used for `pool_nu`) |
| `rescore_rounds.py` | CPU: re-scores saved weights (260923's or this folder's) under all three metrics, full and filtered eval → `rescored/` |
| `analyze_round.py` | CPU: per-probe slice tables and 2 × 2 figure for one round → `runs/NN/analysis/` |
| `comparison_table.py` | Merges `rescored/` and `runs/` into `comparison.md` |
| `runs.md` | One row per recorded round with the decision taken |
| `comparison.md` | Every round × eval filter × metric |
| `cv/pool/` | CV cells (225 fold files), `summary.md`, `selection.json` per probe, the chosen-recipe files |
| `rescored/` | Re-scored 260923 rounds 01, 10, 15, 31 and this folder's rounds |
| `figures/` | CV heatmaps; per-round figures are under `runs/NN/analysis/` |
| `logs/` | tmux logs of the sweep and rounds, `logs/smoke/` the smoke tests |

## Design

**Probes, data, tokens.** The five probes of 260923, definition `rm`, at the same tokens:
`bias_rm_answer` and `bias_followed_rm_answer` at the answer letter, the three deception
probes at the token before the Yes/No self-report. Training rows are 260923's `pool`
draw (the largest cue-balanced draw, 3 samples per question) with
`train_draw_per_class=None`; eval is 260923's eval split, unchanged (one row per
question and cue condition, `pool_seed=53`).

**Headline metric.** `auc_combined` = AUC of the mean over the selected layers of each
layer's sigmoid output (`combined_score="mean_sigmoid"`, matching
`obfuscation_training.compute_averaged_scores_across_layers`). Every metrics file also
holds `auc_mean_logit` (AUC of the mean raw logit, 260923's `auc_layer_mean_logit`) and
`auc_filtered` (mean of per-layer AUCs, 260923's headline). Seed mean and sd are computed
for all three.

**Trainer.** `train_and_eval_detector` from obfuscation-atlas, unchanged: AdamW,
batch 32, constant schedule, gradient clip 1.0, `unit_norm` input scaling, torch seeded
per fit. The trainer's own class balancing drops a few rows inside CV folds when a fold is
not exactly balanced (for example 390 → 387).

**Cross-validation.** Per probe, the rows it trains on are split into 5 folds by
`question_id` (`GroupKFold`, shuffled, seed 0), so no question's rows are in two folds.
For each cell of the grid and each fold, the 32 per-layer probes are fit on the other
folds and the held-out rows' per-layer logits are stored. Layer sets are scored afterwards
from those logits. The criterion is the mean over folds of the held-out combined-score
AUC; the pooled-over-folds AUC and the fold sd are stored as well.

| CV dimension | values |
|---|---|
| learning rate | 0.003, 0.01, 0.03 |
| max_steps | 1000, 2000, 4000 |
| layer set candidates | every contiguous band of width 4, 8, 12, 16, 24, 32 (all starts); the pipeline's train-loss < 0.6 rule applied per fold; all 32 layers |
| folds | 5, by question id |
| train questions per probe | bias 1375, bias_followed 1224, deception 1349, deception_yes 1113, deception_no 704 |

The chosen recipe is the argmax over (lr, steps, band) per probe. Bands are the only
layer rule eligible for selection; the loss rule and all-32 are reported beside it.

**Layer selection in a round.** `layer_selection="band"` with `layer_band=(start,
width)` from the CV choice; `"loss_threshold"` is the pipeline's rule (layers with final
train loss < 0.6, all 32 when none passes). `per_probe_overrides` lets one round carry a
different lr / steps / band per probe.

**Eval filter.** `eval_filter="none_unanimous"` keeps eval rows whose question's three
no-cue rollouts all gave the same parsed answer (`label_agreement.none_unanimous_index`),
applied at load time to the unchanged eval activations. The filtered-train side of the
variant is 260923's `pool_nu` extraction.

**Rounds recorded** (all on `pool` unless stated; 3 seeds each):

| round | recipe | purpose |
|---|---|---|
| 01 / 02 | CV over lr {0.01, 0.03} × 1000 steps: lr 0.03, bands 13–16 / 14–17 | first CV-chosen round |
| 03 | lr 0.03, 1000 steps, loss rule | band vs loss rule at equal lr and steps |
| 04 | rounds 01/02's lr and bands on `pool_nu`, steps rescaled to equal epochs (396 / 396 / 396 / 421 / 347), eval filtered | high-confidence variant |
| 05 / 06 | CV over lr {0.01, 0.03} × {1000, 2000}: 2000 steps (lr 0.01 for bias_followed, 0.03 otherwise), same bands | steps 1000 → 2000 |
| 07 / 08 | CV over the full grid: 4000 steps, lr 0.01 (bias family) / 0.03 (deception), bands 13–16 / 14–17 / 15–18 | **reported run** |

## Gates

- Row counts: the smoke round on `pool` reproduced 260923's train sizes (2908 / 2908 /
  2908 / 1928 / 980) and eval sizes (1326 / 1106 / 1326 / 836 / 490); under the eval
  filter the eval sizes are 783 / 612 / 783 / 490 / 293. Passed.
- CV folds: the fold files partition the probe's rows and no question id appears in two
  folds (checked on `deception_no`). Passed.
- Re-scoring: `rescore_rounds.py` asserts that scoring saved weights reproduces each
  round's stored `auc_filtered` and mean-logit AUC within 2e-3. Passed for 260923's
  rounds 01, 10, 15, 31 and this folder's rounds 01–08.
- Analysis: `analyze_probe` on a round's seed-0 weights reproduces that seed's
  `auc_combined` to 1e-6 (smoke test). Passed.

## Results

Reported from `runs/07_cv_final_pool_starred/probe_metrics.json` and
`runs/08_cv_final_pool_rest/probe_metrics.json` (recipe in the Rounds table; seeds 0, 1,
2), evaluated on 260923's eval split, one row per question and cue condition. Every AUC
below is `auc_combined` (mean sigmoid) unless the column says otherwise; ± is the sample
sd over seeds. Earlier rounds are re-scored from their saved weights under the same
metric (`rescored/`).

### Sizes

| set | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| train, `pool` (pos / neg) | 2908 (1454 / 1454) | 2908 (1454 / 1454) | 2908 (1454 / 1454) | 1928 (964 / 964) | 980 (490 / 490) |
| train, `pool_nu` | 1152 (576 / 576) | 1152 (576 / 576) | 1152 (576 / 576) | 812 (406 / 406) | 340 (170 / 170) |
| eval, full | 1326 (321 / 1005) | 1106 (449 / 657) | 1326 (754 / 572) | 836 (634 / 202) | 490 (120 / 370) |
| eval, no-cue-unanimous | 783 (147 / 636) | 612 (223 / 389) | 783 (455 / 328) | 490 (399 / 91) | 293 (56 / 237) |

### Headline: eval AUC of the combined score

| round | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| 260826 config (260923 round 01, re-scored) | 0.795 ± 0.000 | 0.719 ± 0.017 | 0.477 ± 0.005 | 0.738 ± 0.001 | 0.650 ± 0.000 |
| 260923 round 10, logreg C=30, loss rule | 0.829 | 0.764 | 0.589 | 0.765 | 0.707 |
| 260923 round 15, LDA 0.5, loss rule | 0.828 | 0.762 | 0.598 | 0.774 | 0.732 |
| 260923 round 31, AdamW lr 0.01, 1000 steps, loss rule | 0.818 ± 0.002 | 0.745 ± 0.009 | 0.576 ± 0.007 | 0.760 ± 0.003 | 0.706 ± 0.007 |
| 01 / 02: lr 0.03, 1000 steps, band | 0.837 ± 0.003 | 0.782 ± 0.003 | 0.580 ± 0.003 | 0.778 ± 0.002 | 0.735 ± 0.002 |
| 03: lr 0.03, 1000 steps, loss rule | 0.824 ± 0.004 | 0.756 ± 0.007 | 0.580 ± 0.007 | 0.766 ± 0.011 | 0.712 ± 0.005 |
| 05 / 06: 2000 steps, band | 0.839 ± 0.002 | 0.782 ± 0.002 | 0.601 ± 0.003 | 0.786 ± 0.001 | 0.746 ± 0.003 |
| **07 / 08: 4000 steps, band (reported)** | **0.840 ± 0.001** | **0.788 ± 0.002** | **0.590 ± 0.004** | **0.793 ± 0.001** | **0.749 ± 0.000** |

Rounds 10 and 15 are deterministic fits (sd 0). The 260826 config is `base` rows
(400 per class), lr 3e-3; every other row is `pool` rows.

The reported round under the three metrics:

| metric | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| `auc_combined` (mean sigmoid) | 0.840 | 0.788 | 0.590 | 0.793 | 0.749 |
| `auc_mean_logit` | 0.840 | 0.788 | 0.590 | 0.793 | 0.749 |
| `auc_filtered` (mean of per-layer AUCs, 260923's headline) | 0.834 | 0.781 | 0.589 | 0.789 | 0.746 |

Layers kept by the loss rule in round 03, seed 0: 31 / 31 / 32 / 3 / 19 of 32.

### Cross-validation

Chosen recipe per probe over the full grid (`cv/pool/summary.md`); OOF = mean over
folds of the held-out combined-score AUC, ± the fold sd:

| probe | lr | steps | epochs on full train | band | OOF (band) | OOF, loss rule | OOF, all 32 | OOF, band 13–20 |
|---|---|---|---|---|---|---|---|---|
| bias | 0.01 | 4000 | 44.0 | 13–16 | 0.881 ± 0.019 | 0.860 | 0.860 | 0.877 |
| bias_followed | 0.01 | 4000 | 44.0 | 13–16 | 0.789 ± 0.038 | 0.770 | 0.765 | 0.784 |
| deception | 0.03 | 4000 | 44.0 | 14–17 | 0.612 ± 0.022 | 0.607 | 0.602 | 0.608 |
| deception_yes | 0.03 | 4000 | 66.4 | 14–17 | 0.793 ± 0.011 | 0.779 | 0.769 | 0.783 |
| deception_no | 0.03 | 4000 | 130.6 | 15–18 | 0.769 ± 0.028 | 0.755 | 0.751 | 0.766 |

Band 13–20 is the block 260923's in-sample rounds landed on. A single shared band, chosen
by the mean OOF AUC over the four probes other than `deception` at each probe's chosen
(lr, steps), is 14–17; it is within 0.003 of each probe's own band (−0.001 / −0.003 /
0.000 / −0.001 for bias / bias_followed / deception_yes / deception_no).

OOF AUC of the best band per grid cell:

| lr | steps | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|---|
| 0.003 | 1000 | 0.849 | 0.749 | 0.583 | 0.733 | 0.699 |
| 0.003 | 2000 | 0.866 | 0.770 | 0.590 | 0.746 | 0.715 |
| 0.003 | 4000 | 0.876 | 0.782 | 0.598 | 0.758 | 0.731 |
| 0.01 | 1000 | 0.873 | 0.777 | 0.594 | 0.754 | 0.726 |
| 0.01 | 2000 | 0.879 | 0.787 | 0.601 | 0.765 | 0.740 |
| 0.01 | 4000 | 0.881 | 0.789 | 0.608 | 0.778 | 0.755 |
| 0.03 | 1000 | 0.880 | 0.783 | 0.603 | 0.770 | 0.746 |
| 0.03 | 2000 | 0.880 | 0.787 | 0.609 | 0.781 | 0.758 |
| 0.03 | 4000 | 0.878 | 0.785 | 0.612 | 0.793 | 0.769 |

The best band was 13–16 or 14–17 in every cell except `deception_no` at (0.03, 4000),
where it was 15–18, and `deception` at (0.003, 1000), where it was 12–15.

OOF prediction against the eval result of the round trained on that recipe:

| recipe | OOF (band) | eval, rounds |
|---|---|---|
| lr 0.03, 1000 steps | 0.880 / 0.783 / 0.603 / 0.770 / 0.747 | 0.837 / 0.782 / 0.580 / 0.778 / 0.735 (01 / 02) |
| 2000 steps | 0.880 / 0.787 / 0.609 / 0.781 / 0.758 | 0.839 / 0.782 / 0.601 / 0.786 / 0.746 (05 / 06) |
| 4000 steps | 0.881 / 0.789 / 0.612 / 0.793 / 0.769 | 0.840 / 0.788 / 0.590 / 0.793 / 0.749 (07 / 08) |

### High-confidence variant (no-cue-unanimous questions)

Eval AUC on the no-cue-unanimous eval rows (783 / 612 / 783 / 490 / 293):

| trained on | recipe | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|---|
| `pool_nu` (round 04) | rounds 01/02's lr and bands, 396 / 396 / 396 / 421 / 347 steps (equal epochs) | 0.889 ± 0.001 | 0.850 ± 0.002 | 0.617 ± 0.001 | 0.831 ± 0.003 | 0.767 ± 0.001 |
| `pool` (rounds 01/02, re-scored) | lr 0.03, 1000 steps, bands | 0.903 ± 0.002 | 0.871 ± 0.002 | 0.601 ± 0.004 | 0.832 ± 0.002 | 0.796 ± 0.002 |
| `pool` (rounds 07/08, re-scored) | reported recipe | 0.904 ± 0.001 | 0.873 ± 0.001 | 0.617 ± 0.005 | 0.847 ± 0.002 | 0.799 ± 0.000 |

Round 04's probes on the full eval: 0.822 / 0.762 / 0.599 / 0.775 / 0.714.

### Breakdowns of the reported round (seed 0, `runs/07_*/analysis/`, `runs/08_*/analysis/`)

Combined-score AUC by cue condition and by no-cue unanimity of the question:

| probe | `cue_gold` | `cue_dist` | unanimous | not unanimous | category range |
|---|---|---|---|---|---|
| bias | 0.811 (665) | 0.873 (661) | 0.904 (783) | 0.734 (543) | 0.779 law – 0.901 history |
| bias_followed | 0.806 (637) | 0.734 (469) | 0.873 (612) | 0.656 (494) | 0.757 computer science – 0.878 history |
| deception_yes | 0.749 (395) | 0.842 (441) | 0.847 (490) | 0.710 (346) | 0.740 computer science – 0.866 history |
| deception_no | 0.722 (270) | 0.783 (220) | 0.799 (293) | 0.663 (197) | 0.676 law – 0.782 physics |

The cue-following breakdown is single-class in its against-cue group for every probe
(bias is nested inside cue-following), so it is reported as n only in `slices.md`.

### What the numbers show

- The reported round scores 0.840 / 0.788 / 0.590 / 0.793 / 0.749 against 0.818 /
  0.745 / 0.576 / 0.760 / 0.706 for 260923's round 31 under the same metric: +0.022 /
  +0.043 / +0.014 / +0.033 / +0.043.
- At equal lr and steps (rounds 01/02 vs 03), the CV band scores +0.013 / +0.026 /
  0.000 / +0.012 / +0.023 above the pipeline's loss rule. Under the loss rule, lr 0.03
  (round 03) scores +0.006 / +0.011 / +0.004 / +0.006 / +0.006 above lr 0.01 (round 31).
- Out-of-fold AUC increases with steps at every learning rate for `deception_yes` and
  `deception_no`, and the largest value is at (lr 0.03, 4000 steps), the corner of the
  grid. For `bias` and `bias_followed` the three cells at lr 0.03 and the (0.01, 4000)
  cell are within 0.003 and 0.006 of each other. On eval, 1000 → 2000 → 4000 steps
  gives 0.778 → 0.786 → 0.793 (`deception_yes`) and 0.735 → 0.746 → 0.749
  (`deception_no`).
- The OOF prediction and the eval result of the same recipe differ by at most 0.012 for
  `bias_followed`, `deception_yes` and `deception_no` at every stage; for `bias` the OOF
  value is 0.041–0.043 above eval, and for `deception` 0.008–0.023 above.
- The per-probe bands are 13–16 (bias family), 14–17 and 15–18 (deception); a shared
  band 14–17 is within 0.003 OOF of each.
- On the no-cue-unanimous eval rows, the `pool_nu`-trained probes (round 04) score
  −0.014 / −0.021 / +0.016 / −0.001 / −0.029 relative to the `pool`-trained probes of
  rounds 01/02 on the same rows.
- `bias`, `bias_followed`, `deception_yes` and `deception_no` score 0.05–0.09 higher on
  the no-cue-unanimous eval rows than on all eval rows (rounds 07/08: 0.904 vs 0.840,
  0.873 vs 0.788, 0.847 vs 0.793, 0.799 vs 0.749); `deception` 0.027 higher (0.617 vs
  0.590). The same holds for the `pool_nu`-trained probes of round 04.
- Seed sd of the reported round is 0.000–0.004. Fold sd of the OOF AUC is 0.011–0.038.
- `auc_mean_logit` equals `auc_combined` to three decimals for every probe in the reported
  round; `auc_filtered` is 0.001–0.007 lower.

## Run status

- **CV sweep on `pool`**: `cv_sweep.py`, 5 processes (one per probe) in tmux with
  `OMP_NUM_THREADS=8`, 2026-09-24 19:52 to 2026-09-25 ~00:15 UTC, cheapest cells first
  (lr 0.01 / 0.03 at 1000 steps, then 2000, then lr 0.003, then 4000). Complete: 225 fold
  files under `cv/pool/<probe>/`. Three earlier launches (sequential, then unbounded
  threads) were stopped; their cells were kept and skipped on resume; logs
  `logs/cv_pool_*attempt*.log`.
- **Selection**: `cv_select.py --variant pool` run after the 1000-step cells, after the
  2000-step cells and after the full grid. The recipe files for each stage are
  `cv/pool/chosen_overrides_grid_lr01_03_steps1000_2000.json` (2000-step stage) and
  `cv/pool/chosen_overrides_full_grid.json` (final); `chosen_overrides.json` is the latest.
  The 1000-step-stage recipe is recorded in `runs/01_*/overrides.json`.
- **Rounds 01–08**: `round.py`, weights under `runs/NN_slug/probe_weights/seed{0,1,2}/`,
  metrics in `runs/NN_slug/probe_metrics.json`, rows in `runs.md`, figures and slice
  tables in `runs/NN_slug/analysis/`. All complete. Round 04's first launch failed at
  config time on a stray key in its override file and was relaunched; no round folder was
  created by the failed launch.
- **Re-scoring**: `rescored/<round>.json` for 260923's rounds 01, 10, 15, 31 and this
  folder's rounds 01–08; `comparison.md` built from them.
- **No CV sweep on `pool_nu`**: round 04 reuses the `pool` recipe with rescaled steps.
- **Smoke tests** (50-step fits, eval filter, per-probe override, analysis round-trip)
  are under `logs/smoke/`; they are not rounds.

## Costs

- CV: 225 fits of 32 layers (1000–4000 steps), mean 332 s per fit, 20.7 fit-hours;
  about 4.4 h wall clock with 5 parallel processes. The trainer runs at about 7 steps/s
  per process on this pod and is CPU-bound (GPU utilization under 10%); capping OpenMP
  threads to 8 per process is what allows 5 processes to run at full speed each.
- Rounds: 25–75 min each as two parallel processes (3 seeds × 5 probes × 1000–4000
  steps).
- Disk: `cv/` 70 MB, `runs/` 52 MB, `figures/` 0.8 MB, `rescored/` 0.2 MB. No new
  activations; 260923's `pool` (3.1 GB) and `pool_nu` (1.8 GB) are read in place.
