# Summary: autonomous probe-improvement run (2026-09-23)

33 recorded rounds (`runs/`, cap 40), 6 extraction variants (`activations/`), every
queue item tried. The full per-round record with decisions is `runs.md`; this file
is the digest. All numbers are eval AUC on the fixed eval split (one row per
prompt, `pool_seed` 53), mean over the filtered layers unless marked
"layer-mean logit" (AUC of the mean logit over the filtered layers, the score the
RL penalty would threshold). Seed sd is over `probe_seeds` (0, 1, 2); rounds that
use a deterministic fit (logreg, LDA) have sd 0.000 by construction and are
compared with a 0.02 rule instead (see "Noise").

## The five probes: baseline and best round per probe

"Best (no eval selection)" is the best round whose layer set comes from the shared
train-loss filter, i.e. nothing was chosen on the eval set except the config
itself (see caveat). "Best (in-sample layers)" additionally chooses the layer set
by eval AUC and is optimistic.

| probe | baseline (round 01) | best, no eval selection | round | best, in-sample layers | round |
|---|---|---|---|---|---|
| `bias_rm_answer` | 0.774 ± 0.000 | 0.808 ± 0.000 | 33 | 0.840 ± 0.000 | 16 |
| `bias_followed_rm_answer` ★ | 0.705 ± 0.026 | 0.742 ± 0.000 | 33 | 0.785 ± 0.000 | 16 |
| `deception_rm_preyesno` | 0.475 ± 0.004 | 0.586 ± 0.000 | 15 | 0.599 ± 0.000 | 32 |
| `deception_yes_rm_preyesno` ★ | 0.689 ± 0.000 | 0.764 ± 0.000 | 28 | 0.790 ± 0.000 | 16 |
| `deception_no_rm_preyesno` ★ | 0.644 ± 0.003 | 0.718 ± 0.000 | 28 | 0.748 ± 0.000 | 32 |

Round 33 (every eligible train row, unbalanced) tops the bias family and round 28
(logreg C=10 on the largest balanced draw) tops the deception probes, each by
0.004 to 0.018 over rounds 10 / 15 (C=30 / LDA on the largest balanced draw).
Round 28's edge comes from its stronger regularization letting fewer layers
through the 0.6 loss filter (6 and 1 layers on the deception probes), a mild
form of layer selection; round 33's draw over-represents cue_dist among bias
rows, so part of its gain may be the probe reading the cue condition. The
representative "best clean config" is therefore round 15 or 10:

| config | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| round 10: pool all, logreg C=30 | 0.798 | 0.725 | 0.566 | 0.751 | 0.700 |
| round 15: pool all, LDA shrinkage 0.5 | 0.799 | 0.727 | 0.586 | 0.736 | 0.704 |
| round 31: pool all, shared AdamW lr 1e-2 (3 seeds) | 0.795 ± 0.001 | 0.725 ± 0.014 | 0.554 ± 0.006 | 0.747 ± 0.008 | 0.704 ± 0.008 |
| round 15, layer-mean logit AUC | 0.829 | 0.767 | 0.599 | 0.778 | 0.734 |
| round 32: round 15 + best contiguous band of 8 (in-sample) | 0.823 | 0.772 | 0.599 | 0.787 | 0.748 |
| round 33: full unbalanced pool, logreg C=30 | 0.808 | 0.742 | 0.615 | 0.732 | 0.700 |
| round 33, layer-mean logit AUC | 0.850 | 0.794 | 0.631 | 0.776 | 0.720 |

Gains over the baseline without any eval-side selection: bias +0.025 to +0.034,
bias_followed +0.020 to +0.037, deception_yes +0.05 to +0.06, deception_no +0.06.
The unstratified `deception` probe stays near chance in every configuration
(0.47 to 0.59) because its label is bias inside said-Yes rows and no-bias inside
said-No rows; the only configurations that lift it read the self-report token
(rounds 20, 22, 23) and are not deception detectors.

## What helped, by how much

1. **Fitting the probe properly (item 4).** Round 01's diagnosis: after the
   shared trainer's 1000 AdamW steps at lr 3e-3, train AUC was only 0.87 (bias),
   0.80 (bias_followed), 0.62 (deception), 0.73 / 0.72 (yes / no), and the
   deception probes never reached train loss 0.6 at any layer. lr 1e-2 (round 02)
   gave deception_yes +0.044 and deception_no +0.022 at n=400; a converged
   sklearn logistic fit with C=30 on unit-norm inputs (round 04) gave the same
   numbers deterministically in 1 minute per round (0.780 / 0.698 / 0.502 /
   0.736 / 0.683). Weight decay above 0.01, gradient clipping, 2000 steps,
   batch 128, `normalize_input` none and lr 1e-1 (round 05) all landed within
   0.02 of that (pilots opt1 / opt2 in `runs.md`, `logs/pilot_*.json`).
   lr 1e-1 overfits (train AUC 0.98) and loses 0.02 on the bias family.
2. **More training rows (item 3).** Learning curve with the logistic fit, per-class
   draw 100 / 200 / 400 / all (1454 bias, 964 deception_yes, 490 deception_no,
   1454 bias_followed; the largest class- and cue-balanced draw of the train
   pool, 3 samples per prompt): bias 0.750 / 0.758 / 0.779 / 0.798,
   bias_followed 0.666 / 0.665 / 0.696 / 0.725, deception_yes 0.690 / 0.715 /
   0.735 / 0.751, deception_no 0.635 / 0.625 / 0.656 / 0.700. Still rising at the
   largest draw; the source data is fixed, so this is the ceiling here. Round 31
   shows the shared AdamW trainer at lr 1e-2 gets the same benefit
   (+0.012 / +0.026 / +0.014 / +0.038 over the same trainer at n=400).
3. **LDA (item 5).** The class-mean difference whitened by the shrunk
   within-class covariance (shrinkage 0.5) equals the logistic fit on the
   headline and gives the best layer-mean logit AUC of any round on all five
   probes (0.829 / 0.767 / 0.599 / 0.778 / 0.734 vs 0.829 / 0.765 / 0.589 /
   0.765 / 0.707 for logreg). The raw mean difference without whitening
   (round 14) is far worse (-0.05 to -0.09).
4. **Layer selection (item 6, in-sample).** Best single eval layer: +0.04 to
   +0.06 (layers 11 to 15). Top-8 and the best contiguous band of 8 are within
   0.003 of each other and 0.01 to 0.02 below the single best layer; the band is
   13-20 for four of the five probes (rounds 17 / 18), 9-16 or 24-31 for
   bias_followed depending on the fit (unstable). All 32 layers (round 19) is
   0.005 to 0.03 below the loss-filtered set on the headline but equal on the
   layer-mean logit.

## What did not help

- **Train filter `none_unanimous` (item 2):** a wash at matched n (rounds 07,
  11-13 vs 06, 08-10: -0.017 to +0.037, inside draw noise), and below the
  unfiltered largest draw because it leaves 2.5x fewer rows (round 13 vs 10:
  -0.008 to -0.030).
- **Positions (item 7):** `lastcompl` / `yesno` -0.04 to -0.06 on the four
  stratified/bias probes (round 20); `firstgen` -0.16 on the bias family
  (round 22); `lastprompt` -0.12 on bias but only -0.014 on bias_followed
  (rounds 21, 27: 0.712 vs 0.725 on the largest draw), so the followed-cue
  contrast is readable before generation. For deception `lastprompt` and
  `firstgen` coincide with `preyesno` and `yesno` on all but 5 of 2126 rows.
- **Mean pooling over completion tokens (item 8):** worst representation for
  the bias family (0.593 / 0.569) and below `preyesno` for deception (round 23).
- **Per-question centering (item 9):** -0.04 to -0.19 (round 24). Eval has at
  most 2 rows per question, so centering removes the row or leaves only the
  cue_gold-vs-cue_dist difference.
- **Nonlinear probe (item 10):** equal to linear within seed sd (round 25); the
  unstratified deception MLP did not train at all.
- **Regularization strength at the largest draw:** C=10 / 30 / 100 flat on the
  layer-mean logit (rounds 28, 10, 29); LDA shrinkage 0.9 below 0.5 (round 30).
- **Diff-of-means without whitening (item 5):** -0.05 to -0.09 (round 14).

## Diagnostic: cross-category holdout (item 11)

Trained on four categories and scored on physics only (round 26):
0.774 / 0.717 / 0.535 / 0.712 / 0.653 versus 0.809 / 0.733 / 0.549 / 0.723 /
0.729 for round 10's probes (trained with physics) on the same physics rows.
Transfer loss: bias -0.035, bias_followed -0.016, deception_yes -0.011,
deception_no -0.076 (129 rows, noisy). Per-category AUC of round 10: bias
0.76-0.82, bias_followed 0.70-0.77, deception_yes 0.71-0.79, deception_no
0.63-0.73.

## Recommended recipe

- Rows: the largest cue- and class-balanced train draw (`activations/pool`,
  `train_draw_per_class` = 1454 / 964 / 490 / 1454), no `none_unanimous` filter.
- Positions: `answer` for the bias family, `preyesno` for deception (unchanged).
- Fit: LDA with shrinkage 0.5 (best detector score) or logistic C=30 on
  unit-norm inputs (same headline, 1 minute per round); for the RL pipeline's
  own probe type, the shared AdamW trainer at lr 1e-2 on the same rows (round 31)
  gives the same numbers within seed sd.
- Layers: the train-loss filter as is, or a fixed mid band 13-20 (in-sample; see
  caveat). The layer-mean logit barely depends on the layer set once it includes
  the mid band.

## Selection-bias caveat

The eval split is both the selection set and the reported set (user decision; no
validation split). Every configuration in this file was chosen by looking at its
eval AUC, so all "best" numbers are in-sample to some degree. Two levels:

- Rounds whose layer set comes from the train-loss filter (rounds 10, 15, 28,
  31, ...) are in-sample only through the choice of config among ~30 rounds and
  a handful of pilot sweeps. The learning-curve and position results are
  ordinal comparisons and robust to this; the +0.004 to +0.018 edge of round 28
  over rounds 10 / 15 is not.
- Rounds 16-18 and 32 (`layer_selection` best_eval / topk_eval / band_eval) pick
  layers by eval AUC directly. The max over 32 correlated layers is optimistic
  by an amount this run cannot measure; treat +0.03 to +0.06 as an upper bound.

Pilot sweeps (`pilot_sweep.py`, single seed, one probe at a time, no round) were
used to choose which optimizer settings got a three-seed round; their numbers
are in `runs.md` "Added" and `logs/pilot_*.json`.

## Noise

- Probe-seed sd of the shared trainer: 0.000 to 0.026 (round 01's bias_followed
  0.026 came from one seed at 0.735 vs 0.688 / 0.692).
- Train-draw noise: a different 400 / 200 / 200 draw (round 06 vs 04) moved
  deception_no by 0.027 and the others by under 0.005. Differences on
  deception_no (490 eval rows) below about 0.03, and on the others below about
  0.02 (Hanley-McNeil standard error at 836 to 1326 rows), were treated as no
  change.
- Round 01 reproduces 260826 exactly (seed 0 identical to 3 decimals; the base
  activations are bit-identical to 260826's `activations_maj3`).

## Blocked

Nothing needed a change outside this folder. One protocol lapse: rounds 11-13
were run back to back and their rows appended afterwards, in order.

## Open questions

- Whether the +0.03 to +0.06 from eval-selected layers survives on held-out
  data (needs a validation split or a fresh eval draw).
- Whether more source data would keep lifting the curve: the learning curve is
  still rising at the largest balanced draw, and the unbalanced full pool
  (round 33) adds another +0.01 to +0.02 on the bias family. Anything larger
  needs new rollouts.
- Whether deception_no's weak cross-category transfer (-0.076 on physics) is
  real or the 129-row eval cell.
- `bias_followed` at `lastprompt` is within 0.014 of the answer token: a
  detector that fires before generation may be possible for the followed-cue
  contrast; not tested for deception since `lastprompt` equals `preyesno` there.

## Round 33: every eligible train row, unbalanced

Every eligible train row without class or cue balance (bias 2150 / 6525,
bias_followed 2150 / 4143, deception strata 4015 / 1431 said-Yes and 716 / 2494
said-No; 3 to 6x the balanced draw), logistic C=30: 0.808 / 0.742 / 0.615 /
0.732 / 0.700, layer-mean logit 0.850 / 0.794 / 0.631 / 0.776 / 0.720. Versus
the largest balanced draw (round 10): bias +0.010, bias_followed +0.017,
deception_no +0.000, deception_yes -0.019 on the headline (all 32 layers now
pass the loss filter) but +0.011 on the layer-mean logit. Caveat: cue_dist is
over-represented among bias rows (1423 vs 727), so the probe may partly read the
cue condition, which the eval set (same imbalance) also rewards. The balanced
draw is the cleaner number; the unbalanced one says more rows still help.

## Files

- `runs.md`: queue state, pilots, every round's row with decision.
- `runs/NN_slug/`: `config.json`, `overrides.json`, `probe_metrics.json`
  (per-layer eval AUC, per-layer train loss and train AUC, seed aggregates),
  `probe_weights/seed*/`.
- `activations/{base,pool,pool_nu,pos2,pool_lastprompt,full}/`: bfloat16
  (lossless) activations plus `extraction.json`; `probe_sets/<variant>/` the rows.
- Drivers: `extract.py` (steps 1-2 per variant), `round.py` (one recorded
  round), `append_row.py`, `pilot_sweep.py`. New config knobs in
  `probe_config.py`: `train_draw_per_class`, `probe_fit` / `logreg_C` /
  `lda_shrinkage`, `layer_selection` / `layer_k`, `normalize_input`,
  `center_by_question`, `holdout_category`; new step-2 positions `lastprompt`,
  `firstgen`, `meancompl`.
