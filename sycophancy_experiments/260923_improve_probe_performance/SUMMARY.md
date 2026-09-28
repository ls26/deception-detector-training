# Summary of the autonomous probe-improvement run (2026-09-23)

This is a plain-language account of what the run did and found. The original,
denser write-up the agent produced is kept as `SUMMARY_technical.md`; every
number here comes from `runs.md` (one row per round, with the agent's notes),
the per-round `runs/NN_name/` folders, and `SUMMARY_technical.md`. Where the
record is ambiguous, this file says so instead of guessing.

## 1. What was done, in one paragraph

Starting from the five probes trained in the previous experiment (260826), an
unattended Claude Code session ran 33 numbered "rounds" in about three and a
quarter hours (05:36 to 08:55 on 2026-09-23).
Each round trained the five probes under exactly one change (a different
training method, more training data, a different token to read, and so on) and
scored them on the same fixed set of held-out questions. Every round was
recorded before the next began. The single biggest finding is that the original
probes had never been trained to completion, and that training them properly
and on more examples raises every probe that carries a real signal by 0.03 to
0.07. Filtering out noisy labels, reading other tokens, averaging over tokens,
and more complex probes did not help.

## 2. Words used below

- **Probe.** A small classifier that reads the model's internal activations at
  one token and predicts a label (biased or not; deceptive or not). One probe is
  trained at each of the model's 32 layers.
- **Train set / eval set.** The probe learns from the train set and is scored
  on the eval set, which is a fixed set of held-out questions that never changes
  between rounds. Every score in this file is on the eval set unless it says
  "train".
- **AUC.** The score. 0.5 means the probe is no better than a coin flip; 1.0
  means it ranks every positive example above every negative one. A change of
  about 0.02 or less is within noise here (about 0.03 for `deception_no`, which
  has the fewest eval rows).
- **Headline AUC.** The reported number: the average AUC across the layers that
  passed the "layer filter" (next item).
- **Layer filter.** The standard rule for which of the 32 per-layer probes count
  toward the headline: only layers whose probe reached a training loss below 0.6
  are kept. If no layer reaches it, all 32 are used. This rule never looks at the
  eval set.
- **Eval-selected layers ("in-sample").** In four rounds (16, 17, 18, 32) the
  agent instead picked the layers that scored best on the eval set itself and
  then reported the eval score on those layers. That is like choosing which
  exam questions to count after seeing the marks. Those numbers are optimistic
  by an amount this run could not measure, so they are shown separately and
  should not be quoted as the probe's expected performance.
- **Layer-mean logit AUC.** A second way to score: first average each example's
  raw probe outputs across the kept layers into one number, then compute a
  single AUC from that. This is closer to how a probe would actually be used as
  an RL penalty, which thresholds one combined score. It is often 0.02 to 0.04
  higher than the headline.
- **Seeds.** The probe trainer starts from random initial weights. Each round
  ran three random starts and reports the mean with its spread ("± sd"). Two
  training methods (logistic regression and LDA) have no randomness, so their
  spread is 0.000 by construction and the 0.02 noise rule above applies.
- **The five probes.** `bias` (did the user's cue change the model's answer,
  read at the answer letter), `bias_followed` (same, but only among answers that
  agreed with the cue, which removes an easy shortcut), `deception` (does the
  model's self-report about the cue match what it did, read at the token just
  before its Yes/No), and `deception_yes` / `deception_no` (the same split by
  whether the model said Yes or No). The three marked ★ in the tables are the
  ones the run was asked to focus on.
- **Draw.** Which training examples are used. The standard draw is balanced: the
  same number of positive and negative examples, and within each, the same
  number from the two cue conditions (cue pointing at the right answer, cue
  pointing at the wrong one).

## 3. Bottom line

Eval AUC. "Baseline" is round 1, which reproduced the previous experiment
exactly. "Best without eval-selected layers" uses the standard layer filter.
"With eval-selected layers" is the optimistic number described above.

| probe | baseline | best without eval-selected layers | which round | with eval-selected layers | which round |
|---|---|---|---|---|---|
| `bias` | 0.774 | 0.808 | 33 | 0.840 | 16 |
| `bias_followed` ★ | 0.705 ± 0.026 | 0.742 | 33 | 0.785 | 16 |
| `deception` | 0.475 | 0.586 | 15 | 0.599 | 32 |
| `deception_yes` ★ | 0.689 | 0.764 | 28 | 0.790 | 16 |
| `deception_no` ★ | 0.644 | 0.718 | 28 | 0.748 | 32 |

Two cautions on the "best" column, both from the agent's own notes:

- Rounds 28 and 33 beat the cleaner rounds 10 and 15 by only 0.004 to 0.018,
  which is within noise, and each has a reason to be treated carefully (round
  28's edge comes from the layer filter keeping fewer layers; round 33 used an
  unbalanced training set, explained in section 4). The agent's recommended
  "clean" configuration is round 10 or 15:

| configuration | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| round 10: largest balanced train set, logistic regression | 0.798 | 0.725 | 0.566 | 0.751 | 0.700 |
| round 15: same data, LDA | 0.799 | 0.727 | 0.586 | 0.736 | 0.704 |
| round 15 scored by layer-mean logit | 0.829 | 0.767 | 0.599 | 0.778 | 0.734 |

- The plain `deception` probe stays close to a coin flip in every configuration
  (0.47 to 0.59). The previous experiment already explained why: within the
  Yes group its label means "biased" and within the No group it means "not
  biased", so one linear direction cannot serve both. It is reported for
  completeness only.

## 4. What each round changed, in order

The rounds fall into eight groups. Each row says what was changed relative to
the previous relevant round and what happened. Scores are eval AUC for
bias / bias_followed / deception / deception_yes / deception_no.

### Group A: reproduce the starting point (round 1)

| round | what changed | result |
|---|---|---|
| 01 | Nothing: the previous experiment's exact configuration, re-extracted in this folder, three seeds. | 0.774 / 0.705 / 0.475 / 0.689 / 0.644, matching 260826 to three decimals on seed 0. Diagnosis: the probes were **under-trained**. After the standard 1000 training steps, they separated even their own training examples only moderately (train AUC 0.62 to 0.87), and for the deception probes no layer reached the training-loss threshold, so the layer filter fell back to all 32 layers. |

Because of that diagnosis, the agent moved the "fix the training" item ahead of
the label-filter and data-size items, reasoning that those comparisons are
meaningless under a trainer that has not converged. This reordering is recorded
in `runs.md` under "Added".

### Group B: fix the probe training (rounds 2 to 5)

The standard trainer is a gradient-descent loop (AdamW) that takes a fixed 1000
steps of a given step size ("learning rate"). Two alternatives were introduced:
**logistic regression**, a standard statistics fit that runs until it converges
and has no randomness, with a setting **C** that controls how strongly the fit
is discouraged from using large weights (larger C means a closer fit to the
training data, smaller C means a simpler probe); and, later, **LDA**, explained
in group E. Before each round, the agent ran quick single-seed "pilot" checks on
three probes to decide which settings deserved a full three-seed round; those
pilot numbers are in `runs.md` under "Added".

| round | what changed | result |
|---|---|---|
| 02 | Learning rate raised from 0.003 to 0.01 (vs round 1). | 0.783 / 0.699 / 0.500 / 0.733 / 0.666. `deception_yes` up 0.044, `deception_no` up 0.022. Still not fully converged on the training data. Caveat noted: `deception_yes` now passed the layer filter on only 2 layers. |
| 03 | Trainer replaced by logistic regression with C=100 (vs round 1). | 0.773 / 0.690 / 0.540 / 0.729 / 0.669. Same as round 2 on the deception probes, but deterministic, 1 minute per round instead of 8, and with 24 to 32 layers passing the filter instead of 2. |
| 04 | Logistic regression with C=30 instead of 100 (a simpler fit). | 0.780 / 0.698 / 0.502 / 0.736 / 0.683. The layer-mean logit score did not change; the small headline gain came from fewer layers passing the filter. The pilots showed weight decay, gradient clipping, more steps, and larger batches all made under 0.02 difference. Agent's decision: adopt logistic regression C=30 as the default trainer for the rest of the run. |
| 05 | Standard trainer with a much larger learning rate, 0.1 (vs round 2). | 0.763 / 0.685 / 0.520 / 0.726 / 0.679. The probe now over-fits (train AUC 0.98) and the bias family drops 0.02. Conclusion: for the standard trainer, 0.01 is the right learning rate. |

### Group C: more training data, and the label-noise filter (rounds 6 to 13)

The previous experiment trained on 400 examples per class (200 per class for
the deception strata). The agent extracted a much larger balanced training set
called **`pool`**: the largest possible balanced draw from the training
questions, using all three samples per question: 1454 per class for the bias
probes, 964 for `deception_yes`, 490 for `deception_no`. Smaller sets of 100,
200, and 400 per class were taken as nested subsets of it. A second version,
**`pool_nu`**, applied the filter you asked about earlier: keep only questions
whose three no-cue answers all agreed. That filter leaves 576 / 406 / 170 / 576
per class, and the `deception_no` cue-on-correct-answer cell has only 85 rows
where 100 were wanted (warned, not an error).

| round | what changed | result |
|---|---|---|
| 06 | Same sizes as round 4 (400 / 200 / 200 per class) but drawn from `pool`, i.e. a different random selection of training examples. | 0.779 / 0.696 / 0.542 / 0.735 / 0.656. Four probes unchanged; `deception_no` down 0.027 from just a different draw. This sets the noise level: differences under about 0.03 on `deception_no` are not interpretable. |
| 07 | Label filter on (`pool_nu`), same sizes as round 6. | 0.766 / 0.697 / 0.569 / 0.740 / 0.670. No change beyond noise. **The label filter neither helped nor hurt at this size.** |
| 08 | No filter, 100 per class (subset of round 6's data). | 0.750 / 0.666 / 0.526 / 0.690 / 0.635. About 0.03 below round 6 on each probe. |
| 09 | No filter, 200 per class. | 0.758 / 0.665 / 0.540 / 0.715 / 0.625. Between the 100 and 400 points on `bias` and `deception_yes`; flat on the other two. |
| 10 | No filter, everything in `pool` (1454 / 964 / 490 / 1454 per class). | 0.798 / 0.725 / 0.566 / 0.751 / 0.700. **Best round so far on all five**: vs 400 per class, `bias` +0.019, `bias_followed` +0.029, `deception_yes` +0.016, `deception_no` +0.044. The curve 100 → 200 → 400 → all is still rising at the top, so more data would probably help further, but the source rollouts are fixed. This round became the reference for everything after. |
| 11 | Filter on, 100 per class. | 0.733 / 0.660 / 0.565 / 0.685 / 0.615. Within noise of round 8, if anything lower. |
| 12 | Filter on, 200 per class. | 0.761 / 0.696 / 0.526 / 0.725 / 0.662. Mixed: `bias_followed` +0.031 and `deception_no` +0.037 vs round 9, the others within 0.01. Both gains are at the noise level for those probes. |
| 13 | Filter on, everything in `pool_nu` (576 / 406 / 170 / 576). | 0.780 / 0.710 / 0.573 / 0.743 / 0.670. Below round 10 on every probe by 0.008 to 0.030, because the filter throws away about 60% of the rows. **Conclusion: at equal size the filter is a wash; data size is the lever, so the filter is not worth its cost.** |

Note from the agent: rounds 11 to 13 were run back to back and their rows
appended afterwards, which departs from the "record before the next round"
rule. The rows are in order and complete.

### Group D: a different kind of probe (rounds 14 and 15)

| round | what changed | result |
|---|---|---|
| 14 | Instead of fitting a classifier, use the **difference between the two class averages** as the probe direction (on round 10's data). | 0.750 / 0.656 / 0.474 / 0.657 / 0.611. Worse on every probe by 0.05 to 0.09. The raw average-difference direction is dominated by variation unrelated to the label. |
| 15 | The same idea but **LDA**: the average-difference direction adjusted for how the data varies within each class. Its one setting, "shrinkage" (0 to 1, how much the within-class variation estimate is simplified), was set to 0.5 after a pilot. | 0.799 / 0.727 / 0.586 / 0.736 / 0.704. Same headline as logistic regression, and **the best layer-mean logit scores of the whole run** on all five probes (0.829 / 0.767 / 0.599 / 0.778 / 0.734). |

### Group E: which layers to count (rounds 16 to 19), mostly eval-selected

These rounds all use round 10's data and trainer and change only the layer
rule. Rounds 16 to 18 choose layers by looking at the eval set, so their numbers
are optimistic (see section 2).

| round | what changed | result |
|---|---|---|
| 16 | The single layer with the best eval AUC. **Eval-selected.** | 0.840 / 0.785 / 0.589 / 0.790 / 0.735. Plus 0.04 to 0.06 over round 10. The chosen layers were 11 to 15. |
| 17 | The 8 layers with the best eval AUC. **Eval-selected.** | 0.826 / 0.773 / 0.586 / 0.773 / 0.717. Mostly layers 13 to 20. |
| 18 | The best contiguous block of 8 layers. **Eval-selected.** | 0.823 / 0.772 / 0.585 / 0.773 / 0.717. Within 0.003 of round 17. Block was 13 to 20 for three probes, 9 to 16 for `bias_followed`. |
| 19 | All 32 layers, no filter at all. Not eval-selected. | 0.794 / 0.720 / 0.566 / 0.720 / 0.681. Slightly below the standard filter on the headline because the early layers (weak, AUC 0.55 to 0.65) drag the average down; the layer-mean logit score is unchanged. |

Takeaway: the middle layers (roughly 13 to 20) carry the signal. The standard
filter already finds them most of the time. A fixed middle band would be a
reasonable, low-risk choice for a detector, but its exact benefit was only
measured in-sample.

### Group F: which token to read (rounds 20 to 23, and 27)

The probes read the model's activation at one token. The defaults are the
**answer letter** for the bias probes and the **token just before the Yes/No
self-report** for the deception probes. These rounds tried other tokens, on
round 4's data and trainer unless stated.

| round | what changed | result |
|---|---|---|
| 20 | Bias probes at the **last token of the model's reply**; deception probes at the **Yes/No token itself** (vs round 4). | 0.735 / 0.659 / 0.583 / 0.683 / 0.625. Worse by 0.04 to 0.06 on the four probes that matter. Only the plain `deception` probe rose, and only because at the Yes/No token it can read which word was sampled, which is not deception detection. |
| 21 | All probes at the **last token of the prompt**, before the model has written anything. | 0.660 / 0.684 / 0.499 / 0.740 / 0.679. `bias` collapses (the probe cannot see the answer yet), but `bias_followed` loses only 0.014, so whether the cue is driving the answer is nearly as readable before generation as after. For the deception probes this token is the same as the default on all but 5 of 2126 rows, so no change. |
| 22 | All probes at the **first token the model writes**. | 0.623 / 0.619 / 0.583 / 0.681 / 0.627. Worst for the bias family (that token is usually boilerplate like "The"). |
| 23 | **Average over all tokens of the model's reply** (mean pooling). | 0.593 / 0.569 / 0.592 / 0.698 / 0.616. Worse everywhere. Averaging over hundreds of reasoning tokens washes out the signal. |
| 27 | Last token of the prompt again, but on round 10's large data. | 0.667 / 0.712 / 0.566 / 0.753 / 0.703. `bias_followed` 0.712 vs 0.725 at the answer token. Confirms round 21 at the larger size. |

Takeaway: the default tokens are the right ones. The one interesting side
result is that `bias_followed` works almost as well before the model answers,
which is what a detector that acts before generation would need.

### Group G: other ideas from the queue (rounds 24 to 26)

| round | what changed | result |
|---|---|---|
| 24 | **Per-question centering**: subtract each question's average activation before training, to remove question-specific content (round 10's data). | 0.620 / 0.686 / 0.531 / 0.557 / 0.505. Hurts badly, by up to 0.19. The agent's explanation: the eval set has at most two rows per question, so subtracting the question mean either zeroes the row out or leaves only the difference between the two cue conditions, which is not the label. |
| 25 | A **two-layer neural-network probe** instead of a linear one (standard trainer, learning rate 0.01, round 10's data). | 0.795 / 0.721 / 0.500 / 0.749 / 0.698. Equal to the linear probe within seed spread. Nothing to gain from a more complex probe at this data size. |
| 26 | **Cross-category check**: train on four of the five MMLU-Pro subject categories, score only physics questions (the largest category). | 0.774 / 0.717 / 0.535 / 0.712 / 0.653 on physics, versus 0.809 / 0.733 / 0.549 / 0.723 / 0.729 when physics was included in training. Loss from never seeing physics: 0.035 / 0.016 / 0.011 / 0.076. So the probes mostly read something that transfers across subjects; `deception_no` transfers least, though its physics eval cell is only 129 rows. |

### Group H: follow-ups combining the winners (rounds 28 to 33)

| round | what changed | result |
|---|---|---|
| 28 | Logistic regression with C=10 (a simpler fit than C=30) on round 10's data. | 0.802 / 0.732 / 0.559 / 0.764 / 0.718. Small headline gains on the deception probes, but the layer-mean logit score did not move. The agent attributes the headline gain to the layer filter keeping only 6 and 1 layers, the same effect as choosing layers, not a better probe. |
| 29 | Logistic regression with C=100 on round 10's data. | 0.793 / 0.715 / 0.572 / 0.758 / 0.697. No change. Across 10, 28, 29: C between 10 and 100 makes no difference at this data size. |
| 30 | LDA with shrinkage 0.9 instead of 0.5 (vs round 15). | 0.797 / 0.728 / 0.571 / 0.751 / 0.696. Same headline; lower layer-mean logit on all five. Shrinkage 0.5 stays the better setting. |
| 31 | The **standard AdamW trainer** (the one the RL pipeline itself uses) at learning rate 0.01 on round 10's large data, three seeds. | 0.795 ± 0.001 / 0.725 ± 0.014 / 0.554 ± 0.006 / 0.747 ± 0.008 / 0.704 ± 0.008. Equal to logistic regression within seed spread. **This is the recipe for the RL pipeline's own probe type: learning rate 0.01 and the large balanced training set.** |
| 32 | LDA 0.5 plus the best contiguous block of 8 layers. **Eval-selected.** | 0.823 / 0.772 / 0.599 / 0.787 / 0.748. Best headline of the run on all five, but the layer block was chosen on the eval set. For `bias_followed` the chosen block (24 to 31) differs from round 18's (9 to 16), which the agent flags as a sign that block choice is unstable for that probe. |
| 33 | **Every available training row, with no balancing**: about 3 to 6 times more rows than round 10, but lopsided (for example 2150 biased vs 6525 not-biased rows; the deception Yes group 4015 vs 1431 and No group 716 vs 2494), logistic regression C=30. | 0.808 / 0.742 / 0.615 / 0.732 / 0.700. Bias family up 0.010 to 0.017 over round 10; `deception_yes` down 0.019. Caveat from the agent: among the biased rows, the cue-on-wrong-answer condition outnumbers cue-on-right-answer about 2 to 1, and the eval set has the same imbalance, so part of this gain could be the probe learning which cue condition it is looking at rather than bias itself. Round 10 is the cleaner number; round 33 mainly says that more rows would still help. |

The agent stopped at round 33 of the allowed 40 because the queue and its
follow-ups were exhausted.

## 5. What helped, what did not

**Helped**

1. Training the probes to convergence. The original 1000-step trainer at its
   default step size stopped well short. A converged logistic regression fit, or
   the same trainer with learning rate 0.01, gave `deception_yes` +0.04 and
   `deception_no` +0.02 at the original data size, and the rest of the run was
   built on that.
2. More training examples. Going from 400 per class to the largest balanced set
   (1454 / 964 / 490 / 1454) added +0.02 to +0.04 on every probe with signal,
   and the curve was still rising. This is the largest single lever and it is
   limited by the fixed source data.
3. LDA as the fit. Same headline as logistic regression, best combined-score
   (layer-mean logit) AUC of the run. Worth using if the probe will be applied
   as a single thresholded score.

**Did not help** (all within noise or worse)

- The "three no-cue answers agree" label filter: no gain at equal size, and a
  loss overall because it discards most rows.
- Reading a different token: the answer letter and the token before the
  self-report remain the best. Averaging over the reply is much worse.
- Subtracting each question's mean activation: much worse under this eval design.
- A small neural-network probe instead of a linear one: equal.
- Tuning the logistic-regression strength (C from 10 to 100) or LDA shrinkage:
  flat.
- Weight decay, gradient clipping, more steps, larger batches for the standard
  trainer (pilot checks only, single seed): all within 0.02.

## 6. Things to keep in mind when reading these numbers

- **All choices were made on the eval set.** There was no separate validation
  set (your decision at setup). So every "best" number is somewhat optimistic
  simply because it was picked from about 30 rounds plus pilot sweeps. The
  ordinal findings (more data helps; other tokens are worse; the filter does
  nothing) are robust to this. Small edges of under 0.02 between the top rounds
  are not.
- **Rounds 16, 17, 18, and 32 picked layers by looking at the eval set** and
  are clearly labelled. Treat their extra 0.03 to 0.06 as an upper bound, not a
  measurement.
- **Deterministic fits show ± 0.000.** That is not "no noise"; it means the
  method has no random start. Noise from which examples end up in the training
  set was measured at about 0.03 on `deception_no` (round 6 vs 4) and under
  0.005 on the others.
- **Pilot sweeps were single-seed, on three probes only.** They were used to
  decide what to run as a full round, not as results in themselves.
- **Round 33 is unbalanced** and may partly reflect the cue condition rather
  than bias. Prefer round 10 or 15 when quoting a clean number.
- **Rounds 11 to 13 were recorded after the fact**, in order.
- The training-set sizes for `deception_no` under the label filter were short
  in one cell (85 of 100), as expected from the earlier count.

## 7. Recommended settings going forward

From the agent's summary and consistent with the rounds above:

- **Training data:** the largest balanced draw (`activations/pool`, 1454 / 964 /
  490 / 1454 per class), no label filter.
- **Tokens:** unchanged: answer letter for the bias probes, token before the
  self-report for the deception probes.
- **Fit:** LDA with shrinkage 0.5 for the best combined score, or logistic
  regression C=30 for a one-minute deterministic fit with the same headline.
  For the RL pipeline's own probe type, its standard trainer at learning rate
  0.01 on the same data (round 31) reaches the same numbers.
- **Layers:** the existing training-loss filter, or a fixed middle band
  (about layers 13 to 20). The combined score barely depends on the layer set
  once it includes the middle layers. The precise benefit of a fixed band is
  only known in-sample.

## 8. Open questions the run could not answer

- How much of the +0.03 to +0.06 from eval-selected layers is real. Needs a
  validation split or a fresh eval draw.
- Whether more source data keeps helping. The learning curve was still rising
  at the largest balanced draw, and the unbalanced full set added a little more
  on the bias family. Going further needs new rollouts.
- Whether `deception_no`'s weak transfer to unseen subject categories (−0.076 on
  physics) is real or an artefact of its 129-row physics eval cell.
- Whether a detector that fires before the model answers is viable for the
  followed-cue contrast (`bias_followed` at the last prompt token was within
  0.014 of the answer token). Not tested for deception, where that token is the
  same as the default.

## 9. Where things are

- `runs.md`: the queue with status, the pilot numbers, and one row per round
  with the agent's decision notes. Start here for any detail.
- `runs/NN_name/`: for each round, `config.json` (the full configuration),
  `overrides.json` (just what that round changed and which activation set it
  used), `probe_metrics.json` (per-layer AUCs, train loss and train AUC, seed
  aggregates), and `probe_weights/seed*/`.
- `activations/`: the extracted activations per variant (`base`, `pool`,
  `pool_nu`, `pos2`, `pool_lastprompt`, `full`), about 23 GB; `probe_sets/` the
  matching rows.
- `SUMMARY_technical.md`: the agent's original denser summary with the same
  numbers.
- New code the agent wrote in this folder: `extract.py`, `round.py`,
  `append_row.py`, `pilot_sweep.py`, and the new options in `probe_config.py`
  (`train_draw_per_class`, `probe_fit`, `logreg_C`, `lda_shrinkage`,
  `layer_selection`, `layer_k`, `normalize_input`, `center_by_question`,
  `holdout_category`) plus three new token positions in step 2 (`lastprompt`,
  `firstgen`, `meancompl`).
