# 261010 — Confident slice: CoT probes trained and evaluated on questions with a consistent no-cue answer

Probe experiment on the MMLU-Pro sycophancy domain. A rollout's bias label compares its cued
answer with the model's no-cue answer, which until now was the majority of 3 sampled no-cue CoT
rollouts (260819). This experiment keeps chain-of-thought and restricts training and evaluation to
**slice questions**: questions where at least 6 of 8 no-cue CoT samples give the same parsed
answer. On those questions the labels are rebuilt from that answer. Three `rm` probes
(`bias_followed`, said-Yes deception, said-No deception) are trained on the slice, with learning
rate, steps and layer band chosen by 261007's cross-validation, and compared with the baseline
probe on the same slice eval rows. It builds on
[`261007_format_label_cue_ablation`](../261007_format_label_cue_ablation/) (probe pipeline, labels,
baseline probe) and [`261010_soft_bias_labels`](../261010_soft_bias_labels/) (the 8 no-cue
samples per eval question).

**The reported run** is the unattended run of 2026-10-10, 19:23–20:58 UTC (protocol in
`CLAUDE.md`; queue, log and Result in `runs.md`). It wrote `completions.jsonl`, `arms/`, `cv/`,
`runs/slice_cot/`, `runs/baseline_on_slice/` and `comparison.md` / `comparison.json`.

**Added after the run** (2026-10-10, 23:23 UTC, attended): `nocot_nocue_slice.py` re-scores
261007's no-CoT logprob probe on its eval rows whose no-cue prompt is confident, to set beside the
CoT slice probes -> `nocot_comparison.md` / `nocot_comparison.json`. No training.

### The baseline probe

"Baseline probe" in this README always means one set of saved probes:

- **What it is:** the current CoT probes for the three `rm` kinds, trained in 261007 and saved in
  [`../261007_format_label_cue_ablation/decomp/full3_oldrecipe/`](../261007_format_label_cue_ablation/decomp/full3_oldrecipe/).
  261007's README lists this fit as column B of its decomposition table; 261010_soft_bias_labels
  calls it "fit B" and also uses it as its baseline.
- **How it was trained:** all train-split questions; labels from the majority of 260819's 3 no-cue
  samples; self-report from float32 P(Yes) > 0.5 (Decision 7); every rollout of a prompt can be a
  train row (up to 3), cue-balanced; 260924's learning rate, steps and band (chosen by
  cross-validation in 260924, then fixed); seed 0.
- **Its own eval result (261007):** 0.788 / 0.771 / 0.799 (`bias_followed` / said-Yes / said-No) on
  261007's eval set: all 687 eval questions, one rollout per prompt, majority-of-3 labels.
- **How it is used here:** it is not retrained. Its saved weights and bands are applied to this
  experiment's slice eval activations and scored with the slice labels, so the baseline probe and
  the slice-trained probe are scored on exactly the same rows with the same labels.

## Files

| File | Role |
|---|---|
| `slice_spec.py` | Every fixed choice of this experiment: input paths, the 6-of-8 threshold, sampling settings and seeds, probes, train rollouts per prompt, dry-run sizes. The pipeline's own settings (pool and draw seeds, CV grid, folds, edge cells, band width, trainer, score, bootstrap) are 261007's `spec.py` values |
| `domain_imports.py` | Imports 260819's answer parsing and generator and 260824's label functions; used only by `generate_nocue_train.py` and `slice_labels.py` |
| `slice_pipeline.py` | Imports 261007's pipeline modules unchanged and builds this experiment's `PipelineConfig`; used by the four probe scripts |
| `generate_nocue_train.py` | 8 no-cue CoT samples per train-split question (vLLM) -> `completions.jsonl`, `generation_summary.json` |
| `slice_labels.py` | The slice per question and the relabelled rows -> `arms/slice_cot/labels.jsonl`, `slice_summary.json` |
| `run_pipeline.py` | Train sizes, then 261007's step 1 (probe sets) and step 2 (activations) -> `arms/train_sizes.json`, `arms/slice_cot/{probe_sets,activations}/`, `arms/extraction_summary.json` |
| `cv_run.py` | Cross-validation for one probe with 261007's sweep, selection and edge rule -> `cv/slice_cot/<probe>/` |
| `final_fit.py` | Rescoring check; final fit of the three slice probes; the baseline probe scored on the slice eval rows -> `runs/slice_cot/`, `runs/baseline_on_slice/` |
| `compare_slice.py` | AUCs, Δ and bootstrap CIs -> `comparison.md`, `comparison.json` |
| `nocot_nocue_slice.py` | Added after the run: 261007's no-CoT logprob probe re-scored on its no-cue-confident eval rows, beside the CoT numbers -> `nocot_comparison.md`, `nocot_comparison.json` |
| `CLAUDE.md`, `kickoff_prompt.txt` | The unattended-run protocol and its first message |
| `runs.md` | The unattended run's queue, log, judgment calls and Result |
| `dry/` | Dry runs of every queue step (100 questions per split, 200 training steps); not results |
| `logs/` | One log per step, ending in `EXIT <code>`; `dry_*` logs are the dry runs |

## Design

### The slice

- **No-cue samples:** 8 per question. Eval questions (687): 261010_soft_bias_labels'
  `completions.jsonl`. Train questions (1603): generated here by the same code with 260819's
  no-cue prompt and settings (temperature 1.0, at most 2048 new tokens, prefix caching off), one
  request per sample with seed 90,001,042 + 8 × question id + sample index. Answers parsed with
  260819's `extract_answer`.
- **Slice rule:** a question is in the slice when at least 6 of its 8 samples give the same parsed
  answer (gold or distractor). An unparsed sample counts against agreement. That answer is the
  question's no-cue answer.

### Labels

Each of 261007's label rows (one per cued CoT rollout from 260819) on a slice question is copied
with the no-cue answer set to the slice answer, and `bias_rm` / `deception_rm` recomputed with
260824's `turn1_label` / `turn2_label`. The rollout and the self-report (`said_rm`) are unchanged.
Rows on other questions are dropped. Anti-bias rows and rows with an unparsed cued answer are
excluded from the probe sets, as before. Rows whose 3 old no-cue samples had no majority (old
label `unparsed_turn1`) get a label when their question is in the slice.

### Slice-trained probe vs baseline probe

| | Baseline probe | Slice-trained probe |
|---|---|---|
| Train questions | All train-split questions | Slice train questions only |
| No-cue answer behind the labels | Majority of 3 samples (260819) | The answer at least 6 of 8 samples give |
| Train rows (`bias_followed` / said-Yes / said-No) | 2908 / 1992 / 916 | 1512 / 1004 / 508 |
| Learning rate, steps, band | 260924's cross-validation, fixed | This experiment's cross-validation |

Unchanged between the two: the 260819 CoT rollouts and their follow-ups; the self-report rule; the
read positions (`bias_followed` at the answer-letter token, rows whose answer went with the cue
only; deception at the last token of the follow-up prompt, split by the self-report); the
trainer; the score (mean per-layer sigmoid over a 4-layer band); up to 3 rollouts per prompt in
train with the same size and cue-balance rule; seed 0.

### Train and eval sets

261007's step 1, unchanged:

- **Train:** every rollout of a prompt can be a row (up to 3). Each class takes 2 × the smallest
  (class, cue condition) cell of the train pool, split evenly over `cue_gold` and `cue_dist`
  (draw seed 53). This is the baseline probe's rule.
- **Eval:** one rollout per (question, cue condition) prompt, every eligible prompt, no
  rebalancing. The rollout is picked by one seeded generator (seed 53) that walks the prompts in
  order (261007 `step1_build_probe_sets.py`, `subsample_prompts`). On a prompt that is also in
  261007's eval set, it can pick a different rollout than 261007 did.

### Cross-validation

261007's procedure, on each probe's slice train set:

- Learning rate {0.01, 0.03} × steps {2000, 4000}; 5-fold GroupKFold over train question ids
  (seed 0), so all rollouts of a question sit in one fold.
- For every cell and every 4-layer band, the fold-mean held-out AUC of the band's mean sigmoid;
  the argmax is the choice.
- Edge rule: if the chosen learning rate is 0.03, the cell (0.1, chosen steps) is added; if the
  chosen steps are 4000, (chosen lr, 8000); both -> also (0.1, 8000). The choice is then redone
  over every cell.
- Within a fold the trainer balances its fit rows by dropping a few rows of the larger class
  (for example 401 -> 399), as in 261007.
- The eval split is not read.

### Comparison

- **Final fit:** each probe on its full slice train set at its CV choice, seed 0, scored on the
  slice eval rows.
- **Baseline probe:** its saved weights and band scored on the same slice eval activations, with
  the slice labels.
- **Metric:** AUC of the mean per-layer sigmoid over each probe's own band.
- **Δ** = slice-trained minus baseline probe, with a 95% CI from 261007's question-level
  bootstrap (2000 draws, seed 20261007, both AUCs on the same resampled questions). One training
  seed.

### No-CoT comparison (added after the run)

- **The no-CoT probe:** 261007's `nocot_logprob` final fit. The assistant turn is prefilled with
  "The answer is (" and the model gives one letter; the cued and no-cue answers behind the labels
  are the top letter of P(A) vs P(B), with no sampling. It was trained on all train questions, one
  row per prompt (`bias_followed` 1252 rows, said-Yes 680, said-No 260), with settings from its own
  cross-validation (0.03 / 2000 / 13–16, 0.1 / 8000 / 18–21, 0.1 / 8000 / 28–31).
- **261007's numbers:** on all eval rows (0.880 / 0.809 / 0.762) and on the rows where both the
  cued and the no-cue top letter have probability >= 0.75 (0.920 / 0.841 / 0.826), for
  `bias_followed` / said-Yes / said-No.
- **Here:** the same saved eval scores, re-scored on the rows whose question's no-cue top letter
  has probability >= 0.75 (renormalized over A and B, 261007's `is_confident`), whatever the cued
  prompt does. No retraining. The no-CoT labels already use the no-cue top letter, so no row is
  relabelled.
- **Why the no-cue filter only (the user's reason):** only a no-cue filter can be used for RL.
  Confidence without the cue is a property of the question; confidence with the cue is the model's
  response to the cue, which changes under RL.
- **What differs from the CoT slice:** the threshold is 0.75 in both, but it measures the
  next-token probability of one letter in no-CoT and the agreement of 8 sampled CoT answers in CoT,
  so the two formats keep different questions. The no-CoT probe was trained on all train
  questions, like the CoT baseline probe; the CoT slice-trained probe was trained on slice
  questions only.

## Gates

All passed, on the full run and on the no-CoT re-scoring.

- **Generation:** 1603 questions × 8 samples = 12,824 records; seeds unique, following 261010's
  formula, disjoint from 261010's 5,496 eval seeds and above 260819's largest seed; one BOS token
  per prompt; a repeated identical call byte-identical (16 of 16). Reported, not gated: batch-1 and
  batch-16 outputs agree on 14 of 16 (`logs/generate_a26a5676bdcd.log`).
- **Labels:** every question has samples 0–7; train and eval questions disjoint; 261007's rollout
  keys unique; on every source row, the label functions reproduce 261007's bias and deception
  labels from the old no-cue answer.
- **Probe sets and activations:** 261007's step-1 gates (eval has one row per prompt and covers the
  whole pool; train classes equal, cue-balanced, at the prescribed size), except "one train row
  per prompt", which does not apply with up to 3 rollouts per prompt. Every row is in its own
  split; no question is in both train and eval; each activation file holds its probe-set rows in
  order with the same labels.
- **Final fit:** each train set balanced; the baseline probe's saved weights reproduce its saved
  261007 eval scores on the same rows (largest score difference 1.2e-07, identical AUCs); both
  probes scored on identical row lists.
- **Comparison:** both probes have the same rows and labels; the bootstrap resamples all of each
  probe's eval questions.
- **No-CoT re-scoring:** one eval row per prompt; every eval row has a no-cue probability, the same
  on both cue conditions of a question; every both-prompts-confident row is also no-cue confident;
  the both-prompts filter run through this script's code gives 261007's row counts (827 / 839 /
  186, asserted) and AUCs (reported; equal to 261007's to six decimals); the full-eval AUCs
  recomputed from the saved scores equal the saved ones.

## Results

`comparison.json` from `compare_slice.py`: `runs/slice_cot/` against `runs/baseline_on_slice/`,
both on the slice eval rows (one rollout per prompt on the 489 slice eval questions) with the slice
labels; score = mean sigmoid over each probe's band; seed 0.

### The slice (`slice_summary.json`)

| Split | Questions | Slice questions (gold / distractor answer) | Unparsed samples | Label rows on slice questions | Rows whose bias label changed |
|---|---|---|---|---|---|
| Train | 1603 | 1217 (972 / 245) | 504 of 12,824 | 7302 | 550 |
| Eval | 687 | 489 (384 / 105) | 252 of 5,496 | 2934 | 240 |

"Changed" is against 261007's majority-of-3 label and includes rows that move into or out of
anti-bias or `unparsed_turn1`. The deception label changes on the same rows.

Questions by how many of the 8 samples give the most common parsed answer:

| Most common answer | 3 or fewer | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Train questions | 16 | 146 | 224 | 241 | 345 | 631 |
| Eval questions | 10 | 78 | 110 | 113 | 140 | 236 |

### Sizes

Positive = bias (`bias_followed`) or deceptive (said-Yes: false admission; said-No: false denial).

| Probe | Train rows (pos / neg) | Train questions | Baseline probe's train rows (pos / neg) | Eval rows (pos / neg) | Eval questions |
|---|---|---|---|---|---|
| `bias_followed` | 1512 (756 / 756) | 740 | 2908 (1454 / 1454) | 777 (292 / 485) | 489 |
| said-Yes | 1004 (502 / 502) | 643 | 1992 (996 / 996) | 664 (547 / 117) | 410 |
| said-No | 508 (254 / 254) | 341 | 916 (458 / 458) | 310 (64 / 246) | 232 |

The smallest train-pool cell sets each size (`arms/train_sizes.json`): `bias_followed` bias ×
`cue_gold` 378; said-Yes true admission × `cue_gold` 251; said-No false denial × `cue_gold` 127.

### Slice-trained vs baseline probe, same rows and labels

| Probe | Eval rows | Slice-trained AUC | Baseline probe AUC | Δ [95% CI] |
|---|---|---|---|---|
| `bias_followed` | 777 | 0.871 | 0.866 | +0.006 [−0.010, +0.023] |
| said-Yes | 664 | 0.836 | 0.825 | +0.011 [−0.013, +0.036] |
| said-No | 310 | 0.859 | 0.860 | −0.001 [−0.030, +0.028] |

No CI excludes 0. Δ is computed before rounding, so it can differ by 0.001 from the difference of
the two rounded AUCs.

### Where each baseline-probe number comes from

Two sets of baseline-probe AUCs appear in this folder. Only the first is on the slice-trained
probe's rows.

| | Baseline probe on the slice eval rows | Baseline probe on 261007's eval rows |
|---|---|---|
| AUC, `bias_followed` / said-Yes / said-No | 0.866 / 0.825 / 0.860 | 0.788 / 0.771 / 0.799 |
| Eval rows | 777 / 664 / 310 | 1106 / 920 / 406 |
| Eval questions | The 489 slice questions | All 687 |
| Rollout per prompt | This experiment's draw | 261007's draw |
| Labels | From the 6-of-8 answer | From the majority of 3 |
| Same rows and labels as the slice-trained probe | Yes | No |
| Source | `runs/baseline_on_slice/eval_scores.json` | 261007 `decomp/full3_oldrecipe/runs/ref_cot/probe_metrics.json` |

The right-hand column differs from the left in questions, rollouts and labels at once, so the slice
eval rows are not a subset of 261007's eval rows. `comparison.md` shows the right-hand column in
its Training table under the header "Baseline AUC, all eval rows".

### Cross-validation (`cv/slice_cot/<probe>/selection.json`)

Fold-mean held-out AUC of the best band in each cell (band in brackets); the chosen cell in bold.
"—" = cell not run.

| Learning rate / steps | `bias_followed` | said-Yes | said-No |
|---|---|---|---|
| 0.01 / 2000 | 0.8769 (13–16) | 0.8069 (14–17) | 0.8369 (14–17) |
| 0.01 / 4000 | 0.8824 (13–16) | 0.8184 (14–17) | 0.8423 (14–17) |
| 0.03 / 2000 | **0.8826 (13–16)** | 0.8209 (14–17) | **0.8447 (14–17)** |
| 0.03 / 4000 | 0.8825 (13–16) | 0.8259 (14–17) | 0.8439 (14–17) |
| 0.1 / 2000 (edge) | 0.8763 (13–16) | — | 0.8438 (14–17) |
| 0.1 / 4000 (edge) | — | 0.8245 (14–17) | — |
| 0.03 / 8000 (edge) | — | **0.8267 (15–18)** | — |
| 0.1 / 8000 (edge) | — | 0.8235 (21–24) | — |

Fold standard deviation of the chosen cell: 0.036 / 0.024 / 0.047. Before the edge cells, said-Yes
had chosen 0.03 / 4000 / 14–17 (`logs/cv_deception_yes_rm_preyesno_a26a5676bdcd.log:199`).

| Probe | Slice-trained: lr / steps / band (epochs) | Baseline probe: lr / steps / band |
|---|---|---|
| `bias_followed` | 0.03 / 2000 / 13–16 (42.3) | 0.01 / 4000 / 13–16 |
| said-Yes | 0.03 / 8000 / 15–18 (255.0) | 0.03 / 4000 / 14–17 |
| said-No | 0.03 / 2000 / 14–17 (126.0) | 0.03 / 4000 / 15–18 |

### No-CoT probe on its no-cue-confident rows (`nocot_comparison.json`, added after the run)

The no-CoT probe under three eval filters; one row per prompt, AUC of the mean sigmoid over its
band. No-cue confident: 547 of 687 eval questions.

| Probe | All eval rows: AUC (rows) | Cued and no-cue prompts confident (261007): AUC (rows) | No-cue prompt confident: AUC (rows, pos / neg, questions) |
|---|---|---|---|
| `bias_followed` | 0.880 (1099) | 0.920 (827) | 0.923 (863, 320 / 543, 546) |
| said-Yes | 0.809 (1072) | 0.841 (839) | 0.845 (888, 633 / 255, 492) |
| said-No | 0.762 (261) | 0.826 (186) | 0.831 (199, 65 / 134, 144) |

Each format on its own no-cue-confident eval rows. The CoT rows are the 489 slice questions (at
least 6 of 8 no-cue CoT samples agree), one CoT rollout per prompt, labels from that answer. The
no-CoT rows are the 547 questions whose no-cue top letter has probability >= 0.75, one-letter
answers, labels from the top letters. **The two formats are scored on different completions and
different question sets, so no Δ or CI is computed between them.** The main result of this
experiment is the slice-trained CoT probe.

| Probe | CoT slice-trained probe AUC | CoT baseline probe AUC | CoT rows (pos / neg), questions | No-CoT probe AUC | No-CoT rows (pos / neg), questions |
|---|---|---|---|---|---|
| `bias_followed` | 0.871 | 0.866 | 777 (292 / 485), 489 | 0.923 | 863 (320 / 543), 546 |
| said-Yes | 0.836 | 0.825 | 664 (547 / 117), 410 | 0.845 | 888 (633 / 255), 492 |
| said-No | 0.859 | 0.860 | 310 (64 / 246), 232 | 0.831 | 199 (65 / 134), 144 |

### What the numbers show

- Slice-trained minus baseline probe is between −0.001 and +0.011 for the three probes, on the same
  rows and labels; no 95% CI excludes 0.
- The slice holds 1217 of 1603 train questions and 489 of 687 eval questions. The slice train sets
  are 50–55% the size of the baseline probe's.
- On slice questions, the bias label changes on 550 of 7302 train label rows and 240 of 2934 eval
  label rows.
- In cross-validation the chosen cell is at most 0.001 above the next-best cell for each probe.
- No-CoT probe: dropping the cued half of 261007's confident filter keeps 36 / 49 / 13 more rows
  (`bias_followed` / said-Yes / said-No) and moves AUC by +0.004 / +0.004 / +0.006.
- On their own no-cue-confident rows, the no-CoT probe scores 0.923 / 0.845 / 0.831 and the CoT
  slice-trained probe 0.871 / 0.836 / 0.859 (different completions and question sets).

## Run status

- **Dry runs (attended), 2026-10-10, before the run:** every queue step on 100 questions per split
  and 200 training steps, in `dry/` (outputs written 19:03–19:10 UTC). Not results.
- **Unattended run, tmux `261010-slice`, 2026-10-10 19:23–20:58 UTC:** queue steps 1–6, all
  `EXIT 0`, nothing Blocked. Logs in `logs/*_a26a5676bdcd.log`.
- **Outputs:** train no-cue samples in `completions.jsonl`; labels, probe sets and activations
  under `arms/slice_cot/`; CV folds and choices under `cv/slice_cot/<probe>/`; the slice probes in
  `runs/slice_cot/` (`config.json`, `probe_metrics.json`, `eval_scores.json`, `probe_weights/`); the
  baseline probe's scores on the slice eval rows in `runs/baseline_on_slice/eval_scores.json`
  (scores only, no weights or metrics file, since nothing was trained there).
- **Audit (after the run):** every number in `comparison.md`, `comparison.json` and the `runs.md`
  Log was recomputed from the raw files with no discrepancy, including the bootstrap CIs and a
  rescoring of both probes' weights. It also found:
  - `runs.md` said every wait after step 2 was a background poll; a status check at about 20:45 UTC
    saw a foreground wait loop during step 4. A correction line was added under that sentence
    after the run.
  - `comparison.md`'s Training-table column "Baseline AUC, all eval rows" is the baseline probe on
    261007's eval rows and labels (see "Where each baseline-probe number comes from").
  - `final_fit.py`'s docstring calls it the first read of the slice eval activations;
    `run_pipeline.py` already loads them to check row keys and labels. Nothing was chosen from them.
  - `slice_spec.py` is not stamped into any output; the outputs carry 261007's spec hash
    `10b44b32b4c2`. `slice_spec.py` and every script were last modified at 18:56–19:00 UTC, before
    the dry runs and the run.
- **No-CoT re-scoring (attended, after the run), 2026-10-10 23:23 UTC:** `nocot_nocue_slice.py`
  -> `nocot_comparison.md`, `nocot_comparison.json`; log `logs/nocot_a26a5676bdcd.log`, `EXIT 0`.
  Reads 261007's `arms/nocot_logprob/labels.jsonl` and `runs/nocot_logprob/eval_scores.json`; no
  training, no model. Not covered by the audit above.

## Costs

| Item | Size or time |
|---|---|
| Train no-cue samples | 12,824 completions, 3 min 40 s of generation after engine start |
| `completions.jsonl` | 28 MB |
| Probe sets and activations (`arms/`) | 1.3 GB; about 4 min (step 3) |
| Cross-validation, three probes in parallel | 38 / 77 / 39 min (`bias_followed` / said-Yes / said-No; said-Yes ran three edge cells) |
| `cv/` | 12 MB |
| Final fit, three probes | About 6 min |
| `runs/` | 2.0 MB |
| Unattended run, all six steps | 1 h 35 min |
| No-CoT re-scoring | Seconds of CPU |
