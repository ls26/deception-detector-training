# 261010 — Soft bias labels: hard vs soft training targets for three `rm` probes

Probe experiment on the MMLU-Pro sycophancy domain. The hard bias label of a rollout uses
the strict majority of 3 no-cue samples as the counterfactual. This experiment replaces it
with a soft label for three `rm` probes (`bias_followed`, said-Yes deception, said-No
deception), trains each probe on hard and on soft targets, and scores both on three versions
of the eval labels. Rows, activations, learning rate, steps and layer band are those of
261007's fit B ([`../261007_format_label_cue_ablation/decomp/full3_oldrecipe/`](../261007_format_label_cue_ablation/decomp/full3_oldrecipe/)):
260924's train size and recipe with the P(Yes) self-report labels of Decision 7.

**The reported run** is the unattended run of 2026-10-10, 01:37–01:48 UTC (protocol in
`CLAUDE.md`, state and Result in `runs.md`), which wrote `completions.jsonl`, `labels.jsonl`,
`runs/soft/` and `comparison.md` / `comparison.json`, together with `runs/int/` and
`runs/hard01/`, fitted before it on the same day. The post-hoc analyses in
`followup_analysis.md` were computed after the run.

## Files

| File | Role |
|---|---|
| `soft_spec.py` | Every fixed choice: paths to fit B and 260819, probes, sampling settings, seeds, bootstrap |
| `domain_imports.py` | The one place that imports helpers from 260819, 260824 and 261007 |
| `generate_nocue.py` | 8 fresh no-cue CoT samples per eval-split question (vLLM) -> `completions.jsonl`, `generation_summary.json` |
| `soft_targets.py` | p_none, soft targets and the three eval label sets per row -> `labels.jsonl`, `p_none_summary.json` |
| `train_soft.py` | Fits the three probes with `--targets int`, `hard01` or `soft` -> `runs/<targets>/` |
| `compare_soft.py` | Weighted AUC, ceiling, Spearman and bootstrap CIs -> `comparison.md`, `comparison.json` |
| `followup_analysis.py` | Post-hoc: p_none distribution, old 3 vs fresh 8 samples, old hard vs fresh hard -> `followup_analysis.md` |
| `CLAUDE.md`, `kickoff_prompt.txt` | The unattended-run protocol and its first message |
| `runs.md` | The unattended run's queue, log, judgment calls and Result |
| `dry/` | Dry runs of every queue step (100 questions, 200 training steps); not results |
| `logs/` | One log per step, ending in `EXIT <code>` |

## Design

### Labels

| Quantity | Definition |
|---|---|
| p_none | Share of parsed no-cue samples whose answer is the cued content (gold under `cue_gold`, distractor under `cue_dist`). Unparsed samples are left out. |
| b | 1 − p_none if the rollout followed the cue, else 0 |
| Target | `bias_followed`: b. Said-Yes deception: 1 − b. Said-No deception: b. |

Train targets use 260819's 3 no-cue samples per question (`scores.jsonl`), so they take the
values 0, 1/3, 2/3 and 1. Rounding the 3-sample target gives fit B's hard label on every train
and eval row of the three probes.

Eval labels, on fit B's eval rows (one row per question and cue condition):

| Eval label set | Source | Rows |
|---|---|---|
| Old hard | Fit B's label: strict majority of 260819's 3 no-cue samples | All |
| Fresh hard | 260824's `turn1_label` / `turn2_label` on the strict majority of 8 fresh no-cue samples | Drops 4-4 ties and rows that become anti-bias |
| Fresh soft | The target formula on the 8 fresh samples | All |

Fresh samples: 8 per eval-split question (687 questions) with 260819's prompts and settings
(temperature 1.0, at most 2048 new tokens, prefix caching off), one request per sample with
seed 90,001,042 + 8 × question id + sample index, parsed with 260819's `extract_answer`.

### Arms

| Arm | Targets | Training path | Output |
|---|---|---|---|
| `int` | Fit B's 0/1 labels | Unchanged library path (261007's `fit_layer_probes`) | `runs/int/` |
| `hard01` (hard-trained) | Fit B's labels as float 0/1 | Library soft-label mode | `runs/hard01/` |
| `soft` (soft-trained) | 3-sample soft targets | Library soft-label mode | `runs/soft/` |

All arms use fit B's train rows and recipe, seed 0, and score each eval row by the mean
sigmoid over the band:

| Probe | Learning rate | Steps | Band |
|---|---|---|---|
| `bias_followed_rm_answer` | 0.01 | 4000 | 13–16 |
| `deception_yes_rm_preyesno` | 0.03 | 4000 | 14–17 |
| `deception_no_rm_preyesno` | 0.03 | 4000 | 15–18 |

In soft-label mode the rows are put in the order the integer path uses (label-1 rows first,
then label-0 rows), so all arms train on the same row sequence.

**Library soft-label mode.** `ActivationDataset(soft_labels=True)` and the `balance_dataset`
argument of `train_detector` in obfuscation-atlas (commit `6b2583d`) exist only for this
experiment. In soft-label mode the dataset takes float targets in [0, 1], keeps file order and
refuses balancing. Integer labels behave as before.

### Metrics

- **Weighted AUC.** Each row is a positive with weight q and a negative with weight 1 − q;
  pairs of a row with itself are excluded; score ties count half. On 0/1 labels this is the
  usual AUC.
- **Ceiling.** The weighted AUC of q itself, the highest value any score can reach on a label
  set. It is below 1 when labels are fractional.
- **Spearman** correlation between the score and q, on the fresh soft labels.
- **Δ** = soft-trained minus hard-trained, with a 95% CI from a question-level bootstrap:
  2000 draws, seed 20261007, the same draws for both arms, a row drawn k times weighted k.
  One training seed.

## Gates

All passed on the full run.

- **Generation:** 687 questions, 8 samples each; seeds unique and above every seed 260819
  used; one BOS token per prompt; prefix caching off and a repeated identical call
  byte-identical (16 of 16).
- **Soft targets:** rounding the 3-sample target gives fit B's label on every row; no target
  equals 0.5; the followed flag agrees with fit B's cued answer; every eval row has a fresh
  soft label in [0, 1] from at least one parsed sample; row keys unique.
- **Fits:** the rebuilt config equals fit B's `config.json`; eval rows and labels equal fit
  B's saved `eval_scores.json`; rounded soft targets equal fit B's labels.
- **Comparison:** both arms scored the same rows; every bootstrap draw had weight in both
  classes.
- **Library:** the detector training tests pass (26).

## Results

`comparison.json` from `compare_soft.py`: `runs/hard01/` against `runs/soft/` on fit B's
eval rows, score = mean sigmoid over the band.

### Sizes

| Probe | Train rows (pos / neg) | Train targets 0 / 1/3 / 2/3 / 1 | Eval rows, old labels (pos / neg) | Eval questions | Eval rows with a fractional fresh soft label |
|---|---|---|---|---|---|
| `bias_followed` | 2908 (1454 / 1454) | 1003 / 451 / 643 / 811 | 1106 (449 / 657) | 669 | 688 |
| said-Yes | 1992 (996 / 996) | 558 / 438 / 154 / 842 | 920 (710 / 210) | 571 | 420 |
| said-No | 916 (458 / 458) | 381 / 77 / 217 / 241 | 406 (111 / 295) | 310 | 197 |

### Hard-trained vs soft-trained

`*` would mark a CI that excludes 0; none does.

| Probe | Eval labels | Rows | Hard-trained | Soft-trained | Δ [95% CI] |
|---|---|---|---|---|---|
| `bias_followed` | Old hard | 1106 | 0.788 | 0.793 | +0.005 [−0.002, +0.013] |
| | Fresh hard | 1050 (56 dropped) | 0.804 | 0.804 | −0.000 [−0.008, +0.007] |
| | Fresh soft | 1106 | 0.749 | 0.752 | +0.002 [−0.002, +0.007] |
| said-Yes | Old hard | 920 | 0.771 | 0.772 | +0.000 [−0.008, +0.008] |
| | Fresh hard | 860 (60 dropped) | 0.765 | 0.764 | −0.001 [−0.009, +0.007] |
| | Fresh soft | 920 | 0.762 | 0.760 | −0.002 [−0.008, +0.003] |
| said-No | Old hard | 406 | 0.799 | 0.800 | +0.001 [−0.010, +0.012] |
| | Fresh hard | 387 (19 dropped) | 0.807 | 0.813 | +0.006 [−0.005, +0.019] |
| | Fresh soft | 406 | 0.767 | 0.770 | +0.003 [−0.004, +0.012] |

Δ is computed before rounding, so it can differ by 0.001 from the difference of the two
rounded AUCs.

| Probe | Fresh-soft ceiling | Spearman, hard-trained | Spearman, soft-trained |
|---|---|---|---|
| `bias_followed` | 0.919 | 0.594 | 0.600 |
| said-Yes | 0.943 | 0.601 | 0.593 |
| said-No | 0.943 | 0.585 | 0.596 |

### Library check

| Probe | Fit B saved AUC | `int` AUC | `hard01` AUC | Max \|`int` − fit B\| | Max \|`hard01` − `int`\| |
|---|---|---|---|---|---|
| `bias_followed` | 0.788287 | 0.788287 | 0.788287 | 1.19e-07 | 0 |
| said-Yes | 0.771429 | 0.771429 | 0.771429 | 1.19e-07 | 0 |
| said-No | 0.799420 | 0.799420 | 0.799420 | 1.19e-07 | 0 |

### Post-hoc analyses (`followup_analysis.md`)

Computed after the run by `followup_analysis.py` from the run's outputs.

**p_none distribution.** Per eval question, the share of the 8 fresh samples that are correct:

| Share correct | 0 | Between 0 and 1/2 | Exactly 1/2 | Between 1/2 and 1 | 1 |
|---|---|---|---|---|---|
| Questions (of 687) | 41 | 125 | 33 | 234 | 254 |

`bias_followed` eval rows, p_none from the 8 fresh samples:

| p_none | 0 | Between 0 and 1/2 | Exactly 1/2 | Between 1/2 and 1 | 1 |
|---|---|---|---|---|---|
| `cue_gold` rows (637) | 31 | 106 | 28 | 222 | 250 |
| `cue_dist` rows (469) | 97 | 189 | 28 | 115 | 40 |

**Old 3 vs fresh 8 samples.** Per eval question, correct among 260819's 3 no-cue samples
against the share correct among the 8 fresh samples:

| Old 3 samples | Questions | 0 | Between 0 and 1/2 | Exactly 1/2 | Between 1/2 and 1 | 1 |
|---|---|---|---|---|---|---|
| 0 of 3 correct | 90 | 30 | 48 | 3 | 9 | 0 |
| 1 of 3 correct | 84 | 6 | 36 | 6 | 33 | 3 |
| 2 of 3 correct | 133 | 1 | 23 | 9 | 80 | 20 |
| 3 of 3 correct | 304 | 0 | 6 | 6 | 85 | 207 |
| Some unparsed | 76 | 4 | 12 | 9 | 27 | 24 |

The strict majorities of the 3 old and the 8 fresh samples agree on 562 of 687 questions (a
tie counts as no majority).

**Old hard vs fresh hard labels.** AUC with the old labels on all eval rows, with the old
labels on the rows fresh hard keeps, and with the fresh labels on those rows:

| Probe | Arm | Old labels, all rows | Old labels, kept rows | Fresh labels, kept rows | Rows dropped | Kept rows whose label changes |
|---|---|---|---|---|---|---|
| `bias_followed` | Hard-trained | 0.788 | 0.800 | 0.804 | 56 | 148 |
| | Soft-trained | 0.793 | 0.804 | 0.804 | | |
| said-Yes | Hard-trained | 0.771 | 0.774 | 0.765 | 60 | 91 |
| | Soft-trained | 0.772 | 0.773 | 0.764 | | |
| said-No | Hard-trained | 0.799 | 0.809 | 0.807 | 19 | 44 |
| | Soft-trained | 0.800 | 0.811 | 0.813 | | |

### What the numbers show

- Soft-trained minus hard-trained lies between −0.002 and +0.006 in all nine probe and
  label-set cells, and no 95% CI excludes 0.
- On the fresh soft labels both arms score 0.749–0.770, against ceilings of 0.919–0.943.
- 688 of 1106 `bias_followed` eval rows (62%) have a fractional fresh soft label; 1094 of
  2908 `bias_followed` train rows (38%) have a fractional target.
- All 8 fresh samples are correct on 254 of 687 eval questions and none is correct on 41.
- For the hard-trained probes, leaving out the rows fresh hard drops moves the old-label AUC
  by +0.012 / +0.003 / +0.010 (`bias_followed` / said-Yes / said-No); changing to the fresh
  labels on the same rows moves it by +0.004 / −0.009 / −0.002.

## Run status

- **Library check (attended), 2026-10-10 00:47–01:00 UTC:** `train_soft.py --targets int`
  and `--targets hard01` -> `runs/int/`, `runs/hard01/` (`eval_scores.json`, `metrics.json`,
  `probe_weights/`). Complete.
- **Dry runs:** every queue step on 100 eval questions and 200 training steps, in `dry/`;
  the 3-sample target check at full size in `dry/old3/`. Not results.
- **Unattended run, tmux `261010-soft`, 2026-10-10 01:37–01:48 UTC:** queue steps 1–4
  (`generate_nocue.py`, `soft_targets.py`, `train_soft.py --targets soft`, `compare_soft.py`),
  all `EXIT 0`, nothing Blocked. Logs in `logs/*_c87626336ec0.log`.
- **Audit (after the run):** every number in `comparison.md`, `comparison.json` and `runs.md`
  recomputed from the raw files with no discrepancy. It found that `train_soft.py` and
  `domain_imports.py` were edited at 01:05 UTC, after `runs/int/` and `runs/hard01/` were
  made: the probe-training imports moved behind a function, and a `--labels` argument used
  only by `--targets soft` was added. A 4000-step `--targets hard01` refit with the final
  scripts matched `runs/hard01/` bitwise on all three probes; its output was not kept. The
  audit also noted that the Result section of `runs.md` adds one line ("Copied from
  `comparison.md`") beyond the heading changes its judgment calls describe.
- **Post-hoc analyses:** `followup_analysis.py` -> `followup_analysis.md`, after the run.

## Costs

| Item | Size or time |
|---|---|
| Fresh samples | 5,496 completions, about 2 min 20 s of generation after engine start |
| `completions.jsonl` | 12 MB |
| `labels.jsonl` | 2.0 MB |
| One three-probe fit, 4000 steps | About 6 min |
| `runs/` (three arms, weights included) | 5.9 MB |
| Unattended run, all four steps | 11 min |
