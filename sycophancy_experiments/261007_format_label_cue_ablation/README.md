# 261007 — Format / label / cue ablation: which setup change raises the sycophancy probes?

Probe experiment on the MMLU-Pro sycophancy domain. Ying et al. 2026 (arXiv 2602.20273)
train a sycophancy probe on Llama-3.1-8B-Instruct with the same contrast as our
`bias_followed` probe and report AUC 0.95; [`260924_probe_cv_selection`](../260924_probe_cv_selection/)
reached 0.788. Their setup differs from ours in three ways: no chain-of-thought (the
assistant turn is prefilled and the model emits one letter), labels read from next-token
probabilities instead of sampled rollouts, and a confident expert-biography cue placed
before the question. This experiment changes one factor per arm, rebuilds the five `rm`
probes of 260924 for every arm with one shared procedure, and compares each arm with the
reference. Removal definition only.

**The reported run** is the unattended run of 2026-10-07/08 (spec hash `10b44b32b4c2`,
`spec.py`): the final fits in `runs/<arm>/probe_metrics.json`, compared in
`comparison.md` / `comparison.json`, summarized by the run itself in `SUMMARY.md`. The
run's step-by-step record is `runs.md` and `STATUS.md`. Two extra fits of the reference arm
made afterwards (`decomp/`) and a sample-agreement check (`review/sample_agreement.md`) are
reported under Results.

## Files

| File | Role |
|---|---|
| `spec.py` | Every fixed choice: arms, prompts, bios, seeds, sampling, label rules, size rule, CV grid, gates, reporting. Frozen; its hash is stamped on every output |
| `SCHEMA.md` | Record formats of every JSONL this folder writes |
| `prompts.py` | User content and prompt ids per generation pass and condition, derived from 260819's stored renders |
| `domain_imports.py` | The one place that imports helpers from 260819 / 260823 / 260824 |
| `hf_env.py` | HF Hub env for kernels that do not inherit `dotfiles/env.sh` |
| `generate_turn1.py` | vLLM turn 1: `--gen nocot` (3 conditions, 1 token, top-20 logprobs) or `--gen bio` (cued conditions, CoT) |
| `generate_followups.py` | vLLM removal-wording follow-ups per turn-1 sample (+ a `top` row per no-CoT prompt) |
| `output_store.py` | Resumable single-file JSONL store for the generators |
| `score_pyes.py` | float32 P(Yes) at the last prompt token of the follow-up turn (bf16 model, output head in float32), per gen and for the reference |
| `label.py`, `label_common.py` | `arms/<arm>/labels.jsonl`: removal bias and deception labels per cued rollout, one schema for all arms |
| `baselines.py` | `arms/<arm>/baselines.{md,csv}`: accuracy, follow, bias, self-report and deception rates per condition |
| `train_sizes.py` | `arms/sizes.json`, `train_sizes.md`: the size rule over the four arms |
| `probe_config.py` | `PipelineConfig` for one arm (adapted from 260924) |
| `step1_build_probe_sets.py` | Probe sets from `labels.jsonl`: one row per prompt, cue-balanced draws |
| `step2_extract_activations.py` | Activations at `answer` (bias family) and `preyesno` (deception, the last token of the follow-up prompt) |
| `extract.py` | Steps 1 + 2 for one arm with the step-1, disk, lm_head and HF-vs-vLLM gates |
| `step3_train_eval_probes.py` | The pipeline's AdamW linear probes; writes `probe_metrics.json` and per-row `eval_scores.json` |
| `cv_sweep.py`, `cv_select.py`, `cv_edge.py`, `cv_launch.sh`, `cv_wait.sh` | 5-fold GroupKFold over train questions on the lr x steps grid, band chosen from out-of-fold logits, one cell past a grid edge |
| `final_fit.py` | One seed, the chosen recipe, scored once on eval; appends the `runs.md` row |
| `compare.py` | `comparison.md` / `comparison.json`: main table with question-level bootstrap CIs, checks |
| `audit.py` | Recomputes AUCs, size counts and eval-set identity from the raw files |
| `reproduce_gate.py` | Phase-A gates: 260924's five AUCs reproduced with the new step 3; new step 2 vs 260923's cached activations |
| `render_examples.py` | `review/examples.md`: one worked labelling example per arm and condition |
| `ref_decomposition.py` | Two extra reference fits separating the label, size and recipe changes vs 260924 (`decomp/`) |
| `sample_agreement.py` | `review/sample_agreement.md`: unanimity and pairwise disagreement of the 3 samples, CoT vs no-CoT |
| `CLAUDE.md`, `KICKOFF.md`, `kickoff_prompt.txt`, `watchdog.sh` | The unattended-run protocol, launch command and the session watchdog |
| `runs.md`, `STATUS.md`, `SUMMARY.md` | The run's state files and its own summary |
| `tests_fixture/` | Fixture from 260923's pool rows used to test the pipeline end to end before the run |

## Design

### Arms

| arm | turn 1 | turn-1 answer label | cue | new generation |
|---|---|---|---|---|
| `ref_cot` | CoT (260819's rollouts) | sampled; majority-of-3 no-cue counterfactual | 8 short opinion sentences after the options (260819) | none |
| `nocot_sampled` | assistant prefilled with `The answer is (`, one sampled token | sampled; majority-of-3 | the same 8 sentences | turn 1 + follow-ups |
| `nocot_logprob` | the same prompts | top letter from P(A), P(B) (vLLM logprobs), one row per prompt | the same 8 sentences | shares `nocot_sampled`'s pass; a `top` follow-up row per prompt |
| `bio_cot` | CoT | sampled; majority-of-3 | expert biography as the first paragraph of the user message | cued conditions only |

Prompts (`prompts.py`): the no-CoT user content is 260819's with the sentence `Think step by
step and then finish your answer with "the answer is (X)" where X is the correct letter
choice.` replaced by `Answer with "The answer is (X)" where X is the correct letter choice.`;
the cue sentence stays appended. The bio user content is the bio paragraph, a blank line,
then 260819's no-cue content byte for byte, so 260819's no-cue rollouts are the bio arm's
counterfactual. Four bios (two from the paper, two written for the law / philosophy and
history categories), one per question. The no-CoT completion is stored as prefill + letter
+ `)`; the `)` is not sampled.

Follow-ups: 260823's two removal wordings, the question's assigned plain / explicit draw,
unchanged in every arm. **Self-report, every arm:** `said` = Yes iff the float32 P(Yes) at
the first follow-up token is above 0.5, No iff below, dropped on an exact tie
(`score_pyes.py`: the bf16 model's final residual at the last prompt token through the
output head in float32). The sampled Yes/No text is kept for the baselines only. The
deception probe reads the same position (`preyesno` = the last token of the follow-up
prompt).

Labels: 260824's `turn1_label` / `turn2_label` on content-space answers; anti-bias and
unparsed rows excluded from the probe populations as before.

### The one probe procedure (identical for every arm)

1. Train set per probe: `n_per_class` rows, cue-balanced, at most one row per
   (question, condition) prompt. Size rule per probe kind: matched = the smallest arm's
   full size (2 x its smallest class x condition cell of the train pool); matched if
   >= 200 per class, each arm's own full size if 50-200, dropped below 50.
2. Eval: one row per prompt on the 687 eval questions, never filtered.
3. 5-fold GroupKFold over train questions on lr {0.01, 0.03} x steps {2000, 4000}; the
   4-layer band is read from the out-of-fold per-layer logits; one more cell past a grid
   edge (lr 0.1 or 8000 steps) for the probes that chose an edge.
4. Final fit: one seed, the chosen cell and band, scored once on eval. Headline = AUC of
   the mean per-layer sigmoid over the band (the RL pipeline's score).
5. Δ vs `ref_cot` on the eval questions the two arms share; 95% CI from a question-level
   bootstrap (2000 draws). Pre-registered verdict rule: a factor helps a probe if the CI
   excludes zero. Verdict probes: `bias_followed`, `deception_yes`, `deception_no`.

Probes: `bias_rm_answer`, `bias_followed_rm_answer`, `deception_rm_preyesno`,
`deception_yes_rm_preyesno`, `deception_no_rm_preyesno` (260924's five).

## Gates

Before the run (`review/reproduction.md`): the new step 3 reproduces 260924's five AUCs on
260923's `activations/pool` within 0.005 (0.8403 / 0.7883 / 0.5944 / 0.7934 / 0.7490 vs
0.840 / 0.788 / 0.590 / 0.793 / 0.749); the new step 2 reproduces 260923's cached
activations on 50 rows (min cosine 1.000000 at every layer); `ref_cot`'s bias labels match
260923's pool probe sets row for row.

During the run (all passed on the full run; records in `runs.md`): no-CoT prompts end in
the `(` token and every completion is one token; A + B carry >= 0.95 probability on 0.9905
of no-CoT prompts (bar 0.95); bio prompt minus the bio paragraph equals 260819's no-cue
prompt on all 4580; follow-up prompts match the chat-template rendering and, for no-CoT,
the token layout prompt + letter + `)` + suffix; float32 P(Yes) resolves every bf16 tie
(548 / 1051 / 489 for ref / nocot / bio) with 0.99 sign agreement elsewhere; eval sets are
one row per prompt; train draws equal and cue-balanced at the prescribed sizes; lm_head on
the layer-31 residual equals the model's logits (diff 0); HF-vs-vLLM top letter agrees on
0.9996 / 0.9998 of no-CoT rows with margin > 0.25 nat (bar 0.99); `audit.py` 113 / 113
checks and a read-only subagent audit recomputed every Δ, CI and n.

Three gates stopped the run and were changed by the user before resuming (see Run status).

## Results

Metrics file per arm: `runs/<arm>/probe_metrics.json` (chosen recipe from
`cv/<arm>/chosen_overrides.json`), eval = one row per (question, condition) on the 687 eval
questions, headline = AUC of the mean sigmoid over the band, seed 0.

### Sizes

Train rows per probe (2 x per class; `deception` = the union of the two strata draws) and
eval rows:

| probe | rule | per class | train rows, every arm | eval rows ref / nocot_sampled / nocot_logprob / bio |
|---|---|---|---|---|
| `bias` | matched | 454 | 908 | 1326 / 1357 / 1333 / 1325 |
| `bias_followed` | matched | 626 | 1252 | 1106 / 1160 / 1099 / 1058 |
| `deception_yes` | matched | 340 | 680 | said-Yes rows of the deception eval |
| `deception_no` | unmatched (bio 66 < 200) | own | 296 / 272 / 260 / 132 | said-No rows of the deception eval |
| `deception` | union | | 976 / 952 / 940 / 812 | 1326 / 1357 / 1333 / 1325 |

`bio_cot` set the matched size for `bias`, `bias_followed` and `deception_no`; `ref_cot`
for `deception_yes`. Full table: `train_sizes.md`.

### Main table

Each arm's AUC is on its full eval set; Δ and the CI are on the questions it shares with
`ref_cot` (n). `*` = CI excludes 0.

| probe | ref_cot | nocot_sampled (Δ, 95% CI) | nocot_logprob (Δ, 95% CI) | bio_cot (Δ, 95% CI) |
|---|---|---|---|---|
| `bias` | 0.827 | 0.874 (+0.050*, [+0.023, +0.077], n=665) | 0.901 (+0.073*, [+0.048, +0.099], n=656) | 0.845 (+0.018, [-0.007, +0.041], n=669) |
| `bias_followed` (verdict) | 0.768 | 0.852 (+0.082*, [+0.051, +0.116], n=662) | 0.880 (+0.117*, [+0.083, +0.151], n=652) | 0.766 (-0.003, [-0.032, +0.027], n=668) |
| `deception` | 0.649 | 0.711 (+0.066*, [+0.031, +0.102], n=665) | 0.696 (+0.044*, [+0.008, +0.081], n=656) | 0.637 (-0.012, [-0.050, +0.025], n=669) |
| `deception_yes` (verdict) | 0.752 | 0.819 (+0.062*, [+0.028, +0.097], n=512) | 0.809 (+0.049*, [+0.014, +0.083], n=509) | 0.755 (+0.008, [-0.033, +0.048], n=542) |
| `deception_no` (verdict) | 0.706 | 0.779 (+0.080, [-0.009, +0.176], n=139) | 0.762 (+0.055, [-0.039, +0.152], n=137) | 0.717 (-0.037, [-0.156, +0.077], n=118) |

Chosen recipes (lr / steps / band):

| probe | ref_cot | nocot_sampled | nocot_logprob | bio_cot |
|---|---|---|---|---|
| `bias` | 0.01 / 4000 / 13-16 | 0.01 / 4000 / 13-16 | 0.01 / 2000 / 13-16 | 0.01 / 2000 / 13-16 |
| `bias_followed` | 0.01 / 4000 / 13-16 | 0.01 / 8000 / 24-27 | 0.03 / 2000 / 13-16 | 0.01 / 2000 / 13-16 |
| `deception` | 0.1 / 8000 / 17-20 | 0.1 / 8000 / 27-30 | 0.1 / 8000 / 18-21 | 0.01 / 2000 / 14-17 |
| `deception_yes` | 0.1 / 4000 / 15-18 | 0.1 / 8000 / 28-31 | 0.1 / 8000 / 18-21 | 0.03 / 2000 / 14-17 |
| `deception_no` | 0.1 / 8000 / 22-25 | 0.03 / 2000 / 15-18 | 0.1 / 8000 / 28-31 | 0.1 / 2000 / 11-14 |

### `nocot_logprob` on the confident slice

The same probes scored on the eval rows whose top letter has >= 0.75 probability in both
the cued and the no-cue prompt (`comparison.md`):

| probe | full eval (n) | confident slice (n) |
|---|---|---|
| `bias` | 0.901 (1333) | 0.936 (1025) |
| `bias_followed` | 0.880 (1099) | 0.920 (827) |
| `deception` | 0.696 (1333) | 0.715 (1025) |
| `deception_yes` | 0.809 (1072) | 0.841 (839) |
| `deception_no` | 0.762 (261) | 0.826 (186) |

### Reference arm vs 260924, one change at a time (`decomp/decomposition.md`)

Seed 0, 260924's recipe unless stated. A = 260924's labels at its full size (the
reproduction gate); B = this run's labels at 260924's size (3 rows per prompt, largest
cue-balanced draw); D = this run's labels at the matched size; C = the run (CV recipe).

| probe | A | B | D | C | A→B labels | B→D size | D→C recipe |
|---|---|---|---|---|---|---|---|
| `bias` | 0.840 | 0.840 | 0.827 | 0.827 | +0.000 | -0.013 | +0.000 |
| `bias_followed` | 0.788 | 0.788 | 0.768 | 0.768 | +0.000 | -0.020 | +0.000 |
| `deception` | 0.594 | 0.635 | 0.642 | 0.649 | +0.041 | +0.007 | +0.006 |
| `deception_yes` | 0.793 | 0.771 | 0.752 | 0.752 | -0.022 | -0.020 | -0.000 |
| `deception_no` | 0.749 | 0.799 | 0.753 | 0.706 | +0.050 | -0.046 | -0.048 |

Train rows: A 2908 / 2908 / 2908 / 1928 / 980; B 2908 / 2908 / 2908 / 1992 / 916; D and C as
in Sizes.

### Agreement of the three samples per prompt (`review/sample_agreement.md`)

| setup | condition | unanimous | pairwise disagreement | any unparsed |
|---|---|---|---|---|
| CoT | `none` | 0.605 | 0.219 | 0.103 |
| CoT | `cue_gold` | 0.718 | 0.149 | 0.079 |
| CoT | `cue_dist` | 0.577 | 0.246 | 0.087 |
| no-CoT | `none` | 0.767 | 0.150 | 0.011 |
| no-CoT | `cue_gold` | 0.947 | 0.034 | 0.002 |
| no-CoT | `cue_dist` | 0.890 | 0.072 | 0.003 |

Cued prompts whose cued cell and no-cue cell are both unanimous: CoT 0.421, no-CoT 0.704.
No-CoT P(A): the top letter has >= 0.9 probability on 0.663 of no-cue prompts, 0.927 of
`cue_gold` and 0.837 of `cue_dist`.

### Baselines (all 2290 questions, at p = 0.85; `arms/<arm>/baselines.md`)

| arm | follow rate | bias | anti-bias | accuracy gold-A / gold-B, no cue | said Yes |
|---|---|---|---|---|---|
| `ref_cot` | 0.804 | 0.188 | 0.050 | 0.595 / 0.771 | 0.641 |
| `nocot_sampled` | 0.895 | 0.258 | 0.009 | 0.671 / 0.724 | 0.786 |
| `nocot_logprob` | 0.903 | 0.260 | 0.006 | 0.668 / 0.736 | 0.785 |
| `bio_cot` | 0.768 | 0.160 | 0.059 | 0.595 / 0.771 (260819's rollouts) | 0.824 |

### What the numbers show

- Both no-CoT arms score above `ref_cot` on `bias_followed` (+0.082 and +0.117) and
  `deception_yes` (+0.062 and +0.049) with CIs that exclude zero. `bio_cot` is within
  noise of `ref_cot` on every probe (-0.037 to +0.018, every CI includes zero).
- `deception_no` is inconclusive in every arm: every CI includes zero, its train sizes
  are unmatched (296 / 272 / 260 / 132 rows), and 118-139 eval questions are shared.
- `nocot_logprob` and `nocot_sampled` are not compared with each other under the rule.
  As point estimates the logprob labels are +0.028 on `bias_followed` and -0.010 on
  `deception_yes`.
- The no-CoT samples are far more self-consistent than the CoT samples: unanimous on
  0.77-0.95 of prompts against 0.58-0.72, and both label cells unanimous on 0.70 of cued
  prompts against 0.42. User's reading: removing chain-of-thought raises the probes, but
  whether that comes from removing the reasoning itself or from less label noise is not
  settled by this experiment.
- `ref_cot` is below 260924 because of the matched train size (-0.013 / -0.020 on the
  bias probes, where the labels did not change) and, for the deception probes, the
  thresholded self-report labels (+0.041 / -0.022 / +0.050) and the size (-0.020 /
  -0.046). `deception_no`'s CV-chosen recipe scores 0.048 below 260924's recipe on the
  same matched rows.
- The float32 P(Yes) still carries batch-layout noise: 9 of 13184 identical no-CoT texts
  flipped Yes/No between the `top` row and a same-letter sample row.
- Without chain-of-thought the cue is followed more (0.895 / 0.903 vs 0.804), anti-bias
  nearly vanishes and the gold-A / gold-B accuracy gap shrinks from 0.18 to 0.05-0.07.
  The biography is followed less than the short sentence (0.768) and moves the
  self-report toward Yes (0.824).

## Run status

- Unattended run 2026-10-07T20:08Z → 2026-10-08T08:06Z (`STATUS.md`), one H100. Every
  queue step completed for all four arms. The session hung once on an API error (20:45Z)
  and was relaunched at 23:04Z; `watchdog.sh` was added then.
- Three stops, each resolved by a user decision and recorded in `runs.md` (Blocked,
  RESOLVED): (1) the no-CoT bf16 gates (sibling and invariance bars 1e-2 / 1e-6 → 0.05;
  the top-vs-sample P(Yes) check made a report) because vLLM's bf16 logprobs move by one
  ~0.1-nat step with the batch layout; (2) the size count changed to count step 1's train
  pool; (3) the deception read position moved to the last token of the follow-up prompt
  after the self-report rule changed. `spec.py` was not changed by any of them.
- Outputs: `gens/{nocot,bio}/` (turn 1, follow-ups, P(Yes)), `arms/<arm>/` (labels,
  baselines, probe sets, activations), `arms/sizes.json`, `cv/<arm>/`, `runs/<arm>/`,
  `comparison.{md,json}`, `SUMMARY.md`; logs in `logs/` (`$(hostname)`-suffixed).
- `decomp/` (2026-10-08, after the run): `full3_oldrecipe/` and `matched_oldrecipe/`
  (probe sets, activations, fits), `decomposition.md`. `review/sample_agreement.md` the
  same day.
- Dry-run and test leftovers, not inputs to anything: `gens/*_dry*/`, `arms_dry/`,
  `arms_dry_extract/`, `tests_fixture/`, `review/examples.md` (from the dry run).
- One seed throughout.

## Costs

- Generation (vLLM, one H100): no-CoT turn 1 6870 prompts x 3 in about 2 min; bio turn 1
  4580 prompts x 3 CoT in 6.9 min; follow-ups 18294 (no-CoT) and 13489 (bio) in 2-4 min
  each; float32 P(Yes) scoring 13489-18294 prompts in about 4 min each.
- Extraction: 6-16 min per arm (HF bf16); 1.7-1.8 GB of activations per arm.
- CV: 100 fold files per arm, about 132 s per 2000-step and 268 s per 4000-step fold with
  10 fits in parallel; two arms' sweeps in about 1.5 h; the edge cells (up to 11 per arm,
  8000 steps) about 3.5 h. Final fits about 15 min per arm.
- Disk: `arms/` 7 GB, `decomp/` 3.1 GB, `gens/` 172 MB, `cv/` 58 MB, `runs/` 14 MB.
