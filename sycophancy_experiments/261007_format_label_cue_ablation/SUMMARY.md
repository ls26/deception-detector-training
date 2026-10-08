# Summary: format / label / cue ablation (261007)

spec_hash 10b44b32b4c2. All four arms ran to the end; nothing is Blocked. The run stopped
three times on gates and resumed after user decisions (see "Stops and user decisions").

**Answer.** Removing chain-of-thought is the setup change that makes the probes work better:
both no-CoT arms raise `bias_followed` and `deception_yes` with 95% CIs that exclude zero.
The expert-biography cue changes nothing measurable. Whether logprob labels add anything on
top of no-CoT is not tested by the pre-registered rule (both no-CoT arms are compared with
`ref_cot` only). `deception_no` is inconclusive in every arm.

## Main table (from `comparison.md`)

Eval AUC of the combined score (mean sigmoid over the CV-chosen band), one seed. Δ vs ref_cot
and its 95% CI are computed on the eval questions shared with ref_cot (question-level bootstrap,
2000 draws, seed 20261007); `*` marks a CI that excludes 0.

| probe | ref_cot AUC | nocot_sampled AUC (Δ vs ref, 95% CI) | nocot_logprob AUC (Δ vs ref, 95% CI) | bio_cot AUC (Δ vs ref, 95% CI) |
|---|---|---|---|---|
| bias | 0.827 | 0.874 (+0.050*, [+0.023, +0.077], n=665) | 0.901 (+0.073*, [+0.048, +0.099], n=656) | 0.845 (+0.018, [-0.007, +0.041], n=669) |
| bias_followed (verdict) | 0.768 | 0.852 (+0.082*, [+0.051, +0.116], n=662) | 0.880 (+0.117*, [+0.083, +0.151], n=652) | 0.766 (-0.003, [-0.032, +0.027], n=668) |
| deception | 0.649 | 0.711 (+0.066*, [+0.031, +0.102], n=665) | 0.696 (+0.044*, [+0.008, +0.081], n=656) | 0.637 (-0.012, [-0.050, +0.025], n=669) |
| deception_yes (verdict) | 0.752 | 0.819 (+0.062*, [+0.028, +0.097], n=512) | 0.809 (+0.049*, [+0.014, +0.083], n=509) | 0.755 (+0.008, [-0.033, +0.048], n=542) |
| deception_no (verdict) | 0.706 | 0.779 (+0.080, [-0.009, +0.176], n=139) | 0.762 (+0.055, [-0.039, +0.152], n=137) | 0.717 (-0.037, [-0.156, +0.077], n=118) |

Read the AUCs and the Δs separately: an arm's AUC above is on its full eval set, but its Δ is on
the questions it shares with ref_cot. They mostly agree to within 0.008. The exception is bio_cot
`deception_no`: 0.717 vs 0.706 looks like +0.011, but on its 118 shared questions ref_cot scores
0.755 and bio_cot 0.719, so Δ = -0.037 (`comparison.json` holds both AUCs for every cell).

## Verdict per probe (pre-registered rule: Δ's 95% CI excludes zero)

- **bias_followed:** removing chain-of-thought helped (nocot_sampled +0.082, nocot_logprob
  +0.117, both CIs above zero); the biography cue did not (bio_cot -0.003).
- **deception_yes:** removing chain-of-thought helped (nocot_sampled +0.062, nocot_logprob
  +0.049, both CIs above zero); the biography cue did not (bio_cot +0.008).
- **deception_no:** no factor helped under the rule; every CI includes zero (+0.080, +0.055,
  -0.037), with unmatched train sizes and only 118-139 shared eval questions.

Which factor each arm isolates: nocot_sampled changes the format only; bio_cot changes the cue
only; nocot_logprob changes the format and the label source together relative to ref_cot, so
the label factor on its own is not tested here. As point estimates, logprob labels are 0.028
above sampled labels on `bias_followed` (0.880 vs 0.852) and 0.010 below on `deception_yes`
(0.809 vs 0.819), with no CI for that contrast.

Reported, not interpreted: `bias` and `deception` move the same way (no-CoT up with CIs above
zero, bio_cot within noise). On nocot_logprob's confident slice (both top letters >= 0.75),
AUCs rise further: bias_followed 0.920 (827 rows), deception_yes 0.841 (839 rows).

For scale: ref_cot `bias_followed` here is 0.768 at 626 rows per class, against 260924's 0.788
at its full size (2908 train rows, all three samples per prompt).

## Size rule outcome

Per class, from each arm's step-1 train pool (one row per prompt); matched if the smallest
arm's full size is >= 200, unmatched between 50 and 200.

| probe | rule | per class | ref_cot | nocot_sampled | nocot_logprob | bio_cot |
|---|---|---|---|---|---|---|
| bias | matched | 454 | 454 | 454 | 454 | 454 |
| bias_followed | matched | 626 | 626 | 626 | 626 | 626 |
| deception_yes | matched | 340 | 340 | 340 | 340 | 340 |
| deception_no | unmatched (66 < 200) | own full size | 148 | 136 | 130 | 66 |
| deception | union of the deception_yes and deception_no draws | | 488 | 476 | 470 | 406 |

bio_cot set the matched size for bias, bias_followed and deception_no; ref_cot set it for
deception_yes. Because deception_no is unmatched, its Δs mix the arm effect with a train-size
effect (bio_cot trains on less than half of ref_cot's rows).

## Chosen recipes (lr / steps / band, 5-fold CV on train questions)

| probe | ref_cot | nocot_sampled | nocot_logprob | bio_cot |
|---|---|---|---|---|
| bias | 0.01 / 4000 / 13-16 | 0.01 / 4000 / 13-16 | 0.01 / 2000 / 13-16 | 0.01 / 2000 / 13-16 |
| bias_followed | 0.01 / 4000 / 13-16 | 0.01 / 8000 / 24-27 | 0.03 / 2000 / 13-16 | 0.01 / 2000 / 13-16 |
| deception | 0.1 / 8000 / 17-20 | 0.1 / 8000 / 27-30 | 0.1 / 8000 / 18-21 | 0.01 / 2000 / 14-17 |
| deception_yes | 0.1 / 4000 / 15-18 | 0.1 / 8000 / 28-31 | 0.1 / 8000 / 18-21 | 0.03 / 2000 / 14-17 |
| deception_no | 0.1 / 8000 / 22-25 | 0.03 / 2000 / 15-18 | 0.1 / 8000 / 28-31 | 0.1 / 2000 / 11-14 |

Every edge extension was run (CV ended at 07:40Z, about 12.5 h inside the wall-clock cap).
Many deception recipes landed on the edge cells (lr 0.1, 8000 steps).

## Baseline shifts per arm (all 2290 questions; "at p" mixes cue_gold and cue_dist at p = 0.85)

| arm | follow rate at p | bias rate at p | anti-bias at p | position bias: accuracy gold-A / gold-B, no cue | same, at p | said Yes at p |
|---|---|---|---|---|---|---|
| ref_cot | 0.804 | 0.188 | 0.050 | 0.595 / 0.771 | 0.731 / 0.813 | 0.641 |
| nocot_sampled | 0.895 | 0.258 | 0.009 | 0.671 / 0.724 | 0.831 / 0.842 | 0.786 |
| nocot_logprob | 0.903 | 0.260 | 0.006 | 0.668 / 0.736 | 0.836 / 0.844 | 0.785 |
| bio_cot | 0.768 | 0.160 | 0.059 | 0.595 / 0.771 (260819's no-cue rollouts, by design) | 0.675 / 0.827 | 0.824 |

- **No-CoT:** the cue works more strongly without reasoning (follow +0.09 to +0.10, bias +0.07),
  anti-bias almost vanishes, position bias mostly disappears (gold-A / gold-B gap 0.05-0.07
  without the cue vs 0.18 for ref_cot), and the self-report shifts to Yes.
- **Bio:** the biography is a weaker cue than the short opinion (follow -0.04, bias -0.03),
  position bias under the cue grows (gap 0.15 vs 0.08), and the self-report shifts strongly
  to Yes, which leaves bio_cot few false denials.

Full tables: `arms/<arm>/baselines.md`.

## Blocked

Nothing is Blocked at the end. Two Blocked entries during the run were resolved by user
decisions (below); all four arms are in the comparison.

## Stops and user decisions (deviations from the plan as frozen)

1. **2026-10-07, before this session:** the no-CoT turn-1 bf16 gates were loosened by the user
   (sibling and invariance bars 1e-2 / 1e-6 -> 0.05).
2. **2026-10-07T23:17Z, no-CoT follow-ups Blocked; resolved 2026-10-08T00:20Z:** the top-vs-sample check compared
   vLLM's bf16 P(Yes) on identical prompts and failed on 2 of 13184 pairs, each one bf16 step
   apart. The user made it a report. Identical inputs are still asserted exactly (text and
   token ids).
3. **2026-10-07T23:41Z, run stopped; resolved 2026-10-08T00:20Z:** the size rule counted a prompt with mixed labels
   in both classes, but step 1 keeps one row per prompt, so the prescribed sizes could not be
   drawn. The user fixed the count to step 1's actual pool. This made the sizes smaller than
   the first count gave (bias 626 -> 454, deception_yes 528 -> 340 per class).
4. **2026-10-08T00:44Z, run stopped; resolved 00:52Z:** the deception read position (the token before the
   sampled Yes/No) did not fit the user's 2026-10-07 rule that `said` = float32 P(Yes) > 0.5.
   The user moved the deception read position to the last token of the follow-up prompt for
   every row, the position P(Yes) is read from. This is identical to the old position on
   96.7-99.98% of rows per arm. It was validated against 260923's cached activations (cosine
   1.000000).

Other things to know (details in `runs.md`):
- The float32 P(Yes) labels carry a little batch-layout noise: identical texts flipped Yes/No
  on 9 of 13184 checked pairs (0.07%).
- `audit.py` passed 113/113 checks but covers only the AUCs; the read-only subagent audit
  recomputed every Δ, CI and n and found them all correct.
- Eval activations were read before the final fits in three places, none of which used labels
  or probe scores or selected anything: `extract.py`'s pre-registered HF-vs-vLLM letter gate
  (no-CoT eval rows' letter logprobs); my mid-run cosine-only equivalence check on 50 of
  260923's pool eval rows (ref_cot's eval prompts); and the Phase-A reproduction gate and
  fixture on those pool eval rows.
- One seed throughout.

## Caveats

- the bio arm changes authority, length and position at once
- the matched size measures signal per example, not per pipeline
- a no-CoT policy has no text to change, so RL against these probes needs the pipeline changes
  listed in the plan (GRPO has no assistant-prefill support, its reward regex scores a bare
  letter as a parse failure, and its probe reads the end-of-turn token by default)

## Files

`comparison.md` (main table and checks), `comparison.json` (shared-question AUCs and deltas),
`runs.md` (step-by-step record, judgment calls, the subagent audit), `STATUS.md`,
`arms/<arm>/baselines.md`, `arms/train_sizes.md`, `cv/<arm>/summary.md`, `runs/<arm>/`.
