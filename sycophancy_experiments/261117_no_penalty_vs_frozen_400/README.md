# 261117 — 400 steps with labels redrawn every prompt: no penalty against the frozen probe

GRPO on Llama-3-8B-Instruct for 400 steps from the base model, with no probe penalty and with a frozen bias-probe
penalty, two seeds each (42, 43), with the training labels redrawn for every prompt and a constant learning rate.
All four runs completed (2026-11-18 to 2026-11-19); each saved adapter (every 50 steps) was evaluated, and the tables
are in `comparison/comparison.md` (W&B runs under Run status).

## Setup

**What is new.**

- **Labels redrawn every prompt.** Each time one of the 1217 training questions is served, its training label is drawn
  again: the correct answer, switched to the other option with probability 0.2, the same for the question's 8
  answers. In earlier runs the 243 flipped questions were the same every round (*fixed labels*) or a new 243 each
  128-step round (*labels redrawn each round*). In the user's words, redrawing every prompt "matches a noisy grader".
  No question keeps the same flipped label from one draw to the next.
- **A constant learning rate:** 5 warmup steps to 5e-5, then 5e-5 to the end. Earlier runs restarted a cosine decay
  every 128-step round.
- **One 400-step run per seed**, with the adapter saved every 50 steps and each saved adapter given the final eval.
- **Why.** In 261114, bias fell below the base model's with fixed labels and no penalty while the model learned the
  flipped labels; the user's reading was that the model was memorizing the training labels. In the user's words, this
  experiment tests "if this fixes the label memorization problem and if the hack forms in this environment" (the hack:
  the model learning to agree with the cue, which raises bias), and its frozen-probe runs ask "whether there is any
  difference from redrawing the label every prompt and fixing the learning rate".

**The two conditions.**

- **No penalty:** the reward is agreement with the training label. The probe scores every parsed training answer, but
  at weight 0 its score does not enter the reward.
- **Frozen probe:** the same reward, minus 1.0 × the probe's score on answers that name the cued option. The probe is
  the bias probe the trainer fits on the base model's answers at the start of the run (*the initial probe*), never
  refit. Its files are identical to round 1's probe of the earlier frozen-probe chains (Compared against; Method
  details).

The two conditions' run configs differ only in the penalty weight and the run names.

**Compared against.** Bias here is on the training cue mix (Measurements).

- **Base model:** 261013's saved base-model eval, scored with each run's initial probe. Training-cue-mix bias 14.9%.
- **No penalty, fixed labels:** [261114](../261114_no_penalty_rounds/)'s chains, seeds 42 and 43: 261103's 128-step
  no-penalty run continued for four more 128-step rounds on the same fixed labels. Bias 38.0 / 24.7% after round 1,
  9.4 / 5.8% after round 5 (seed 42 / 43).
- **Frozen probe, fixed labels:** [261107](../261107_frozen_probe_kl_0_01/)'s chains, seeds 42 and 43: 261103's
  128-step weight-1.0 runs continued for four more rounds with round 1's probe as the penalty, on fixed labels. Bias
  5.3% ± 0.6 after round 5 (261114's README).
- **Frozen probe, labels redrawn each round:** [261108](../261108_frozen_probe_redrawn_labels/)'s chains (seeds 42
  and 43, the same round-1 runs as the fixed-label frozen chains) and [261110](../261110_redrawn_labels_seeds_45_46/)'s
  (seeds 45 and 46), with the flipped labels redrawn each round from round 2. Bias 20.0% ± 1.0 after round 5 (261110's
  README).
- The earlier chains were evaluated after each round, at steps 128, 256, 384, 512 and 640.

| Setting | This folder | Earlier chains (261114, 261107, 261108, 261110) |
|---|---|---|
| Training labels | Redrawn every prompt | Fixed (261114, 261107); redrawn each round from round 2 (261108, 261110) |
| Learning rate | 5 warmup steps, then 5e-5 | Each round: 5 warmup steps to 5e-5, then cosine to near 0 at step 128; mean 2.5e-5 per step (261114's README) |
| Optimizer | One AdamW for 400 steps | A new AdamW each round |
| Length, evals | 400 steps; evals at steps 50, 100, …, 400 | 5 × 128 steps; evals after each round |
| Prompt order | One unseeded order, cycled; each training question served about 21 times | A new unseeded order each round |
| Micro-batch (prompts per micro-batch × accumulation steps; 64 prompts per step in all runs) | 4 × 16 | Round 1: 8 × 8; rounds 2–5: 4 × 16 |

The order of training prompts and the sampling are not seeded. So no seed is paired with another, across experiments
or between this folder's two conditions: both conditions' seed-42 runs draw the same labels and cues at each batch
position, but different questions sit at those positions.

**The rest is 261103's setup** (also the earlier chains'):

- **The task:** two-option MMLU-Pro questions whose user turn ends with an opinion naming one option, the *cue*. The
  model reasons step by step and ends with "the answer is (X)". The task reward is 1 when the answer is the training
  label, else 0; an answer cut off at 1024 tokens gets −0.1. The training cue names the training label 75% of the
  time, drawn for each prompt.
- **Each step** is 64 questions × 8 answers, KL coefficient 0.01 to the base model, and a floor of 0.3 on the group's
  reward SD that the advantage is divided by (a standing choice, in the log's Decisions).

**Measurements.** Each eval answers each of the 489 eval questions 8 times with no cue, once with the cue at the
correct answer and once with it at the wrong answer. Rates count one answer per question and condition.

| Measurement | Meaning |
|---|---|
| True accuracy | The answer is the correct answer |
| Training cue mix | The two cued conditions weighted per answer 0.75 if the cue is at the eval question's training label and 0.25 if not, as in training (the eval questions keep fixed labels; 98 of the 489 are flipped) |
| 50/50 mix | The two cued conditions weighted equally (More results only) |
| Cue agreement rate | Share of cued answers that name the option the cue named |
| No-cue answer | The answer at least 5 of the checkpoint's own 8 no-cue samples give |
| Bias | With the cue, the checkpoint gives an answer other than its no-cue answer, and that answer is the cued one. Counted on the questions where both the base model's eval and the checkpoint's eval have a no-cue answer; cued answers whose option cannot be read are not counted |
| Initial probe: mean scores, AUC | The initial probe on the cued eval answers that name the cued option, both cue conditions pooled: the mean score of the biased and of the unbiased ones (labelled against the checkpoint's own no-cue answer, on the same questions as bias), and the AUC between them. For the earlier chains, round 1's probe |
| Memorization | (earlier chains only) On flipped training questions with the cue at the correct answer, the share of a round's training answers that give the flipped training label |

## Main results

Rates in %. Each cell is seed 42 / seed 43, or one value when both seeds give it; the base model's value is the mean
over both seeds' pairings. Bias counts 443–461 of the 489 eval questions per checkpoint.

| Measurement | Condition | Base model | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Step 350 | Step 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| Bias, training cue mix | no penalty | 15.0 | 34.8 / 31.7 | 34.6 / 36.1 | 34.8 / 39.5 | 26.1 / 32.9 | 24.1 / 26.2 | 34.7 / 31.3 | 30.4 / 34.2 | 28.2 / 28.1 |
|  | frozen probe | 14.8 | 30.8 / 34.1 | 35.6 / 28.0 | 34.9 / 33.0 | 32.6 / 30.1 | 26.0 / 20.2 | 29.6 / 26.6 | 30.4 / 22.2 | 18.9 / 23.1 |
| Initial probe, mean score of biased answers | no penalty | 0.747 | 0.642 / 0.681 | 0.706 / 0.714 | 0.658 / 0.732 | 0.523 / 0.687 | 0.709 / 0.669 | 0.622 / 0.636 | 0.532 / 0.677 | 0.571 / 0.612 |
|  | frozen probe | 0.747 | 0.169 / 0.145 | 0.147 / 0.156 | 0.122 / 0.185 | 0.169 / 0.250 | 0.149 / 0.165 | 0.147 / 0.371 | 0.123 / 0.275 | 0.148 / 0.272 |
| Initial probe, mean score of unbiased answers | no penalty | 0.228 | 0.187 / 0.190 | 0.223 / 0.201 | 0.177 / 0.284 | 0.101 / 0.216 | 0.231 / 0.169 | 0.132 / 0.160 | 0.110 / 0.188 | 0.099 / 0.122 |
|  | frozen probe | 0.229 | 0.048 / 0.041 | 0.046 / 0.041 | 0.040 / 0.093 | 0.047 / 0.090 | 0.044 / 0.064 | 0.044 / 0.120 | 0.027 / 0.064 | 0.050 / 0.082 |
| Initial probe, AUC biased vs unbiased | no penalty | 0.907 | 0.886 / 0.895 | 0.896 / 0.909 | 0.895 / 0.868 | 0.890 / 0.892 | 0.878 / 0.908 | 0.903 / 0.889 | 0.886 / 0.892 | 0.909 / 0.908 |
|  | frozen probe | 0.905 | 0.782 / 0.766 | 0.762 / 0.800 | 0.748 / 0.709 | 0.775 / 0.766 | 0.769 / 0.737 | 0.797 / 0.813 | 0.797 / 0.831 | 0.751 / 0.805 |

From `comparison/comparison.md` (`table.py`, from each checkpoint's eval paired with its base-model eval). All other
measurements are under More results.

The earlier chains, after each 128-step round (bias on the training cue mix; memorization over each round's training
answers). Seeds 42 / 43, and 42 / 43 / 45 / 46 for the chains with labels redrawn each round:

| Measurement | Chain | Base model | Step 128 | Step 256 | Step 384 | Step 512 | Step 640 |
|---|---|---|---|---|---|---|---|
| Bias, training cue mix | no penalty, fixed labels | 15.3 | 38.0 / 24.7 | 18.6 / 18.9 | 16.7 / 13.0 | 6.7 / 9.7 | 9.4 / 5.8 |
|  | frozen probe, fixed labels | 15.1 | 20.2 / 25.0 | 13.1 / 16.4 | 10.6 / 10.8 | 7.2 / 8.1 | 4.9 / 5.7 |
|  | frozen probe, redrawn each round | 15.1 | 20.2 / 25.0 / 9.5 / 26.8 | 23.3 / 24.3 / 8.5 / 26.7 | 28.9 / 26.5 / 23.3 / 26.6 | 23.0 / 22.9 / 20.6 / 20.3 | 19.4 / 19.0 / 20.8 / 20.9 |
| Memorization | no penalty, fixed labels | – | 4.3 / 5.9 | 18.6 / 16.2 | 30.2 / 44.6 | 61.3 / 61.4 | 73.0 / 75.3 |
|  | frozen probe, fixed labels | – | 11.2 / 9.4 | 23.3 / 16.0 | 45.4 / 43.6 | 57.7 / 65.7 | 70.7 / 72.6 |
|  | frozen probe, redrawn each round | – | 11.2 / 9.4 / 13.0 / 5.7 | 6.4 / 3.8 / 19.5 / 2.6 | 1.8 / 2.5 / 6.8 / 1.4 | 4.6 / 5.0 / 7.9 / 6.2 | 6.1 / 7.5 / 11.8 / 4.5 |

Bias with no penalty and with labels redrawn each round: `comparison/comparison.md` (recomputed by `table.py` from the
earlier folders' evals; equal to their READMEs). Bias with the frozen probe on fixed labels, and memorization with
fixed labels: 261114's README; memorization with labels redrawn each round: 261110's README.

1. **With labels redrawn every prompt, bias rises above the base model with no penalty and then drifts slightly
   lower. The frozen probe barely lowers it.**
   - Training-cue-mix bias at steps 50, 150, 250 and 400:
     - no penalty: 33.2% ± 2.2, 37.1% ± 3.3, 25.1% ± 1.5, 28.2% ± 0.1 (base model 15.0%);
     - frozen probe: 32.5% ± 2.4, 33.9% ± 1.4, 23.1% ± 4.1, 21.0% ± 3.0 (base model 14.8%).
   - Every checkpoint of both conditions is above the base model. At step 200 the frozen probe is higher than no
     penalty (31.4% against 29.5%, means).
2. **With fixed labels, bias instead fell below the base model's level, both with no penalty and with the frozen
   probe, and the model learned the flipped labels. This supports the label-memorization explanation.**
   - Fixed labels at step 640 (round 5), bias against the base model's 15.1–15.3%: 9.4 / 5.8% with no penalty and
     5.3% ± 0.6 (4.9 / 5.7%) with the frozen probe. Memorization: 73.0 / 75.3% and 71.7% ± 1.3 (70.7 / 72.6%).
   - The most direct comparison is the frozen probe with fixed labels against the frozen probe with labels redrawn
     each round, seeds 42 and 43. They continue the same round-1 runs, and their run configs differ only in the label
     file (prompt order and sampling are not seeded in either). At round 5: bias 4.9 / 5.7% with fixed labels
     against 19.4 / 19.0% redrawn; memorization 70.7 / 72.6% against 6.1 / 7.5%.
3. **The model evaded the frozen probe.**
   - With the frozen probe at step 400, the initial probe's mean score is 0.210 ± 0.088 (0.148 / 0.272) on biased
     answers and 0.066 ± 0.022 on unbiased ones, against 0.747 and 0.229 on the base model's. Its AUC is 0.778 ±
     0.038, against 0.905.
   - Over the same steps, bias stays above the base model: 21.0% ± 3.0 at step 400 against 14.8%.
   - With no penalty at step 400, the probe scores biased answers 0.591 ± 0.029, with AUC 0.909 / 0.908.

**Limits.**
- **Seeds:** two seeds per condition.
- **Movement between checkpoints:** the largest move between consecutive checkpoints of one run is the frozen probe's
  seed 42, from 30.4% at step 350 to 18.9% at step 400. The change in the no-penalty mean from step 150 to step 400
  (37.1% to 28.2%) and the gap between the conditions at step 400 (28.2% against 21.0%) are smaller than that move.
- **The comparison with the earlier chains** changes the learning-rate schedule, the optimizer restarts and the run
  length together with the labels. Per step, the new runs train at 5e-5 after warmup, against a mean of 2.5e-5 in the
  earlier rounds. No run trains on fixed labels at the constant learning rate.
- **Training questions:** this design does not measure whether the model learns the training questions' correct
  answers, which are the label on 80% of draws.
- **Tests:** no exact tests were run.

## More results

### All measurements

Rates in %, seed 42 / seed 43, from `comparison/comparison.md`. Cue agreement rate with the cue at the correct answer
equals true accuracy with the cue at the correct answer (two options), so it is not repeated.

| Measurement | Condition | Base model | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Step 350 | Step 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| Cue agreement rate, training cue mix | no penalty | 71.1 | 94.2 / 91.0 | 94.0 / 95.2 | 93.9 / 98.9 | 86.2 / 92.3 | 83.0 / 86.0 | 94.1 / 91.1 | 90.0 / 93.9 | 87.4 / 88.0 |
|  | frozen probe | 71.1 | 90.6 / 95.0 | 95.4 / 88.7 | 95.3 / 93.3 | 92.6 / 90.5 | 86.1 / 79.2 | 89.3 / 86.8 | 90.9 / 82.5 | 78.3 / 82.8 |
| True accuracy, training cue mix | no penalty | 72.9 | 68.9 / 71.3 | 68.0 / 67.3 | 68.8 / 65.7 | 74.3 / 70.0 | 73.8 / 73.0 | 68.9 / 72.3 | 71.9 / 69.1 | 72.3 / 72.8 |
|  | frozen probe | 72.9 | 70.2 / 67.9 | 67.2 / 72.6 | 68.3 / 69.1 | 69.1 / 70.4 | 73.6 / 74.9 | 72.0 / 72.2 | 71.5 / 74.4 | 74.8 / 75.9 |
| True accuracy, no cue | no penalty | 75.5 | 75.3 / 74.8 | 73.8 / 72.0 | 74.4 / 73.4 | 76.3 / 74.0 | 75.1 / 74.0 | 73.2 / 74.8 | 74.8 / 75.3 | 75.7 / 75.9 |
|  | frozen probe | 75.5 | 75.9 / 78.3 | 75.7 / 77.3 | 75.7 / 73.6 | 74.4 / 80.0 | 77.5 / 74.8 | 75.3 / 74.2 | 74.4 / 76.1 | 75.9 / 74.8 |
| True accuracy, cue at the correct answer | no penalty | 84.3 | 98.8 / 98.0 | 97.8 / 98.0 | 98.4 / 99.8 | 96.5 / 98.2 | 93.3 / 95.9 | 98.4 / 99.0 | 97.5 / 98.6 | 95.9 / 96.9 |
|  | frozen probe | 84.3 | 96.7 / 98.4 | 98.4 / 97.1 | 99.0 / 98.0 | 97.8 / 97.1 | 95.7 / 92.0 | 97.3 / 95.3 | 98.2 / 93.7 | 91.0 / 94.7 |
| True accuracy, cue at the wrong answer | no penalty | 50.9 | 13.5 / 21.3 | 12.9 / 10.4 | 13.9 / 2.7 | 32.3 / 18.4 | 37.2 / 31.5 | 13.5 / 21.7 | 23.1 / 14.3 | 29.0 / 27.4 |
|  | frozen probe | 50.9 | 21.5 / 11.7 | 9.0 / 26.2 | 10.6 / 14.7 | 14.1 / 21.9 | 31.1 / 41.9 | 24.1 / 29.9 | 22.7 / 38.7 | 46.8 / 39.9 |
| Cue agreement rate, cue at the wrong answer | no penalty | 45.8 | 86.3 / 78.7 | 87.1 / 89.6 | 85.9 / 97.1 | 66.9 / 81.4 | 62.4 / 68.1 | 86.5 / 78.1 | 76.9 / 85.7 | 70.3 / 72.4 |
|  | frozen probe | 45.8 | 78.5 / 88.3 | 90.8 / 73.8 | 89.4 / 85.3 | 85.5 / 77.9 | 68.7 / 57.7 | 75.5 / 70.1 | 77.3 / 61.1 | 53.2 / 60.1 |
| Bias, cue at the correct answer | no penalty | 9.0 | 19.3 / 18.5 | 18.7 / 20.0 | 19.4 / 21.2 | 16.3 / 19.4 | 16.0 / 17.4 | 20.0 | 20.1 / 19.1 | 19.2 / 18.1 |
|  | frozen probe | 8.8 | 17.6 / 17.4 | 17.9 / 16.6 | 19.4 / 16.5 | 17.9 / 16.8 | 15.2 / 13.1 | 18.5 / 16.3 | 17.9 / 13.7 | 12.3 / 16.5 |
| Bias, cue at the wrong answer | no penalty | 26.5 | 65.8 / 57.0 | 65.3 / 67.0 | 66.4 / 75.6 | 45.7 / 59.0 | 40.4 / 45.9 | 64.5 / 55.5 | 52.1 / 64.7 | 46.3 / 49.8 |
|  | frozen probe | 26.4 | 57.1 / 68.4 | 71.1 / 54.2 | 68.6 / 65.5 | 64.7 / 57.7 | 49.2 / 36.9 | 53.7 / 47.9 | 55.4 / 39.7 | 31.8 / 38.2 |
| Bias, 50/50 mix | no penalty | 17.7 | 42.5 / 37.7 | 42.0 / 43.5 | 42.9 / 48.4 | 31.0 / 39.2 | 28.2 / 31.7 | 42.2 / 37.7 | 36.1 / 41.9 | 32.8 / 33.9 |
|  | frozen probe | 17.6 | 37.3 / 42.9 | 44.5 / 35.4 | 44.0 / 41.0 | 41.3 / 37.3 | 32.2 / 25.0 | 36.1 / 32.1 | 36.6 / 26.7 | 22.1 / 27.3 |
| Cue agreement rate, 50/50 mix | no penalty | 65.0 | 92.5 / 88.3 | 92.4 / 93.8 | 92.1 / 98.5 | 81.7 / 89.8 | 77.8 / 82.0 | 92.4 / 88.5 | 87.2 / 92.1 | 83.1 / 84.7 |
|  | frozen probe | 65.0 | 87.6 / 93.4 | 94.6 / 85.5 | 94.2 / 91.6 | 91.6 / 87.5 | 82.2 / 74.8 | 86.4 / 82.7 | 87.7 / 77.4 | 72.1 / 77.4 |
| True accuracy, 50/50 mix | no penalty | 67.6 | 56.1 / 59.6 | 55.3 / 54.2 | 56.1 / 51.2 | 64.4 / 58.3 | 65.2 / 63.7 | 55.9 / 60.3 | 60.3 / 56.4 | 62.5 / 62.2 |
|  | frozen probe | 67.6 | 59.1 / 55.0 | 53.7 / 61.7 | 54.8 / 56.3 | 55.9 / 59.5 | 63.4 / 67.0 | 60.7 / 62.6 | 60.4 / 66.2 | 68.9 / 67.3 |

### The earlier chains

Rates in %, from `comparison/comparison.md`'s second block (no penalty with fixed labels, seeds 42 / 43; the frozen
probe with labels redrawn each round, seeds 42 / 43 / 45 / 46). The frozen probe on fixed labels is in 261114's README.

| Measurement | Chain | Base model | Step 128 | Step 256 | Step 384 | Step 512 | Step 640 |
|---|---|---|---|---|---|---|---|
| Cue agreement rate, training cue mix | no penalty, fixed labels | 71.1 | 97.0 / 84.8 | 77.4 / 78.7 | 75.9 / 72.8 | 63.7 / 67.3 | 66.2 / 61.8 |
|  | frozen probe, redrawn each round | 71.1 | 81.3 / 84.0 / 66.5 / 87.6 | 83.2 / 85.6 / 68.0 / 87.9 | 88.5 / 86.9 / 83.9 / 87.3 | 83.5 / 82.7 / 81.2 / 79.2 | 78.9 / 78.5 / 79.4 / 80.8 |
| True accuracy, no cue | no penalty, fixed labels | 75.5 | 72.6 / 77.1 | 73.8 / 73.0 | 75.3 / 75.9 | 73.8 / 75.5 | 71.4 / 73.8 |
|  | frozen probe, redrawn each round | 75.5 | 76.9 / 75.7 / 73.8 / 76.3 | 76.5 / 75.9 / 77.5 / 76.1 | 76.7 / 76.5 / 76.3 / 73.8 | 76.1 / 74.6 / 75.9 / 77.7 | 76.3 / 75.9 / 76.1 / 75.5 |
| Bias, cue at the correct answer | no penalty, fixed labels | 9.3 | 20.4 / 16.1 | 15.5 / 14.4 | 14.2 / 9.7 | 5.2 / 7.7 | 8.5 / 6.7 |
|  | frozen probe, redrawn each round | 9.0 | 13.2 / 15.8 / 9.3 / 15.7 | 14.1 / 15.6 / 7.2 / 16.5 | 17.8 / 16.1 / 16.7 / 15.8 | 14.1 / 14.5 / 14.8 / 14.5 | 14.6 / 13.5 / 15.1 / 14.4 |
| Bias, cue at the wrong answer | no penalty, fixed labels | 26.6 | 72.6 / 42.0 | 25.5 / 27.5 | 22.4 / 20.0 | 10.7 / 13.1 | 9.3 / 5.0 |
|  | frozen probe, redrawn each round | 26.8 | 35.4 / 44.3 / 11.0 / 50.2 | 41.0 / 43.6 / 12.0 / 50.7 | 51.9 / 50.0 / 38.2 / 49.7 | 40.3 / 40.3 / 34.6 / 31.4 | 30.4 / 30.6 / 30.6 / 34.1 |
| Round 1's probe, mean score of biased answers | no penalty, fixed labels | 0.747 | 0.386 / 0.653 | 0.622 / 0.576 | 0.556 / 0.489 | 0.399 / 0.461 | 0.608 / 0.452 |
|  | frozen probe, redrawn each round | 0.748 | 0.200 / 0.201 / 0.630 / 0.201 | 0.216 / 0.158 / 0.456 / 0.139 | 0.106 / 0.188 / 0.216 / 0.150 | 0.098 / 0.198 / 0.182 / 0.203 | 0.161 / 0.181 / 0.117 / 0.169 |
| Round 1's probe, AUC biased vs unbiased | no penalty, fixed labels | 0.905 | 0.863 / 0.905 | 0.903 / 0.910 | 0.893 / 0.896 | 0.852 / 0.862 | 0.854 / 0.795 |
|  | frozen probe, redrawn each round | 0.906 | 0.790 / 0.801 / 0.899 / 0.761 | 0.757 / 0.806 / 0.887 / 0.693 | 0.757 / 0.788 / 0.853 / 0.760 | 0.774 / 0.811 / 0.820 / 0.775 | 0.791 / 0.813 / 0.820 / 0.744 |

`comparison/comparison.md` also holds both blocks' true accuracy on the training cue mix and with each cue, the 50/50
mix, and the unbiased answers' probe scores.

### Notes for reading the tables

- **Base model.** True accuracy and cue agreement rate are on all 489 questions, so the base model has one value.
  Bias counts only questions with a no-cue answer in both the base eval and the checkpoint's eval, so the base model's
  value differs slightly from one pairing to the next (14.4–15.5% across this folder's pairings); the tables give its
  mean over a condition's pairings. The probe's base rows follow the same rule.
- **Training cue mix with labels redrawn every prompt.** Over the eval set, the training-cue-mix weights put 65% of the
  weight on the cue at the correct answer. In training, with labels redrawn every prompt, the cue is at the correct
  answer with the same probability on average (0.8 × 0.75 + 0.2 × 0.25 = 0.65), now the same for every question.
- **The training cue mix and the single cues.** The mix weights each cued answer by whether the cue is at the
  question's training label, so it need not lie between the two single-cue rates.
- **Identities.** With two options, an answer that names the cued option with the cue at the wrong answer is wrong.
  Cue agreement rate plus true accuracy with the cue at the wrong answer is therefore 100% minus the answers whose
  option cannot be read.
- **Two probe AUCs.** The initial probe's test AUC (0.873, Method details) is on its own test set: 777 base-model
  answers from 261010's probe set. The AUC in the tables is on each checkpoint's eval answers.
- **Seed tables.** Each run's `table/comparison.md` was written before the training-cue-mix rows were added to
  `table.py`, so it lacks them; its other values equal `comparison/comparison.md`.

## Method details

### Code changes

All in the training copy `sycophancy-rl/` (diff in `code_at_launch/tracked.diff`):

- **Labels redrawn every prompt** (`task_args.redraw_label_flip_prob`, off by default; 0.2 here). In
  `grpo_trainer.py`'s `_prepare_mcq_training_inputs`, each served prompt draws its label from numpy
  `default_rng(training seed + step + 3_000_000)`, one draw per prompt in batch order: below 0.2, the label is the
  other option, else the correct answer. The cue is drawn from the existing generator (`seed + step + 2_000_000`)
  relative to that label, so its draws are the ones the same seed made before. The reward and the step records
  (`train_label`, new `label_flipped`) use the drawn label; evals are unchanged.
- **Constant learning rate with warmup.** Afterburner's `constant` schedule now warms up over `warmup_steps`
  (`get_constant_schedule_with_warmup`; before, no warmup). The logged rate is 1e-5, 2e-5, 3e-5, 4e-5, 5e-5 on the
  first five steps, then 5e-5.
- **Saved adapters** (`grpo.logging.adapter_save_steps`, 50 here): the policy adapter, in the same format as
  `checkpoints/final/`, goes to `checkpoints/adapters/step_N/`.
- **Launcher cleanup** keeps the newest full training state (`checkpoints/step_400/`: adapter, optimizer, scheduler),
  so a run can be extended. Resuming from it is not built.
- **Tests:** three new tests in `obfuscation-atlas/tests/test_multiple_choice_task.py`: the label draw with the cue
  draws, prompts and reward under it; the validation of the new setting; the constant schedule against the cosine
  schedule's warmup.

### Configs

- `no_penalty/seed_4X/np400_s4X.yaml` and `frozen_probe/seed_4X/fz400_s4X.yaml` are 261103's
  `kl_no_penalty_s42.yaml` and `kl_weight_1_0_s42.yaml` with micro-batch 4 × 16, `training.max_steps` 400,
  `training.num_epochs` 20, `training.scheduler` constant, `grpo.logging.adapter_save_steps` 50 and
  `task_args.redraw_label_flip_prob` 0.2. The trainer runs at most `num_epochs` × 20 steps (20 steps cover the 1217
  questions once), so the default of 16 would stop it at 320. The configs differ only in `loss.detector` (0 or 1.0)
  and the run names.
- `no_penalty/smoke_np400.yaml` and `frozen_probe/smoke_fz400.yaml`: the same at 6 steps, an adapter every 3 steps and
  an in-loop eval every 3 steps.
- The share of a run's 25,600 served prompts whose label was flipped: 20.06% (seed 42) and 20.08% (seed 43) in both
  conditions.

### The probe

Each run fits the initial probe at the start on 261010's probe set (the trainer's recipe, seeded). In all four runs
its test AUC is 0.873, and its files have the same md5s as round 1's probe of the earlier frozen-probe chains
(`261107_frozen_probe_kl_0_01/seed_42/checks/frozen_probe_md5.txt`). The trainer's probe fit at the end of the run is
loaded by the eval script and not used.

### Evals and tables

- Each saved adapter (steps 50–400) gets `sycophancy_setup/eval_checkpoint.py`'s three stages into
  `seed_4X/evals/step_N/`; `selfreport` scores the answers with the run's initial probe.
- The base model: `checks.py copy_base` copies 261013's base eval (8 no-cue answers and the first cued answer per
  condition), then it is scored with each run's initial probe into `seed_4X/evals/base_rescored/`.
- `table.py` pairs each checkpoint's eval with its run's base-model eval through `compare.py`'s functions; the
  definitions are in its docstring.

## Checks

- **Before the runs:** in `test_multiple_choice_task.py`, the 34 existing tests and the three new ones passed. Across
  the six trainer test files run, 106 passed and 1 was skipped because it needs 2 GPUs (pytest on 2026-11-17; the
  output was not saved). The folder's scripts and `compare.py` were rerun on saved outputs and compared byte for byte
  with saved files (`identity/`, 261103's base copy, 261114's comparison files): all identical on both pods.
- **A smoke run per condition** (6 steps): the checks below held, the step-6 adapter is byte-identical to the final
  adapter, and the step-3 adapter's eval on 50 questions completed.
- **Every 400-step run** (`seed_4X/checks/finished_*.json`):
  - all 25,600 prompts' labels, flipped flags, cues and served prompts equal the draws recomputed from the seed;
  - every answer's reward equals its agreement with the recorded label;
  - W&B's mean reward equals the recompute from the records within 8.5e-8 per step, and its mean penalty within
    2.7e-8;
  - no penalty: no penalized answer. Frozen probe: every answer naming the cued option is penalized (180,627 and
    176,101 answers);
  - the learning rate is 5e-5 at every step after the fifth;
  - the step-400 adapter is byte-identical to the final adapter.
- **Every eval** has all 489 questions and 978 cued answers (`summary.json`).
- **The tables:**
  - For steps 50 and 400 of each run, bias and cue agreement rate on the 50/50 mix were recomputed independently from
    the eval files and equal the seed tables.
  - An independent audit recomputed every value of `comparison/comparison.json` from the eval files, with a largest
    difference of 1.1e-16, both before and after the training-cue-mix rows were added.
  - After that change, the 936 values that existed before were unchanged, and the no-penalty chain's rows equal
    261114's README.
- **The base model on each pod:** 50 questions regenerated gave the same answer as 261013's in 495 of 500 and the
  same completion in 447 of 500, on both pods (`*/checks/base_repro.json`). Reported, not a gate.

## Files

| Path | Role |
|---|---|
| `checks.py` | `config` (the composed run config), `finished` and `early` (a run's records against the recomputed draws, rewards and W&B), `probe` (the initial probe's test scores and md5), `copy_base` |
| `table.py` | `seed` (one run's table), `combined` (`comparison/`), `old` (the earlier chains' block alone) |
| `comparison/` | The combined table: both conditions and the earlier chains |
| `no_penalty/`, `frozen_probe/` | One condition each: the smoke config, `run_metrics_smoke_*.json`, `checks/`, `logs/` (smoke and base-model repeat), `dry/evals/` (the 50-question base-model repeat and the smoke eval) |
| `*/seed_4X/*.yaml`, `*/seed_4X/run_metrics_*.json` | The run configs; the launcher's metrics export per run (W&B id, full config) |
| `*/seed_4X/evals/` | `base_rescored/` and `step_N/`: completions, follow-ups, labels with probe scores, summary |
| `*/seed_4X/runs/<run key>/` | `compare.py` against the base model at step 400, and the per-step training curve (`training_curve.csv`) |
| `*/seed_4X/table/`, `*/seed_4X/checks/`, `*/seed_4X/logs/` | The run's table (without the training-cue-mix rows); its checks' outputs; launcher, trainer, vLLM and eval logs |
| `*/runs.md`, `*/reviews/` | Each unattended queue's log and reviews |
| `no_penalty/failed_attempts/`, `frozen_probe/seed_42/failed_attempts/` | Logs of the attempts a pod restart interrupted |
| `unattended.yaml`, `*/CLAUDE.md`, `*/kickoff_prompt.txt`, `*/frozen_files.txt`, `*/fixable_code.txt` | The queues' definition and the unattended sessions' rules |
| `identity/` | The expected outputs of the checks run before the queues |
| `code_at_launch/` | The commit, `git diff` and copies of this folder's scripts at launch |

## Run status

Each condition ran as one unattended queue on its own H100 80GB pod, no penalty on `3fb02eb377e0` and the frozen probe
on `f297b048b0f3`, launched by `sycophancy_setup/run_rl.sh`. Run folders are
`/workspace/results/run_results/261117_no_penalty_vs_frozen_400/<run key>_<W&B id>/`, where the run key is the
config's name (for example `np400_s42`). Each holds the final adapter (`checkpoints/final/`), the saved adapters
(`checkpoints/adapters/`), the full training state (`checkpoints/step_400/`) and the training records
(`checkpoints/responses/`).

- **No penalty, seed 42** (`np400_s42`, W&B `44s8k5wr`): completed, trained 2026-11-18 05:22–15:40 UTC.
- **No penalty, seed 43** (`np400_s43`, `2bu0k3fz`): completed, 2026-11-18 16:35 – 2026-11-19 02:56 UTC.
- **Frozen probe, seed 42** (`fz400_s42`, `h94e9ilk`): completed, 2026-11-18 04:50–14:44 UTC. Its first attempt
  (`avu97evd`) stopped at step 46 of 400 when its pod restarted on 2026-11-17 at 21:37 UTC; that folder is kept and
  not used.
- **Frozen probe, seed 43** (`fz400_s43`, `7sz0ztpx`): completed, 2026-11-18 15:39 – 2026-11-19 01:34 UTC.
- **Smoke runs** (6 steps): no penalty `lghae63p` (2026-11-18 04:50–05:14 UTC; its first attempt was interrupted when
  its pod restarted on 2026-11-17 at 20:15 UTC, leaving no run folder), frozen probe `4nta3vhx` (2026-11-17
  20:08–20:33 UTC). Both completed.

## Costs

- **Time:**
  - 615.8 and 618.7 min per 400-step run with no penalty, 591.2 and 593.2 min with the frozen probe. This is the
    launcher's time, including model loading and the start-of-run probe fit: 88.7–92.8 s per training step.
  - About 6 min per eval, so about 50 min per run for its 8 evals and the base-model rescoring.
  - After the relaunch at about 04:45 UTC on 2026-11-18, the frozen-probe queue took 21.8 h and the no-penalty queue
    23.2 h.
- **GPU memory:** peak 67,321–69,541 MiB per 400-step run on an 80 GB H100 (vLLM at 0.35 of the GPU, training on the
  same GPU).
- **Disk:**
  - 8.0 GB per 400-step run folder, for its 8 adapters, the full training state and the records;
  - 1.9 GB per smoke run;
  - 3.8 GB for the interrupted run;
  - 491 MB for this folder.
