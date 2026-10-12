# Rerun of August's sycophancy baseline from the code copy vs August's run

August: W&B `lx6gcbcl` (260825_grpo_det0). Rerun: W&B `l52s88q8`.

| metric | August | rerun |
|---|---|---|
| in-loop eval accuracy, logged at step 30 | 0.7715 | 0.7482 |
| in-loop eval cue-following, logged at step 30 | 0.7802 | 0.7642 |
| in-loop eval accuracy, logged at step 60 | 0.8413 | 0.8530 |
| in-loop eval cue-following, logged at step 60 | 0.9971 | 0.9738 |
| in-loop eval accuracy, logged at step 90 | 0.8428 | 0.8530 |
| in-loop eval cue-following, logged at step 90 | 0.9985 | 0.9185 |
| in-loop eval accuracy, logged at step 120 | 0.8428 | 0.8617 |
| in-loop eval cue-following, logged at step 120 | 0.9389 | 0.9243 |
| final eval accuracy | 0.8370 | 0.8559 |
| final eval cue-following | 0.9229 | 0.8981 |
| final eval parse failures | 0 | 0.0015 |
| first step with train cue-following >= 0.95 | 7 | 7 |
| trainer wall-clock (h) | 4.2769 | 3.5953 |

In-loop eval rows score the policy from the previous eval point (the row logged at step 30
scores the starting model): the trainer's eval queue is first-in-first-out with one
submission before training. Final eval: one sample per held-out question, temperature 1.
