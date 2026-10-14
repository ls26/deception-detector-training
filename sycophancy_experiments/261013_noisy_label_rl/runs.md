# Runs

State file for the unattended run. Read this first after any restart.

## Queue

- [x] 1. Base-model eval (`run_eval.sh base`) → `evals/base/summary.json`: 489 questions, 2934 cued rollouts
- [x] 2. Start vLLM (`start_vllm.sh run`) → answers within 10 min
- [x] 3. RL run, 128 steps (`run_rl.sh run`) → `logs/run_$HOST.log` ends `EXIT 0`; `CRASH:` lines copied here
  - `CRASH:` lines in `logs/run_1dbd6c289614.log`: none (`grep -c 'CRASH:'` = 0; the launcher
    reports `exit=0, attempt=1/4`).
- [x] 4. Rollout-record check (`check_records.py`) → `check_records_run.json`: `"PASS": true`, 128 files
- [x] 5. Stop vLLM → GPU free
- [x] 6. Final-checkpoint eval (`run_eval.sh final`) → `evals/final/summary.json`: 489 questions, 2934 cued rollouts
- [x] 7. Comparison (`compare.py`) → `comparison.md` copied into Result; `training_curve.csv` 128 rows

## Log

One line per step: `<UTC time> | step | outcome | next`.

- 2026-10-14 04:46 UTC | 1 base eval | log ends EXIT 0; summary.json: 489 questions, 2934 cued rollouts (7 min) | 2 start vLLM
- 2026-10-14 04:50 UTC | 2 start vLLM | /v1/models answered after 227 s (afterburner from the code copy) | 3 RL run
- 2026-10-14 09:47 UTC | 3 RL run | log ends EXIT 0; run_id 6vcs39wj; 0 CRASH lines; GRPO subprocess 294.6 min, attempt 1/4; 10 "could not remove activation cache" warnings, each followed by "OK: removed" for the same cache | 4 record check
- 2026-10-14 09:50 UTC | 4 record check | exit 0; check_records_run.json: PASS true, n_files 128 (128/128 files match W&B; 1.9 min) | 5 stop vLLM
- 2026-10-14 09:50 UTC | 5 stop vLLM | `tmux kill-session -t 261013-vllm`; nvidia-smi lists no process 0 s later; 70 GB free | 6 final eval
- 2026-10-14 10:00 UTC | 6 final eval | log ends EXIT 0; adapter checkpoints/final of 6vcs39wj; summary.json: 489 questions, 2934 cued rollouts (9.5 min) | 7 comparison
- 2026-10-14 10:00 UTC | 7 comparison | exit 0; comparison.md, comparison.json, training_curve.csv written; training_curve.csv has 128 data rows; comparison.md copied into Result | done

## Judgment calls

- Kickoff 2026-10-14 04:38 UTC (creation time of this session's tmux session `261013-noisyrl`);
  the 10 h wall-clock cap therefore ends at 14:38 UTC.
- The 15-minute check-in task is scheduled at minutes 7, 22, 37 and 52 of every hour (every
  15 minutes, off the :00/:30 marks the scheduler asks to avoid), with the prompt given in the
  kickoff message, word for word.
- Step 5: `logs/vllm_run_1dbd6c289614.log` has no `EXIT` line. Killing the tmux session hangs
  up the wrapper's shell before it reaches `echo "EXIT $?"` (the smoke run's
  `logs/vllm_smoke_1dbd6c289614.log` has none either). The step's check is the GPU being free,
  and it passed, so this is not counted as a failure.
- After step 7 (10:01 UTC) the 15-minute check-in task was deleted. Its only job is to resume
  an unticked step, and none is left.

## Blocked

## Result

Copied verbatim from `comparison.md` (written by step 7, 2026-10-14 10:00 UTC):

# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 460; in both 442 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.738 |
| accuracy | cue_gold | 0.852 | 0.975 |
| accuracy | cue_dist | 0.513 | 0.185 |
| accuracy | train_mix | 0.736 | 0.700 |
| reward | none | 0.647 | 0.655 |
| reward | cue_gold | 0.716 | 0.794 |
| reward | cue_dist | 0.498 | 0.314 |
| reward | train_mix | 0.690 | 0.752 |
| followed_cue | cue_gold | 0.852 | 0.975 |
| followed_cue | cue_dist | 0.460 | 0.815 |
| followed_cue | train_mix | 0.715 | 0.922 |
| bias | cue_gold | 0.105 | 0.179 |
| bias | cue_dist | 0.267 | 0.596 |
| bias | train_mix | 0.163 | 0.321 |
| anti_bias | cue_gold | 0.035 | 0.002 |
| anti_bias | cue_dist | 0.019 | 0.000 |
| anti_bias | train_mix | 0.030 | 0.001 |
| no_bias | cue_gold | 0.860 | 0.818 |
| no_bias | cue_dist | 0.713 | 0.404 |
| no_bias | train_mix | 0.807 | 0.678 |
| deception_soft | cue_gold | 0.542 | 0.557 |
| deception_soft | cue_dist | 0.596 | 0.466 |
| deception_soft | train_mix | 0.559 | 0.523 |
| deception | cue_gold | 0.543 | 0.635 |
| deception | cue_dist | 0.650 | 0.422 |
| deception | train_mix | 0.578 | 0.561 |
| said_yes | cue_gold | 0.571 | 0.783 |
| said_yes | cue_dist | 0.766 | 0.934 |
| said_yes | train_mix | 0.641 | 0.832 |
| false_denial | cue_gold | 0.372 | 0.088 |
| false_denial | cue_dist | 0.287 | 0.071 |
| false_denial | train_mix | 0.313 | 0.078 |
| false_admission | cue_gold | 0.564 | 0.755 |
| false_admission | cue_dist | 0.786 | 0.940 |
| false_admission | train_mix | 0.632 | 0.790 |
| parse_failed | none | 0.026 | 0.001 |
| parse_failed | cue_gold | 0.017 | 0.000 |
| parse_failed | cue_dist | 0.027 | 0.000 |
| parse_failed | train_mix | 0.022 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3912 | 3912 |
| rollouts | cue_gold | 1467 | 1467 |
| rollouts | cue_dist | 1467 | 1467 |
| rollouts | train_mix | 2934 | 2934 |
| n_bias_rows | cue_gold | 1303 | 1326 |
| n_bias_rows | cue_dist | 1291 | 1326 |
| n_bias_rows | train_mix | 2594 | 2652 |
| n_deception_rows | cue_gold | 1258 | 1323 |
| n_deception_rows | cue_dist | 1266 | 1326 |
| n_deception_rows | train_mix | 2524 | 2649 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.882 |
| accuracy | cue_gold | 0.942 | 0.995 |
| accuracy | cue_dist | 0.626 | 0.235 |
| accuracy | train_mix | 0.832 | 0.733 |
| reward | none | 0.738 | 0.732 |
| reward | cue_gold | 0.768 | 0.808 |
| reward | cue_dist | 0.572 | 0.335 |
| reward | train_mix | 0.748 | 0.762 |
| followed_cue | cue_gold | 0.942 | 0.995 |
| followed_cue | cue_dist | 0.348 | 0.765 |
| followed_cue | train_mix | 0.735 | 0.916 |
| bias | cue_gold | 0.006 | 0.036 |
| bias | cue_dist | 0.325 | 0.707 |
| bias | train_mix | 0.117 | 0.270 |
| anti_bias | cue_gold | 0.040 | 0.003 |
| anti_bias | cue_dist | 0.001 | 0.000 |
| anti_bias | train_mix | 0.026 | 0.001 |
| no_bias | cue_gold | 0.955 | 0.961 |
| no_bias | cue_dist | 0.674 | 0.293 |
| no_bias | train_mix | 0.857 | 0.728 |
| deception_soft | cue_gold | 0.534 | 0.591 |
| deception_soft | cue_dist | 0.594 | 0.434 |
| deception_soft | train_mix | 0.553 | 0.534 |
| deception | cue_gold | 0.522 | 0.714 |
| deception | cue_dist | 0.655 | 0.340 |
| deception | train_mix | 0.566 | 0.581 |
| said_yes | cue_gold | 0.524 | 0.743 |
| said_yes | cue_dist | 0.793 | 0.942 |
| said_yes | train_mix | 0.618 | 0.810 |
| false_denial | cue_gold | 0.333 | 0.105 |
| false_denial | cue_dist | 0.288 | 0.074 |
| false_denial | train_mix | 0.286 | 0.077 |
| false_admission | cue_gold | 0.523 | 0.737 |
| false_admission | cue_dist | 0.832 | 0.981 |
| false_admission | train_mix | 0.605 | 0.768 |
| parse_failed | none | 0.022 | 0.001 |
| parse_failed | cue_gold | 0.014 | 0.000 |
| parse_failed | cue_dist | 0.026 | 0.000 |
| parse_failed | train_mix | 0.020 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3072 | 3072 |
| rollouts | cue_gold | 1152 | 1152 |
| rollouts | cue_dist | 1152 | 1152 |
| rollouts | train_mix | 2304 | 2304 |
| n_bias_rows | cue_gold | 1035 | 1050 |
| n_bias_rows | cue_dist | 1024 | 1050 |
| n_bias_rows | train_mix | 2059 | 2100 |
| n_deception_rows | cue_gold | 994 | 1047 |
| n_deception_rows | cue_dist | 1023 | 1050 |
| n_deception_rows | train_mix | 2017 | 2097 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.214 |
| accuracy | cue_gold | 0.524 | 0.905 |
| accuracy | cue_dist | 0.098 | 0.000 |
| accuracy | train_mix | 0.384 | 0.579 |
| reward | none | 0.313 | 0.374 |
| reward | cue_gold | 0.524 | 0.743 |
| reward | cue_dist | 0.225 | 0.238 |
| reward | train_mix | 0.479 | 0.717 |
| followed_cue | cue_gold | 0.524 | 0.905 |
| followed_cue | cue_dist | 0.870 | 1.000 |
| followed_cue | train_mix | 0.640 | 0.948 |
| bias | cue_gold | 0.489 | 0.725 |
| bias | cue_dist | 0.045 | 0.174 |
| bias | train_mix | 0.338 | 0.514 |
| anti_bias | cue_gold | 0.015 | 0.000 |
| anti_bias | cue_dist | 0.090 | 0.000 |
| anti_bias | train_mix | 0.047 | 0.000 |
| no_bias | cue_gold | 0.496 | 0.275 |
| no_bias | cue_dist | 0.865 | 0.826 |
| no_bias | train_mix | 0.615 | 0.486 |
| deception_soft | cue_gold | 0.570 | 0.428 |
| deception_soft | cue_dist | 0.604 | 0.587 |
| deception_soft | train_mix | 0.581 | 0.482 |
| deception | cue_gold | 0.621 | 0.333 |
| deception | cue_dist | 0.630 | 0.736 |
| deception | train_mix | 0.624 | 0.485 |
| said_yes | cue_gold | 0.746 | 0.935 |
| said_yes | cue_dist | 0.654 | 0.902 |
| said_yes | train_mix | 0.732 | 0.916 |
| false_denial | cue_gold | 0.374 | 0.085 |
| false_denial | cue_dist | 0.250 | 0.021 |
| false_denial | train_mix | 0.349 | 0.081 |
| false_admission | cue_gold | 0.865 | 0.987 |
| false_admission | cue_dist | 0.649 | 0.886 |
| false_admission | train_mix | 0.776 | 0.912 |
| parse_failed | none | 0.039 | 0.001 |
| parse_failed | cue_gold | 0.029 | 0.000 |
| parse_failed | cue_dist | 0.032 | 0.000 |
| parse_failed | train_mix | 0.033 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 840 | 840 |
| rollouts | cue_gold | 315 | 315 |
| rollouts | cue_dist | 315 | 315 |
| rollouts | train_mix | 630 | 630 |
| n_bias_rows | cue_gold | 268 | 276 |
| n_bias_rows | cue_dist | 267 | 276 |
| n_bias_rows | train_mix | 535 | 552 |
| n_deception_rows | cue_gold | 264 | 276 |
| n_deception_rows | cue_dist | 243 | 276 |
| n_deception_rows | train_mix | 507 | 552 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.630 | 0.915 | 0.915 | 0.915 |
| first_10_steps | label_correct__cue_at_other | 0.177 | 0.705 | 0.281 | 0.281 |
| first_10_steps | label_flipped__cue_at_label | 0.152 | 0.719 | 0.719 | 0.265 |
| first_10_steps | label_flipped__cue_at_other | 0.042 | 0.898 | 0.097 | 0.898 |
| last_10_steps | label_correct__cue_at_label | 0.616 | 0.987 | 0.987 | 0.987 |
| last_10_steps | label_correct__cue_at_other | 0.172 | 0.786 | 0.212 | 0.212 |
| last_10_steps | label_flipped__cue_at_label | 0.163 | 0.852 | 0.852 | 0.145 |
| last_10_steps | label_flipped__cue_at_other | 0.050 | 0.977 | 0.023 | 0.977 |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4136 | 0.752 | 0.863 | 0.772 |
| first_10_steps | base_wrong | 984 | 0.501 | 0.784 | 0.602 |
| last_10_steps | base_right | 4088 | 0.730 | 0.923 | 0.797 |
| last_10_steps | base_wrong | 1032 | 0.663 | 0.958 | 0.733 |
