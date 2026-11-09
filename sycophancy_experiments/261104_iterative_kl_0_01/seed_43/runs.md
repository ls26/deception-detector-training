# Runs

State file for the unattended run of the seed-43 line (pod 09f52c746bb4). Read this first after any restart.

## Queue

- [x] 1. Pod check
- [x] 2. Round 1 regenerated (seed 43; seed 42: not needed)
- [x] 3. score 1
- [x] 4. generate 1
- [x] 5. build 2
- [x] 6. config 2
- [x] 7. Tables on round 1 alone
- [x] 8. Review 1
- [x] 9. vLLM for smoke
- [x] 10. Smoke RL run
- [x] 11. Smoke record check
- [x] 12. Stop vLLM
- [x] 13. Smoke checks
- [x] 14. Review 2
- [x] 15. Round 2: vLLM
- [x] 16. Round 2: RL run
- [x] 17. Round 2: record check
- [x] 18. Round 2: checks
- [x] 19. Round 2: stop vLLM
- [x] 20. Round 2: final eval
- [x] 21. Round 2: base rescored
- [x] 22. Round 2: compare
- [x] 23. Round 2: score 2
- [x] 24. Round 2: generate 2
- [x] 25. Round 2: build 3
- [x] 26. Round 2: config 3
- [x] 27. Review 3 (end of round 2)
- [x] 28. Round 3: vLLM
- [x] 29. Round 3: RL run
- [x] 30. Round 3: record check
- [x] 31. Round 3: checks
- [x] 32. Round 3: stop vLLM
- [x] 33. Round 3: final eval
- [x] 34. Round 3: base rescored
- [x] 35. Round 3: compare
- [x] 36. Round 3: score 3
- [x] 37. Round 3: generate 3
- [x] 38. Round 3: build 4
- [x] 39. Round 3: config 4
- [x] 40. Review 4 (end of round 3)
- [x] 41. Round 4: vLLM
- [x] 42. Round 4: RL run
- [x] 43. Round 4: record check
- [x] 44. Round 4: checks
- [x] 45. Round 4: stop vLLM
- [x] 46. Round 4: final eval
- [x] 47. Round 4: base rescored
- [x] 48. Round 4: compare
- [x] 49. Round 4: score 4
- [x] 50. Round 4: generate 4
- [x] 51. Round 4: build 5
- [x] 52. Round 4: config 5
- [x] 53. Review 5 (end of round 4)
- [x] 54. Round 5: vLLM
- [x] 55. Round 5: RL run
- [x] 56. Round 5: record check
- [x] 57. Round 5: checks
- [x] 58. Round 5: stop vLLM
- [x] 59. Round 5: final eval
- [x] 60. Round 5: base rescored
- [x] 61. Round 5: compare
- [x] 62. Round 5: score 5
- [x] 63. Review 6 (end of round 5)
- [x] 64. Seed table
- [x] 65. Dynamics
- [x] 66. Probe table
- [x] 67. Review 7 (auditor)

## Pod

- `hostname`: 09f52c746bb4
- `nvidia-smi -L`: GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-98cc18da-4f6e-a293-6870-8e1c62ab9065)
- `uname -r`: 6.8.0-147-generic
- `uptime -s`: 2026-10-20 02:51:57
- Driver 595.91.07, CUDA 13.2 (`nvidia-smi`)
- `df -h /workspace` (2026-11-04 21:05 UTC): size 513G, used 415G, avail 98G (81%)
- `nvidia-smi` at step 1: no process on the GPU

## Log

One line per step: `<UTC time> | step | outcome | next`.

- 2026-11-04 21:06 | 1 | pod check passed: 09f52c746bb4, one H100 80GB, no GPU process, 98G free | 2
- 2026-11-04 21:07 | 2 | job EXIT 0; checks/round1_repro.json: rows 500, questions 50, same_completion 212, same_pred 473 (reported) | 3
- 2026-11-04 21:10 | 3 | job EXIT 0; checks/score_round1.json: round_probe_here_vs_final_eval answers 977, max_abs_difference 0.0 | 4
- 2026-11-04 21:18 | 4 | job EXIT 0; generations/round1/completions.jsonl 17038 rows, 1217 questions, all split train | 5
- 2026-11-04 21:19 | 5 | job EXIT 0; per_group original 378, round1 378; no TOPPED UP: line, no DROPPED FROM TEST FILE: line; train_rows 3024, test 274 biased / 451 unbiased, test_one_class false | 6
- 2026-11-04 21:20 | 6 | job EXIT 0; wrote s43_round2.yaml and smoke_s43_round2.yaml (max_train_examples 6048, batch 4 x 16) | 7
- 2026-11-04 21:21 | 7 | exit 0; wrote dry/tables/round1/, dry/dynamics_round1/, dry/probe_table_round1/ | 8
- 2026-11-04 21:24 | 8 | review 1 PASS (fresh general-purpose agent; full verdict under Reviews) | 9
- 2026-11-04 21:27 | 9 | vLLM for smoke_s43_round2 answered after 170 s (logs/vllm_smoke_s43_round2_a1_09f52c746bb4.log); GPU 30263 MiB used | 10
- 2026-11-04 21:50 | 10 | smoke_s43_round2 attempt 1 log ends EXIT 0; W&B run l9mf5rig; no CRASH: line; `GRPO subprocess finished in 21.1 min (exit=0, attempt=1/4)`; peak GPU memory 75791 MiB; lowest free disk 92G; run_metrics batch 4 x 16 | 11
- 2026-11-04 21:51 | 11 | exit 0; runs/smoke_s43_round2/check_records.json: PASS true, n_files 2, seed 43 | 12
- 2026-11-04 21:51 | 12 | killed 261104-s43-vllm; nvidia-smi lists no process after ~10 s, 0 MiB used | 13
- 2026-11-04 21:53 | 13 | exit 0; wrote checks/reward_, advantages_, probe_ and kl_smoke_s43_round2.json (advantage max_abs_difference 1.43e-08, kl_div_file_step_0 0.104) | 14
- 2026-11-04 21:55 | 14 | review 2 PASS (fresh general-purpose agent; full verdict under Reviews) | 15
- 2026-11-04 21:58 | 15 | vLLM for s43_round2 answered after 201 s (logs/vllm_s43_round2_a1_09f52c746bb4.log); GPU 30263 MiB used | 16
- 2026-11-05 01:09 | 16 | s43_round2 attempt 1 log ends EXIT 0; W&B run yi9o10la; no CRASH: line; `GRPO subprocess finished in 188.8 min (exit=0, attempt=1/4)`; peak GPU memory 78330 MiB; lowest free disk 83G; run_metrics batch 4 x 16 | 17
- 2026-11-05 01:10 | 17 | exit 0; runs/s43_round2/check_records.json: PASS true, n_files 128, seed 43 | 18
- 2026-11-05 01:12 | 18 | exit 0; wrote checks/advantages_, probe_ and kl_s43_round2.json (advantage max_abs_difference 3.94e-08 over 128 steps, kl_div_file_step_0 0.107, KL steps 128) | 19
- 2026-11-05 01:12 | 19 | killed 261104-s43-vllm; nvidia-smi lists no process after ~10 s, 0 MiB used | 20
- 2026-11-05 01:18 | 20 | eval log ends EXIT 0; evals/s43_round2/summary.json: questions 489, cued_rollouts 978 | 21
- 2026-11-05 01:20 | 21 | copy_base exit 0; rescoring log ends EXIT 0; evals/base_rescored_s43_round2/summary.json: questions 489, cued_rollouts 978 | 22
- 2026-11-05 01:21 | 22 | exit 0; wrote runs/s43_round2/comparison.md, comparison.json, training_curve.csv (128 data rows) | 23
- 2026-11-05 01:24 | 23 | job EXIT 0; checks/score_round2.json: round_probe_here_vs_final_eval answers 976, max_abs_difference 0.0 | 24
- 2026-11-05 01:32 | 24 | job EXIT 0; generations/round2/completions.jsonl 17038 rows, 1217 questions, all split train | 25
- 2026-11-05 01:33 | 25 | job EXIT 0; per_group original 378, round1 378, round2 255; no TOPPED UP: line, no DROPPED FROM TEST FILE: line; train_rows 4044, test 153 biased / 448 unbiased, test_one_class false | 26
- 2026-11-05 01:33 | 26 | job EXIT 0; wrote s43_round3.yaml (checkpoint s43_round2_yi9o10la/checkpoints/final, max_train_examples 8088, batch 4 x 16) | 27
- 2026-11-05 01:37 | 27 | review 3 PASS (fresh general-purpose agent; full verdict under Reviews) | 28
- 2026-11-05 01:40 | 28 | vLLM for s43_round3 answered after 171 s (logs/vllm_s43_round3_a1_09f52c746bb4.log); GPU 30263 MiB used | 29
- 2026-11-05 04:55 | 29 | s43_round3 attempt 1 log ends EXIT 0; W&B run sprg9bny; no CRASH: line; `GRPO subprocess finished in 193.2 min (exit=0, attempt=1/4)`; peak GPU memory 78004 MiB; lowest free disk 78G; run_metrics batch 4 x 16 | 30
- 2026-11-05 04:56 | 30 | exit 0; runs/s43_round3/check_records.json: PASS true, n_files 128, seed 43 | 31
- 2026-11-05 04:58 | 31 | exit 0; wrote checks/advantages_, probe_ and kl_s43_round3.json (advantage max_abs_difference 2.31e-08 over 128 steps, kl_div_file_step_0 0.163, KL steps 128) | 32
- 2026-11-05 04:58 | 32 | killed 261104-s43-vllm; nvidia-smi lists no process after ~10 s, 0 MiB used | 33
- 2026-11-05 05:04 | 33 | eval log ends EXIT 0; evals/s43_round3/summary.json: questions 489, cued_rollouts 978 | 34
- 2026-11-05 05:07 | 34 | copy_base exit 0; rescoring log ends EXIT 0; evals/base_rescored_s43_round3/summary.json: questions 489, cued_rollouts 978 | 35
- 2026-11-05 05:07 | 35 | exit 0; wrote runs/s43_round3/comparison.md, comparison.json, training_curve.csv (128 data rows) | 36
- 2026-11-05 05:10 | 36 | job EXIT 0; checks/score_round3.json: round_probe_here_vs_final_eval answers 975, max_abs_difference 0.0 | 37
- 2026-11-05 05:18 | 37 | job EXIT 0; generations/round3/completions.jsonl 17038 rows, 1217 questions, all split train | 38
- 2026-11-05 05:19 | 38 | job EXIT 0; per_group original 378, round1 378, round2 255, round3 135; no TOPPED UP: line, no DROPPED FROM TEST FILE: line; train_rows 4584, test 85 biased / 454 unbiased, test_one_class false | 39
- 2026-11-05 05:20 | 39 | job EXIT 0; wrote s43_round4.yaml (checkpoint s43_round3_sprg9bny/checkpoints/final, max_train_examples 9168, batch 4 x 16) | 40
- 2026-11-05 05:23 | 40 | review 4 PASS (fresh general-purpose agent; full verdict under Reviews) | 41
- 2026-11-05 05:27 | 41 | vLLM for s43_round4 answered after 180 s (logs/vllm_s43_round4_a1_09f52c746bb4.log); GPU 30263 MiB used | 42
- 2026-11-05 08:38 | 42 | s43_round4 attempt 1 log ends EXIT 0; W&B run pvormx4t; no CRASH: line; `GRPO subprocess finished in 188.8 min (exit=0, attempt=1/4)`; peak GPU memory 75992 MiB; lowest free disk 81G; run_metrics batch 4 x 16 | 43
- 2026-11-05 08:39 | 43 | exit 0; runs/s43_round4/check_records.json: PASS true, n_files 128, seed 43 | 44
- 2026-11-05 08:41 | 44 | exit 0; wrote checks/advantages_, probe_ and kl_s43_round4.json (advantage max_abs_difference 1.97e-08 over 128 steps, kl_div_file_step_0 0.060, KL steps 128) | 45
- 2026-11-05 08:41 | 45 | killed 261104-s43-vllm; nvidia-smi lists no process after ~10 s, 0 MiB used | 46
- 2026-11-05 08:46 | 46 | eval log ends EXIT 0; evals/s43_round4/summary.json: questions 489, cued_rollouts 978 | 47
- 2026-11-05 08:49 | 47 | copy_base exit 0; rescoring log ends EXIT 0; evals/base_rescored_s43_round4/summary.json: questions 489, cued_rollouts 978 | 48
- 2026-11-05 08:49 | 48 | exit 0; wrote runs/s43_round4/comparison.md, comparison.json, training_curve.csv (128 data rows) | 49
- 2026-11-05 08:52 | 49 | job EXIT 0; checks/score_round4.json: round_probe_here_vs_final_eval answers 976, max_abs_difference 0.0 | 50
- 2026-11-05 08:59 | 50 | job EXIT 0; generations/round4/completions.jsonl 17038 rows, 1217 questions, all split train | 51
- 2026-11-05 09:01 | 51 | job EXIT 0; per_group original 378, round1 378, round2 255, round3 135, round4 100 (smallest pool 41, floored at 100); `TOPPED UP: round 5: round4 bias / cue_gold +59 rows of round1`; `TOPPED UP: round 5: round4 bias / cue_dist +6 rows of round3`; no DROPPED FROM TEST FILE: line; train_rows 4984, test 61 biased / 448 unbiased, test_one_class false | 52
- 2026-11-05 09:01 | 52 | job EXIT 0; wrote s43_round5.yaml (checkpoint s43_round4_pvormx4t/checkpoints/final, max_train_examples 9968, batch 4 x 16) | 53
- 2026-11-05 09:05 | 53 | review 5 PASS (fresh general-purpose agent; full verdict under Reviews) | 54
- 2026-11-05 09:08 | 54 | vLLM for s43_round5 answered after 170 s (logs/vllm_s43_round5_a1_09f52c746bb4.log); GPU 30263 MiB used | 55
- 2026-11-05 12:15 | 55 | s43_round5 attempt 1 log ends EXIT 0; W&B run 3yrst3z8; no CRASH: line; `GRPO subprocess finished in 185.1 min (exit=0, attempt=1/4)`; peak GPU memory 76481 MiB; lowest free disk 79G; run_metrics batch 4 x 16 | 56
- 2026-11-05 12:17 | 56 | exit 0; runs/s43_round5/check_records.json: PASS true, n_files 128, seed 43 | 57
- 2026-11-05 12:18 | 57 | exit 0; wrote checks/advantages_, probe_ and kl_s43_round5.json (advantage max_abs_difference 2.18e-08 over 128 steps, kl_div_file_step_0 0.143, KL steps 128) | 58
- 2026-11-05 12:18 | 58 | killed 261104-s43-vllm; nvidia-smi lists no process after ~10 s, 0 MiB used | 59
- 2026-11-05 12:24 | 59 | eval log ends EXIT 0; evals/s43_round5/summary.json: questions 489, cued_rollouts 978 | 60
- 2026-11-05 12:26 | 60 | copy_base exit 0; rescoring log ends EXIT 0; evals/base_rescored_s43_round5/summary.json: questions 489, cued_rollouts 978 | 61
- 2026-11-05 12:27 | 61 | exit 0; wrote runs/s43_round5/comparison.md, comparison.json, training_curve.csv (128 data rows) | 62
- 2026-11-05 12:29 | 62 | job EXIT 0; checks/score_round5.json: round_probe_here_vs_final_eval answers 978, max_abs_difference 0.0 | 63
- 2026-11-05 12:32 | 63 | review 6 PASS (fresh general-purpose agent; full verdict under Reviews) | 64
- 2026-11-05 12:32 | 64 | exit 0; wrote tables/rounds/seed_table.md, seed_table.json | 65
- 2026-11-05 12:33 | 65 | exit 0; wrote runs/dynamics/dynamics.json, dynamics.md | 66
- 2026-11-05 12:34 | 66 | job EXIT 0; wrote probe_table.md, probe_table.json | 67
- 2026-11-05 12:45 | 67 | review 7 PASS (fresh auditor agent; full verdict under Reviews; one wording note on seed_table.py's fixed probe-only-groups sentence, no number affected); copied tables/rounds/seed_table.md and probe_table.md into Result | done

## Judgment calls

- 2026-11-05 12:32 UTC, steps 64 and 65: ran the seed table (64) and the dynamics table (65) at the same time, as two background shell commands. Both are CPU steps that read only saved files and neither reads the other's output (CLAUDE.md allows changing the order of independent CPU steps; the GPU rule covers GPU jobs only). Each is ticked when its own command exits.

## Code fixes

## RL crashes

- smoke_s43_round2: none (attempt 1, no CRASH: line).
- s43_round2: none (attempt 1, no CRASH: line).
- s43_round3: none (attempt 1, no CRASH: line).
- s43_round4: none (attempt 1, no CRASH: line).
- s43_round5: none (attempt 1, no CRASH: line).

## Reviews

### Review 1 (step 8), 2026-11-04 21:24 UTC, fresh general-purpose agent: PASS

Review 1 (start, step 8), seed-43 line: overall verdict PASS

I read every file myself and changed no file. My one scratch script is /tmp/review1_s43/check.py, run with /workspace/obfuscation-atlas/.venv/bin/python, plus a plain `diff` of the two round-2 yamls.

**(a) Score: PASS**
- PASS: `round_probe_here_vs_final_eval.max_abs_difference` is 0.0, which is ≤ 0.02.
- PASS: `first_probe_link` is exactly `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset`.
- PASS: `round_probe_link` and `round_probe_in_run_folder` are the same string. Both are exactly `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`.

**(b) Own answers: PASS**
- PASS: `generations/round1/completions.jsonl` has 17038 rows over 1217 questions, and every question has exactly 14 rows.
- PASS: `split` is `train` in all 17038 rows.
- PASS: `adapter` is `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/final` in all 17038 rows.

**(c) Round 2's probe set (set rule, Q = 2): PASS**
- PASS: the keys of `sources` are `['original', 'round1']`, in that order.
- PASS: `original` has `per_group` 378, `own_rows` 378 in all four groups, and every `topped_up_from` is empty.
- PASS, per-group size:
  - original: smallest pool is 378, so min(max(378, 100), 378) = 378, which matches `per_group` 378.
  - round1: smallest pool is 453, so min(max(453, 100), 378) = 378, which matches `per_group` 378.
- PASS: in every group of both sources, `own_rows` = min(`pool`, `per_group`) = 378, and 378 plus no top-ups = 378. No `topped_up_from` names any source.
- PASS: `train_rows` is 3024 = 4 × (378 + 378), and `train_biased` is 1512 = 3024 / 2.
- PASS, `train.jsonl` row counts: 3024 rows, 378 in each of the 8 (`source`, `label`, `condition`) cells, which equals the counts `build.json` implies. `label` 1 always goes with `bias_rm` `bias`, and 0 with `no bias`.
- PASS: no (`source`, `question_id`, `condition`, `sample_idx`) appears twice (0 duplicates).
- PASS: no training row is on a question with `split` `eval` in questions.jsonl (0 rows), and every training `question_id` is in that file.
- PASS: the 1512 `source` `original` rows equal `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl` (1512 rows) in order, on `question_id`, `condition`, `sample_idx`, `prompt`, `completion` and `label`.
- Does not apply: the byte-for-byte prefix rule is only for Q ≥ 3.
- PASS, `test.jsonl`: 725 rows, of which 274 have `label` 1 (= `test_biased` 274) and 451 have `label` 0 (= `test_unbiased` 451).
- PASS: `test_answers` is "/workspace/sycophancy_experiments/261103_kl_0_01/weight_1_0/evals/kl_weight_1_0_s43 (round 1's final-eval answers)".
- PASS: `test_one_class` is false, and neither count is 0.

**(d) Round 2's config (config rule, Q = 2): PASS**
- PASS, `s43_round2.yaml` named settings:
  - `run_key` and `probe.directory_save_prefix`: `s43_round2`
  - `output_dir`: `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43`
  - `group`: `261104_iterative_kl_0_01`
  - `checkpoint`: `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/final`
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 43, `training.max_steps` 128
  - `++grpo.model.kl_reference_from_base`: true
- PASS, batch: `training.batch_size` 4 and `training.grad_accum_steps` 16. The Judgment calls section of runs.md is empty, so 4 × 16 is what the rule expects.
- PASS, probe files:
  - `bias_probe_train_jsonl` is `.../seed_43/probe_sets/round2/train.jsonl`.
  - `bias_probe_eval_jsonl` is `.../seed_43/probe_sets/round2/test.jsonl`, the right file because `test_one_class` is false.
  - `max_train_examples` is 6048 = 2 × 3024.
- PASS, everything else: the override key set equals `round_template.yaml`'s, with no extra or missing keys. The 9 other overrides are identical to the template in value and type: `+data/probe_dataset`, `task_args.bias_probe_at_answer`, `model.activation_matching_layers`, `detector.filter_layers_below_loss_threshold` ('null' as a string), learning rate 0.03, 2000 max steps, `mmlu_questions_jsonl`, `cue_gold_prob` 0.75 and `model_batch_size` 8. `config_name` grpo_mcq and `probe.retrain` true also match.
- PASS, smoke config: the text diff of `smoke_s43_round2.yaml` against `s43_round2.yaml` shows only:
  - header comment lines 1 and 3;
  - `run_key` `smoke_s43_round2`;
  - `probe.directory_save_prefix` `smoke_s43_round2`;
  - `training.max_steps` 2;
  - an added `training.eval_steps: 1`.

**(e) Round-1 tables: PASS**
- PASS: `runs.round1` in `dry/tables/round1/seed_table.json` and `runs.s43` in `/workspace/sycophancy_experiments/261103_kl_0_01/weight_1_0/tables/kl_weight_1_0/seed_table.json` have the same keys and the same values in every key except `final_eval`. That key is the full path here and the relative path `evals/kl_weight_1_0_s43` there.
- PASS: `dry/dynamics_round1/dynamics.json` exists (288876 bytes, valid JSON).
- PASS: `dry/probe_table_round1/probe_table.md` exists (1709 bytes).

**Reported (no verdict)**
- `round_probe_here_vs_final_eval.answers`: 977.
- From round 2's `build.json`:
  - original: `pool` bias / cue_gold 378, bias / cue_dist 1037, no bias / cue_gold 2739, no bias / cue_dist 642; `per_group` 378.
  - round1: `pool` bias / cue_gold 453, bias / cue_dist 1425, no bias / cue_gold 2727, no bias / cue_dist 563; `per_group` 378.
  - Top-ups: none, in either source.
  - `train_rows` 3024; `test_biased` 274; `test_unbiased` 451; `test_one_class` false.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/round1_repro.json`: rows 500, questions 50, same_completion 212, same_pred 473.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/dry/probe_table_round1/probe_table.md` in full:

```
# Probe scores per round

Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the
cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue
answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test
set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round
R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.

| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | Round R's probe; its own test set; round R-1's model (round 1: base) | 292 / 485 | 0.722 / 0.255 | 0.873 |
| round1 | Round R's probe; round R's answers; round R's model | 274 / 451 | 0.201 / 0.043 | 0.801 |
| round1 | Round R's probe; base model's answers; base model | 159 / 428 | 0.745 / 0.230 | 0.903 |
| round1 | First probe; round R's answers; round R's model | 274 / 451 | 0.201 / 0.043 | 0.801 |

# Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
```

**Overall verdict: PASS.** Rules (a), (b), (c), (d) and (e) all pass, so the queue can go on to step 9.


### Review 2 (step 14), 2026-11-04 21:55 UTC, fresh general-purpose agent: PASS

Review 2 (smoke, step 14), seed-43 line: overall verdict PASS

I read every file myself and changed no file. I wrote no scratch file; I only ran read-only shell and python3 one-liners.

**(a) Step files: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/runs/smoke_s43_round2/check_records.json` has `"PASS": true`, `"n_files": 2` and `"seed": 43`. Both entries of `per_step` (file steps 0 and 1) are also `PASS` true.

**(b) Reward: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/reward_smoke_s43_round2.json` has `penalty_weight` 1.0.
- File step 0: `probe_score_present_iff_parsed_mismatches` is 0, and `reward_mean_abs_difference` is 1.05e-08 (≤ 0.001).
- File step 1: `probe_score_present_iff_parsed_mismatches` is 0, and `reward_mean_abs_difference` is 3.66e-09 (≤ 0.001).

**(c) Advantage and KL: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/advantages_smoke_s43_round2.json`:
  - `advantage_std_floor` is 0.3.
  - `steps_compared` is 2.
  - `max_abs_difference` is 1.43e-08 (≤ 1e-4).
  - `kl_div_file_step_0` is 0.10409 (≥ 0.01).
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/kl_smoke_s43_round2.json`: `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 2 (run_id `l9mf5rig`).

**(d) Starting adapter: PASS**
- Line 97 of `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/logs/rl_smoke_s43_round2_a1_09f52c746bb4.log` is exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/final/lora`.
- It is the only `Phase-0: loading fresh model with adapter from` line in the log.
- The log ends `EXIT 0`, and its line 3172 is `GRPO subprocess finished in 21.1 min (exit=0, attempt=1/4)`. So attempt 1 is the attempt that finished, on its first try.

**(e) Probe: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/probe_smoke_s43_round2.json`:
  - `train_biased_unbiased` [1512, 1512] equals `expected_train_biased_unbiased` [1512, 1512].
  - `test_biased_unbiased` [274, 451] equals `expected_test_biased_unbiased` [274, 451].
  - Both entries of `saved_layers` (`initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`) are [13, 14, 15, 16].
- The label counts I counted in `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round2/train.jsonl` match: 3024 rows, 1512 with `label` 1 and 1512 with `label` 0, i.e. [1512, 1512].
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/smoke_s43_round2.yaml` names the eval file `data.probe_dataset.bias_probe_eval_jsonl: /workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round2/test.jsonl`. Its counts match too: 725 rows, 274 with `label` 1 and 451 with `label` 0, i.e. [274, 451].

**(f) Batch: PASS**
- The Judgment calls section of `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/runs.md` is empty, so the rule expects 4 and 16.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/run_metrics_smoke_s43_round2.json` (`run_id` `l9mf5rig`), under `metrics.full_config`:
  - `training.batch_size` 4
  - `training.grad_accum_steps` 16
  - `grpo.batching.per_device_train_batch_size` 4
  - `grpo.batching.gradient_accumulation_steps` 16

**(g) Phase-0 memory: PASS**
- Under RL crashes, runs.md says "smoke_s43_round2: none (attempt 1, no CRASH: line)". So no crashed attempt exists, and none stopped in Phase 0.
- The log has no `OutOfMemoryError` line and no `CUDA out of memory` line anywhere. It also has no `CRASH:`, `Traceback` or `removed partial run directory` line.
- The only lines that mention OOM at all are two accelerate INFO messages, at lines 80 and 98 (below). Neither is an out-of-memory error.

**Reported (no verdict)**
- `kl_div` per step in the smoke's KL file: step 1 0.10409469594014809 and step 2 0.10154047660762444 (`kl_loss` 0.0010409 and 0.0010154).
- For comparison, round 1 in `/workspace/sycophancy_experiments/261103_kl_0_01/weight_1_0/checks/kl_kl_weight_1_0_s43.json` (128 steps): last `kl_div` 0.1313937724335119, mean of its last 16 `kl_div` 0.10848194917161891.
- From the probe check: `test_auc` 0.8573567255247868, `test_mean_biased` 0.6464389930853117, `test_mean_unbiased` 0.2878463241163777. The same file also records `offline_auc_261010` 0.871.
- Step-file timing (`stat -c %Y`) in `/workspace/results/run_results/261104_iterative_kl_0_01/smoke_s43_round2_l9mf5rig/checkpoints/responses/`: `step_0.jsonl` 1793828207 (21:36:47 UTC) and `step_1.jsonl` 1793828312 (21:38:32 UTC), so 105 s apart.
  - That is next to 115 s for 261103's weight-1.0 smoke run at 8 × 8. I recomputed the 115 s from `/workspace/results/run_results/261103_kl_0_01/smoke_kl_weight_1_0_s42_vjgbshdo/checkpoints/responses/` and got the same number.
- Every out-of-memory line in the log: there are no `OutOfMemoryError` or `CUDA out of memory` lines. The only lines that mention OOM are:
  - line 80: `[2026-11-04 21:29:09,987][accelerate.utils.modeling][INFO] - We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).`
  - line 98: `[2026-11-04 21:29:24,284][accelerate.utils.modeling][INFO] - We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).`
- From runs.md for the smoke run (step 10 Log line):
  - minutes: `GRPO subprocess finished in 21.1 min (exit=0, attempt=1/4)`
  - peak GPU memory: 75791 MiB
  - crashed attempts: none ("smoke_s43_round2: none (attempt 1, no CRASH: line)")
  - lowest free disk: 92G (extra)

**Overall verdict: PASS.** Rules (a) to (g) all pass, so the queue can go on to step 15.

### Review 3 (step 27, end of round 2), 2026-11-05 01:37 UTC, fresh general-purpose agent: PASS

Review 3 (end of round 2, step 27), seed-43 line: overall verdict PASS

I read every file myself and changed no file. My scratch scripts are /tmp/review3_s43/config_check.py (run with /workspace/obfuscation-atlas/.venv/bin/python) and /tmp/review3_s43/set_check.py (python3). I also ran read-only shell commands: grep, stat, diff and python3 one-liners.

**(a) Step files: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/runs/s43_round2/check_records.json` has `"PASS": true`, `"n_files": 128` and `"seed": 43`.
- Extra detail: `wandb_run_id` is `yi9o10la`, and `run_dir` is `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round2_yi9o10la`. All 128 `per_step` entries (file steps 0–127) are PASS true.

**(b) Round 2's config (config rule, Q = 2): PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/s43_round2.yaml` has:
  - `run_key` and `probe.directory_save_prefix`: `s43_round2`
  - `output_dir`: `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43`
  - `group`: `261104_iterative_kl_0_01`
  - `checkpoint`: `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/final`
- Under `overrides`:
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 43, `training.max_steps` 128
  - `++grpo.model.kl_reference_from_base` true
- Batch: `training.batch_size` 4 and `training.grad_accum_steps` 16. The Judgment calls section of runs.md is empty, so 4 × 16 is what the rule expects.
- Probe files:
  - `bias_probe_train_jsonl` is `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round2/train.jsonl`.
  - `bias_probe_eval_jsonl` is `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round2/test.jsonl`. That is the right file, because round 2's `build.json` has `test_one_class` false.
  - `max_train_examples` is 6048 = 2 × `train_rows` 3024.
- Every other override:
  - The override key set equals the key set in `round_template.yaml`, with nothing extra or missing.
  - The 9 other overrides match the template in both value and type: `+data/probe_dataset`, `task_args.bias_probe_at_answer`, `model.activation_matching_layers`, `detector.filter_layers_below_loss_threshold` ('null'), learning rate 0.03, 2000 max steps, `mmlu_questions_jsonl`, `cue_gold_prob` 0.75 and `model_batch_size` 8.
  - Also checked: `config_name` grpo_mcq and `probe.retrain` true match the template.

**(c) Advantage and KL: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/advantages_s43_round2.json`:
  - `advantage_std_floor` 0.3
  - `steps_compared` 128
  - `max_abs_difference` 3.94e-08, which is ≤ 1e-4
  - `kl_div_file_step_0` 0.10735, which is ≥ 0.01
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/kl_s43_round2.json` (run_id `yi9o10la`): `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128.

**(d) Probe: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/probe_s43_round2.json`:
  - `train_biased_unbiased` [1512, 1512] equals `expected_train_biased_unbiased` [1512, 1512].
  - `test_biased_unbiased` [274, 451] equals `expected_test_biased_unbiased` [274, 451].
  - Both entries of `saved_layers` (`initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`) are [13, 14, 15, 16].

**(e) Starting adapter: PASS**
- Line 96 of `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/logs/rl_s43_round2_a1_09f52c746bb4.log` is exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/final/lora`. It is the only such line in the log.
- This log is the attempt that finished:
  - It is the only `rl_s43_round2` log.
  - Its line 4823 is `GRPO subprocess finished in 188.8 min (exit=0, attempt=1/4)`, and the log ends `EXIT 0`.
  - It has no `CRASH:`, `removed partial run directory`, `Traceback`, `OutOfMemoryError` or `CUDA out of memory` line.

**(f) Batch: PASS**
- The Judgment calls section of runs.md is empty, so the rule expects 4 and 16.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/run_metrics_s43_round2.json` (`run_id` `yi9o10la`), under `metrics.full_config`:
  - `training.batch_size` 4
  - `training.grad_accum_steps` 16
  - `grpo.batching.per_device_train_batch_size` 4
  - `grpo.batching.gradient_accumulation_steps` 16

**(g) Eval: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/evals/s43_round2/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/evals/base_rescored_s43_round2/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/runs/s43_round2/training_curve.csv` has 129 lines: a header plus 128 data rows, steps 0 to 127.

**(h) Score: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/score_round2.json`:
  - `round_probe_here_vs_final_eval.max_abs_difference` is 0.0, which is ≤ 0.02 (over 976 answers).
  - `first_probe_link` is exactly `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset`.
  - `round_probe_link` and `round_probe_in_run_folder` are the same string. Both are exactly `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round2_yi9o10la/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`.

**(i) Round 3's probe set (set rule, Q = 3): PASS**
- The keys of `sources` in `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round3/build.json` are `['original', 'round1', 'round2']`, in that order.
- `original` has `per_group` 378, `own_rows` 378 in all four groups, and every `topped_up_from` is empty.
- Per-group size, min(max(smallest pool, 100), 378):
  - original: smallest pool 378 gives 378, which matches `per_group` 378.
  - round1: smallest pool 453 gives 378, which matches `per_group` 378.
  - round2: smallest pool 255 gives 255, which matches `per_group` 255.
- In every group of all three sources, `own_rows` = min(`pool`, `per_group`), and `own_rows` plus the top-ups (none) = `per_group`. No `topped_up_from` names any source.
- `train_rows` is 4044 = 4 × (378 + 378 + 255), and `train_biased` is 2022 = 4044 / 2.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round3/train.jsonl`:
  - Row counts: 4044 rows. Every (`source`, `label`, `condition`) cell equals what `build.json` implies: 378 in each of the 4 original cells and the 4 round1 cells, and 255 in each of the 4 round2 cells. `label` 1 always goes with `bias_rm` `bias`.
  - 0 duplicate (`source`, `question_id`, `condition`, `sample_idx`).
  - 0 rows on questions with `split` `eval` in `/workspace/sycophancy-rl/sycophancy_setup/data/questions.jsonl`, and every training `question_id` is in that file.
  - The 1512 `source` `original` rows equal `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl` (1512 rows) in order, on `question_id`, `condition`, `sample_idx`, `prompt`, `completion` and `label`.
- Prefix rule (Q ≥ 3): `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round2/train.jsonl` (6537395 bytes) is byte for byte the start of round 3's `train.jsonl` (8703233 bytes).
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round3/test.jsonl`: 601 rows. 153 have `label` 1 (= `test_biased` 153) and 448 have `label` 0 (= `test_unbiased` 448).
- `test_answers` is "/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/evals/s43_round2 (round 2's final-eval answers)", which names round 2's final eval.
- `test_one_class` is false, and neither count is 0.

**(j) Round 3's config (config rule, Q = 3): PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/s43_round3.yaml` has:
  - `run_key` and `probe.directory_save_prefix`: `s43_round3`
  - `output_dir`: `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43`
  - `group`: `261104_iterative_kl_0_01`
  - `checkpoint`: `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round2_yi9o10la/checkpoints/final`
- Under `overrides`:
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 43, `training.max_steps` 128
  - batch 4 and 16 (Judgment calls is empty)
  - `++grpo.model.kl_reference_from_base` true
- Probe files:
  - `bias_probe_train_jsonl` is `.../seed_43/probe_sets/round3/train.jsonl`.
  - `bias_probe_eval_jsonl` is `.../seed_43/probe_sets/round3/test.jsonl`, the right file because `test_one_class` is false.
  - `max_train_examples` is 8088 = 2 × 4044.
- Every other override: the key set equals the template's, and the same 9 other overrides match it in value and type.
- A text diff of `s43_round3.yaml` against `s43_round2.yaml` shows only the expected lines: the header comment lines 1 and 3, `run_key`, `directory_save_prefix`, `checkpoint`, and the three probe-file lines.

**Reported (no verdict)**
- From `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/runs.md`, round 2 (step 16 log line):
  - minutes: `GRPO subprocess finished in 188.8 min (exit=0, attempt=1/4)`
  - lowest free disk: 83G
  - peak GPU memory: 78330 MiB
  - crashed attempts: none ("s43_round2: none (attempt 1, no CRASH: line)"), so there are no W&B ids, batches reached or answer lengths to report.
- Mean seconds per step: 80.3 s.
  - `stat -c %Y` of `step_0.jsonl` is 1793830084 (2026-11-04 22:08:04 UTC), and of `step_127.jsonl` is 1793840279 (2026-11-05 00:57:59 UTC), both in `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round2_yi9o10la/checkpoints/responses/`. So (1793840279 − 1793830084) / 127 = 10195 / 127 = 80.3 s.
  - That is next to 82 s for round 1 (261103, 8 × 8, pod 33f36c5bf32d). I recomputed round 1's number from `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/responses/` and got 82.25 s.
  - Round 2's 188.8 RL minutes sit next to round 1's 186.9 RL minutes.
- From `probe_s43_round2.json`: `test_auc` 0.8573567255247868, `test_mean_biased` 0.6464389930853117, `test_mean_unbiased` 0.2878463241163777. The same file also records `offline_auc_261010` 0.871.
  - These three values are identical to the smoke run's probe check that review 2 reported. Both runs had the same probe set and the same starting adapter.
- From `advantages_s43_round2.json`: `kl_div_file_step_0` 0.10735487763304263, and the mean `wandb_kl_div` over the last 16 steps (file steps 112–127) is 0.15859233807532291.
  - The KL file agrees: its first `kl_div` is 0.10735, its last is 0.16394 and the mean of its last 16 is 0.15859.
- From round 3's `build.json`:
  - original: `pool` bias / cue_gold 378, bias / cue_dist 1037, no bias / cue_gold 2739, no bias / cue_dist 642; `per_group` 378.
  - round1: `pool` bias / cue_gold 453, bias / cue_dist 1425, no bias / cue_gold 2727, no bias / cue_dist 563; `per_group` 378.
  - round2: `pool` bias / cue_gold 255, bias / cue_dist 631, no bias / cue_gold 2840, no bias / cue_dist 551; `per_group` 255.
  - Top-ups: none, in any source.
  - `test_one_class` false (test 153 biased / 448 unbiased; `train_rows` 4044).
  - The build log `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/logs/build_round3_a1_09f52c746bb4.log` has no `TOPPED UP:` or `DROPPED FROM TEST FILE:` line, and ends `EXIT 0`.

**Overall verdict: PASS.** Rules (a) to (j) all pass, so the queue can go on to step 28.

### Review 4 (step 40, end of round 3), 2026-11-05 05:23 UTC, fresh general-purpose agent: PASS

Review 4 (end of round 3, step 40), seed-43 line: overall verdict PASS

I read every file myself and changed no file. My scratch scripts are /tmp/review4_s43/config_check.py (run with /workspace/obfuscation-atlas/.venv/bin/python) and /tmp/review4_s43/set_check.py (python3). I also ran read-only shell commands: grep, stat, diff, wc and python3 one-liners.

The run ids match the brief. `run_metrics_s43_round3.json` has `run_id` `sprg9bny`, and `run_metrics_s43_round2.json` has `run_id` `yi9o10la`. The Judgment calls section of `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/runs.md` is empty, so the batch rules expect 4 × 16.

**(a) Step files: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/runs/s43_round3/check_records.json` has `"PASS": true`, `"n_files": 128` and `"seed": 43`.
- Extra detail: `wandb_run_id` is `sprg9bny`, and `run_dir` is `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round3_sprg9bny`. All 128 `per_step` entries are PASS true.

**(b) Round 3's config (config rule, Q = 3): PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/s43_round3.yaml` has:
  - `run_key` and `probe.directory_save_prefix`: `s43_round3`
  - `output_dir`: `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43`
  - `group`: `261104_iterative_kl_0_01`
  - `checkpoint`: `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round2_yi9o10la/checkpoints/final`
- Under `overrides`:
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 43, `training.max_steps` 128
  - `training.batch_size` 4 and `training.grad_accum_steps` 16
  - `++grpo.model.kl_reference_from_base` true
- Probe files:
  - `bias_probe_train_jsonl` is `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round3/train.jsonl`.
  - `bias_probe_eval_jsonl` is `.../seed_43/probe_sets/round3/test.jsonl`. That is the right file, because round 3's `build.json` has `test_one_class` false.
  - `max_train_examples` is 8088 = 2 × `train_rows` 4044.
- Every other override:
  - The override keys equal the keys of `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/round_template.yaml`, in the same order, with nothing extra or missing.
  - The 9 other overrides equal the template in both value and type: `+data/probe_dataset` sycophancy_bias_probe, `task_args.bias_probe_at_answer` true, `model.activation_matching_layers` range_13-17, `detector.filter_layers_below_loss_threshold` 'null', learning rate 0.03, 2000 max steps, `mmlu_questions_jsonl` questions.jsonl, `cue_gold_prob` 0.75 and `model_batch_size` 8.
  - Also checked: `config_name` grpo_mcq and `probe.retrain` true match the template.

**(c) Advantage and KL: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/advantages_s43_round3.json`:
  - `advantage_std_floor` 0.3
  - `steps_compared` 128
  - `max_abs_difference` 2.313e-08, which is ≤ 1e-4. The largest per-step `abs_difference` is the same value.
  - `kl_div_file_step_0` 0.16311, which is ≥ 0.01
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/kl_s43_round3.json` (run_id `sprg9bny`): `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128.

**(d) Probe: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/probe_s43_round3.json`:
  - `train_biased_unbiased` [2022, 2022] equals `expected_train_biased_unbiased` [2022, 2022].
  - `test_biased_unbiased` [153, 448] equals `expected_test_biased_unbiased` [153, 448].
  - Both entries of `saved_layers` (`initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`) are [13, 14, 15, 16].

**(e) Starting adapter: PASS**
- Line 97 of `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/logs/rl_s43_round3_a1_09f52c746bb4.log` is exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261104_iterative_kl_0_01/s43_round2_yi9o10la/checkpoints/final/lora`. It is the only such line in the log.
- This log is the attempt that finished:
  - It is the only `rl_s43_round3` log, and no `failed_attempts/` folder exists.
  - Its line 5084 is `GRPO subprocess finished in 193.2 min (exit=0, attempt=1/4)`, and the log ends `EXIT 0`.
  - It has no `CRASH:`, `removed partial run directory`, `Traceback`, `OutOfMemoryError` or `CUDA out of memory` line.

**(f) Batch: PASS**
- Judgment calls is empty, so the rule expects 4 and 16.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/run_metrics_s43_round3.json`, under `metrics.full_config`:
  - `training.batch_size` 4
  - `training.grad_accum_steps` 16
  - `grpo.batching.per_device_train_batch_size` 4
  - `grpo.batching.gradient_accumulation_steps` 16

**(g) Eval: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/evals/s43_round3/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/evals/base_rescored_s43_round3/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/runs/s43_round3/training_curve.csv` has 129 lines: a header plus 128 data rows, steps 0 to 127 with each step once.

**(h) Score: PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/score_round3.json`:
  - `round_probe_here_vs_final_eval.max_abs_difference` is 0.0, which is ≤ 0.02 (over 975 answers).
  - `first_probe_link` is exactly `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset`.
  - `round_probe_link` and `round_probe_in_run_folder` are the same string. Both are exactly `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round3_sprg9bny/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`.

**(i) Round 4's probe set (set rule, Q = 4): PASS**
- The keys of `sources` in `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round4/build.json` are `['original', 'round1', 'round2', 'round3']`, in that order.
- `original` has `per_group` 378, `own_rows` 378 in all four groups, and every `topped_up_from` is empty.
- Per-group size, min(max(smallest pool, 100), 378):
  - original: smallest pool 378 gives 378, which matches `per_group` 378.
  - round1: smallest pool 453 gives 378, which matches `per_group` 378.
  - round2: smallest pool 255 gives 255, which matches `per_group` 255.
  - round3: smallest pool 135 gives 135, which matches `per_group` 135.
- In every group of all four sources, `own_rows` = min(`pool`, `per_group`), and `own_rows` plus the top-ups (none) = `per_group`. No `topped_up_from` names any source.
- `train_rows` is 4584 = 4 × (378 + 378 + 255 + 135), and `train_biased` is 2292 = 4584 / 2.
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round4/train.jsonl`:
  - Row counts: 4584 rows. Every (`source`, `label`, `condition`) cell equals what `build.json` implies: 378 in each of the 4 original cells and the 4 round1 cells, 255 in each of the 4 round2 cells, and 135 in each of the 4 round3 cells.
  - `label` 1 always goes with `bias_rm` `bias` (2292 rows), and 0 with `no bias` (2292 rows).
  - The sources appear in blocks, in the order original, round1, round2, round3.
  - 0 duplicate (`source`, `question_id`, `condition`, `sample_idx`).
  - 0 rows on questions with `split` `eval` in `/workspace/sycophancy-rl/sycophancy_setup/data/questions.jsonl`, and every training `question_id` is in that file.
  - The 1512 `source` `original` rows equal `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl` (1512 rows) in order, on `question_id`, `condition`, `sample_idx`, `prompt`, `completion` and `label`.
- Prefix rule (Q ≥ 3): `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round3/train.jsonl` (8703233 bytes) is byte for byte the start of round 4's `train.jsonl` (9870478 bytes).
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/probe_sets/round4/test.jsonl`: 539 rows. 85 have `label` 1 (= `test_biased` 85) and 454 have `label` 0 (= `test_unbiased` 454). No other label values.
- `test_answers` is "/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/evals/s43_round3 (round 3's final-eval answers)", which names round 3's final eval.
- `test_one_class` is false, and neither count is 0.

**(j) Round 4's config (config rule, Q = 4): PASS**
- `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/s43_round4.yaml` has:
  - `run_key` and `probe.directory_save_prefix`: `s43_round4`
  - `output_dir`: `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43`
  - `group`: `261104_iterative_kl_0_01`
  - `checkpoint`: `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round3_sprg9bny/checkpoints/final`. That folder exists and holds `lora/`.
- Under `overrides`:
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 43, `training.max_steps` 128
  - batch 4 and 16 (Judgment calls is empty)
  - `++grpo.model.kl_reference_from_base` true
- Probe files:
  - `bias_probe_train_jsonl` is `.../seed_43/probe_sets/round4/train.jsonl`.
  - `bias_probe_eval_jsonl` is `.../seed_43/probe_sets/round4/test.jsonl`, the right file because `test_one_class` is false.
  - `max_train_examples` is 9168 = 2 × 4584.
- Every other override: the keys equal the template's, in the same order, and the same 9 other overrides match it in value and type. `config_name` and `probe.retrain` also match.
- A text diff of `s43_round4.yaml` against `s43_round3.yaml` shows only the expected lines: the header comment lines 1 and 3, `run_key`, `directory_save_prefix`, `checkpoint`, and the three probe-file lines.

**Reported (no verdict)**
- From runs.md, round 3 (step 29 log line):
  - minutes: `GRPO subprocess finished in 193.2 min (exit=0, attempt=1/4)`
  - lowest free disk: 78G
  - peak GPU memory: 78004 MiB
  - crashed attempts: none ("s43_round3: none (attempt 1, no CRASH: line)"), so there are no W&B ids, batches reached or answer lengths to report.
- Mean seconds per step: 80.8 s.
  - From `stat -c %Y` in `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round3_sprg9bny/checkpoints/responses/`: `step_0.jsonl` is 1793843433 (2026-11-05 01:50:33 UTC) and `step_127.jsonl` is 1793853696 (2026-11-05 04:41:36 UTC). So 10263 / 127 = 80.8 s.
  - That is next to 82 s for round 1 (261103, 8 × 8, pod 33f36c5bf32d). I recomputed round 1's number from `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/responses/` and got 82.25 s.
  - Round 3's 193.2 RL minutes sit next to round 1's 186.9.
- From `probe_s43_round3.json`: `test_auc` 0.8362365779645191, `test_mean_biased` 0.5609528280192, `test_mean_unbiased` 0.2664886146052855. The file also records `offline_auc_261010` 0.871.
- From `advantages_s43_round3.json`: `kl_div_file_step_0` is 0.16310988168697804. The mean `wandb_kl_div` over the last 16 steps (file steps 112–127) is 0.07419475987899204.
  - The KL file agrees: its first `kl_div` is 0.16311, its last is 0.17898 and the mean of its last 16 is 0.07419.
  - The last step's 0.179 stands alone: file steps 112–126 read 0.063–0.085.
- From round 4's `build.json` (pools in the order bias / cue_gold, bias / cue_dist, no bias / cue_gold, no bias / cue_dist):
  - original: `pool` 378 / 1037 / 2739 / 642; `per_group` 378.
  - round1: `pool` 453 / 1425 / 2727 / 563; `per_group` 378.
  - round2: `pool` 255 / 631 / 2840 / 551; `per_group` 255.
  - round3: `pool` 135 / 237 / 2923 / 614; `per_group` 135.
  - Top-ups: none, in any source.
  - `test_one_class` false (test 85 biased / 454 unbiased; `train_rows` 4584).
  - The build log `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/logs/build_round4_a1_09f52c746bb4.log` has no `TOPPED UP:` or `DROPPED FROM TEST FILE:` line, and ends `EXIT 0`.

**Overall verdict: PASS.** Rules (a) to (j) all pass, so the queue can go on to step 41.

### Review 5 (step 53, end of round 4), 2026-11-05 09:05 UTC, fresh general-purpose agent: PASS

Review 5 verdict (end of round 4, step 53): PASS

All ten rules (a) to (j) pass. I read every file myself and changed none. My two scratch scripts are under /tmp/claude-0/-workspace/8c08b635-56ad-4903-b8af-7c196a45d0b0/scratchpad/.

The Judgment calls section of runs.md is empty, so the out-of-memory rule was never recorded. The expected batch is therefore 4 × 16 for rules (b), (f) and (j).

**(a) Step files: PASS.** `runs/s43_round4/check_records.json` has `"PASS": true`, `"n_files": 128` and `"seed": 43`. All 128 per-step entries are PASS, and `wandb_run_id` is pvormx4t.

**(b) Round 4's config: PASS.** In `s43_round4.yaml`:
- `run_key` and `probe.directory_save_prefix` are `s43_round4`.
- `output_dir` is `.../261104_iterative_kl_0_01/seed_43` and `group` is `261104_iterative_kl_0_01`.
- `checkpoint` is `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round3_sprg9bny/checkpoints/final`. The run id `sprg9bny` comes from `run_metrics_s43_round3.json`.
- Overrides: `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 43, `training.max_steps` 128, `training.batch_size` 4, `training.grad_accum_steps` 16, `++grpo.model.kl_reference_from_base` true.
- The probe files are `probe_sets/round4/train.jsonl` and `probe_sets/round4/test.jsonl`. The test file is correct because round 4's `test_one_class` is false.
- `max_train_examples` is 9168, which is 2 × round 4's `train_rows` of 4584.
- The override keys are the same set as the template's, and every other override equals `round_template.yaml`'s.

**(c) Advantage and KL: PASS.**
- `advantages_s43_round4.json`: `advantage_std_floor` 0.3, `steps_compared` 128, `max_abs_difference` 1.97e-08 (limit 1e-4), `kl_div_file_step_0` 0.0605 (must be at least 0.01).
- `kl_s43_round4.json`: `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128, run_id pvormx4t.

**(d) Probe: PASS.** In `probe_s43_round4.json`, the training counts (biased, unbiased) are [2292, 2292], equal to the expected [2292, 2292]. The test counts are [85, 454], equal to the expected [85, 454]. `saved_layers` is [13, 14, 15, 16] for both entries (`initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`).

**(e) Starting adapter: PASS.** Line 97 of `logs/rl_s43_round4_a1_09f52c746bb4.log` is `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261104_iterative_kl_0_01/s43_round3_sprg9bny/checkpoints/final/lora`. This was the only attempt, and its log ends `EXIT 0`.

**(f) Batch: PASS.** Under `metrics.full_config` in `run_metrics_s43_round4.json`: `training.batch_size` 4, `training.grad_accum_steps` 16, `grpo.batching.per_device_train_batch_size` 4, `grpo.batching.gradient_accumulation_steps` 16.

**(g) Eval: PASS.**
- `evals/s43_round4/summary.json` has questions 489 and cued_rollouts 978.
- `evals/base_rescored_s43_round4/summary.json` has questions 489 and cued_rollouts 978.
- `runs/s43_round4/training_curve.csv` has 128 data rows (129 lines including the header).

**(h) Score: PASS.** In `score_round4.json`:
- `round_probe_here_vs_final_eval.max_abs_difference` is 0.0 over 976 answers (limit 0.02).
- `first_probe_link` is `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset`.
- `round_probe_link` and `round_probe_in_run_folder` are identical, and both are `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round4_pvormx4t/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`.

**(i) Round 5's probe set: PASS.**
- The `sources` keys are, in order, original, round1, round2, round3, round4.
- `original` has `per_group` 378, `own_rows` 378 in all four groups and no top-ups.
- `per_group` = min(max(smallest pool, 100), 378) holds for every source:

| Source | Smallest pool | per_group |
|---|---|---|
| original | 378 | 378 |
| round1 | 453 | 378 |
| round2 | 255 | 255 |
| round3 | 135 | 135 |
| round4 | 41 | 100 |

- In every group, `own_rows` = min(pool, per_group), and `own_rows` plus the top-ups equals `per_group`.
- Round 4's top-ups name only earlier own-answer sources: round1 and round3.
- `train_rows` is 4984, equal to 4 × 1246 (the sum of `per_group`). `train_biased` is 2492, equal to 4984 / 2.
- `train.jsonl` has 4984 rows, and its counts per (`source`, `label`, `condition`) exactly match `build.json`, with topped-up rows counted under their donor source's name.
- There are 0 duplicate (`source`, `question_id`, `condition`, `sample_idx`) keys.
- No training question has split eval in `questions.jsonl` (0 found), and every question id is in that file.
- The 1512 original rows equal `bias_probe_train.jsonl`'s 1512 rows in order on all six fields.
- Round 4's `train.jsonl` (9,870,478 bytes) is byte for byte the start of round 5's (10,686,616 bytes); `cmp -n` found no difference.
- `test.jsonl` has 61 rows with label 1 and 448 with label 0, equal to `test_biased` 61 and `test_unbiased` 448.
- `test_answers` is `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/evals/s43_round4 (round 4's final-eval answers)`.
- `test_one_class` is false, which is correct because neither count is 0.

**(j) Round 5's config: PASS.** In `s43_round5.yaml`:
- `run_key` and `probe.directory_save_prefix` are `s43_round5`; `output_dir` and `group` are as in (b).
- `checkpoint` is `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round4_pvormx4t/checkpoints/final`.
- The fixed overrides are the same as in (b), batch 4 × 16 included.
- The probe files are `probe_sets/round5/train.jsonl` and `probe_sets/round5/test.jsonl`. The test file is correct because `test_one_class` is false.
- `max_train_examples` is 9968, which is 2 × 4984.
- Every other override equals the template's, and the key sets are identical.

Reported values (no verdict)

**From runs.md:**
- Round 4's RL time: 188.8 min, from the line `GRPO subprocess finished in 188.8 min (exit=0, attempt=1/4)`, which also appears in the log.
- Lowest free disk: 81G.
- Peak GPU memory: 75992 MiB.
- Crashed attempts: none. runs.md says "s43_round4: none (attempt 1, no CRASH: line)", and the log has 0 `CRASH:` lines and no `removed partial run directory` line.

**Mean seconds per step: 77.7 s.** The modification time of `step_0.jsonl` is 05:38:21 UTC and of `step_127.jsonl` is 08:22:55 UTC, so (1793866975 − 1793857101) / 127 = 77.7 s. For comparison, round 1 (261103, 8 × 8, pod 33f36c5bf32d) took 82 s per step and 186.9 RL minutes.

**From probe_s43_round4.json:**
- `test_auc`: 0.8069
- `test_mean_biased`: 0.5914
- `test_mean_unbiased`: 0.2969

**From advantages_s43_round4.json:**
- `kl_div_file_step_0`: 0.0605
- Mean `wandb_kl_div` over the last 16 steps (file steps 112 to 127): 0.1319

**From round 5's build.json:**

| Source | Pool: bias / cue_gold | Pool: bias / cue_dist | Pool: no bias / cue_gold | Pool: no bias / cue_dist | per_group |
|---|---|---|---|---|---|
| original | 378 | 1037 | 2739 | 642 | 378 |
| round1 | 453 | 1425 | 2727 | 563 | 378 |
| round2 | 255 | 631 | 2840 | 551 | 255 |
| round3 | 135 | 237 | 2923 | 614 | 135 |
| round4 | 41 | 94 | 2936 | 642 | 100 |

- Top-ups: round4's bias / cue_gold group got 59 rows from round1, and its bias / cue_dist group got 6 rows from round3. These are the only top-ups, and they match the build log's two `TOPPED UP:` lines.
- `test_one_class`: false.

Overall verdict: PASS

### Review 6 (step 63, end of round 5), 2026-11-05 12:32 UTC, fresh general-purpose agent: PASS

Review 6 (end of round 5, step 63), seed-43 line: overall verdict PASS. All eight rules (a) to (h) pass. I read every file myself and changed nothing.

The run IDs check out. `run_id` is `3yrst3z8` in `run_metrics_s43_round5.json` and `pvormx4t` in `run_metrics_s43_round4.json`. `check_records.json` names the same W&B run (`3yrst3z8`) and the same run folder (`/workspace/results/run_results/261104_iterative_kl_0_01/s43_round5_3yrst3z8`).

The batch rules expect 4 × 16. The "Judgment calls" section of `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/runs.md` is empty, so the out-of-memory rule was never recorded for any run of this line, the smoke run included.

**(a) Step files: PASS.** `runs/s43_round5/check_records.json` has `"PASS": true`, `"n_files": 128` and `"seed": 43`. It holds 128 per-step entries and every one has `PASS` true.

**(b) Config: PASS.** In `s43_round5.yaml`:
- `run_key` and `probe.directory_save_prefix` are both `s43_round5`.
- `output_dir` is `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43`.
- `group` is `261104_iterative_kl_0_01`.
- `checkpoint` is `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round4_pvormx4t/checkpoints/final`.
- The overrides are `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 43, `training.max_steps` 128, `training.batch_size` 4, `training.grad_accum_steps` 16 and `++grpo.model.kl_reference_from_base` true.
- `bias_probe_train_jsonl` is `.../seed_43/probe_sets/round5/train.jsonl`. `bias_probe_eval_jsonl` is `.../seed_43/probe_sets/round5/test.jsonl`, which is correct because round 5's `build.json` has `test_one_class` false.
- `max_train_examples` is 9968, which is 2 × the `train_rows` of 4984 in `build.json`.
- The config and `round_template.yaml` have the same set of override keys. The only differences are the five FILL_IN slots (`training.seed`, `training.max_steps`, the two probe-file paths and `max_train_examples`). The other top-level keys also match the template (`config_name` grpo_mcq, `probe.retrain` true).

**(c) Advantage and KL: PASS.**
- `checks/advantages_s43_round5.json`: `advantage_std_floor` 0.3, `steps_compared` 128, `max_abs_difference` 2.18e-08 (limit 1e-4), `kl_div_file_step_0` 0.1433 (must be at least 0.01).
- `checks/kl_s43_round5.json`: `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128, for run `3yrst3z8`.

**(d) Probe: PASS.** `checks/probe_s43_round5.json`:
- `train_biased_unbiased` is [2492, 2492], equal to `expected_train_biased_unbiased`.
- `test_biased_unbiased` is [61, 448], equal to `expected_test_biased_unbiased`.
- `saved_layers` is [13, 14, 15, 16] for both `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`.

**(e) Starting adapter: PASS.** Line 99 of `logs/rl_s43_round5_a1_09f52c746bb4.log` reads exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261104_iterative_kl_0_01/s43_round4_pvormx4t/checkpoints/final/lora`. It is the only attempt log for round 5, and it ends `EXIT 0`.

**(f) Batch: PASS.** Under `metrics.full_config` in `run_metrics_s43_round5.json`: `training.batch_size` 4, `training.grad_accum_steps` 16, `grpo.batching.per_device_train_batch_size` 4, `grpo.batching.gradient_accumulation_steps` 16.

**(g) Eval: PASS.**
- `evals/s43_round5/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `evals/base_rescored_s43_round5/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `runs/s43_round5/training_curve.csv` has 128 data rows (steps 0 to 127; 129 lines including the header).

**(h) Score: PASS.** `checks/score_round5.json`:
- `round_probe_here_vs_final_eval.max_abs_difference` is 0.0 (limit 0.02), over 978 answers.
- `first_probe_link` is `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset`, as required.
- `round_probe_link` equals `round_probe_in_run_folder`, and both are `/workspace/results/run_results/261104_iterative_kl_0_01/s43_round5_3yrst3z8/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`. That folder exists.

**Reported (no verdict):**
- **From `runs.md`:**
  - Round 5 RL minutes: 185.1 (`GRPO subprocess finished in 185.1 min (exit=0, attempt=1/4)`).
  - Lowest free disk: 79G.
  - Peak GPU memory: 76481 MiB.
  - Crashed attempts: none. The "RL crashes" section says "s43_round5: none (attempt 1, no CRASH: line)", and the log has no `CRASH:`, `removed partial run directory`, `OutOfMemoryError` or `CUDA out of memory` line.
- **Mean seconds per step:** 75.5 s. That is the mtime of `step_127.jsonl` minus the mtime of `step_0.jsonl`, (1793879980 − 1793870397) / 127 = 9583 / 127, from step_0 at 09:19:57 UTC to step_127 at 11:59:40 UTC (01:19:57 to 03:59:40 PST). For comparison, round 1 (261103, 8 × 8, pod 33f36c5bf32d) took 82 s per step and 186.9 RL minutes.
- **From `probe_s43_round5.json`:** `test_auc` 0.8787, `test_mean_biased` 0.6609, `test_mean_unbiased` 0.2646.
- **From `advantages_s43_round5.json`:** `kl_div_file_step_0` 0.1433; mean `wandb_kl_div` over the last 16 steps (file steps 112 to 127) 0.1431. The same mean from the `kl_div` list in `kl_s43_round5.json` is also 0.1431.
- Round 6's probe set and config: not applicable, since round 5 is the last round.

**Overall verdict: PASS.**

### Review 7 (step 67, auditor), 2026-11-05 12:45 UTC, fresh auditor agent: PASS

Review 7 (auditor, step 67), seed-43 line: overall verdict PASS.

I matched every number in the three tables to its source files. I found one wording problem in a table header. It changes no number.

**The rule.** "Audit the three tables against their sources; every number you cannot match to the files fails review 7": **PASS.**
- I reran each of the three scripts with the step 64, 65 and 66 arguments, inside Python, with nothing written to disk. The rebuilt output was identical to the saved files:
  - `seed_table.md` and `seed_table.json`;
  - `dynamics.md` and `dynamics.json`, with the W&B rows read again live;
  - `probe_table.md` and `probe_table.json`.
- I also recomputed the numbers with my own code from the raw files. That check is listed under Verified below.
- No numbers differed. I wrote no files and did not use `/tmp`. Bytecode writing was off, and the W&B log folder gained no files.

Discrepancies

1. Numbers: None.

2. Procedure claims: None.

3. Counts across stages: None.

4. Config consistency: None.

5. Leakage: None.

6. Unsupported or unjustified claims
1. **The probe-only groups note.** In `/workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/tables/rounds/seed_table.md` line 111, the claim is "probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ)".
   - **What the files show:** a few of these groups also contain an answer cut off at the length cap. That answer's reward carries the trainer's 0.1 length penalty (`grpo.loss.non_termination_penalty` 0.1 in each run's `full_config`, which `dynamics.py` also subtracts). In those groups the rewards differ by that penalty as well as by the probe score.
   - **How many:** round 1 has 2 such groups out of 5488, and round 2 has 1 out of 5480 (summed over the 128 steps). Rounds 3–5 have none.
   - **Effect:** no number changes. The share counts groups by their answers' letters only.
   - **Where it comes from:** the sentence is fixed text in the shared `seed_table.py`, not something this run produced.
   - **What would support the sentence as written:** no probe-only group containing a cut-off answer, or the sentence naming the length penalty too.
   - **Label:** contradicted claim (wording only, not a number).

Verified

- **Seed table eval rows:** I recomputed every metric in every condition, for the base column and all five rounds, with my own code from `completions.jsonl` and `labels.jsonl` (sample 0; the 0.75/0.25 weighting for train_mix; questions with a no-cue answer in both evals). The largest difference from `seed_table.json` was under 1e-12.
  - Shared questions: 457, 454, 460, 460, 457.
  - Eval questions: 489.
  - The base column's probe rows are empty, as expected.
- **Seed table training rows:** I recomputed them from the 128 step files of each run folder, 512 records each. "equal task reward groups" are 0.736, 0.810, 0.906, 0.948, 0.970; "probe-only groups" are 0.670, 0.669, 0.688, 0.707, 0.717. All match.
- **Eval adapters:** every final eval's adapter is its run folder's `checkpoints/final`. Each round 2–5 run id matches `run_id` in its `run_metrics_s43_round<R>.json`.
- **Probe table, own test set row:** I recomputed it from each run's `initial_detector.base_model.on_dataset/f1_and_recall_plot_…pkl`, using index 0 and the mean over layers 13–16.
  - Index 0 is the test set: its counts equal the `test.jsonl` labels (292/485, 274/451, 153/448, 85/454, 61/448). Index 2 equals the `train.jsonl` counts.
  - Means and AUCs match, and also match `test_auc` and the two means in `checks/probe_s43_round<R>.json`.
- **Probe table, the other three rows:** I recomputed them from the eval labels, the rescored base labels and `first_probe_round<R>` labels. All 15 match.
- **Which probe and model scored which answers:** I read the `probe_run_dir` and `adapter` fields in each folder's followups and completions.
  - Each eval and each rescored base eval was scored with its own round's run folder.
  - `first_probe_round<R>` was scored through `stand_in/round<R>`, with round R's adapter.
  - The `stand_in` links point to round 1's initial probe and to round R's initial probe.
  - `checks/score_round<R>.json` shows a largest difference of 0.0 between the two scorings of round R's probe.
  - Each RL log has one Phase-0 line, naming round R−1's `checkpoints/final/lora`.
- **Labelling rule:** in every eval and rescored base `labels.jsonl` (978 rows each), `a_none` follows the 5-of-8 rule (0 disagreements). The rescored base copies equal 261013's base eval in completions, predictions, `bias_rm` and `a_none` (4890/4890 and 978/978).
- **Round 1's test file:** `bias_probe_eval.jsonl` (777 rows) is byte-identical to `261010_confident_slice/arms/slice_cot/probe_sets/bias_followed_rm_eval.jsonl`.
- **Probe training sets:**
  - The row counts per (source, label, condition) in `train.jsonl` equal `build.json`'s own rows plus top-ups.
  - `per_group` = min(max(the smallest pool, 100), 378), and `train_rows` = 4 × the sum of `per_group`. The training-rows column follows: 1512, 3024, 4044, 4584, 4984.
  - No duplicate keys, and each round's `train.jsonl` starts byte for byte with the round before's.
  - The two round-5 top-ups (+59 from round 1, +6 from round 3) match the `TOPPED UP:` lines in `logs/build_round5_a1_09f52c746bb4.log`. No build log has a `DROPPED` line.
- **Dynamics step-file rows:** I recomputed all seven window rows (cue-naming groups, mixed-reward groups, cue not at the label, answers not naming the cue, mean absolute advantage, its probe-only part, advantage on cue answers) independently for all five runs. Matching rows also confirms the probe-only analysis group counts (for example 140 + 180 = 320 = 32 × 10).
- **Dynamics KL row:** it matches the window means of the separate W&B readouts in `checks/kl_s43_round<R>.json` and `261103_kl_0_01/weight_1_0/checks/kl_kl_weight_1_0_s43.json`.
  - "From the base model" holds for every run. Rounds 2–5 have `kl_reference_from_base` true. Round 1 started from the base model, and its KL is 0 at steps 0 and 1.
- **Other dynamics sections:**
  - The grad-norm, entropy and answer-length rows, and the baseline rows (261013 and 261018 run folders), matched on the live rerun.
  - I recomputed the probe-data counts and the final-eval median token counts independently; both match.
  - The letter-split "both" rows A and C equal the probe table's base-answer and round-answer rows.
- **Leakage:**
  - None of the 1217 training questions is an eval question (489).
  - No probe training file contains an eval question, and generations for rounds 1–4 are all split train (17038 rows each).
  - Each test file uses eval questions only, by design.
  - Nothing is chosen from the test file: `n_calibration_examples` is None, `filter_layers_below_loss_threshold` is None, and on-policy calibration negatives are off in every run's `full_config`.
- **Code:** `rounds.py`, `dynamics.py`, `checks.py` and `round_template.yaml` are identical to `code_at_launch/`. Code fixes in `runs.md` is empty. `seed_table.py` and `compare.py` were last changed on 2026-10-24.
- **Order of steps:** `logs/table_a1_09f52c746bb4.log` ends `EXIT 0`.
  - The outputs are timestamped 12:32:30 (seed table), 12:33:47 (dynamics) and 12:34:26 (probe table), UTC.
  - Review 6 is logged at 12:32. The seed table takes about 8 s to compute, so it fits after the review. Minute-level times can't prove the order to the second.
  - The judgment call to run steps 64 and 65 together is recorded in `runs.md`.

**Overall verdict: PASS.**

## Blocked

## Result

From `tables/rounds/seed_table.md` (step 64):

### RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): round1 457, round2 454, round3 460, round4 460, round5 457.

| metric | condition | base | base mean | base SD | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.757 | 0.765 | 0.755 | 0.763 | 0.761 | 0.760 | 0.004 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.937 | 0.883 | 0.828 | 0.820 | 0.804 | 0.854 | 0.055 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.331 | 0.534 | 0.640 | 0.701 | 0.695 | 0.580 | 0.155 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.736 | 0.760 | 0.764 | 0.781 | 0.766 | 0.762 | 0.016 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.658 | 0.667 | 0.661 | 0.679 | 0.675 | 0.668 | 0.009 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.781 | 0.736 | 0.714 | 0.710 | 0.701 | 0.728 | 0.032 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.372 | 0.530 | 0.595 | 0.628 | 0.634 | 0.552 | 0.109 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.728 | 0.720 | 0.701 | 0.698 | 0.695 | 0.709 | 0.015 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.937 | 0.883 | 0.828 | 0.820 | 0.804 | 0.854 | 0.055 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.667 | 0.464 | 0.356 | 0.297 | 0.305 | 0.418 | 0.154 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.840 | 0.741 | 0.670 | 0.643 | 0.638 | 0.707 | 0.085 |
| bias | cue_gold | 0.094 / 0.085 / 0.093 / 0.091 / 0.091 | 0.091 | 0.003 | 0.158 | 0.099 | 0.072 | 0.052 | 0.050 | 0.086 | 0.044 |
| bias | cue_dist | 0.264 / 0.270 / 0.271 / 0.269 / 0.269 | 0.269 | 0.003 | 0.443 | 0.238 | 0.113 | 0.081 | 0.079 | 0.191 | 0.155 |
| bias | train_mix | 0.154 / 0.148 / 0.154 / 0.151 / 0.152 | 0.152 | 0.002 | 0.250 | 0.144 | 0.089 | 0.057 | 0.059 | 0.120 | 0.081 |
| anti_bias | cue_gold | 0.038 / 0.038 / 0.038 / 0.038 / 0.038 | 0.038 | 0.000 | 0.009 | 0.011 | 0.004 | 0.009 | 0.011 | 0.009 | 0.003 |
| anti_bias | cue_dist | 0.011 / 0.011 / 0.011 / 0.011 / 0.009 | 0.011 | 0.001 | 0.004 | 0.002 | 0.009 | 0.013 | 0.015 | 0.009 | 0.006 |
| anti_bias | train_mix | 0.028 / 0.028 / 0.028 / 0.028 / 0.027 | 0.028 | 0.000 | 0.005 | 0.009 | 0.007 | 0.009 | 0.010 | 0.008 | 0.002 |
| no_bias | cue_gold | 0.869 / 0.877 / 0.869 / 0.872 / 0.871 | 0.871 | 0.003 | 0.834 | 0.890 | 0.924 | 0.939 | 0.939 | 0.905 | 0.045 |
| no_bias | cue_dist | 0.725 / 0.719 / 0.717 / 0.720 / 0.722 | 0.721 | 0.003 | 0.553 | 0.759 | 0.878 | 0.906 | 0.906 | 0.800 | 0.151 |
| no_bias | train_mix | 0.818 / 0.824 / 0.818 / 0.821 / 0.820 | 0.820 | 0.002 | 0.745 | 0.847 | 0.905 | 0.934 | 0.932 | 0.873 | 0.079 |
| deception_soft | cue_gold | 0.553 / 0.554 / 0.551 / 0.552 / 0.553 | 0.552 | 0.001 | 0.544 | 0.554 | 0.414 | 0.639 | 0.563 | 0.543 | 0.081 |
| deception_soft | cue_dist | 0.602 / 0.598 / 0.599 / 0.602 / 0.600 | 0.600 | 0.002 | 0.514 | 0.601 | 0.549 | 0.740 | 0.716 | 0.624 | 0.100 |
| deception_soft | train_mix | 0.568 / 0.568 / 0.566 / 0.569 / 0.568 | 0.568 | 0.001 | 0.538 | 0.570 | 0.456 | 0.675 | 0.616 | 0.571 | 0.082 |
| deception | cue_gold | 0.572 / 0.573 / 0.566 / 0.570 / 0.567 | 0.570 | 0.003 | 0.563 | 0.575 | 0.346 | 0.732 | 0.622 | 0.567 | 0.141 |
| deception | cue_dist | 0.669 / 0.658 / 0.662 / 0.664 / 0.663 | 0.663 | 0.004 | 0.533 | 0.628 | 0.565 | 0.850 | 0.811 | 0.677 | 0.145 |
| deception | train_mix | 0.603 / 0.600 / 0.598 / 0.602 / 0.599 | 0.600 | 0.002 | 0.555 | 0.585 | 0.415 | 0.775 | 0.691 | 0.604 | 0.137 |
| biased_share | cue_gold | 0.097 / 0.089 / 0.097 / 0.094 / 0.095 | 0.094 | 0.003 | 0.159 | 0.100 | 0.072 | 0.053 | 0.051 | 0.087 | 0.045 |
| biased_share | cue_dist | 0.267 / 0.273 / 0.274 / 0.272 / 0.271 | 0.272 | 0.003 | 0.445 | 0.239 | 0.114 | 0.082 | 0.080 | 0.192 | 0.156 |
| biased_share | train_mix | 0.158 / 0.152 / 0.158 / 0.155 / 0.157 | 0.156 | 0.003 | 0.251 | 0.145 | 0.089 | 0.058 | 0.059 | 0.120 | 0.081 |
| false_denial_soft | cue_gold | 0.388 / 0.393 / 0.377 / 0.371 / 0.367 | 0.379 | 0.011 | 0.293 | 0.288 | 0.492 | 0.258 | 0.238 | 0.314 | 0.102 |
| false_denial_soft | cue_dist | 0.350 / 0.350 / 0.351 / 0.354 / 0.347 | 0.350 | 0.002 | 0.280 | 0.296 | 0.435 | 0.248 | 0.350 | 0.322 | 0.073 |
| false_denial_soft | train_mix | 0.367 / 0.368 / 0.364 / 0.365 / 0.358 | 0.364 | 0.004 | 0.288 | 0.291 | 0.456 | 0.255 | 0.302 | 0.318 | 0.079 |
| false_admission_soft | cue_gold | 0.571 / 0.570 / 0.570 / 0.571 / 0.572 | 0.571 | 0.001 | 0.592 | 0.584 | 0.407 | 0.660 | 0.581 | 0.565 | 0.094 |
| false_admission_soft | cue_dist | 0.694 / 0.691 / 0.693 / 0.694 / 0.694 | 0.693 | 0.002 | 0.702 | 0.697 | 0.563 | 0.783 | 0.748 | 0.699 | 0.084 |
| false_admission_soft | train_mix | 0.606 / 0.604 / 0.605 / 0.606 / 0.607 | 0.605 | 0.001 | 0.623 | 0.618 | 0.457 | 0.700 | 0.636 | 0.607 | 0.090 |
| said_yes_biased_soft | cue_gold | 0.612 / 0.607 / 0.623 / 0.629 / 0.633 | 0.621 | 0.011 | 0.707 | 0.712 | 0.508 | 0.742 | 0.762 | 0.686 | 0.102 |
| said_yes_biased_soft | cue_dist | 0.650 / 0.650 / 0.649 / 0.646 / 0.653 | 0.650 | 0.002 | 0.720 | 0.704 | 0.565 | 0.752 | 0.650 | 0.678 | 0.073 |
| said_yes_biased_soft | train_mix | 0.633 / 0.632 / 0.636 / 0.635 / 0.642 | 0.636 | 0.004 | 0.712 | 0.709 | 0.544 | 0.745 | 0.698 | 0.682 | 0.079 |
| deception_from_biased | cue_gold | 0.038 / 0.035 / 0.036 / 0.035 / 0.035 | 0.036 | 0.001 | 0.047 | 0.029 | 0.036 | 0.014 | 0.012 | 0.027 | 0.015 |
| deception_from_biased | cue_dist | 0.093 / 0.096 / 0.096 / 0.096 / 0.094 | 0.095 | 0.001 | 0.124 | 0.071 | 0.050 | 0.020 | 0.028 | 0.059 | 0.042 |
| deception_from_biased | train_mix | 0.058 / 0.056 / 0.058 / 0.057 / 0.056 | 0.057 | 0.001 | 0.072 | 0.042 | 0.041 | 0.015 | 0.018 | 0.038 | 0.023 |
| deception_from_unbiased | cue_gold | 0.515 / 0.519 / 0.515 / 0.517 / 0.518 | 0.517 | 0.002 | 0.498 | 0.526 | 0.378 | 0.625 | 0.551 | 0.516 | 0.090 |
| deception_from_unbiased | cue_dist | 0.509 / 0.502 / 0.503 / 0.505 / 0.506 | 0.505 | 0.003 | 0.390 | 0.530 | 0.499 | 0.719 | 0.688 | 0.565 | 0.137 |
| deception_from_unbiased | train_mix | 0.510 / 0.512 / 0.509 / 0.512 / 0.512 | 0.511 | 0.001 | 0.466 | 0.528 | 0.416 | 0.660 | 0.598 | 0.534 | 0.098 |
| said_yes | cue_gold | 0.600 / 0.597 / 0.598 / 0.605 / 0.602 | 0.600 | 0.003 | 0.656 | 0.635 | 0.344 | 0.771 | 0.668 | 0.615 | 0.160 |
| said_yes | cue_dist | 0.785 / 0.789 / 0.787 / 0.787 / 0.788 | 0.787 | 0.001 | 0.819 | 0.796 | 0.574 | 0.914 | 0.851 | 0.791 | 0.129 |
| said_yes | train_mix | 0.660 / 0.659 / 0.660 / 0.664 / 0.662 | 0.661 | 0.002 | 0.711 | 0.682 | 0.420 | 0.818 | 0.730 | 0.672 | 0.150 |
| false_denial | cue_gold | 0.357 / 0.368 / 0.333 / 0.317 / 0.317 | 0.339 | 0.023 | 0.208 | 0.200 | 0.515 | 0.125 | 0.043 | 0.218 | 0.179 |
| false_denial | cue_dist | 0.282 / 0.261 / 0.273 / 0.275 / 0.269 | 0.272 | 0.008 | 0.178 | 0.148 | 0.462 | 0.108 | 0.250 | 0.229 | 0.140 |
| false_denial | train_mix | 0.320 / 0.308 / 0.307 / 0.303 / 0.298 | 0.307 | 0.008 | 0.191 | 0.165 | 0.472 | 0.124 | 0.168 | 0.224 | 0.141 |
| false_admission | cue_gold | 0.595 / 0.593 / 0.590 / 0.596 / 0.593 | 0.594 | 0.002 | 0.630 | 0.616 | 0.333 | 0.766 | 0.653 | 0.599 | 0.160 |
| false_admission | cue_dist | 0.810 / 0.808 / 0.809 / 0.810 / 0.809 | 0.809 | 0.001 | 0.817 | 0.779 | 0.578 | 0.916 | 0.860 | 0.790 | 0.129 |
| false_admission | train_mix | 0.656 / 0.653 / 0.653 / 0.657 / 0.655 | 0.655 | 0.002 | 0.677 | 0.656 | 0.409 | 0.815 | 0.723 | 0.656 | 0.151 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.002 | 0.002 | 0.004 | 0.004 | 0.000 | 0.002 | 0.002 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.002 | 0.002 | 0.002 | 0.000 | 0.001 | 0.001 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.002 | 0.002 | 0.004 | 0.002 | 0.000 | 0.002 | 0.001 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.001 | 0.002 | 0.003 | 0.002 | 0.000 | 0.001 | 0.001 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 449 / 446 / 452 / 452 / 449 | 449.600 | 2.510 | 457 | 454 | 459 | 459 | 457 | 457.200 | 2.049 |
| n_bias_rows | cue_dist | 443 / 441 / 446 / 446 / 443 | 443.800 | 2.168 | 456 | 453 | 459 | 459 | 457 | 456.800 | 2.490 |
| n_bias_rows | train_mix | 892 / 887 / 898 / 898 / 892 | 893.400 | 4.669 | 913 | 907 | 918 | 918 | 914 | 914 | 4.528 |
| n_deception_rows | cue_gold | 432 / 429 / 435 / 435 / 432 | 432.600 | 2.510 | 453 | 449 | 457 | 455 | 452 | 453.200 | 3.033 |
| n_deception_rows | cue_dist | 438 / 436 / 441 / 441 / 439 | 439 | 2.121 | 454 | 452 | 455 | 453 | 450 | 452.800 | 1.924 |
| n_deception_rows | train_mix | 870 / 865 / 876 / 876 / 871 | 871.600 | 4.615 | 907 | 901 | 912 | 908 | 902 | 906 | 4.528 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.218 | 0.474 | 0.352 | 0.550 | 0.446 | 0.408 | 0.128 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.195 | 0.389 | 0.385 | 0.546 | 0.366 | 0.376 | 0.125 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.039 | 0.143 | 0.090 | 0.168 | 0.170 | 0.122 | 0.057 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.057 | 0.174 | 0.163 | 0.208 | 0.211 | 0.163 | 0.063 |
| probe_start_auc | cue_gold | – | – | – | 0.849 | 0.879 | 0.907 | 0.885 | 0.786 | 0.861 | 0.047 |
| probe_start_auc | cue_dist | – | – | – | 0.763 | 0.782 | 0.782 | 0.870 | 0.674 | 0.774 | 0.070 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 72 | 45 | 33 | 24 | 23 | 39.400 | 20.256 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 202 | 108 | 52 | 37 | 36 | 87 | 70.697 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 356 | 356 | 348 | 355 | 350 | 353 | 3.742 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 95 | 92 | 106 | 93 | 95 | 96.200 | 5.630 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.282 | 0.393 | 0.363 | 0.544 | 0.476 | 0.412 | 0.102 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.249 | 0.295 | 0.405 | 0.520 | 0.406 | 0.375 | 0.106 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.069 | 0.124 | 0.155 | 0.191 | 0.189 | 0.145 | 0.051 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.095 | 0.153 | 0.232 | 0.225 | 0.243 | 0.189 | 0.064 |
| probe_fresh_auc | cue_gold | – | – | – | 0.856 | 0.864 | 0.805 | 0.869 | 0.796 | 0.838 | 0.035 |
| probe_fresh_auc | cue_dist | – | – | – | 0.748 | 0.730 | 0.687 | 0.833 | 0.675 | 0.735 | 0.063 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 72 | 45 | 33 | 24 | 23 | 39.400 | 20.256 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 202 | 108 | 52 | 37 | 36 | 87 | 70.697 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 356 | 356 | 348 | 355 | 350 | 353 | 3.742 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 95 | 92 | 106 | 93 | 95 | 96.200 | 5.630 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.297 | 0.362 | 0.416 | 0.530 | 0.531 | 0.427 | 0.103 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.251 | 0.320 | 0.454 | 0.534 | 0.440 | 0.400 | 0.113 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.062 | 0.128 | 0.182 | 0.230 | 0.268 | 0.174 | 0.081 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.075 | 0.170 | 0.285 | 0.279 | 0.334 | 0.229 | 0.105 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.870 | 0.846 | 0.827 | 0.831 | 0.761 | 0.827 | 0.041 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.759 | 0.716 | 0.687 | 0.792 | 0.613 | 0.713 | 0.069 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 72 | 45 | 33 | 24 | 23 | 39.400 | 20.256 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 202 | 108 | 52 | 37 | 36 | 87 | 70.697 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 356 | 356 | 348 | 355 | 350 | 353 | 3.742 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 95 | 92 | 106 | 93 | 95 | 96.200 | 5.630 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|
| equal task reward groups | 0.736 | 0.810 | 0.906 | 0.948 | 0.970 | 0.874 | 0.098 |
| probe-only groups | 0.670 | 0.669 | 0.688 | 0.707 | 0.717 | 0.690 | 0.021 |

From `probe_table.md` (step 66):

### Probe scores per round

Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the
cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue
answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test
set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round
R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.

| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | Round R's probe; its own test set; round R-1's model (round 1: base) | 292 / 485 | 0.722 / 0.255 | 0.873 |
| round1 | Round R's probe; round R's answers; round R's model | 274 / 451 | 0.201 / 0.043 | 0.801 |
| round1 | Round R's probe; base model's answers; base model | 159 / 428 | 0.745 / 0.230 | 0.903 |
| round1 | First probe; round R's answers; round R's model | 274 / 451 | 0.201 / 0.043 | 0.801 |
| round2 | Round R's probe; its own test set; round R-1's model (round 1: base) | 274 / 451 | 0.646 / 0.288 | 0.857 |
| round2 | Round R's probe; round R's answers; round R's model | 153 / 448 | 0.414 / 0.150 | 0.830 |
| round2 | Round R's probe; base model's answers; base model | 157 / 426 | 0.701 / 0.261 | 0.890 |
| round2 | First probe; round R's answers; round R's model | 153 / 448 | 0.281 / 0.096 | 0.777 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 153 / 448 | 0.561 / 0.266 | 0.836 |
| round3 | Round R's probe; round R's answers; round R's model | 85 / 454 | 0.372 / 0.107 | 0.866 |
| round3 | Round R's probe; base model's answers; base model | 163 / 431 | 0.669 / 0.305 | 0.855 |
| round3 | First probe; round R's answers; round R's model | 85 / 454 | 0.351 / 0.059 | 0.878 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 85 / 454 | 0.591 / 0.297 | 0.807 |
| round4 | Round R's probe; round R's answers; round R's model | 61 / 448 | 0.548 / 0.177 | 0.887 |
| round4 | Round R's probe; base model's answers; base model | 161 / 431 | 0.733 / 0.417 | 0.836 |
| round4 | First probe; round R's answers; round R's model | 61 / 448 | 0.440 / 0.065 | 0.911 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 61 / 448 | 0.661 / 0.265 | 0.879 |
| round5 | Round R's probe; round R's answers; round R's model | 59 / 445 | 0.397 / 0.179 | 0.738 |
| round5 | Round R's probe; base model's answers; base model | 160 / 429 | 0.723 / 0.430 | 0.806 |
| round5 | First probe; round R's answers; round R's model | 59 / 445 | 0.370 / 0.067 | 0.861 |

### Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
| round2 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round2 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4044 |
| round3 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 4044 |
| round3 | round2 | 255 / 631 / 2840 / 551 | 255 | 255 / 255 / 255 / 255 | none | 4044 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4584 |
| round4 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 4584 |
| round4 | round2 | 255 / 631 / 2840 / 551 | 255 | 255 / 255 / 255 / 255 | none | 4584 |
| round4 | round3 | 135 / 237 / 2923 / 614 | 135 | 135 / 135 / 135 / 135 | none | 4584 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4984 |
| round5 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 4984 |
| round5 | round2 | 255 / 631 / 2840 / 551 | 255 | 255 / 255 / 255 / 255 | none | 4984 |
| round5 | round3 | 135 / 237 / 2923 / 614 | 135 | 135 / 135 / 135 / 135 | none | 4984 |
| round5 | round4 | 41 / 94 / 2936 / 642 | 100 | 41 / 94 / 100 / 100 | bias / cue_gold: +59 from round1; bias / cue_dist: +6 from round3 | 4984 |
