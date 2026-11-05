# Runs

State file for the unattended run (`CLAUDE.md` in this folder). Read this first after any restart.

Done before launch (not queue steps): `rounds.py build 1`, `rounds.py config 1`, `rounds.py selftest`, and every identity check in `CLAUDE.md` that could run (all identical; trainer tests 34 passed).

## Queue

- [x] 1. Pod check
- [x] 2. Start vLLM for smoke_round1
- [x] 3. Smoke RL run smoke_round1
- [x] 4. Record check smoke_round1
- [x] 5. Stop vLLM
- [x] 6. Checks smoke_round1 (reward, advantages, probe)
- [x] 7. **Review 1** (smoke 1)
- [x] 8. Start vLLM for round1
- [x] 9. RL run round1 (128 steps)
- [x] 10. Record check round1
- [x] 11. Checks round1 (advantages, probe)
- [x] 12. Stop vLLM
- [x] 13. Final eval round1
- [x] 14. Base answers rescored with round 1's probe
- [x] 15. compare.py round1
- [x] 16. rounds.py score 1
- [x] 17. rounds.py generate 1
- [x] 18. rounds.py build 2
- [x] 19. rounds.py config 2
- [x] 20. **Review 2** (end of round 1, round 2's probe set and config)
- [x] 21. Start vLLM for smoke_round2
- [x] 22. Smoke RL run smoke_round2
- [x] 23. Record check smoke_round2
- [x] 24. Stop vLLM
- [x] 25. Checks smoke_round2 (reward, advantages, probe)
- [x] 26. **Review 3** (smoke 2)
- [x] 27. Start vLLM for round2
- [x] 28. RL run round2 (128 steps)
- [x] 29. Record check round2
- [x] 30. Checks round2 (advantages, probe)
- [x] 31. Stop vLLM
- [x] 32. Final eval round2
- [x] 33. Base answers rescored with round 2's probe
- [x] 34. compare.py round2
- [x] 35. rounds.py score 2
- [x] 36. rounds.py generate 2
- [x] 37. rounds.py build 3
- [x] 38. rounds.py config 3
- [x] 39. **Review 4** (end of round 2, round 3's probe set and config)
- [x] 40. Start vLLM for round3
- [x] 41. RL run round3 (128 steps)
- [x] 42. Record check round3
- [x] 43. Checks round3 (advantages, probe)
- [x] 44. Stop vLLM
- [x] 45. Final eval round3
- [x] 46. Base answers rescored with round 3's probe
- [x] 47. compare.py round3
- [x] 48. rounds.py score 3
- [x] 49. rounds.py generate 3
- [x] 50. rounds.py build 4
- [x] 51. rounds.py config 4
- [x] 52. **Review 5** (end of round 3, round 4's probe set and config)
- [x] 53. Start vLLM for round4
- [x] 54. RL run round4 (128 steps)
- [x] 55. Record check round4
- [x] 56. Checks round4 (advantages, probe)
- [x] 57. Stop vLLM
- [x] 58. Final eval round4
- [x] 59. Base answers rescored with round 4's probe
- [x] 60. compare.py round4
- [x] 61. rounds.py score 4
- [x] 62. rounds.py generate 4
- [x] 63. rounds.py build 5
- [x] 64. rounds.py config 5
- [x] 65. **Review 6** (end of round 4, round 5's probe set and config)
- [x] 66. Start vLLM for round5
- [x] 67. RL run round5 (128 steps)
- [x] 68. Record check round5
- [x] 69. Checks round5 (advantages, probe)
- [x] 70. Stop vLLM
- [x] 71. Final eval round5
- [x] 72. Base answers rescored with round 5's probe
- [x] 73. compare.py round5
- [x] 74. rounds.py score 5
- [x] 75. **Review 7** (end of round 5)
- [x] 76. seed_table.py (tables/rounds)
- [x] 77. dynamics.py (runs/dynamics)
- [x] 78. rounds.py table
- [x] 79. **Review 8** (auditor), then copy the tables into Result

## Pod

Step 1, 2026-11-03T02:50Z: hostname `33f36c5bf32d`; `nvidia-smi -L`: one `NVIDIA H100 80GB HBM3`; `uname -r` 6.8.0-147-generic;
`uptime -s` 2026-10-20 02:51:57; driver 595.91.07, CUDA 13.2; `df -h /workspace`: 513G size, 394G used, 119G available (77%).
This session runs in tmux session `261102-rounds` (its own Claude process, started by the user's session in `261102-iter`).

## Log

One line per step: `<UTC time> | step | outcome | next`.

2026-11-03T02:50Z | 1 Pod check | pass: 33f36c5bf32d, one H100 80GB, 119G free; made logs/ checks/ runs/ evals/ | 2 start vLLM for smoke_round1
2026-11-03T02:56Z | 2 Start vLLM for smoke_round1 | pass: started 02:50:25, answered at 02:55:59 (5.6 min); GPU 30405 MiB used; log logs/vllm_smoke_round1_a1_33f36c5bf32d.log | 3 smoke RL run smoke_round1
2026-11-03T03:13Z | 3 Smoke RL run smoke_round1 | pass: log ends `EXIT 0`; no `CRASH:` lines; `GRPO subprocess finished in 15.1 min (exit=0, attempt=1/4)`; run id x9l5z72g, run dir /workspace/results/run_results/261102_iterative_probe_retraining/smoke_round1_x9l5z72g; peak GPU memory 78273 MiB, lowest free disk 116 GB (polled every 15 s) | 4 record check smoke_round1
2026-11-03T03:14Z | 4 Record check smoke_round1 | pass: runs/smoke_round1/check_records.json written, `PASS: 2/2 files` | 5 stop vLLM
2026-11-03T03:14Z | 5 Stop vLLM | pass: session 261102-rounds-vllm killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 6 checks smoke_round1
2026-11-03T03:15Z | 6 Checks smoke_round1 | pass: exit 0 (probe check read logs/rl_smoke_round1_a1_33f36c5bf32d.log); wrote checks/reward_smoke_round1.json, advantages_smoke_round1.json, probe_smoke_round1.json | 7 review 1
2026-11-03T03:16Z | 7 Review 1 | PASS (all of (a)-(e); test AUC 0.8730, step-0 KL 0) | 8 start vLLM for round1
2026-11-03T03:20Z | 8 Start vLLM for round1 | pass: started 03:16:59, answered at 03:19:54 (2.9 min); log logs/vllm_round1_a1_33f36c5bf32d.log | 9 RL run round1
2026-11-03T06:25Z | 9 RL run round1 | pass: log ends `EXIT 0`, attempt a1. Lines: `GRPO subprocess finished in 103.6 min (exit=1, attempt=1/4)`, `CRASH: round1 attempt 1 failed with exit code 1` (out of memory at batch 73, W&B rirez2ko; see RL crashes), `GRPO subprocess finished in 79.4 min (exit=0, attempt=2/4)`; run id jmf1ks33, run dir /workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33; step wall time 03:20-06:24 (3.1 h); peak GPU memory 80939 MiB, lowest free disk 111 GB (polled every 15 s) | 10 record check round1
2026-11-03T06:25Z | 10 Record check round1 | pass: runs/round1/check_records.json written, `PASS: 128/128 files` | 11 checks round1
2026-11-03T06:26Z | 11 Checks round1 | pass: exit 0; wrote checks/advantages_round1.json, probe_round1.json (the probe check read logs/rl_round1_a1_33f36c5bf32d.log, which holds both launcher tries; it collects the printed training counts as a set, and both tries printed (756, 756)) | 12 stop vLLM
2026-11-03T06:27Z | 12 Stop vLLM | pass: session 261102-rounds-vllm killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 13 final eval round1
2026-11-03T06:31Z | 13 Final eval round1 | pass: log ends `EXIT 0` (06:27-06:31); adapter round1_jmf1ks33/checkpoints/final; evals/round1/summary.json `"questions": 489`, `"cued_rollouts": 978` | 14 base answers rescored with round 1's probe
2026-11-03T06:34Z | 14 Base answers rescored with round 1's probe | pass: copy_base wrote 4890 rows; log ends `EXIT 0` (06:31-06:33); evals/base_rescored_round1/summary.json `"questions": 489`, `"cued_rollouts": 978`; `cmp evals/base_rescored_round1/labels.jsonl` with 261025's `evals/base_rescored_floor_s42/labels.jsonl`: identical | 15 compare.py round1
2026-11-03T06:34Z | 15 compare.py round1 | pass: exit 0; runs/round1/comparison.md, comparison.json, training_curve.csv (128 data rows) | 16 rounds.py score 1
2026-11-03T06:36Z | 16 rounds.py score 1 | pass: log ends `EXIT 0` (06:34-06:36); wrote evals/first_probe_round1/, stand_in/round1/, checks/score_round1.json | 17 rounds.py generate 1
2026-11-03T06:40Z | 17 rounds.py generate 1 | pass: log ends `EXIT 0` (06:36-06:40); generations/round1/completions.jsonl has 17038 rows | 18 rounds.py build 2
2026-11-03T06:41Z | 18 rounds.py build 2 | pass: log ends `EXIT 0`; per_group: original 378, round1 378; no `TOPPED UP:` lines; train_rows 3024, test 432 biased / 432 unbiased, test_one_class false | 19 rounds.py config 2
2026-11-03T06:42Z | 19 rounds.py config 2 | pass: log ends `EXIT 0`; wrote round2.yaml and smoke_round2.yaml | 20 review 2
2026-11-03T06:45Z | 20 Review 2 | PASS (all of (a)-(i); round 1 test AUC 0.8730, step-0 KL 0, last-16-step KL mean 2.9154; round 2 probe set 3024 rows, no top-ups) | 21 start vLLM for smoke_round2
2026-11-03T06:48Z | 21 Start vLLM for smoke_round2 | pass: started 06:45:30, answered at 06:48:20 (2.8 min); log logs/vllm_smoke_round2_a1_33f36c5bf32d.log | 22 smoke RL run smoke_round2
2026-11-03T07:08Z | 22 Smoke RL run smoke_round2 | pass: log ends `EXIT 0`; no `CRASH:` lines; `GRPO subprocess finished in 18.0 min (exit=0, attempt=1/4)`; `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/checkpoints/final/lora`; run id 43a2heq5, run dir /workspace/results/run_results/261102_iterative_probe_retraining/smoke_round2_43a2heq5; peak GPU memory 75985 MiB, lowest free disk 112 GB (polled every 15 s) | 23 record check smoke_round2
2026-11-03T07:09Z | 23 Record check smoke_round2 | pass: runs/smoke_round2/check_records.json written, `PASS: 2/2 files` | 24 stop vLLM
2026-11-03T07:09Z | 24 Stop vLLM | pass: session 261102-rounds-vllm killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 25 checks smoke_round2
2026-11-03T07:11Z | 25 Checks smoke_round2 | pass: exit 0 (probe check read logs/rl_smoke_round2_a1_33f36c5bf32d.log); wrote checks/reward_smoke_round2.json, advantages_smoke_round2.json, probe_smoke_round2.json | 26 review 3
2026-11-03T07:12Z | 26 Review 3 | PASS (all of (a)-(f); step-0 KL 2.9453, smoke 2 probe test AUC 0.7357 on round 2's test set) | 27 start vLLM for round2
2026-11-03T07:15Z | 27 Start vLLM for round2 | pass: started 07:12:32, answered at 07:15:26 (2.9 min); log logs/vllm_round2_a1_33f36c5bf32d.log | 28 RL run round2
2026-11-03T08:37Z | 28 RL run round2 | pass: log ends `EXIT 0`, attempt a1; no `CRASH:` lines; `GRPO subprocess finished in 79.8 min (exit=0, attempt=1/4)`; `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/checkpoints/final/lora`; run id kgc82y7d, run dir /workspace/results/run_results/261102_iterative_probe_retraining/round2_kgc82y7d; step wall time 07:15-08:37 (1.4 h); peak GPU memory 77782 MiB, lowest free disk 107 GB (polled every 15 s) | 29 record check round2
2026-11-03T08:38Z | 29 Record check round2 | pass: runs/round2/check_records.json written, `PASS: 128/128 files` | 30 checks round2
2026-11-03T08:39Z | 30 Checks round2 | pass: exit 0 (probe check read logs/rl_round2_a1_33f36c5bf32d.log); wrote checks/advantages_round2.json, probe_round2.json | 31 stop vLLM
2026-11-03T08:39Z | 31 Stop vLLM | pass: session 261102-rounds-vllm killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 32 final eval round2
2026-11-03T08:44Z | 32 Final eval round2 | pass: log ends `EXIT 0` (08:39-08:43); adapter round2_kgc82y7d/checkpoints/final; evals/round2/summary.json `"questions": 489`, `"cued_rollouts": 978` | 33 base answers rescored with round 2's probe
2026-11-03T08:46Z | 33 Base answers rescored with round 2's probe | pass: copy_base wrote 4890 rows; log ends `EXIT 0` (08:44-08:46); evals/base_rescored_round2/summary.json `"questions": 489`, `"cued_rollouts": 978` | 34 compare.py round2
2026-11-03T08:47Z | 34 compare.py round2 | pass: exit 0; runs/round2/comparison.md, comparison.json, training_curve.csv (128 data rows) | 35 rounds.py score 2
2026-11-03T08:49Z | 35 rounds.py score 2 | pass: log ends `EXIT 0` (08:46-08:49); wrote evals/first_probe_round2/, stand_in/round2/, checks/score_round2.json | 36 rounds.py generate 2
2026-11-03T08:53Z | 36 rounds.py generate 2 | pass: log ends `EXIT 0` (08:49-08:52); generations/round2/completions.jsonl has 17038 rows | 37 rounds.py build 3
2026-11-03T08:54Z | 37 rounds.py build 3 | pass: log ends `EXIT 0`; per_group: original 378, round1 378, round2 378; no `TOPPED UP:` lines; train_rows 4536, test 425 biased / 455 unbiased, test_one_class false | 38 rounds.py config 3
2026-11-03T08:55Z | 38 rounds.py config 3 | pass: log ends `EXIT 0`; wrote round3.yaml | 39 review 4
2026-11-03T08:57Z | 39 Review 4 | PASS (all of (a)-(i); round 2 probe test AUC 0.7357, step-0 KL 2.888, last-16-step KL mean 4.0696; round 3 probe set 4536 rows, no top-ups) | 40 start vLLM for round3
2026-11-03T09:01Z | 40 Start vLLM for round3 | pass: started 08:57:20, answered at 09:00:44 (3.4 min); log logs/vllm_round3_a1_33f36c5bf32d.log | 41 RL run round3
2026-11-03T10:29Z | 41 RL run round3 | pass: log ends `EXIT 0`, attempt a1; no `CRASH:` lines; `GRPO subprocess finished in 86.0 min (exit=0, attempt=1/4)`; `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round2_kgc82y7d/checkpoints/final/lora`; run id 5xj7bsnh, run dir /workspace/results/run_results/261102_iterative_probe_retraining/round3_5xj7bsnh; step wall time 09:01-10:29 (1.5 h); peak GPU memory 78886 MiB, lowest free disk 104 GB (polled every 15 s) | 42 record check round3
2026-11-03T10:30Z | 42 Record check round3 | pass: runs/round3/check_records.json written, `PASS: 128/128 files` | 43 checks round3
2026-11-03T10:31Z | 43 Checks round3 | pass: exit 0 (probe check read logs/rl_round3_a1_33f36c5bf32d.log); wrote checks/advantages_round3.json, probe_round3.json | 44 stop vLLM
2026-11-03T10:31Z | 44 Stop vLLM | pass: session 261102-rounds-vllm killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 45 final eval round3
2026-11-03T10:36Z | 45 Final eval round3 | pass: log ends `EXIT 0` (10:31-10:35); adapter round3_5xj7bsnh/checkpoints/final; evals/round3/summary.json `"questions": 489`, `"cued_rollouts": 978` | 46 base answers rescored with round 3's probe
2026-11-03T10:38Z | 46 Base answers rescored with round 3's probe | pass: copy_base wrote 4890 rows; log ends `EXIT 0` (10:36-10:38); evals/base_rescored_round3/summary.json `"questions": 489`, `"cued_rollouts": 978` | 47 compare.py round3
2026-11-03T10:38Z | 47 compare.py round3 | pass: exit 0; runs/round3/comparison.md, comparison.json, training_curve.csv (128 data rows) | 48 rounds.py score 3
2026-11-03T10:41Z | 48 rounds.py score 3 | pass: log ends `EXIT 0` (10:38-10:40); wrote evals/first_probe_round3/, stand_in/round3/, checks/score_round3.json | 49 rounds.py generate 3
2026-11-03T10:45Z | 49 rounds.py generate 3 | pass: log ends `EXIT 0` (10:41-10:44); generations/round3/completions.jsonl has 17038 rows | 50 rounds.py build 4
2026-11-03T10:45Z | 50 rounds.py build 4 | pass: log ends `EXIT 0`; per_group: original 378, round1 378, round2 378, round3 378; no `TOPPED UP:` lines; train_rows 6048, test 454 biased / 456 unbiased, test_one_class false | 51 rounds.py config 4
2026-11-03T10:46Z | 51 rounds.py config 4 | pass: log ends `EXIT 0`; wrote round4.yaml | 52 review 5
2026-11-03T10:49Z | 52 Review 5 | PASS (all of (a)-(i); round 3 probe test AUC 0.9184, step-0 KL 3.9855, last-16-step KL mean 4.0439; round 4 probe set 6048 rows, no top-ups) | 53 start vLLM for round4
2026-11-03T10:52Z | 53 Start vLLM for round4 | pass: started 10:49:04, answered at 10:51:48 (2.7 min); log logs/vllm_round4_a1_33f36c5bf32d.log | 54 RL run round4
2026-11-03T12:36Z | 54 RL run round4 | pass: log ends `EXIT 0`, attempt a1; no `CRASH:` lines; `GRPO subprocess finished in 101.7 min (exit=0, attempt=1/4)`; `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round3_5xj7bsnh/checkpoints/final/lora`; run id zzpngiif, run dir /workspace/results/run_results/261102_iterative_probe_retraining/round4_zzpngiif; step wall time 10:52-12:35 (1.7 h); peak GPU memory 76944 MiB, lowest free disk 102 GB (polled every 15 s) | 55 record check round4
2026-11-03T12:36Z | 55 Record check round4 | pass: runs/round4/check_records.json written, `PASS: 128/128 files` | 56 checks round4
2026-11-03T12:37Z | 56 Checks round4 | pass: exit 0 (probe check read logs/rl_round4_a1_33f36c5bf32d.log); wrote checks/advantages_round4.json, probe_round4.json | 57 stop vLLM
2026-11-03T12:38Z | 57 Stop vLLM | pass: session 261102-rounds-vllm killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 58 final eval round4
2026-11-03T12:42Z | 58 Final eval round4 | pass: log ends `EXIT 0` (12:38-12:42); adapter round4_zzpngiif/checkpoints/final; evals/round4/summary.json `"questions": 489`, `"cued_rollouts": 978` | 59 base answers rescored with round 4's probe
2026-11-03T12:45Z | 59 Base answers rescored with round 4's probe | pass: copy_base wrote 4890 rows; log ends `EXIT 0` (12:42-12:44); evals/base_rescored_round4/summary.json `"questions": 489`, `"cued_rollouts": 978` | 60 compare.py round4
2026-11-03T12:45Z | 60 compare.py round4 | pass: exit 0; runs/round4/comparison.md, comparison.json, training_curve.csv (128 data rows) | 61 rounds.py score 4
2026-11-03T12:47Z | 61 rounds.py score 4 | pass: log ends `EXIT 0` (12:45-12:47); wrote evals/first_probe_round4/, stand_in/round4/, checks/score_round4.json | 62 rounds.py generate 4
2026-11-03T12:51Z | 62 rounds.py generate 4 | pass: log ends `EXIT 0` (12:47-12:51); generations/round4/completions.jsonl has 17038 rows | 63 rounds.py build 5
2026-11-03T12:52Z | 63 rounds.py build 5 | pass: log ends `EXIT 0`; per_group: original 378, round1 378, round2 378, round3 378, round4 312 (round4's smallest pool, bias / cue_gold, is 312); no `TOPPED UP:` lines; train_rows 7296, test 164 biased / 450 unbiased, test_one_class false | 64 rounds.py config 5
2026-11-03T12:53Z | 64 rounds.py config 5 | pass: log ends `EXIT 0`; wrote round5.yaml | 65 review 6
2026-11-03T12:56Z | 65 Review 6 | PASS (all of (a)-(i); round 4 probe test AUC 0.7906, step-0 KL 4.0449, last-16-step KL mean 1.6594; round 5 probe set 7296 rows, round4 per_group 312, no top-ups) | 66 start vLLM for round5
2026-11-03T13:00Z | 66 Start vLLM for round5 | pass: started 12:56:15, answered at 12:59:44 (3.5 min); log logs/vllm_round5_a1_33f36c5bf32d.log | 67 RL run round5
2026-11-03T15:01Z | 67 RL run round5 | pass: log ends `EXIT 0`, attempt a1; no `CRASH:` lines; `GRPO subprocess finished in 119.0 min (exit=0, attempt=1/4)`; `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round4_zzpngiif/checkpoints/final/lora`; run id f02rok2w, run dir /workspace/results/run_results/261102_iterative_probe_retraining/round5_f02rok2w; step wall time 13:00-15:00 (2.0 h); peak GPU memory 76818 MiB, lowest free disk 100 GB (polled every 15 s) | 68 record check round5
2026-11-03T15:01Z | 68 Record check round5 | pass: runs/round5/check_records.json written, `PASS: 128/128 files` | 69 checks round5
2026-11-03T15:02Z | 69 Checks round5 | pass: exit 0 (probe check read logs/rl_round5_a1_33f36c5bf32d.log); wrote checks/advantages_round5.json, probe_round5.json | 70 stop vLLM
2026-11-03T15:03Z | 70 Stop vLLM | pass: session 261102-rounds-vllm killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 71 final eval round5
2026-11-03T15:08Z | 71 Final eval round5 | pass: log ends `EXIT 0` (15:03-15:07); adapter round5_f02rok2w/checkpoints/final; evals/round5/summary.json `"questions": 489`, `"cued_rollouts": 978` | 72 base answers rescored with round 5's probe
2026-11-03T15:10Z | 72 Base answers rescored with round 5's probe | pass: copy_base wrote 4890 rows; log ends `EXIT 0` (15:08-15:10); evals/base_rescored_round5/summary.json `"questions": 489`, `"cued_rollouts": 978` | 73 compare.py round5
2026-11-03T15:11Z | 73 compare.py round5 | pass: exit 0; runs/round5/comparison.md, comparison.json, training_curve.csv (128 data rows) | 74 rounds.py score 5
2026-11-03T15:13Z | 74 rounds.py score 5 | pass: log ends `EXIT 0` (15:11-15:13); wrote evals/first_probe_round5/, stand_in/round5/, checks/score_round5.json | 75 review 7
2026-11-03T15:15Z | 75 Review 7 | PASS (all of (a)-(g); round 5 probe test AUC 0.8247, step-0 KL 1.5961, last-16-step KL mean 1.4785) | 76 seed_table.py
2026-11-03T15:16Z | 76 seed_table.py | pass: exit 0; wrote tables/rounds/seed_table.md, seed_table.json | 77 dynamics.py
2026-11-03T15:17Z | 77 dynamics.py | pass: exit 0; wrote runs/dynamics/dynamics.json, dynamics.md | 78 rounds.py table
2026-11-03T15:17Z | 78 rounds.py table | pass: log ends `EXIT 0`; wrote probe_table.md, probe_table.json | 79 review 8
2026-11-03T15:29Z | 79 Review 8 | PASS (auditor matched every number in seed_table.md, probe_table.md and dynamics.md to its files; two wording notes and one gap, not failures); tables copied into Result | done

## RL crashes

Every crashed RL attempt, the launcher's own retries included (Infrastructure rules).

- **round1, step 9, attempt a1, launcher try 1/4** (log `logs/rl_round1_a1_33f36c5bf32d.log`):
  - `GRPO subprocess finished in 103.6 min (exit=1, attempt=1/4)`
  - `CRASH: round1 attempt 1 failed with exit code 1`
  - W&B run id `rirez2ko` (`removed partial run directory for crashed run rirez2ko: /workspace/results/run_results/261102_iterative_probe_retraining/round1_rirez2ko`)
  - Reached training batch 73 of 128 (72 steps logged), in the training update, not Phase 0. Last error line:
    `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.46 GiB. GPU 0 has a total capacity of 79.18 GiB of which 6.15 GiB is free. Process 64139 has 29.55 GiB memory in use. Including non-PyTorch memory, this process has 43.46 GiB memory in use.`
    (raised in `F.log_softmax` inside `compute_logprobs_from_logits`, from `train_on_generation_batch`'s `policy_adapter.logprobs`).
  - Mean answer length over the 72 logged steps: 215.7 tokens (W&B `response_length_mean`).
  - The launcher restarted the run from scratch as try 2/4 (W&B run `jmf1ks33`) inside the same attempt. The out-of-memory
    fallback (batch 4 x 16) is for after the launcher's retries, so nothing is changed yet.
  - Measured at 05:59Z, no verdict: try 2 (`jmf1ks33`) starts like try 1 in the log (base model, the same eight probe
    dataset hashes, 128 steps, `Found 53278343168 / 85520809984 = 62.3% memory available`), and runs at about 29 s per
    step against try 1's 83-104 s. W&B `response_length_mean` (mean tokens per answer): try 1 `rirez2ko` steps 1-10 237.1,
    steps 41-60 190.6, last 10 of 72 235.7; try 2 `jmf1ks33` steps 1-10 163.9, steps 41-60 6.2, steps 81-90 6.4.

## Judgment calls

- 2026-11-03T06:42Z, review 2's prompt (and the same for reviews 4-7). Filling the round row's `<…>` for one round:
  `<R>`, `<R+1>`, `<N>` and the run folders become their values. Two spots needed more than a substitution. Rule (h)'s
  source list `original`, `round1`, …, `round<R>` is written out as the actual list. Rule (i), "meets (b) for round
  `<R+1>`", gets round `<R+1>`'s probe files and `max_train_examples` source spelled out in a parenthesis, because (b)
  as filled in names round `<R>`'s files. For round 1, the branches that name `<run dir R−1>` (round 0) say "no earlier
  round" / "not applicable to round 1". The rule text is otherwise unchanged.

- 2026-11-03T15:12Z, review 7's prompt. Review 7's row says "Rules (a)–(g) of the row above with `<R>` = 5, and their
  reported items except those about round 6". The row directly above it in the table is review 3's (smoke 2), which has
  rules (a)–(f) only and no round items, so "the row above" is read as the round row (reviews 2, 4, 5, 6), the only row
  with rules (a)–(g) and round `<R+1>` items. The prompt holds that row's rules (a)–(g) with round 5's values and its
  reported items without round 6's `build.json` item and without step 14's `cmp` (round 1 only).

- 2026-11-03T15:18Z, review 8's prompt. Its row says "per round `<R>`" and "`<run dir R>`'s step records", and "Use
  the commands of steps 76–78". The prompt keeps the `<R>` path patterns, defines `<R>` = 1 … 5 and lists each round's
  run folder, and adds the three commands of steps 76–78 as run (full paths, the run folders filled in), with a note to
  point a rerun's output at `/tmp`. The rule text is otherwise unchanged.

## Code fixes

## Reviews

### Review 1 (smoke 1, step 7): PASS (fresh general-purpose agent, 2026-11-03T03:16Z)

Review 1 (smoke 1, step 7): overall verdict PASS. I read every file myself and changed nothing.

**(a) Step files: PASS.** `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/runs/smoke_round1/check_records.json` has `"PASS": true`, `"n_files": 2` and `"seed": 42`. Both per-step entries (file steps 0 and 1) also have `"PASS": true`.

**(b) Config: PASS.** I parsed `smoke_round1.yaml` and `round_template.yaml` with `yaml.safe_load` and checked both the values and their types.
- `checkpoint` is null (None).
- `loss.detector` 0.6, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 2, `training.eval_steps` 1, `++grpo.model.kl_reference_from_base` true.
- `data.probe_dataset.bias_probe_train_jsonl` is `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/probe_sets/round1/train.jsonl`.
- `data.probe_dataset.bias_probe_eval_jsonl` is `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/probe_sets/round1/test.jsonl`.
- `data.probe_dataset.max_train_examples` is 3024.
- The other 9 overrides all equal the template's: `+data/probe_dataset`, `data.task_dataset.mmlu_questions_jsonl`, `detector.filter_layers_below_loss_threshold` ('null'), `detector.model_batch_size` 8, `detector.pretrain_cfg.learning_rate` 0.03, `detector.pretrain_cfg.max_steps` 2000, `model.activation_matching_layers` range_13-17, `task_args.bias_probe_at_answer` true, `task_args.cue_gold_prob` 0.75.
- The only key in the smoke config that the template lacks is `training.eval_steps`, which the rule names. The template has no key the smoke config lacks.
- `cmp` returned exit 0 for `probe_sets/round1/train.jsonl` against `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl`, and exit 0 for `test.jsonl` against `bias_probe_eval.jsonl`. Both pairs are identical.

**(c) Reward: PASS.** `checks/reward_smoke_round1.json` has `penalty_weight` 0.6.
- File step 0: `probe_score_present_iff_parsed_mismatches` is 0 and `reward_mean_abs_difference` is 2.77e-08, within the 0.001 limit.
- File step 1: mismatches 0 and `reward_mean_abs_difference` 1.06e-08.

**(d) Advantage and KL: PASS.** `checks/advantages_smoke_round1.json` has `advantage_std_floor` 0.3, `steps_compared` 2, `max_abs_difference` 9.19e-09 (within the 1e-4 limit) and `kl_div_file_step_0` 0 (below 0.01).

**(e) First probe: PASS.** `checks/probe_smoke_round1.json`:
- `train_biased_unbiased` is [756, 756], equal to `expected_train_biased_unbiased` [756, 756].
- `test_biased_unbiased` is [292, 485], equal to `expected_test_biased_unbiased` [292, 485].
- `saved_layers` is [13, 14, 15, 16] for both `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`.

**Reported (no verdict):**
- `penalty_mean_abs_difference` (the gap between the penalty recomputed from the step files and the one W&B logged): 2.08e-09 at file step 0 and 4.33e-09 at file step 1.
- From `probe_smoke_round1.json`:
  - `test_auc` is 0.8730052252506708. 261025's, 261027's and 261031's runs all gave 0.8730, so this matches to four decimals.
  - `test_mean_biased` is 0.7221844251680647 and `test_mean_unbiased` is 0.25478438520600644.
  - The file also lists `offline_auc_261010` as 0.871.
- From `runs.md`, step 3's log line:
  - The run took 15.1 min (`GRPO subprocess finished in 15.1 min (exit=0, attempt=1/4)`).
  - Peak GPU memory was 78273 MiB, and the lowest free disk was 116 GB.
  - Run id `x9l5z72g`, run folder `/workspace/results/run_results/261102_iterative_probe_retraining/smoke_round1_x9l5z72g`.
- Crashed attempts: none. `runs.md` says "no `CRASH:` lines", and grepping `logs/rl_smoke_round1_a1_33f36c5bf32d.log` finds no `CRASH:` or `removed partial run directory` line. The log ends `EXIT 0`, and that log is the only RL attempt in `logs/`.

**Overall verdict: PASS.** Every rule from (a) to (e) holds.

### Review 2 (end of round 1 and round 2's probe set and config, step 20): PASS (fresh general-purpose agent, 2026-11-03T06:45Z)

Review 2 (end of round 1 at step 20, plus round 2's probe set and config): overall verdict PASS. I read every file myself and changed nothing.

**(a) Step files: PASS.** `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/runs/round1/check_records.json` has `"PASS": true`, `"n_files": 128` and `"seed": 42`. Its `wandb_run_id` is `jmf1ks33`, the run that finished.

**(b) Config: PASS.** I parsed `round1.yaml` and `round_template.yaml` with yaml.safe_load.
- `checkpoint` is null (None), as required for round 1.
- `loss.detector` 0.6, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 128, `++grpo.model.kl_reference_from_base` true (a bool).
- `bias_probe_train_jsonl` is `.../probe_sets/round1/train.jsonl`.
- `bias_probe_eval_jsonl` is `.../probe_sets/round1/test.jsonl`. `probe_sets/round1/build.json` has `test_one_class` false, so test.jsonl is the right file.
- `max_train_examples` is 3024, which is 2 × round 1's `train_rows` of 1512.
- The override keys match the template's (same set, same order). Every override the template does not leave as FILL_IN has the template's value. `output_dir`, `group` and `config_name` also match the template.

**(c) Advantage and KL: PASS.** `checks/advantages_round1.json` has:
- `advantage_std_floor` 0.3;
- `steps_compared` 128 (`per_step` covers file steps 0 to 127);
- `max_abs_difference` 2.26e-08, within the 1e-4 limit;
- `kl_div_file_step_0` 0, below the 0.01 limit for round 1.

**(d) Probe: PASS.** In `checks/probe_round1.json`:
- `train_biased_unbiased` is [756, 756], equal to the expected [756, 756].
- `test_biased_unbiased` is [292, 485], equal to the expected [292, 485].
- `saved_layers` is [13, 14, 15, 16] for both `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`.

**(e) Starting adapter: PASS.** `logs/rl_round1_a1_33f36c5bf32d.log` is the only RL attempt for round 1, and it ends `EXIT 0`. `grep -c "Phase-0: loading fresh model with adapter"` gives 0. The log holds both launcher tries, and neither has the line.

**(f) Eval: PASS.**
- `evals/round1/summary.json` has `"questions": 489` and `"cued_rollouts": 978`.
- `evals/base_rescored_round1/summary.json` has `"questions": 489` and `"cued_rollouts": 978`.
- `runs/round1/training_curve.csv` has 129 lines: one header and 128 data rows (steps 0 to 127), with no empty rows.

**(g) Score: PASS.** In `checks/score_round1.json`:
- `round_probe_here_vs_final_eval.max_abs_difference` is 0.0 over 978 answers, within the 0.02 limit.
- `first_probe_link` is `/workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/initial_detector.base_model.on_dataset`, as required.
- `round_probe_link` and `round_probe_in_run_folder` are the same string, `/workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`, as required.

**(h) Round 2's probe set: PASS.** `probe_sets/round2/build.json`:
- The keys of `sources` are exactly [`original`, `round1`], in that order.
- `original` has `per_group` 378, `own_rows` 378 in all four groups, and every `topped_up_from` is empty.
- Per-group size rule, per_group = min(max(smallest pool, 100), 378):
  - `original`: the smallest pool is 378, so min(max(378, 100), 378) = 378, matching `per_group`.
  - `round1`: the smallest pool is 972, so the rule gives 378, matching `per_group`.
- In every group of both sources, `own_rows` (378) = min(pool, 378), and `own_rows` + top-ups = 378. There are no top-ups, so the "only earlier sources" rule holds trivially.
- `train_rows` is 3024 = 4 × (378 + 378), and `train_biased` is 1512 = 3024 / 2.

`probe_sets/round2/train.jsonl`:
- It has 3024 rows. The counts per (source, label, condition) equal what `build.json` implies: 378 in each of the 8 cells of original/round1 × label 1/0 × cue_gold/cue_dist.
- Each row's label agrees with its `bias_rm`.
- No (source, question_id, condition, sample_idx) appears twice.
- Every question_id is in `questions.jsonl`, and all 3024 rows are train-split. None is eval-split.
- The 1512 `original` rows (rows 0 to 1511, one block) equal `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl` in order, on question_id, condition, sample_idx, prompt, completion and label.
- The byte-prefix rule applies only from round 2 on, so it does not apply here.

`probe_sets/round2/test.jsonl`:
- It has 864 rows, 432 with label 1 and 432 with label 0, matching `test_biased` 432 and `test_unbiased` 432.
- `test_answers` names `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/evals/round1 (round 1's final-eval answers)`.
- `test_one_class` is false, and neither count is 0, so the one-class rule holds.
- Not required by the rule: all 864 rows are eval-split, and all are found in `evals/round1/completions.jsonl`.

**(i) Round 2's config: PASS.**
- `round2.yaml`:
  - `checkpoint` is `/workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/checkpoints/final`. `run_metrics_round1.json` has `run_id` jmf1ks33, and the folder and its `lora/` exist.
  - `loss.detector` 0.6, floor 0.3, seed 42, `max_steps` 128, KL setting true.
  - `bias_probe_train_jsonl` is `.../probe_sets/round2/train.jsonl`. `bias_probe_eval_jsonl` is `.../probe_sets/round2/test.jsonl`, the right file since `test_one_class` is false.
  - `max_train_examples` is 6048 = 2 × 3024.
  - Every other override, and the other top-level fields, equal the template's.
- `smoke_round2.yaml` differs from `round2.yaml` only in:
  - `run_key` `smoke_round2`;
  - `probe.directory_save_prefix` `smoke_round2` (`retrain` is unchanged);
  - `training.max_steps` 2;
  - an added `training.eval_steps` 1.

**Reported (no verdict):**
- **From `runs.md`, step 9:**
  - Round time: launcher try 1 ran 103.6 min and crashed, and try 2 ran 79.4 min and finished (`GRPO subprocess finished in 79.4 min (exit=0, attempt=2/4)`). The step's wall time was 03:20 to 06:24 UTC (3.1 h).
  - Lowest free disk: 111 GB. Peak GPU memory: 80939 MiB.
- **Crashed attempts:** one, inside attempt a1 (launcher try 1/4).
  - W&B id `rirez2ko`.
  - It reached training batch 73 of 128, with 72 steps logged, in the training update, not Phase 0.
  - Error: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.46 GiB ...` in `F.log_softmax`. I confirmed this at log lines 2057 to 2067.
  - Mean answer length over the 72 logged steps: 215.7 tokens.
  - The launcher then restarted from scratch as try 2/4 (`jmf1ks33`), which finished.
- **From `probe_round1.json`:** `test_auc` 0.8730052252506708, `test_mean_biased` 0.7221844251680647, `test_mean_unbiased` 0.25478438520600644. The file also lists `offline_auc_261010` 0.871.
- **From `advantages_round1.json`:** `kl_div_file_step_0` is 0. The mean `wandb_kl_div` over the last 16 steps (file steps 112 to 127) is 2.9154.
- **From round 2's `build.json`:**
  - `original`: pools are bias/cue_gold 378, bias/cue_dist 1037, no bias/cue_gold 2739 and no bias/cue_dist 642, with `per_group` 378.
  - `round1`: pools are bias/cue_gold 972, bias/cue_dist 1839, no bias/cue_gold 1839 and no bias/cue_dist 972, with `per_group` 378.
  - There are no top-ups, and `test_one_class` is false.
  - The original pool differs from round 1's `build.json` (378 in each group) because the code builds them differently. In round 2 `rounds.py` draws the original set from 261010's base-model pool, while round 1's build recorded the original file's counts. This is consistent with `rounds.py` lines 319 to 330, and no rule covers it.
- **Step 14's `cmp`:** `runs.md` says "identical". I reran `cmp evals/base_rescored_round1/labels.jsonl /workspace/sycophancy_experiments/261025_probe_penalty_sd_floor/evals/base_rescored_floor_s42/labels.jsonl` and it also printed nothing (identical).

**Overall verdict: PASS.** Every rule from (a) to (i) holds.

### Review 3 (smoke 2, step 26): PASS (fresh general-purpose agent, 2026-11-03T07:12Z)

Review 3 (smoke 2, step 26): overall verdict PASS. Every rule passes. I read every file myself and changed nothing.

**(a) Step files: PASS.** `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/runs/smoke_round2/check_records.json` has `"PASS": true`, `"n_files": 2` and `"seed": 42`. Both per-step entries also show `"PASS": true`.

**(b) Reward: PASS.** In `checks/reward_smoke_round2.json`:
- `penalty_weight` is 0.6.
- `probe_score_present_iff_parsed_mismatches` is 0 at file step 0 and 0 at file step 1.
- `reward_mean_abs_difference` is 3.11e-9 at step 0 and 2.40e-9 at step 1, both at or below 0.001.

**(c) Advantage and KL: PASS.** In `checks/advantages_smoke_round2.json`:
- `advantage_std_floor` is 0.3.
- `steps_compared` is 2.
- `max_abs_difference` is 5.31e-9, at or below 1e-4.
- `kl_div_file_step_0` is 2.9453, at or above 0.01.

**(d) Starting adapter: PASS.**
- Run dir 1 is `/workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33`; `run_metrics_round1.json` has `run_id` jmf1ks33.
- `logs/rl_smoke_round2_a1_33f36c5bf32d.log` line 97 reads exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/checkpoints/final/lora`.
- a1 is the only smoke_round2 RL log, and it ends `EXIT 0`.

**(e) Probe: PASS.** In `checks/probe_smoke_round2.json`:
- `train_biased_unbiased` is [1512, 1512] and `expected_train_biased_unbiased` is [1512, 1512].
- `test_biased_unbiased` is [432, 432] and `expected_test_biased_unbiased` is [432, 432].
- `saved_layers` is [13, 14, 15, 16] in both entries: `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`.
- The expected counts match the label counts I made from the files:
  - `probe_sets/round2/train.jsonl`: 3024 rows, 1512 with label 1 and 1512 with label 0.
  - `smoke_round2.yaml` names `probe_sets/round2/test.jsonl` as its eval file. It has 864 rows, 432 with label 1 and 432 with label 0.

**(f) Memory: PASS.** Searching `logs/rl_smoke_round2_a1_33f36c5bf32d.log` for `OutOfMemoryError|CUDA out of memory` finds 0 lines.

**Reported (no verdict):**
- **KL to the base model per step, from `advantages_smoke_round2.json` (`wandb_kl_div`):**
  - file step 0: 2.945336990058422
  - file step 1: 3.04017423838377
- **Round 1 KL, from `checks/advantages_round1.json` (128 steps, for comparison):**
  - last step (file step 127): 3.058913379907608
  - mean of the last 16 steps (file steps 112–127): 2.915423830738291
- **Smoke 2 probe test scores, from `probe_smoke_round2.json`:**
  - `test_auc`: 0.7356663665980796
  - `test_mean_biased`: 0.5425752643840732
  - `test_mean_unbiased`: 0.37163845397299156
  - These are on round 2's test set (round 1's final-eval answers, 432 biased and 432 unbiased). That set differs from the one behind review 1's 0.8730, which was 292 biased and 485 unbiased base-model answers.
- **From `runs.md` (step 22 log line):**
  - minutes: `GRPO subprocess finished in 18.0 min (exit=0, attempt=1/4)`, which is also the only such line in the log (line 2926).
  - peak GPU memory: 75985 MiB.
  - crashed attempts: none. `runs.md` says "no `CRASH:` lines", and the log has no `CRASH:` or `removed partial run directory` line. The "RL crashes" section of `runs.md` lists only round 1's crashed launcher try (W&B `rirez2ko`), nothing for smoke 2.

**OVERALL: PASS.**

### Review 4 (end of round 2 and round 3's probe set and config, step 39): PASS (fresh general-purpose agent, 2026-11-03T08:57Z)

Review 4 (end of round 2 at step 39, plus round 3's probe set and config). Overall verdict: **PASS**. I read every file myself and changed nothing.

**(a) Step files: PASS.** `runs/round2/check_records.json` has `"PASS": true`, `"n_files": 128` and `"seed": 42`. The `per_step` list also has 128 entries, and the W&B id is kgc82y7d.

**(b) Round 2 config: PASS.** In `round2.yaml`:
- `checkpoint` is `/workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/checkpoints/final`. That matches `run_metrics_round1.json`'s `run_id`, jmf1ks33.
- Under `overrides`:
  - `loss.detector` 0.6, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 128.
  - `++grpo.model.kl_reference_from_base` true.
  - The probe training file is `.../probe_sets/round2/train.jsonl` and the probe eval file is `.../probe_sets/round2/test.jsonl`. The test file is right because `probe_sets/round2/build.json` has `test_one_class` false.
  - `max_train_examples` is 6048, which is 2 × `train_rows` 3024.
- The other 9 overrides are identical in value and type to `round_template.yaml`'s: `+data/probe_dataset`, `task_args.bias_probe_at_answer`, `model.activation_matching_layers`, `detector.filter_layers_below_loss_threshold` ('null'), `detector.pretrain_cfg.learning_rate`, `detector.pretrain_cfg.max_steps`, `data.task_dataset.mmlu_questions_jsonl`, `task_args.cue_gold_prob` and `detector.model_batch_size`. The override keys are the same set as the template's, in the same order.

**(c) Advantage and KL: PASS.** `checks/advantages_round2.json` has `advantage_std_floor` 0.3, `steps_compared` 128 and `max_abs_difference` 3.45e-09, which is ≤ 1e-4. Round 2 is not round 1, so the KL must be at least 0.01: `kl_div_file_step_0` is 2.888.

**(d) Probe: PASS.** `checks/probe_round2.json`:
- `train_biased_unbiased` is [1512, 1512], equal to its expected value.
- `test_biased_unbiased` is [432, 432], equal to its expected value.
- `saved_layers` is [13, 14, 15, 16] for both `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`.

**(e) Starting adapter: PASS.** Only one attempt exists, `logs/rl_round2_a1_33f36c5bf32d.log`, and it ends `EXIT 0`. Its line 97 is exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/checkpoints/final/lora`, with nothing after it on the line.

**(f) Eval: PASS.**
- `evals/round2/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `evals/base_rescored_round2/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `runs/round2/training_curve.csv` has 129 lines: the header plus 128 data rows (steps 0–127), none empty.

**(g) Score: PASS.** `checks/score_round2.json`:
- `round_probe_here_vs_final_eval.max_abs_difference` is 0.0 over 978 answers, which is ≤ 0.02.
- `first_probe_link` is `/workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/initial_detector.base_model.on_dataset`.
- `round_probe_link` and `round_probe_in_run_folder` are identical, and both are `/workspace/results/run_results/261102_iterative_probe_retraining/round2_kgc82y7d/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`.

**(h) Round 3's probe set: PASS.**

`probe_sets/round3/build.json`:
- The keys of `sources` are exactly [`original`, `round1`, `round2`], in that order.
- `original` has `per_group` 378, `own_rows` 378 in all four groups, and empty `topped_up_from` everywhere.
- Per-group size, min(max(smallest pool, 100), 378):
  - `original`: smallest pool 378, so 378, matching.
  - `round1`: smallest pool 972, so 378, matching.
  - `round2`: smallest pool 951, so 378, matching.
- In every group of every source, `own_rows` (378) = min(pool, 378), and `own_rows` + top-ups = 378. There are no top-ups at all, so the rule that top-ups name only earlier sources holds trivially.
- `train_rows` is 4536 = 4 × (378 × 3), and `train_biased` is 2268 = 4536 / 2.

`probe_sets/round3/train.jsonl`:
- It has 4536 rows. The counts per (`source`, `label`, `condition`) are 378 in each of the 12 cells (3 sources × label 0/1 × cue_gold/cue_dist), matching `build.json`.
- No (`source`, `question_id`, `condition`, `sample_idx`) appears twice.
- No row is on a question with `split` `eval`: `questions.jsonl` has 489 eval questions, and none of them appear. Every `question_id` exists in `questions.jsonl`.
- The 1512 `original` rows equal `sycophancy_setup/data/bias_probe_train.jsonl` (1512 rows), in order, on `question_id`, `condition`, `sample_idx`, `prompt`, `completion` and `label`.
- Byte prefix: `probe_sets/round2/train.jsonl` (5,035,252 bytes) is exactly the start of `probe_sets/round3/train.jsonl` (6,676,555 bytes).

`probe_sets/round3/test.jsonl`:
- It has 880 rows: 425 with label 1, matching `test_biased` 425, and 455 with label 0, matching `test_unbiased` 455. No other labels.
- `test_answers` is `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/evals/round2 (round 2's final-eval answers)`, so it names `evals/round2`.
- `test_one_class` is false, which is correct because neither count is 0.

**(i) Round 3's config: PASS.** In `round3.yaml`:
- `checkpoint` is `/workspace/results/run_results/261102_iterative_probe_retraining/round2_kgc82y7d/checkpoints/final`, which matches `run_metrics_round2.json`'s `run_id`.
- `loss.detector` 0.6, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 128, `++grpo.model.kl_reference_from_base` true.
- The probe training file is `.../probe_sets/round3/train.jsonl` and the probe eval file is `.../probe_sets/round3/test.jsonl` (`test_one_class` is false).
- `max_train_examples` is 9072 = 2 × 4536.
- The other 9 overrides are identical to the template's, with the same key set and order.

The `smoke_round2.yaml` part of (i) and step 14's `cmp` apply only to round 1, so they do not apply here.

**Reported (no verdict):**
- **From `runs.md`, round 2's RL step (step 28):**
  - Minutes: `GRPO subprocess finished in 79.8 min (exit=0, attempt=1/4)`. The step's wall time was 07:15–08:37 UTC (1.4 h).
  - Lowest free disk: 107 GB.
  - Peak GPU memory: 77782 MiB.
  - Crashed attempts: none. `runs.md` says "no `CRASH:` lines", and the log has 0 `CRASH:` or `removed partial run directory` lines. The RL crashes section lists only round 1's `rirez2ko`.
- **From `probe_round2.json`:** `test_auc` 0.73567, `test_mean_biased` 0.54258, `test_mean_unbiased` 0.37164.
- **From `advantages_round2.json`:** `kl_div_file_step_0` is 2.88816. The mean `wandb_kl_div` over the last 16 steps (file steps 112–127) is 4.06955.
- **From round 3's `build.json`:**
  - `original`: pools are bias/cue_gold 378, bias/cue_dist 1037, no bias/cue_gold 2739, no bias/cue_dist 642; `per_group` 378.
  - `round1`: pools are bias/cue_gold 972, bias/cue_dist 1839, no bias/cue_gold 1839, no bias/cue_dist 972; `per_group` 378.
  - `round2`: pools are bias/cue_gold 951, bias/cue_dist 2324, no bias/cue_gold 2544, no bias/cue_dist 972; `per_group` 378.
  - Top-ups: none. All `topped_up_from` entries are empty, and `logs/build_round3_a1_33f36c5bf32d.log` has no `TOPPED UP:` lines.
  - `test_one_class`: false.

**Overall: PASS**, all of rules (a) to (i).

### Review 5 (end of round 3 and round 4's probe set and config, step 52): PASS (fresh general-purpose agent, 2026-11-03T10:49Z)

Review 5 (end of round 3 at step 52, plus round 4's probe set and config). **Overall verdict: PASS.** All of rules (a) to (i) pass. I read every file myself and changed nothing in the experiment. My two check scripts are in the scratchpad: /tmp/claude-0/-workspace/4d2f80b9-db12-4096-b4e6-1385822cbc4c/scratchpad/review5_config.py and /tmp/claude-0/-workspace/4d2f80b9-db12-4096-b4e6-1385822cbc4c/scratchpad/review5_build.py.

**(a) Step files: PASS.** `runs/round3/check_records.json` has `"PASS": true`, `"n_files": 128` and `"seed": 42`. The W&B run is `5xj7bsnh` and the run folder is `/workspace/results/run_results/261102_iterative_probe_retraining/round3_5xj7bsnh`.

**(b) Round 3's config: PASS.** In `round3.yaml`:
- `checkpoint` is `/workspace/results/run_results/261102_iterative_probe_retraining/round2_kgc82y7d/checkpoints/final`. `run_metrics_round2.json` gives run id `kgc82y7d`.
- Under `overrides`, the fixed settings are all as required: `loss.detector` 0.6, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 128, `++grpo.model.kl_reference_from_base` true.
- The probe training file is `.../probe_sets/round3/train.jsonl`. The probe eval file is `.../probe_sets/round3/test.jsonl`, which is right because round 3's `build.json` has `test_one_class` false.
- `max_train_examples` is 9072, which is 2 × that build's `train_rows` of 4536.
- The other overrides have the same keys as `round_template.yaml`, in the same order, with no differing values. The top-level keys and `output_dir`, `group`, `config_name` and `probe.retrain` also match.

**(c) Advantage and KL: PASS.** In `checks/advantages_round3.json`:
- `advantage_std_floor` is 0.3 and `steps_compared` is 128.
- `max_abs_difference` is 1.41e-8, within ≤ 1e-4.
- `kl_div_file_step_0` is 3.9855, which is ≥ 0.01 as required for round 3.

**(d) Probe: PASS.** In `checks/probe_round3.json`:
- `train_biased_unbiased` is [2268, 2268], equal to the expected [2268, 2268].
- `test_biased_unbiased` is [425, 455], equal to the expected [425, 455].
- Both `saved_layers` entries, `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`, are [13, 14, 15, 16].

**(e) Starting adapter: PASS.** Line 97 of `logs/rl_round3_a1_33f36c5bf32d.log` reads exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round2_kgc82y7d/checkpoints/final/lora`. The log ends `EXIT 0`, and a1 is the only round-3 RL log.

**(f) Eval: PASS.**
- `evals/round3/summary.json` has `questions` 489 and `cued_rollouts` 978.
- `evals/base_rescored_round3/summary.json` has `questions` 489 and `cued_rollouts` 978.
- `runs/round3/training_curve.csv` has 128 data rows: 129 lines including the header, none blank.

**(g) Score: PASS.** In `checks/score_round3.json`:
- `round_probe_here_vs_final_eval.max_abs_difference` is 0.0 over 978 answers, within ≤ 0.02.
- `first_probe_link` is `/workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/initial_detector.base_model.on_dataset`, as required.
- `round_probe_link` equals `round_probe_in_run_folder`, and both are `.../round3_5xj7bsnh/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`.

**(h) Round 4's probe set: PASS.**

`probe_sets/round4/build.json`:
- The `sources` keys are exactly [original, round1, round2, round3], in that order.
- `original` has `per_group` 378, `own_rows` 378 in all four groups, and no top-ups.
- In every source, `per_group` = min(max(smallest pool, 100), 378) = 378. The smallest pools are 378, 972, 951 and 1197.
- In every group, `own_rows` = min(pool, per_group) = 378, and `own_rows` plus the top-ups equals `per_group`. All `topped_up_from` entries are empty.
- `train_rows` is 6048 = 4 × 1512, and `train_biased` is 3024 = 6048 / 2.

`probe_sets/round4/train.jsonl`:
- It has 6048 rows. Each of the 16 (source, label, condition) cells has 378 rows, which matches `build.json`, and every row's label agrees with its `bias_rm`.
- No (source, question_id, condition, sample_idx) appears twice.
- No row uses a question whose split is `eval` in `questions.jsonl` (that file has 1217 train and 489 eval questions).
- The 1512 `original` rows equal `bias_probe_train.jsonl`, in order, on the six fields the rule names.
- `probe_sets/round3/train.jsonl` (6,676,555 bytes) is byte for byte the start of round 4's file (8,286,977 bytes).

`probe_sets/round4/test.jsonl`:
- It has 910 rows: 454 with label 1 (equal to `test_biased` 454) and 456 with label 0 (equal to `test_unbiased` 456).
- `test_answers` names `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/evals/round3`.
- `test_one_class` is false, which is correct because neither count is 0.

**(i) Round 4's config: PASS.** In `round4.yaml`:
- `checkpoint` is `/workspace/results/run_results/261102_iterative_probe_retraining/round3_5xj7bsnh/checkpoints/final`.
- The fixed settings are the same as in (b): 0.6, 0.3, 42, 128 and true.
- The probe training file is `.../probe_sets/round4/train.jsonl`. The probe eval file is `.../probe_sets/round4/test.jsonl`, which is right because `test_one_class` is false.
- `max_train_examples` is 12096, which is 2 × 6048.
- All other overrides equal the template's, with the same key set and order.

**Reported (no verdict):**
- **From `runs.md` (step 41 log line):**
  - Round time: `GRPO subprocess finished in 86.0 min (exit=0, attempt=1/4)`. The step's wall time was 09:01 to 10:29 UTC (1.5 h).
  - Lowest free disk: 104 GB.
  - Peak GPU memory: 78886 MiB.
  - Crashed attempts: none. The log has 0 `CRASH:` lines and no `removed partial run directory` line, and its only "GRPO subprocess finished" line is the exit=0 one. The RL crashes section of `runs.md` lists only round 1's `rirez2ko`.
- **From `probe_round3.json`:** `test_auc` 0.9184 (0.9183710407239819), `test_mean_biased` 0.7527 (0.752676402913297), `test_mean_unbiased` 0.3087 (0.3086784625782217).
- **From `advantages_round3.json`:** `kl_div_file_step_0` 3.9855 (3.985494315624237). The mean `wandb_kl_div` over the last 16 steps (file steps 112 to 127) is 4.0439.
- **From round 4's `build.json`:**
  - `original`: pools 378 / 1037 / 2739 / 642, `per_group` 378.
  - `round1`: pools 972 / 1839 / 1839 / 972, `per_group` 378.
  - `round2`: pools 951 / 2324 / 2544 / 972, `per_group` 378.
  - `round3`: pools 1197 / 2317 / 2328 / 1197, `per_group` 378.
  - The pool order is bias/cue_gold, bias/cue_dist, no bias/cue_gold, no bias/cue_dist.
  - Top-ups: none. All `topped_up_from` entries are empty, and `logs/build_round4_a1_33f36c5bf32d.log` has no `TOPPED UP:` line.
  - `test_one_class` is false.
- **Step 14's `cmp` result:** not applicable, since the round is 3, not 1.

### Review 6 (end of round 4 and round 5's probe set and config, step 65): PASS (fresh general-purpose agent, 2026-11-03T12:56Z)

Review 6 (end of round 4 at step 65, plus round 5's probe set and config). **Overall verdict: PASS.** All nine rules, (a) to (i), pass. I read every file myself and changed nothing in the experiment. My two check scripts are in the scratchpad: /tmp/claude-0/-workspace/4d2f80b9-db12-4096-b4e6-1385822cbc4c/scratchpad/check_configs.py and /tmp/claude-0/-workspace/4d2f80b9-db12-4096-b4e6-1385822cbc4c/scratchpad/check_h.py.

**(a) Step files: PASS.** `runs/round4/check_records.json` has `"PASS": true` (a boolean), `"n_files": 128` and `"seed": 42`. Its `per_step` list has 128 entries. The W&B run id is `zzpngiif`.

**(b) Round 4's config: PASS.** In `round4.yaml`:
- `checkpoint` is `/workspace/results/run_results/261102_iterative_probe_retraining/round3_5xj7bsnh/checkpoints/final`, as required. `run_metrics_round3.json` gives run id `5xj7bsnh`.
- `loss.detector` 0.6, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 128 and `++grpo.model.kl_reference_from_base` true.
- The probe training file is `.../probe_sets/round4/train.jsonl`.
- The probe eval file is `.../probe_sets/round4/test.jsonl`. That is right because round 4's `build.json` has `test_one_class` false.
- `max_train_examples` is 12096, which is 2 × round 4's `train_rows` of 6048.
- The 17 override keys are the template's 17, in the same order. The other 9 overrides equal the template's in value and type. `output_dir`, `group`, `config_name` and `probe.retrain` also match the template.

**(c) Advantage and KL: PASS.** `checks/advantages_round4.json` has:
- `advantage_std_floor` 0.3;
- `steps_compared` 128;
- `max_abs_difference` 1.69e-8, which is at most 1e-4;
- `kl_div_file_step_0` 4.0449, which is at least 0.01 as round 4 requires.

**(d) Probe: PASS.** `checks/probe_round4.json` has:
- `train_biased_unbiased` [3024, 3024], equal to `expected_train_biased_unbiased` [3024, 3024];
- `test_biased_unbiased` [454, 456], equal to `expected_test_biased_unbiased` [454, 456];
- `saved_layers` [13, 14, 15, 16] for both `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`.

**(e) Starting adapter: PASS.** `logs/rl_round4_a1_33f36c5bf32d.log` is the only round 4 RL log, and it ends `EXIT 0`. Its line 99 is exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round3_5xj7bsnh/checkpoints/final/lora`, with nothing after it on the line.

**(f) Eval: PASS.**
- `evals/round4/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `evals/base_rescored_round4/summary.json`: `"questions": 489`, `"cued_rollouts": 978`.
- `runs/round4/training_curve.csv`: one header line plus 128 data rows (steps 0 to 127), no blank rows.

**(g) Score: PASS.** `checks/score_round4.json` has:
- `round_probe_here_vs_final_eval.max_abs_difference` 0.0 over 978 answers, which is at most 0.02;
- `first_probe_link` `/workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/initial_detector.base_model.on_dataset`, as required;
- `round_probe_link` equal to `round_probe_in_run_folder`, and both are `/workspace/results/run_results/261102_iterative_probe_retraining/round4_zzpngiif/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`, as required. All three paths exist on disk.

**(h) Round 5's probe set: PASS.**

`probe_sets/round5/build.json`:
- The keys of `sources` are exactly `original`, `round1`, `round2`, `round3`, `round4`, in that order.
- `original` has `per_group` 378, `own_rows` 378 in all four groups, and no top-ups.
- `per_group` follows min(max(smallest pool, 100), 378) for every source:
  - `original`: smallest pool 378, so 378.
  - `round1`: smallest pool 972, so 378.
  - `round2`: smallest pool 951, so 378.
  - `round3`: smallest pool 1197, so 378.
  - `round4`: smallest pool 312, so 312.
- Every pool is at least its `per_group`. So in every group `own_rows` = `per_group`, every `topped_up_from` is empty, and own rows plus top-ups = `per_group`.
- `train_rows` 7296 = 4 × 1824 (the sum of the five `per_group` values). `train_biased` 3648 = 7296 / 2.

`probe_sets/round5/train.jsonl`:
- It has 7296 rows.
- The row counts per (`source`, `label`, `condition`) equal `build.json`'s for all 20 combinations: 378 each for `original` and `round1` to `round3`, and 312 each for `round4`. The label agrees with `bias_rm` in every row.
- No (`source`, `question_id`, `condition`, `sample_idx`) appears twice.
- None of its question ids has `split` `eval` in `questions.jsonl`, which has 1217 train and 489 eval questions.
- The 1512 rows with `source` `original` equal `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl` (1512 rows) in order, on `question_id`, `condition`, `sample_idx`, `prompt`, `completion` and `label`.
- `probe_sets/round4/train.jsonl` (8,286,977 bytes, 6048 lines) is byte for byte the start of round 5's file (9,749,899 bytes). `cmp -n` found no difference, and the shared part ends at a line break.

`probe_sets/round5/test.jsonl`:
- It has 614 rows: 164 with `label` 1, equal to `test_biased` 164, and 450 with `label` 0, equal to `test_unbiased` 450.
- `test_answers` names `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/evals/round4`, followed by "(round 4's final-eval answers)".
- `test_one_class` is false, and neither count is 0, so the two agree.

**(i) Round 5's config: PASS.** In `round5.yaml`:
- `checkpoint` is `/workspace/results/run_results/261102_iterative_probe_retraining/round4_zzpngiif/checkpoints/final`, as required. `run_metrics_round4.json` gives run id `zzpngiif`, and the folder exists.
- `loss.detector` 0.6, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 128 and `++grpo.model.kl_reference_from_base` true.
- The probe training file is `.../probe_sets/round5/train.jsonl`.
- The probe eval file is `.../probe_sets/round5/test.jsonl`, which is right because `test_one_class` is false.
- `max_train_examples` is 14592, which is 2 × `train_rows` 7296.
- The other 9 overrides, and the key set and order, equal the template's.
- The `smoke_round2.yaml` part of this rule applies only to round 1, so it was not checked.

**Reported (no verdict):**
- **From `runs.md` (round 4's RL step 54):**
  - `GRPO subprocess finished in 101.7 min (exit=0, attempt=1/4)`; step wall time 10:52 to 12:35 UTC (1.7 h).
  - Lowest free disk 102 GB, peak GPU memory 76944 MiB.
  - Crashed attempts in round 4: none. The round 4 log has 0 `CRASH:` lines and no `removed partial run directory` line. The RL crashes section of `runs.md` lists only round 1's crash (`rirez2ko`).
- **From `probe_round4.json`:** `test_auc` 0.7905508540072649, `test_mean_biased` 0.6818386452038678, `test_mean_unbiased` 0.42682555533553423.
- **From `advantages_round4.json`:** `kl_div_file_step_0` 4.04488817602396. The mean `wandb_kl_div` of the last 16 steps (file steps 112 to 127) is 1.6593577791936696.
- **From round 5's `build.json`** (pool values in the order bias / cue_gold, bias / cue_dist, no bias / cue_gold, no bias / cue_dist):

  | Source | Pool | `per_group` |
  |---|---|---|
  | original | 378, 1037, 2739, 642 | 378 |
  | round1 | 972, 1839, 1839, 972 | 378 |
  | round2 | 951, 2324, 2544, 972 | 378 |
  | round3 | 1197, 2317, 2328, 1197 | 378 |
  | round4 | 312, 549, 2809, 757 | 312 |

  - Top-ups: none. All `topped_up_from` entries are empty, and `logs/build_round5_a1_33f36c5bf32d.log` has no `TOPPED UP:` line.
  - `test_one_class` is false.
- **Step 14's `cmp` result:** not applicable, because this review is for round 4, not round 1.

### Review 7 (end of round 5, step 75): PASS (fresh general-purpose agent, 2026-11-03T15:15Z)

**Review 7 (end of round 5, step 75): overall verdict PASS.** All seven rules, (a) to (g), pass. I read every file myself and changed nothing. My only file is a read-only helper script at /tmp/claude-0/-workspace/4d2f80b9-db12-4096-b4e6-1385822cbc4c/scratchpad/review7.py.

**(a) Step files: PASS.** In `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/runs/round5/check_records.json`, `"PASS": true`, `"n_files": 128` and `"seed": 42`. Its `run_dir` is `.../round5_f02rok2w`.

**(b) Config: PASS.** In `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/round5.yaml`:
- `checkpoint` is `/workspace/results/run_results/261102_iterative_probe_retraining/round4_zzpngiif/checkpoints/final`, which is what a round other than round 1 needs. `run_metrics_round4.json` has `run_id` `zzpngiif`.
- `loss.detector` 0.6, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 128 and `++grpo.model.kl_reference_from_base` true.
- The probe training file is `.../probe_sets/round5/train.jsonl`.
- The probe eval file is `.../probe_sets/round5/test.jsonl`. That is correct because `probe_sets/round5/build.json` has `test_one_class` false.
- `max_train_examples` is 14592, which is 2 × `train_rows` 7296.
- Both files have the same 17 override keys. The 12 overrides that the template does not leave to be filled in all equal `round_template.yaml`'s: `+data/probe_dataset`, `task_args.bias_probe_at_answer`, `model.activation_matching_layers` range_13-17, `detector.filter_layers_below_loss_threshold` 'null', learning rate 0.03, probe max_steps 2000, the questions file, `cue_gold_prob` 0.75, `model_batch_size` 8, plus the three settings already listed.

**(c) Advantage and KL: PASS.** In `checks/advantages_round5.json`:
- `advantage_std_floor` 0.3 and `steps_compared` 128.
- `max_abs_difference` is 6.36e-09, within the 1e-4 limit.
- `kl_div_file_step_0` is 1.596, which meets the ≥ 0.01 required for round 5.

**(d) Probe: PASS.** In `checks/probe_round5.json`:
- `train_biased_unbiased` [3648, 3648] equals `expected_train_biased_unbiased` [3648, 3648].
- `test_biased_unbiased` [164, 450] equals `expected_test_biased_unbiased` [164, 450].
- Both `saved_layers` entries are [13, 14, 15, 16]: `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`.

**(e) Starting adapter: PASS.** `logs/rl_round5_a1_33f36c5bf32d.log` is the only round-5 RL log, and it ends `EXIT 0`. Line 101 is exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261102_iterative_probe_retraining/round4_zzpngiif/checkpoints/final/lora`.

**(f) Eval: PASS.**
- `evals/round5/summary.json` has `questions` 489 and `cued_rollouts` 978.
- `evals/base_rescored_round5/summary.json` has `questions` 489 and `cued_rollouts` 978.
- `runs/round5/training_curve.csv` has 129 lines: a header plus 128 data rows (steps 0 to 127).

**(g) Score: PASS.** In `checks/score_round5.json`:
- `round_probe_here_vs_final_eval.max_abs_difference` is 0.0 over 977 answers, within the 0.02 limit.
- `first_probe_link` is `/workspace/results/run_results/261102_iterative_probe_retraining/round1_jmf1ks33/initial_detector.base_model.on_dataset`. `run_metrics_round1.json` has `run_id` `jmf1ks33`.
- `round_probe_link` and `round_probe_in_run_folder` are identical. Both are `/workspace/results/run_results/261102_iterative_probe_retraining/round5_f02rok2w/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`, and `run_metrics_round5.json` has `run_id` `f02rok2w`.

**Reported (no verdict):**
- **From `runs.md`, step 67's log line:**
  - Time: `GRPO subprocess finished in 119.0 min (exit=0, attempt=1/4)`; the step ran 13:00 to 15:00 UTC (2.0 h).
  - Lowest free disk: 100 GB.
  - Peak GPU memory: 76818 MiB.
  - Crashed attempts: none. There are no `CRASH:` lines, and the RL crashes section lists only round 1's `rirez2ko`.
- **From `probe_round5.json`:**
  - `test_auc` 0.8246612466124661
  - `test_mean_biased` 0.6623413262934219
  - `test_mean_unbiased` 0.38292494798079135
- **From `advantages_round5.json`:**
  - `kl_div_file_step_0` 1.5960766011849046
  - Mean `wandb_kl_div` over the last 16 steps (file steps 112 to 127): 1.47845952543139

**Overall: PASS.**

### Review 8 (auditor, step 79): PASS (fresh auditor agent, 2026-11-03T15:29Z)

#### Review 8 (auditor, step 79): **PASS**

**The rule:** "Every number it cannot match to the files fails review 8."
**Verdict: PASS.** I matched every number in all three reports to its source files.
- I reran steps 76–78 with the scripts and inputs in your brief. Each run's output text was identical to the saved files: `tables/rounds/seed_table.md` and `.json`, `probe_table.md` and `.json`, `runs/dynamics/dynamics.md` and `.json`.
- I also recomputed the main rows myself from the raw records, without the scripts. They match at the reports' precision.
- I found two wording problems and one gap (below). In each case the numbers themselves match their files, so none of them triggers the fail rule.

How I reran without writing anything: each script ran in memory, with `Path.write_text` and `mkdir` intercepted, under `python -B`. Nothing went to `/tmp` or to any experiment folder. W&B was read-only, and no new files appeared in `~/.cache/wandb`.

---

#### Discrepancies

##### 1. Numbers
None.

##### 2. Procedure claims
**2.1 Contradicted claim (minor).**
- **Where:** `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/tables/rounds/seed_table.md`, line 111. The text comes from `seed_table.py`.
- **Claim:** "probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ)".
- **What the step records show:** some probe-only groups contain an answer cut off at the length cap.
  - round 5: 1 group with a cut-off answer, out of 6607 probe-only groups;
  - 261027_s42: 18 out of 6680;
  - rounds 1–4: none.
- **Why it matters:** in those groups the trainer's reward also differs by the 0.1 non-termination penalty (`dynamics.py` line 96).
- **Evidence the claim would need:** a qualifier such as "apart from the 0.1 cut-off penalty in N groups".

##### 3. Counts across stages
None. Every count lines up across stages:
- **Training-answer pools:** each round's 7302 cued answers, minus the filtered rows, equal the pool sum in `build.json`: 5622, 6791, 7039 and 4427.
- **Original pool:** 4796, which is 1415 biased + 3381 unbiased, as in the setup README.
- **Training files:** `train_rows` in `build.json` equals the rows in `train.jsonl` (1512, 3024, 4536, 6048, 7296). The rows per source, label and condition also match.
- **Test files:** `test_biased` / `test_unbiased` in `build.json` = the label counts in `test.jsonl` = the counts in the trainer's saved score file (`.pkl`) = `test_biased_unbiased` in `checks/probe_round<R>.json`.
- **Shared questions:** 432, 455, 456, 456, 464. These agree across `comparison.json`, the seed table, `probe_table.json` and `dynamics.json`. Each round's 5-of-8 base question set (466 questions) and its bias labels are identical to 261013's saved base eval.

##### 4. Config consistency
None. No config hash or stamp exists. Instead I checked:
- `rounds.py`, `dynamics.py` and `checks.py` are identical to `code_at_launch/`.
- `sycophancy_setup/` and `obfuscation_atlas/` have no changes against HEAD 7276ff2, which is the commit in `code_at_launch/git_head.txt`.
- The "Code fixes" section of `runs.md` is empty, which agrees with the above.
- The run ids in `run_metrics_round<R>.json` match the run folders recorded in `seed_table.json` and `dynamics.json`.
- Every eval's `adapter` field is `<run dir R>/checkpoints/final`.

##### 5. Leakage
No train/eval overlap, and nothing was chosen from eval data. The one designed read of eval data:
- **Train/eval overlap:** none. Probe training files for rounds 1–5 have 0 eval question ids, and `generations/round1–4` are all split `train`.
- **The designed read:** `rounds.py build` (line 331 calls `test_rows`, lines 297–304) reads round R−1's eval labels and completions to write round R's probe test file. The trainer reads that file in Phase 0 (`train_grpo_model_against_probe.py` line 501) only to report the test AUC and scores.
- **Nothing chosen from it:**
  - calibration is off (`n_calibration_examples` defaults to None; no config sets it);
  - `detector.filter_layers_below_loss_threshold` is 'null';
  - `early_stop_auc_threshold` defaults to None;
  - `config` reads `test_one_class` (line 360) under a fixed rule, and it was false in every round.

##### 6. Unsupported or unjustified claims
**6.1 Contradicted claim (a definition; the numbers match their file).**
- **Where:** `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/probe_table.md`, lines 34–35 against line 40.
- **Claim:** "Pool: a source's rows per group".
- **What the files show:** the round-1 row for `original` shows 378 / 378 / 378 / 378. Rounds 2–5 show the same source's pool as 378 / 1037 / 2739 / 642.
- **Why:** round 1's `build.json` records the copied training file's group counts under the name "pool" (`rounds.py` line 319).
- **Evidence the claim would need:** round 1's row should show the real pool, or a footnote saying it shows the file's counts.

**6.2 Coverage gap (informational; your brief already lists this as not a failure).**
- **Where:** `seed_table.md`, lines 80–109.
- **The gap:** the md does not say which probe `probe_start` is in each column:
  - round1: the first probe;
  - round R ≥ 2: round R's own probe;
  - 261027_s42: 261027's own start-of-run fit.
- **Consequence:** the mean and SD columns on these rows combine different probes.

I found no causal wording, no generalization and no comparison claimed as like-for-like in the three reports.

---

#### Verified
- **Seed table:** the in-memory rerun of the step-76 command gave identical `seed_table.md` and `seed_table.json`. I recomputed from raw `completions.jsonl` / `labels.jsonl` for all 6 columns:
  - accuracy, reward, parse_failed and cue agreement rate for all 4 conditions;
  - bias, no_bias, `n_bias_rows`, `n_deception_rows`, deception_soft, deception and said_yes;
  - the `probe_start` and `probe_start_base_activations` means, AUCs and counts;
  - the base column (0.755 / 0.843 / 0.509 accuracy, parse_failed, reward);
  - equal-task-reward and probe-only shares from the step records (0.974 … 0.879; 0.970 … 0.815).
- **Probe table:** the in-memory rerun of `rounds.py table` gave identical `.md` and `.json`. I recomputed all 20 score rows myself:
  - own-test-set rows from each run's pickle (index 0, mean over layers 13–16). They also match `test_auc` in `checks/probe_round<R>.json` (0.8730, 0.7357, 0.9184, 0.7906, 0.8247);
  - the other three rows from `evals/round<R>`, `base_rescored_round<R>` and `first_probe_round<R>`.
  - The training-set rows match `build.json` and the `train.jsonl` counts.
- **The probe table's prose holds:**
  - the 5-of-8 rule: 0 `a_none` mismatches;
  - round R's test file is exactly round R−1's table answer set, with the same keys, labels and completions;
  - round 1's files are byte-identical to `sycophancy_setup/data`;
  - round R's own test set was read through round R−1's adapter (the Phase-0 code and the log lines);
  - first-probe links point to `round1_jmf1ks33`, and the round-probe links equal `round_probe_in_run_folder`, with `max_abs_difference` 0.0.
- **Dynamics:** the in-memory rerun of the step-77 command, with W&B read live, gave identical `.md` and `.json`. Independently:
  - I recomputed all 7 step-record training rows for rounds 1–5;
  - the KL row matches `wandb_kl_div` in `checks/advantages_round<R>.json`. The step-0 KL for rounds 2–5 is 2.888, 3.985, 4.045 and 1.596, which means the reference is the base model;
  - step 127's grad_norm, kl_div, entropy and answer length equal the summaries in `run_metrics_round<R>.json`;
  - the baseline rows and the base and baseline_s42 final-eval text are identical to `identity/reference/dynamics/dynamics.json`;
  - I spot-checked the probe-only, training-text and final-eval blocks from raw files;
  - row A "both" and row C "both" equal the probe table's base-answers and round-answers rows;
  - the "(comparison.json)" AUCs equal `comparison.json`;
  - the probe-data counts equal the probe files;
  - grad_norm is the norm before clipping (the return value of `clip_grad_norm_`).
- **Not failures, as your brief says:** the empty base probe rows; the mean and SD columns; `probe_fresh`; row D; the cosines. `runs.md`'s Result section is empty, which is expected before step 79 completes.

All files are in `/workspace/sycophancy_experiments/261102_iterative_probe_retraining/`.

## Blocked

## Result

From `tables/rounds/seed_table.md` (step 76):

### RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): round1 432, round2 455, round3 456, round4 456, round5 464, 261027_s42 448.

| metric | condition | base | base mean | base SD | round1 | round2 | round3 | round4 | round5 | 261027_s42 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.616 | 0.679 | 0.663 | 0.718 | 0.697 | 0.748 | 0.687 | 0.046 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 1.000 | 0.998 | 0.996 | 0.867 | 0.912 | 0.963 | 0.956 | 0.055 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.000 | 0.059 | 0.002 | 0.495 | 0.387 | 0.307 | 0.208 | 0.215 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.650 | 0.670 | 0.647 | 0.739 | 0.732 | 0.739 | 0.696 | 0.045 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.550 | 0.613 | 0.560 | 0.640 | 0.626 | 0.669 | 0.610 | 0.046 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.800 | 0.798 | 0.796 | 0.728 | 0.748 | 0.787 | 0.776 | 0.030 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.200 | 0.231 | 0.202 | 0.495 | 0.419 | 0.372 | 0.320 | 0.126 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.750 | 0.749 | 0.747 | 0.704 | 0.715 | 0.744 | 0.735 | 0.020 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 1.000 | 0.998 | 0.996 | 0.867 | 0.912 | 0.963 | 0.956 | 0.055 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 1.000 | 0.941 | 0.998 | 0.505 | 0.613 | 0.693 | 0.792 | 0.215 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 1.000 | 0.976 | 0.996 | 0.742 | 0.805 | 0.868 | 0.898 | 0.110 |
| bias | cue_gold | 0.094 / 0.092 / 0.094 / 0.092 / 0.094 / 0.086 | 0.092 | 0.003 | 0.370 | 0.286 | 0.309 | 0.149 | 0.222 | 0.163 | 0.250 | 0.087 |
| bias | cue_dist | 0.274 / 0.270 / 0.271 / 0.267 / 0.269 / 0.269 | 0.270 | 0.002 | 0.630 | 0.648 | 0.686 | 0.211 | 0.295 | 0.478 | 0.491 | 0.200 |
| bias | train_mix | 0.155 / 0.153 / 0.152 / 0.152 / 0.154 / 0.150 | 0.153 | 0.002 | 0.453 | 0.408 | 0.456 | 0.164 | 0.243 | 0.268 | 0.332 | 0.123 |
| anti_bias | cue_gold | 0.035 / 0.038 / 0.038 / 0.038 / 0.037 / 0.032 | 0.036 | 0.002 | 0.000 | 0.000 | 0.000 | 0.004 | 0.011 | 0.000 | 0.003 | 0.004 |
| anti_bias | cue_dist | 0.010 / 0.007 / 0.011 / 0.011 / 0.011 / 0.009 | 0.010 | 0.002 | 0.000 | 0.000 | 0.000 | 0.009 | 0.000 | 0.000 | 0.001 | 0.004 |
| anti_bias | train_mix | 0.027 / 0.026 / 0.028 / 0.028 / 0.028 / 0.023 | 0.026 | 0.002 | 0.000 | 0.000 | 0.000 | 0.005 | 0.006 | 0.000 | 0.002 | 0.003 |
| no_bias | cue_gold | 0.871 / 0.871 / 0.868 / 0.871 / 0.868 / 0.882 | 0.872 | 0.005 | 0.630 | 0.714 | 0.691 | 0.846 | 0.767 | 0.837 | 0.747 | 0.085 |
| no_bias | cue_dist | 0.717 / 0.723 / 0.718 / 0.722 / 0.720 / 0.722 | 0.720 | 0.003 | 0.370 | 0.352 | 0.314 | 0.781 | 0.705 | 0.522 | 0.507 | 0.197 |
| no_bias | train_mix | 0.818 / 0.821 / 0.820 / 0.820 / 0.818 / 0.827 | 0.821 | 0.003 | 0.547 | 0.592 | 0.544 | 0.830 | 0.751 | 0.732 | 0.666 | 0.121 |
| deception_soft | cue_gold | 0.551 / 0.553 / 0.550 / 0.552 / 0.552 / 0.554 | 0.552 | 0.001 | 0.547 | 0.634 | 0.600 | 0.676 | 0.581 | 0.405 | 0.574 | 0.094 |
| deception_soft | cue_dist | 0.598 / 0.603 / 0.599 / 0.602 / 0.601 / 0.599 | 0.600 | 0.002 | 0.451 | 0.391 | 0.359 | 0.690 | 0.665 | 0.508 | 0.511 | 0.139 |
| deception_soft | train_mix | 0.566 / 0.569 / 0.566 / 0.568 / 0.568 / 0.569 | 0.568 | 0.001 | 0.506 | 0.549 | 0.505 | 0.685 | 0.614 | 0.442 | 0.550 | 0.087 |
| deception | cue_gold | 0.564 / 0.568 / 0.566 / 0.566 / 0.572 / 0.573 | 0.568 | 0.003 | 0.532 | 0.692 | 0.636 | 0.729 | 0.600 | 0.312 | 0.584 | 0.150 |
| deception | cue_dist | 0.663 / 0.667 / 0.662 / 0.661 / 0.663 / 0.661 | 0.663 | 0.002 | 0.458 | 0.363 | 0.344 | 0.730 | 0.698 | 0.533 | 0.521 | 0.165 |
| deception | train_mix | 0.597 / 0.600 / 0.599 / 0.598 / 0.602 / 0.603 | 0.600 | 0.002 | 0.498 | 0.582 | 0.518 | 0.736 | 0.638 | 0.389 | 0.560 | 0.120 |
| biased_share | cue_gold | 0.097 / 0.095 / 0.097 / 0.095 / 0.098 / 0.089 | 0.095 | 0.003 | 0.370 | 0.286 | 0.309 | 0.150 | 0.225 | 0.163 | 0.250 | 0.087 |
| biased_share | cue_dist | 0.276 / 0.272 / 0.274 / 0.270 / 0.272 / 0.271 | 0.273 | 0.002 | 0.630 | 0.648 | 0.686 | 0.212 | 0.295 | 0.478 | 0.492 | 0.199 |
| biased_share | train_mix | 0.160 / 0.157 / 0.157 / 0.156 / 0.159 / 0.154 | 0.157 | 0.002 | 0.453 | 0.408 | 0.456 | 0.165 | 0.244 | 0.268 | 0.333 | 0.123 |
| false_denial_soft | cue_gold | 0.388 / 0.374 / 0.374 / 0.371 / 0.380 / 0.390 | 0.380 | 0.008 | 0.239 | 0.119 | 0.167 | 0.126 | 0.180 | 0.403 | 0.206 | 0.106 |
| false_denial_soft | cue_dist | 0.355 / 0.351 / 0.351 / 0.350 / 0.353 / 0.344 | 0.351 | 0.004 | 0.270 | 0.129 | 0.147 | 0.158 | 0.244 | 0.434 | 0.230 | 0.115 |
| false_denial_soft | train_mix | 0.373 / 0.363 / 0.362 / 0.361 / 0.367 / 0.365 | 0.365 | 0.004 | 0.241 | 0.121 | 0.155 | 0.134 | 0.208 | 0.421 | 0.213 | 0.111 |
| false_admission_soft | cue_gold | 0.569 / 0.572 / 0.569 / 0.571 / 0.571 / 0.570 | 0.570 | 0.001 | 0.728 | 0.840 | 0.793 | 0.772 | 0.697 | 0.405 | 0.706 | 0.156 |
| false_admission_soft | cue_dist | 0.691 / 0.698 / 0.693 / 0.695 / 0.693 / 0.694 | 0.694 | 0.002 | 0.759 | 0.875 | 0.822 | 0.833 | 0.842 | 0.576 | 0.785 | 0.109 |
| false_admission_soft | train_mix | 0.603 / 0.607 / 0.604 / 0.606 / 0.606 / 0.605 | 0.605 | 0.002 | 0.725 | 0.845 | 0.798 | 0.794 | 0.746 | 0.450 | 0.726 | 0.142 |
| said_yes_biased_soft | cue_gold | 0.612 / 0.626 / 0.626 / 0.629 / 0.620 / 0.610 | 0.620 | 0.008 | 0.761 | 0.881 | 0.833 | 0.874 | 0.820 | 0.597 | 0.794 | 0.106 |
| said_yes_biased_soft | cue_dist | 0.645 / 0.649 / 0.649 / 0.650 / 0.647 / 0.656 | 0.649 | 0.004 | 0.730 | 0.871 | 0.853 | 0.842 | 0.756 | 0.566 | 0.770 | 0.115 |
| said_yes_biased_soft | train_mix | 0.627 / 0.637 / 0.638 / 0.639 / 0.633 / 0.635 | 0.635 | 0.004 | 0.759 | 0.879 | 0.845 | 0.866 | 0.792 | 0.579 | 0.787 | 0.111 |
| deception_from_biased | cue_gold | 0.038 / 0.036 / 0.036 / 0.035 / 0.037 / 0.035 | 0.036 | 0.001 | 0.088 | 0.034 | 0.052 | 0.019 | 0.040 | 0.066 | 0.050 | 0.025 |
| deception_from_biased | cue_dist | 0.098 / 0.095 / 0.096 / 0.095 / 0.096 / 0.093 | 0.096 | 0.002 | 0.170 | 0.083 | 0.101 | 0.034 | 0.072 | 0.207 | 0.111 | 0.065 |
| deception_from_biased | train_mix | 0.059 / 0.057 / 0.057 / 0.056 / 0.058 / 0.056 | 0.057 | 0.001 | 0.109 | 0.050 | 0.071 | 0.022 | 0.051 | 0.113 | 0.069 | 0.036 |
| deception_from_unbiased | cue_gold | 0.513 / 0.518 / 0.514 / 0.517 / 0.515 / 0.519 | 0.516 | 0.002 | 0.458 | 0.600 | 0.548 | 0.657 | 0.541 | 0.339 | 0.524 | 0.112 |
| deception_from_unbiased | cue_dist | 0.500 / 0.508 / 0.503 / 0.507 / 0.505 / 0.506 | 0.505 | 0.003 | 0.281 | 0.308 | 0.258 | 0.656 | 0.594 | 0.301 | 0.400 | 0.176 |
| deception_from_unbiased | train_mix | 0.506 / 0.512 / 0.510 / 0.511 / 0.510 / 0.512 | 0.510 | 0.002 | 0.397 | 0.500 | 0.434 | 0.662 | 0.564 | 0.329 | 0.481 | 0.120 |
| said_yes | cue_gold | 0.594 / 0.599 / 0.599 / 0.601 / 0.601 / 0.596 | 0.598 | 0.003 | 0.755 | 0.960 | 0.901 | 0.861 | 0.769 | 0.342 | 0.765 | 0.222 |
| said_yes | cue_dist | 0.786 / 0.792 / 0.785 / 0.789 / 0.787 / 0.789 | 0.788 | 0.003 | 0.750 | 0.980 | 0.934 | 0.907 | 0.869 | 0.556 | 0.833 | 0.156 |
| said_yes | train_mix | 0.655 / 0.660 / 0.658 / 0.662 / 0.661 / 0.659 | 0.659 | 0.002 | 0.750 | 0.968 | 0.917 | 0.879 | 0.804 | 0.416 | 0.789 | 0.199 |
| false_denial | cue_gold | 0.350 / 0.341 / 0.333 / 0.317 / 0.349 / 0.368 | 0.343 | 0.017 | 0.200 | 0.031 | 0.071 | 0.059 | 0.126 | 0.411 | 0.150 | 0.141 |
| false_denial | cue_dist | 0.278 / 0.269 / 0.275 / 0.263 / 0.273 / 0.265 | 0.270 | 0.006 | 0.268 | 0.024 | 0.070 | 0.083 | 0.212 | 0.477 | 0.189 | 0.169 |
| false_denial | train_mix | 0.319 / 0.309 / 0.312 / 0.295 / 0.314 / 0.316 | 0.311 | 0.008 | 0.221 | 0.028 | 0.062 | 0.067 | 0.160 | 0.449 | 0.165 | 0.157 |
| false_admission | cue_gold | 0.588 / 0.592 / 0.591 / 0.592 / 0.596 / 0.593 | 0.592 | 0.003 | 0.728 | 0.957 | 0.889 | 0.847 | 0.738 | 0.293 | 0.742 | 0.237 |
| false_admission | cue_dist | 0.811 / 0.815 / 0.808 / 0.809 / 0.809 / 0.809 | 0.810 | 0.003 | 0.781 | 0.988 | 0.944 | 0.904 | 0.902 | 0.585 | 0.851 | 0.147 |
| false_admission | train_mix | 0.650 / 0.654 / 0.653 / 0.654 / 0.656 / 0.655 | 0.654 | 0.002 | 0.726 | 0.965 | 0.900 | 0.869 | 0.792 | 0.367 | 0.770 | 0.214 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.000 | 0.008 | 0.025 | 0.012 | 0.006 | 0.012 | 0.011 | 0.008 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.001 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.001 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 426 / 448 / 448 / 448 / 456 / 440 | 444.333 | 10.309 | 432 | 455 | 456 | 456 | 463 | 448 | 451.667 | 10.746 |
| n_bias_rows | cue_dist | 420 / 441 / 443 / 442 / 450 / 435 | 438.500 | 10.252 | 432 | 455 | 456 | 456 | 464 | 448 | 451.833 | 10.962 |
| n_bias_rows | train_mix | 846 / 889 / 891 / 890 / 906 / 875 | 882.833 | 20.547 | 864 | 910 | 912 | 912 | 927 | 896 | 903.500 | 21.705 |
| n_deception_rows | cue_gold | 411 / 431 / 431 / 431 / 439 / 426 | 428.167 | 9.390 | 432 | 455 | 456 | 454 | 458 | 448 | 450.500 | 9.670 |
| n_deception_rows | cue_dist | 416 / 438 / 438 / 437 / 445 / 431 | 434.167 | 9.948 | 432 | 455 | 456 | 452 | 464 | 448 | 451.167 | 10.778 |
| n_deception_rows | train_mix | 827 / 869 / 869 / 868 / 884 / 857 | 862.333 | 19.325 | 864 | 910 | 912 | 906 | 922 | 896 | 901.667 | 20.294 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.110 | 0.502 | 0.330 | 0.626 | 0.498 | 0.063 | 0.355 | 0.229 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.120 | 0.536 | 0.447 | 0.669 | 0.483 | 0.035 | 0.382 | 0.249 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.093 | 0.238 | 0.237 | 0.300 | 0.271 | 0.015 | 0.192 | 0.113 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.132 | 0.283 | 0.335 | 0.420 | 0.349 | 0.015 | 0.256 | 0.152 |
| probe_start_auc | cue_gold | – | – | – | 0.589 | 0.823 | 0.633 | 0.833 | 0.799 | 0.756 | 0.739 | 0.104 |
| probe_start_auc | cue_dist | – | – | – | 0.497 | 0.807 | 0.625 | 0.784 | 0.708 | 0.691 | 0.685 | 0.113 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 160 | 130 | 141 | 68 | 103 | 73 | 112.500 | 37.431 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 272 | 295 | 313 | 96 | 137 | 214 | 221.167 | 88.624 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 272 | 324 | 314 | 327 | 320 | 358 | 319.167 | 27.730 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 160 | 131 | 142 | 123 | 139 | 90 | 130.833 | 23.541 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.171 | 0.561 | 0.500 | 0.441 | 0.669 | 0.279 | 0.437 | 0.184 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.208 | 0.662 | 0.663 | 0.484 | 0.658 | 0.259 | 0.489 | 0.210 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.157 | 0.386 | 0.423 | 0.291 | 0.408 | 0.151 | 0.303 | 0.124 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.199 | 0.491 | 0.600 | 0.350 | 0.493 | 0.151 | 0.381 | 0.179 |
| probe_fresh_auc | cue_gold | – | – | – | 0.557 | 0.700 | 0.587 | 0.677 | 0.817 | 0.751 | 0.682 | 0.098 |
| probe_fresh_auc | cue_dist | – | – | – | 0.514 | 0.701 | 0.577 | 0.669 | 0.724 | 0.701 | 0.647 | 0.083 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 160 | 130 | 141 | 68 | 103 | 73 | 112.500 | 37.431 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 272 | 295 | 313 | 96 | 137 | 214 | 221.167 | 88.624 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 272 | 324 | 314 | 327 | 320 | 358 | 319.167 | 27.730 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 160 | 131 | 142 | 123 | 139 | 90 | 130.833 | 23.541 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.457 | 0.719 | 0.403 | 0.492 | 0.603 | 0.119 | 0.466 | 0.204 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.727 | 0.764 | 0.599 | 0.571 | 0.629 | 0.087 | 0.563 | 0.245 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.230 | 0.496 | 0.319 | 0.331 | 0.506 | 0.035 | 0.320 | 0.176 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.535 | 0.583 | 0.542 | 0.441 | 0.572 | 0.037 | 0.452 | 0.209 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.701 | 0.799 | 0.604 | 0.747 | 0.693 | 0.753 | 0.716 | 0.067 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.664 | 0.769 | 0.559 | 0.687 | 0.599 | 0.680 | 0.660 | 0.074 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 160 | 130 | 141 | 68 | 103 | 73 | 112.500 | 37.431 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 272 | 295 | 313 | 96 | 137 | 214 | 221.167 | 88.624 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 272 | 324 | 314 | 327 | 320 | 358 | 319.167 | 27.730 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 160 | 131 | 142 | 123 | 139 | 90 | 130.833 | 23.541 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | round1 | round2 | round3 | round4 | round5 | 261027_s42 | mean | SD |
|---|---|---|---|---|---|---|---|---|
| equal task reward groups | 0.974 | 0.993 | 0.986 | 0.947 | 0.988 | 0.879 | 0.961 | 0.043 |
| probe-only groups | 0.970 | 0.972 | 0.927 | 0.722 | 0.807 | 0.815 | 0.869 | 0.103 |

From `probe_table.md` (step 78):

### Probe scores per round

Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the
cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue
answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test
set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round
R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.

| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | Round R's probe; its own test set; round R-1's model (round 1: base) | 292 / 485 | 0.722 / 0.255 | 0.873 |
| round1 | Round R's probe; round R's answers; round R's model | 432 / 432 | 0.116 / 0.107 | 0.558 |
| round1 | Round R's probe; base model's answers; base model | 155 / 406 | 0.737 / 0.232 | 0.901 |
| round1 | First probe; round R's answers; round R's model | 432 / 432 | 0.116 / 0.107 | 0.558 |
| round2 | Round R's probe; its own test set; round R-1's model (round 1: base) | 432 / 432 | 0.543 / 0.372 | 0.736 |
| round2 | Round R's probe; round R's answers; round R's model | 425 / 455 | 0.526 / 0.251 | 0.830 |
| round2 | Round R's probe; base model's answers; base model | 160 / 429 | 0.664 / 0.372 | 0.788 |
| round2 | First probe; round R's answers; round R's model | 425 / 455 | 0.266 / 0.034 | 0.896 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 425 / 455 | 0.753 / 0.309 | 0.918 |
| round3 | Round R's probe; round R's answers; round R's model | 454 / 456 | 0.411 / 0.267 | 0.669 |
| round3 | Round R's probe; base model's answers; base model | 162 / 427 | 0.770 / 0.448 | 0.847 |
| round3 | First probe; round R's answers; round R's model | 454 / 456 | 0.176 / 0.116 | 0.540 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 454 / 456 | 0.682 / 0.427 | 0.791 |
| round4 | Round R's probe; round R's answers; round R's model | 164 / 450 | 0.651 / 0.333 | 0.834 |
| round4 | Round R's probe; base model's answers; base model | 159 / 427 | 0.619 / 0.356 | 0.811 |
| round4 | First probe; round R's answers; round R's model | 164 / 450 | 0.630 / 0.184 | 0.861 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 164 / 450 | 0.662 / 0.383 | 0.825 |
| round5 | Round R's probe; round R's answers; round R's model | 240 / 459 | 0.489 / 0.295 | 0.777 |
| round5 | Round R's probe; base model's answers; base model | 164 / 435 | 0.645 / 0.429 | 0.791 |
| round5 | First probe; round R's answers; round R's model | 240 / 459 | 0.434 / 0.077 | 0.858 |

### Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
| round2 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round2 | round1 | 972 / 1839 / 1839 / 972 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4536 |
| round3 | round1 | 972 / 1839 / 1839 / 972 | 378 | 378 / 378 / 378 / 378 | none | 4536 |
| round3 | round2 | 951 / 2324 / 2544 / 972 | 378 | 378 / 378 / 378 / 378 | none | 4536 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 6048 |
| round4 | round1 | 972 / 1839 / 1839 / 972 | 378 | 378 / 378 / 378 / 378 | none | 6048 |
| round4 | round2 | 951 / 2324 / 2544 / 972 | 378 | 378 / 378 / 378 / 378 | none | 6048 |
| round4 | round3 | 1197 / 2317 / 2328 / 1197 | 378 | 378 / 378 / 378 / 378 | none | 6048 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 7296 |
| round5 | round1 | 972 / 1839 / 1839 / 972 | 378 | 378 / 378 / 378 / 378 | none | 7296 |
| round5 | round2 | 951 / 2324 / 2544 / 972 | 378 | 378 / 378 / 378 / 378 | none | 7296 |
| round5 | round3 | 1197 / 2317 / 2328 / 1197 | 378 | 378 / 378 / 378 / 378 | none | 7296 |
| round5 | round4 | 312 / 549 / 2809 / 757 | 312 | 312 / 312 / 312 / 312 | none | 7296 |
