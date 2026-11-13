# Runs

State file for the unattended run of queue `seed_42`. Read this first after any restart.

## Queue

- [x] 1. Start check
- [x] 2. Folders and inputs
- [x] 3. Round 2: build
- [x] 4. Round 2: config
- [x] 5. Round-1 tables
- [x] 6. Review 1
- [x] 7. Smoke: vLLM
- [x] 8. Smoke: RL
- [x] 9. Smoke: record check
- [x] 10. Smoke: stop vLLM
- [x] 11. Smoke: checks
- [x] 12. Review 2
- [x] 13. Round 2: vLLM
- [x] 14. Round 2: RL
- [x] 15. Round 2: record check
- [x] 16. Round 2: checks
- [x] 17. Round 2: stop vLLM
- [x] 18. Round 2: eval
- [x] 19. Round 2: base rescore
- [x] 20. Round 2: compare
- [x] 21. Round 2: score
- [x] 22. Round 2: generate
- [x] 23. Round 3: build
- [x] 24. Round 3: config
- [x] 25. Review 3
- [x] 26. Round 3: vLLM
- [x] 27. Round 3: RL
- [x] 28. Round 3: record check
- [x] 29. Round 3: checks
- [x] 30. Round 3: stop vLLM
- [x] 31. Round 3: eval
- [x] 32. Round 3: base rescore
- [x] 33. Round 3: compare
- [x] 34. Round 3: score
- [x] 35. Round 3: generate
- [x] 36. Round 4: build
- [x] 37. Round 4: config
- [x] 38. Review 4
- [x] 39. Round 4: vLLM
- [x] 40. Round 4: RL
- [x] 41. Round 4: record check
- [x] 42. Round 4: checks
- [x] 43. Round 4: stop vLLM
- [x] 44. Round 4: eval
- [x] 45. Round 4: base rescore
- [x] 46. Round 4: compare
- [x] 47. Round 4: score
- [x] 48. Round 4: generate
- [x] 49. Round 5: build
- [x] 50. Round 5: config
- [x] 51. Review 5
- [x] 52. Round 5: vLLM
- [x] 53. Round 5: RL
- [x] 54. Round 5: record check
- [x] 55. Round 5: checks
- [x] 56. Round 5: stop vLLM
- [x] 57. Round 5: eval
- [x] 58. Round 5: base rescore
- [x] 59. Round 5: compare
- [x] 60. Round 5: score
- [x] 61. Review 6
- [x] 62. Seed table
- [x] 63. Dynamics
- [x] 64. Probe table
- [x] 65. Review 7

## Log

One line per step: `<UTC time> | step | outcome | next`.

Start check, 2026-11-09 19:37 UTC: `hostname` da70f8166885; `nvidia-smi -L` GPU 0: NVIDIA H200 (UUID: GPU-02d193f6-c3ca-0be4-8676-4b9e229d888e); driver 595.91.07, CUDA 13.2; `uname -r` 6.8.0-147-generic; `uptime -s` 2026-10-20 05:01:27; `df -h /workspace` 513G size, 298G used, 215G avail (59%). Pod and GPU match the Goal. Resume check scheduled at kickoff (cron job 89f70048, minutes 7 and 37 of every hour, session-only).

2026-11-09 19:45 UTC | 1. Start check | done: pod/GPU match; fence test rejected by the hook (no file created); all 8 identity checks exit 0 and all 17 compared files identical (scratchpad `start/`) | step 2
2026-11-09 19:46 UTC | 2. Folders and inputs | done: no `261108-rd*` tmux session on this pod; folders made; `checks/round1_inputs_md5.txt` (d786c8fc… completions, 02e719e9… first_probe labels) and `checks/labels_md5_check.txt` (4 × OK) written | step 3
2026-11-09 19:48 UTC | 3. Round 2: build | EXIT 0 (`logs/build_round2_a1_da70f8166885.log`): train_rows 3024 (original and round1 at per_group 378), test 221 biased / 454 unbiased, test_one_class false; no SHRUNK:, TOPPED UP: or DROPPED FROM TEST FILE: lines | step 4
2026-11-09 19:49 UTC | 4. Round 2: config | EXIT 0 (`logs/config_round2_a1_da70f8166885.log`): `s42_refit_round2.yaml`, `smoke_s42_refit_round2.yaml` written | step 5
2026-11-09 19:51 UTC | 5. Round-1 tables | exit 0: `dry/tables/round1/`, `dry/dynamics_round1/`, `dry/probe_table_round1/` written | step 6
2026-11-09 19:52 UTC | 6. Review 1 | PASS (`reviews/1_a1.md`; training code still 261108's) | step 7
2026-11-09 19:57 UTC | 7. Smoke: vLLM | server up (started 19:52:47, `curl` answered 19:57:52, ~5 min; `logs/vllm_smoke_s42_refit_round2_a1_da70f8166885.log`); 215 GB free before | step 8 (launched 19:58:01)
2026-11-09 20:24 UTC | 8. Smoke: RL | EXIT 0 (`logs/rl_smoke_s42_refit_round2_a1_da70f8166885.log`): `GRPO subprocess finished in 24.3 min (exit=0, attempt=1/4)`; no CRASH: lines, no out-of-memory lines; W&B `g2igkbp0`, run dir `/workspace/results/run_results/261109_refit_probe_redrawn_labels/smoke_s42_refit_round2_g2igkbp0`; peak GPU memory 94837 MiB; lowest free disk 210 GB | step 9
2026-11-09 20:25 UTC | 9. Smoke: record check | exit 0: `runs/smoke_s42_refit_round2/check_records.json` written (script prints PASS: 2/2 files) | step 10
2026-11-09 20:26 UTC | 10. Smoke: stop vLLM | session killed; `nvidia-smi` lists no process after ~10 s | step 11
2026-11-09 20:28 UTC | 11. Smoke: checks | exit 0: `checks/reward_smoke_s42_refit_round2.json`, `advantages_smoke_s42_refit_round2.json`, `probe_smoke_s42_refit_round2.json` (from log a1), `kl_smoke_s42_refit_round2.json` written | step 12
2026-11-09 20:30 UTC | 12. Review 2 | PASS (`reviews/2_a1.md`; training code still 261108's) | step 13 (vLLM launched 20:30:34, 213 GB free, GPU empty)
2026-11-09 20:34 UTC | 13. Round 2: vLLM | server up (`curl` answered 20:34:54, ~4 min; `logs/vllm_s42_refit_round2_a1_da70f8166885.log`) | step 14 (RL launched 20:34:58, 213 GB free)
2026-11-10 01:07 UTC | 14. Round 2: RL | EXIT 0 (`logs/rl_s42_refit_round2_a1_da70f8166885.log`): `GRPO subprocess finished in 269.6 min (exit=0, attempt=1/4)`; no CRASH: lines, no out-of-memory lines; W&B `3p610ssb`, run dir `/workspace/results/run_results/261109_refit_probe_redrawn_labels/s42_refit_round2_3p610ssb`; peak GPU memory 98830 MiB; lowest free disk 200 GB | steps 15–16 (CPU, background) alongside 17–18
2026-11-10 01:07 UTC | 17. Round 2: stop vLLM | session killed; `nvidia-smi` lists no process after ~10 s (run before steps 15–16 finished: they only read saved files) | step 18 (eval launched, 207 GB free, GPU empty)
2026-11-10 01:08 UTC | 15. Round 2: record check | exit 0: `runs/s42_refit_round2/check_records.json` written (script prints PASS: 128/128 files) | (step 18 running)
2026-11-10 01:10 UTC | 16. Round 2: checks | exit 0: `checks/advantages_s42_refit_round2.json`, `probe_s42_refit_round2.json` (from log a1), `kl_s42_refit_round2.json` written | (step 18 running)
2026-11-10 01:15 UTC | 18. Round 2: eval | EXIT 0 (`logs/eval_s42_refit_round2_a1_da70f8166885.log`, ~7.5 min): `evals/s42_refit_round2/summary.json` questions 489, cued_rollouts 978 | step 19 (copy_base wrote 4890 rows; rescore launched, 207 GB free, GPU empty)
2026-11-10 01:19 UTC | 19. Round 2: base rescore | EXIT 0 (`logs/eval_base_rescored_s42_refit_round2_a1_da70f8166885.log`, ~3 min): `evals/base_rescored_s42_refit_round2/summary.json` questions 489, cued_rollouts 978 | step 20 (CPU, background) alongside step 21 (score launched)
2026-11-10 01:19 UTC | 20. Round 2: compare | exit 0: `runs/s42_refit_round2/comparison.json`, `comparison.md`, `training_curve.csv` (128 data rows) | (step 21 running)
2026-11-10 01:23 UTC | 21. Round 2: score | EXIT 0 (`logs/score_round2_a1_da70f8166885.log`, ~4 min): `evals/first_probe_round2/`, `stand_in/round2/`, `checks/score_round2.json` written | step 22 (generate launched, 207 GB free, GPU empty)
2026-11-10 01:32 UTC | 22. Round 2: generate | EXIT 0 (`logs/generate_round2_a1_da70f8166885.log`, ~9 min): `generations/round2/completions.jsonl` 17038 rows | step 23
2026-11-10 01:33 UTC | 23. Round 3: build | EXIT 0 (`logs/build_round3_a1_da70f8166885.log`): train_rows 4208 (original 378, round1 378, round2 296 per group), test 156 biased / 454 unbiased, test_one_class false; no SHRUNK:, TOPPED UP: or DROPPED FROM TEST FILE: lines | step 24
2026-11-10 01:34 UTC | 24. Round 3: config | EXIT 0 (`logs/config_round3_a1_da70f8166885.log`): `s42_refit_round3.yaml` written; no out-of-memory fallback in effect, yaml unchanged | step 25
2026-11-10 01:38 UTC | 25. Review 3 | PASS (`reviews/3_a1.md`); its code item (not a gate) now prints a difference: see Judgment calls | step 26 (vLLM launched 01:38:27, 207 GB free, GPU empty)
2026-11-10 01:43 UTC | 26. Round 3: vLLM | server up (`curl` answered 01:43:01, ~4.5 min; `logs/vllm_s42_refit_round3_a1_da70f8166885.log`) | step 27 (RL launched 01:43:05, 207 GB free)
2026-11-10 05:50 UTC | 27. Round 3: RL | EXIT 0 (`logs/rl_s42_refit_round3_a1_da70f8166885.log`): `GRPO subprocess finished in 245.1 min (exit=0, attempt=1/4)`; no CRASH: lines, no out-of-memory lines; W&B `ojvmixvx`, run dir `/workspace/results/run_results/261109_refit_probe_redrawn_labels/s42_refit_round3_ojvmixvx`; peak GPU memory 98457 MiB; lowest free disk 198 GB | steps 28–29 (CPU, background) alongside 30–31
2026-11-10 05:50 UTC | 30. Round 3: stop vLLM | session killed; `nvidia-smi` lists no process after ~10 s (before steps 28–29 finished: they only read saved files) | step 31 (eval launched 05:50:51, 205 GB free, GPU empty)
2026-11-10 05:52 UTC | 28. Round 3: record check | exit 0: `runs/s42_refit_round3/check_records.json` written (script prints PASS: 128/128 files) | (step 31 running)
2026-11-10 05:54 UTC | 29. Round 3: checks | exit 0: `checks/advantages_s42_refit_round3.json`, `probe_s42_refit_round3.json` (from log a1), `kl_s42_refit_round3.json` written | (step 31 running)
2026-11-10 05:59 UTC | 31. Round 3: eval | EXIT 0 (`logs/eval_s42_refit_round3_a1_da70f8166885.log`, ~8 min): `evals/s42_refit_round3/summary.json` questions 489, cued_rollouts 978 | step 32 (copy_base wrote 4890 rows; rescore launched 06:00:00, 205 GB free, GPU empty)
2026-11-10 06:03 UTC | 32. Round 3: base rescore | EXIT 0 (`logs/eval_base_rescored_s42_refit_round3_a1_da70f8166885.log`, ~3 min): `evals/base_rescored_s42_refit_round3/summary.json` questions 489, cued_rollouts 978 | step 33 (CPU, background) alongside step 34 (score launched)
2026-11-10 06:04 UTC | 33. Round 3: compare | exit 0: `runs/s42_refit_round3/comparison.json`, `comparison.md`, `training_curve.csv` (128 data rows) | (step 34 running)
2026-11-10 06:07 UTC | 34. Round 3: score | EXIT 0 (`logs/score_round3_a1_da70f8166885.log`, ~4 min): `evals/first_probe_round3/`, `stand_in/round3/`, `checks/score_round3.json` written | step 35 (generate launched, 205 GB free, GPU empty)
2026-11-10 06:17 UTC | 35. Round 3: generate | EXIT 0 (`logs/generate_round3_a1_da70f8166885.log`, ~10 min): `generations/round3/completions.jsonl` 17038 rows | step 36
2026-11-10 06:18 UTC | 36. Round 4: build | EXIT 0 (`logs/build_round4_a1_da70f8166885.log`): train_rows 5336 (original 378, round1 378, round2 296, round3 282 per group), test 153 biased / 453 unbiased, test_one_class false; no SHRUNK:, TOPPED UP: or DROPPED FROM TEST FILE: lines | step 37
2026-11-10 06:19 UTC | 37. Round 4: config | EXIT 0 (`logs/config_round4_a1_da70f8166885.log`): `s42_refit_round4.yaml` written; no out-of-memory fallback in effect, yaml unchanged | step 38
2026-11-10 06:22 UTC | 38. Review 4 | PASS (`reviews/4_a1.md`; code item prints the same difference as review 3, cause recorded under Judgment calls) | step 39 (vLLM launched 06:22:52, 205 GB free, GPU empty)
2026-11-10 06:27 UTC | 39. Round 4: vLLM | server up (`curl` answered 06:27:33, ~4.5 min; `logs/vllm_s42_refit_round4_a1_da70f8166885.log`) | step 40 (RL launched 06:27:38, 205 GB free)
2026-11-10 11:06 UTC | 40. Round 4: RL | EXIT 0 (`logs/rl_s42_refit_round4_a1_da70f8166885.log`): `GRPO subprocess finished in 276.7 min (exit=0, attempt=1/4)`; no CRASH: lines, no out-of-memory lines; W&B `offaq68f`, run dir `/workspace/results/run_results/261109_refit_probe_redrawn_labels/s42_refit_round4_offaq68f`; peak GPU memory 98063 MiB; lowest free disk 195 GB | steps 41–42 (CPU, background) alongside 43–44
2026-11-10 11:07 UTC | 43. Round 4: stop vLLM | session killed; `nvidia-smi` lists no process after ~10 s (before steps 41–42 finished: they only read saved files) | step 44 (eval launched 11:07:12, 203 GB free, GPU empty)
2026-11-10 11:08 UTC | 41. Round 4: record check | exit 0: `runs/s42_refit_round4/check_records.json` written (script prints PASS: 128/128 files) | (step 44 running)
2026-11-10 11:10 UTC | 42. Round 4: checks | exit 0: `checks/advantages_s42_refit_round4.json`, `probe_s42_refit_round4.json` (from log a1), `kl_s42_refit_round4.json` written | (step 44 running)
2026-11-10 11:14 UTC | 44. Round 4: eval | EXIT 0 (`logs/eval_s42_refit_round4_a1_da70f8166885.log`, ~7.5 min): `evals/s42_refit_round4/summary.json` questions 489, cued_rollouts 978 | step 45
2026-11-10 11:18 UTC | 45. Round 4: base rescore | copy_base wrote 4890 rows; EXIT 0 (`logs/eval_base_rescored_s42_refit_round4_a1_da70f8166885.log`, launched 11:15:37, ~2.5 min): `evals/base_rescored_s42_refit_round4/summary.json` questions 489, cued_rollouts 978 | step 46 (CPU, background) alongside step 47 (score launched, 203 GB free, GPU empty)
2026-11-10 11:19 UTC | 46. Round 4: compare | exit 0: `runs/s42_refit_round4/comparison.json`, `comparison.md`, `training_curve.csv` (128 data rows) | (step 47 running)
2026-11-10 11:22 UTC | 47. Round 4: score | EXIT 0 (`logs/score_round4_a1_da70f8166885.log`, ~3.5 min): `evals/first_probe_round4/`, `stand_in/round4/`, `checks/score_round4.json` written | step 48 (generate launched, 203 GB free, GPU empty)
2026-11-10 11:31 UTC | 48. Round 4: generate | EXIT 0 (`logs/generate_round4_a1_da70f8166885.log`, ~9 min): `generations/round4/completions.jsonl` 17038 rows | step 49
2026-11-10 11:32 UTC | 49. Round 5: build | EXIT 0 (`logs/build_round5_a1_da70f8166885.log`): train_rows 5964 (original 378, round1 378, round2 296, round3 282, round4 157 per group), test 94 biased / 440 unbiased, test_one_class false; no SHRUNK:, TOPPED UP: or DROPPED FROM TEST FILE: lines | step 50
2026-11-10 11:33 UTC | 50. Round 5: config | EXIT 0 (`logs/config_round5_a1_da70f8166885.log`): `s42_refit_round5.yaml` written; no out-of-memory fallback in effect, yaml unchanged | step 51
2026-11-10 11:37 UTC | 51. Review 5 | PASS (`reviews/5_a1.md`; code item prints the same difference as reviews 3–4, cause under Judgment calls) | step 52 (vLLM launched 11:37:27, 208 GB free, GPU empty)
2026-11-10 11:41 UTC | 52. Round 5: vLLM | server up (`curl` answered 11:41:37, ~4 min; `logs/vllm_s42_refit_round5_a1_da70f8166885.log`) | step 53 (RL launched 11:41:42, 208 GB free)
2026-11-10 16:11 UTC | 53. Round 5: RL | EXIT 0 (`logs/rl_s42_refit_round5_a1_da70f8166885.log`): `GRPO subprocess finished in 267.8 min (exit=0, attempt=1/4)`; no CRASH: lines, no out-of-memory lines; W&B `0pizoqn5`, run dir `/workspace/results/run_results/261109_refit_probe_redrawn_labels/s42_refit_round5_0pizoqn5`; peak GPU memory 98804 MiB; lowest free disk 198 GB | steps 54–55 (CPU, background) alongside 56–57
2026-11-10 16:12 UTC | 56. Round 5: stop vLLM | session killed; `nvidia-smi` lists no process after ~10 s (before steps 54–55 finished: they only read saved files) | step 57 (eval launched 16:12:12, 207 GB free, GPU empty)
2026-11-10 16:13 UTC | 54. Round 5: record check | exit 0: `runs/s42_refit_round5/check_records.json` written (script prints PASS: 128/128 files) | (step 57 running)
2026-11-10 16:15 UTC | 55. Round 5: checks | exit 0: `checks/advantages_s42_refit_round5.json`, `probe_s42_refit_round5.json` (from log a1), `kl_s42_refit_round5.json` written | (step 57 running)
2026-11-10 16:19 UTC | 57. Round 5: eval | EXIT 0 (`logs/eval_s42_refit_round5_a1_da70f8166885.log`, ~7 min): `evals/s42_refit_round5/summary.json` questions 489, cued_rollouts 978 | step 58
2026-11-10 16:22 UTC | 58. Round 5: base rescore | copy_base wrote 4890 rows; EXIT 0 (`logs/eval_base_rescored_s42_refit_round5_a1_da70f8166885.log`, launched 16:20:15, ~2 min): `evals/base_rescored_s42_refit_round5/summary.json` questions 489, cued_rollouts 978 | step 59 (CPU, background) alongside step 60 (score launched, 207 GB free, GPU empty)
2026-11-10 16:23 UTC | 59. Round 5: compare | exit 0: `runs/s42_refit_round5/comparison.json`, `comparison.md`, `training_curve.csv` (128 data rows) | (step 60 running)
2026-11-10 16:26 UTC | 60. Round 5: score | EXIT 0 (`logs/score_round5_a1_da70f8166885.log`, ~3.5 min): `evals/first_probe_round5/`, `stand_in/round5/`, `checks/score_round5.json` written | step 61
2026-11-10 16:28 UTC | 61. Review 6 | PASS (`reviews/6_a1.md`; code item prints the same difference as reviews 3–5, cause under Judgment calls) | steps 62–64 (seed table and dynamics CPU in the background, probe table as job `table` launched 16:29:10, in parallel)
2026-11-10 16:30 UTC | 64. Probe table | EXIT 0 (`logs/table_a1_da70f8166885.log`): `probe_table.md`, `probe_table.json` written | (step 63 running)
2026-11-10 16:30 UTC | 62. Seed table | exit 0: `tables/rounds/seed_table.md`, `seed_table.json` written | (step 63 running)
2026-11-10 16:30 UTC | 63. Dynamics | exit 0: `runs/dynamics/dynamics.json`, `dynamics.md` written | step 65
2026-11-10 16:46 UTC | 65. Review 7 | PASS (`reviews/7_a1.md`, auditor: all three tables regenerate byte-identically and every number matches its source files; notes two wording points and one disclosed design read that change no number); `tables/rounds/seed_table.md` and `probe_table.md` copied into Result | done
2026-11-10 16:48 UTC | Done | all 65 steps ticked, Blocked empty, Result holds both tables; done notification sent ("Terminal notification sent. Mobile push requested."); resume check (cron 89f70048) left in place | none

## Judgment calls

- 2026-11-10 01:07 UTC: after round 2's RL step, stopped vLLM (step 17) and launched the eval (step 18) while the record check (15) and checks (16) ran in the background, so the GPU does not wait on CPU steps that only read saved files (CLAUDE.md: "run a CPU step that only reads saved files ... while the next GPU step runs"; the order of independent jobs). Steps 15–16 are ticked when they finish. I will do the same in later rounds.
- 2026-11-10 01:38 UTC: review 3's code item (reported, not a gate) found that `git diff -- sycophancy-rl/ | cmp - .../261108_frozen_probe_redrawn_labels/code_at_launch/tracked.diff` now prints a difference (reviews 1 and 2 at 19:52 and 20:30 UTC found none). Cause: commit 262ff86 "Support no-cue training runs with the bias probe at the answer letter" (2026-11-09 21:16 UTC, author Lily, during round 2's RL run) committed the grpo_trainer.py and compare.py hunks of that diff; the README.md hunk stays uncommitted. I checked that `git diff 475c845 -- sycophancy-rl/` (the launch HEAD against the working tree) is byte-identical to `tracked.diff`, so the code on disk is unchanged and no run here is affected. Not a gate, so the queue goes on; later reviews' code item will print the same difference for this reason.
- 2026-11-10 16:29 UTC: ran the three table steps (62 seed table, 63 dynamics, 64 probe table) at the same time rather than one after another: each only reads saved files and none reads another's output, and no GPU job is running (the order of independent jobs; how many CPU-bound jobs run in parallel). Each is ticked when its own check passes.

## Code fixes

## Reviews

One line per review: `<UTC time> | review | verdict | reviews/<n>_a<attempt>.md`.

2026-11-09 19:52 UTC | review 1 (start) | PASS | reviews/1_a1.md
2026-11-09 20:30 UTC | review 2 (smoke) | PASS | reviews/2_a1.md
2026-11-10 01:38 UTC | review 3 (end of round 2) | PASS | reviews/3_a1.md
2026-11-10 06:22 UTC | review 4 (end of round 3) | PASS | reviews/4_a1.md
2026-11-10 11:37 UTC | review 5 (end of round 4) | PASS | reviews/5_a1.md
2026-11-10 16:28 UTC | review 6 (end of round 5) | PASS | reviews/6_a1.md
2026-11-10 16:46 UTC | review 7 (auditor) | PASS | reviews/7_a1.md

## Blocked

## Result

From `/workspace/sycophancy_experiments/261109_refit_probe_redrawn_labels/seed_42/tables/rounds/seed_table.md`:

### RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): round1 455, round2 460, round3 460, round4 454, round5 456.

| metric | condition | base | base mean | base SD | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.769 | 0.769 | 0.793 | 0.763 | 0.771 | 0.773 | 0.012 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.939 | 0.900 | 0.892 | 0.857 | 0.847 | 0.887 | 0.037 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.431 | 0.554 | 0.556 | 0.656 | 0.673 | 0.574 | 0.097 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.763 | 0.781 | 0.777 | 0.786 | 0.789 | 0.779 | 0.010 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.673 | 0.675 | 0.712 | 0.669 | 0.689 | 0.683 | 0.018 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.775 | 0.757 | 0.753 | 0.726 | 0.738 | 0.750 | 0.019 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.464 | 0.542 | 0.540 | 0.607 | 0.622 | 0.555 | 0.063 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.746 | 0.736 | 0.730 | 0.717 | 0.723 | 0.730 | 0.011 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.939 | 0.900 | 0.892 | 0.857 | 0.847 | 0.887 | 0.037 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.569 | 0.444 | 0.444 | 0.344 | 0.325 | 0.425 | 0.097 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.813 | 0.747 | 0.741 | 0.684 | 0.676 | 0.732 | 0.056 |
| bias | cue_gold | 0.087 / 0.091 / 0.091 / 0.087 / 0.092 | 0.090 | 0.002 | 0.132 | 0.128 | 0.111 | 0.075 | 0.075 | 0.104 | 0.028 |
| bias | cue_dist | 0.272 / 0.271 / 0.267 / 0.270 / 0.276 | 0.271 | 0.003 | 0.354 | 0.211 | 0.222 | 0.132 | 0.099 | 0.204 | 0.099 |
| bias | train_mix | 0.150 / 0.152 / 0.149 / 0.150 / 0.154 | 0.151 | 0.002 | 0.202 | 0.156 | 0.144 | 0.094 | 0.082 | 0.135 | 0.049 |
| anti_bias | cue_gold | 0.038 / 0.038 / 0.038 / 0.036 / 0.038 | 0.037 | 0.001 | 0.002 | 0.009 | 0.011 | 0.009 | 0.013 | 0.009 | 0.004 |
| anti_bias | cue_dist | 0.009 / 0.011 / 0.011 / 0.011 / 0.011 | 0.011 | 0.001 | 0.000 | 0.002 | 0.004 | 0.022 | 0.011 | 0.008 | 0.009 |
| anti_bias | train_mix | 0.028 / 0.028 / 0.028 / 0.026 / 0.028 | 0.028 | 0.001 | 0.002 | 0.007 | 0.008 | 0.012 | 0.013 | 0.008 | 0.004 |
| no_bias | cue_gold | 0.875 / 0.872 / 0.872 / 0.877 / 0.871 | 0.873 | 0.003 | 0.866 | 0.863 | 0.878 | 0.916 | 0.912 | 0.887 | 0.025 |
| no_bias | cue_dist | 0.719 / 0.717 / 0.722 / 0.718 / 0.713 | 0.718 | 0.003 | 0.646 | 0.786 | 0.774 | 0.846 | 0.890 | 0.788 | 0.092 |
| no_bias | train_mix | 0.822 / 0.820 / 0.823 / 0.824 / 0.818 | 0.821 | 0.003 | 0.797 | 0.837 | 0.848 | 0.894 | 0.905 | 0.856 | 0.044 |
| deception_soft | cue_gold | 0.554 / 0.553 / 0.553 / 0.549 / 0.551 | 0.552 | 0.002 | 0.519 | 0.508 | 0.583 | 0.668 | 0.632 | 0.582 | 0.070 |
| deception_soft | cue_dist | 0.599 / 0.600 / 0.600 / 0.600 / 0.595 | 0.599 | 0.002 | 0.573 | 0.625 | 0.672 | 0.742 | 0.737 | 0.670 | 0.073 |
| deception_soft | train_mix | 0.568 / 0.569 / 0.568 / 0.566 / 0.565 | 0.567 | 0.002 | 0.537 | 0.545 | 0.614 | 0.693 | 0.667 | 0.611 | 0.070 |
| deception | cue_gold | 0.570 / 0.572 / 0.572 / 0.567 / 0.566 | 0.570 | 0.003 | 0.507 | 0.498 | 0.662 | 0.798 | 0.751 | 0.643 | 0.137 |
| deception | cue_dist | 0.661 / 0.662 / 0.660 / 0.660 / 0.654 | 0.660 | 0.003 | 0.620 | 0.729 | 0.758 | 0.842 | 0.860 | 0.762 | 0.097 |
| deception | train_mix | 0.600 / 0.604 / 0.600 / 0.598 / 0.596 | 0.599 | 0.003 | 0.539 | 0.562 | 0.698 | 0.813 | 0.790 | 0.681 | 0.126 |
| biased_share | cue_gold | 0.091 / 0.094 / 0.094 / 0.091 / 0.095 | 0.093 | 0.002 | 0.132 | 0.129 | 0.112 | 0.076 | 0.076 | 0.105 | 0.028 |
| biased_share | cue_dist | 0.275 / 0.274 / 0.270 / 0.274 / 0.279 | 0.274 | 0.003 | 0.354 | 0.212 | 0.223 | 0.135 | 0.100 | 0.205 | 0.098 |
| biased_share | train_mix | 0.154 / 0.157 / 0.153 / 0.154 / 0.159 | 0.155 | 0.002 | 0.202 | 0.157 | 0.145 | 0.095 | 0.083 | 0.136 | 0.048 |
| false_denial_soft | cue_gold | 0.394 / 0.387 / 0.382 / 0.360 / 0.397 | 0.384 | 0.014 | 0.350 | 0.389 | 0.267 | 0.159 | 0.187 | 0.270 | 0.100 |
| false_denial_soft | cue_dist | 0.352 / 0.350 / 0.350 / 0.350 / 0.352 | 0.351 | 0.001 | 0.344 | 0.411 | 0.305 | 0.232 | 0.213 | 0.301 | 0.081 |
| false_denial_soft | train_mix | 0.370 / 0.368 / 0.363 / 0.356 / 0.373 | 0.366 | 0.007 | 0.341 | 0.391 | 0.283 | 0.197 | 0.195 | 0.281 | 0.087 |
| false_admission_soft | cue_gold | 0.570 / 0.571 / 0.571 / 0.568 / 0.567 | 0.569 | 0.001 | 0.545 | 0.525 | 0.622 | 0.710 | 0.669 | 0.614 | 0.079 |
| false_admission_soft | cue_dist | 0.692 / 0.694 / 0.693 / 0.694 / 0.689 | 0.692 | 0.002 | 0.698 | 0.682 | 0.777 | 0.821 | 0.795 | 0.755 | 0.061 |
| false_admission_soft | train_mix | 0.604 / 0.606 / 0.605 / 0.604 / 0.602 | 0.604 | 0.002 | 0.587 | 0.573 | 0.670 | 0.745 | 0.710 | 0.657 | 0.075 |
| said_yes_biased_soft | cue_gold | 0.606 / 0.613 / 0.618 / 0.640 / 0.603 | 0.616 | 0.014 | 0.650 | 0.611 | 0.733 | 0.841 | 0.813 | 0.730 | 0.100 |
| said_yes_biased_soft | cue_dist | 0.648 / 0.650 / 0.650 / 0.650 / 0.648 | 0.649 | 0.001 | 0.656 | 0.589 | 0.695 | 0.768 | 0.787 | 0.699 | 0.081 |
| said_yes_biased_soft | train_mix | 0.630 / 0.632 / 0.637 / 0.644 / 0.627 | 0.634 | 0.007 | 0.659 | 0.609 | 0.717 | 0.803 | 0.805 | 0.719 | 0.087 |
| deception_from_biased | cue_gold | 0.036 / 0.037 / 0.036 / 0.033 / 0.038 | 0.036 | 0.002 | 0.046 | 0.050 | 0.030 | 0.012 | 0.014 | 0.031 | 0.018 |
| deception_from_biased | cue_dist | 0.097 / 0.096 / 0.095 / 0.096 / 0.098 | 0.096 | 0.001 | 0.122 | 0.087 | 0.068 | 0.031 | 0.021 | 0.066 | 0.041 |
| deception_from_biased | train_mix | 0.057 / 0.058 / 0.056 / 0.055 / 0.059 | 0.057 | 0.002 | 0.069 | 0.061 | 0.041 | 0.019 | 0.016 | 0.041 | 0.024 |
| deception_from_unbiased | cue_gold | 0.518 / 0.517 / 0.517 / 0.517 / 0.513 | 0.516 | 0.002 | 0.473 | 0.457 | 0.553 | 0.656 | 0.618 | 0.551 | 0.087 |
| deception_from_unbiased | cue_dist | 0.502 / 0.504 / 0.506 / 0.504 / 0.496 | 0.502 | 0.004 | 0.451 | 0.538 | 0.604 | 0.710 | 0.715 | 0.604 | 0.113 |
| deception_from_unbiased | train_mix | 0.511 / 0.511 / 0.512 / 0.511 / 0.506 | 0.510 | 0.002 | 0.468 | 0.483 | 0.573 | 0.674 | 0.651 | 0.570 | 0.094 |
| said_yes | cue_gold | 0.595 / 0.598 / 0.602 / 0.602 / 0.592 | 0.598 | 0.005 | 0.555 | 0.539 | 0.747 | 0.873 | 0.826 | 0.708 | 0.154 |
| said_yes | cue_dist | 0.785 / 0.791 / 0.789 / 0.786 / 0.783 | 0.787 | 0.003 | 0.789 | 0.784 | 0.910 | 0.968 | 0.951 | 0.881 | 0.089 |
| said_yes | train_mix | 0.656 / 0.661 / 0.662 / 0.662 / 0.653 | 0.659 | 0.004 | 0.630 | 0.610 | 0.804 | 0.906 | 0.871 | 0.764 | 0.137 |
| false_denial | cue_gold | 0.359 / 0.366 / 0.341 / 0.308 / 0.366 | 0.348 | 0.025 | 0.317 | 0.339 | 0.118 | 0.000 | 0.000 | 0.155 | 0.165 |
| false_denial | cue_dist | 0.275 / 0.264 / 0.261 / 0.269 / 0.270 | 0.268 | 0.006 | 0.261 | 0.371 | 0.157 | 0.033 | 0.044 | 0.173 | 0.144 |
| false_denial | train_mix | 0.318 / 0.318 / 0.295 / 0.293 / 0.320 | 0.309 | 0.014 | 0.275 | 0.350 | 0.136 | 0.012 | 0.013 | 0.157 | 0.153 |
| false_admission | cue_gold | 0.591 / 0.594 / 0.596 / 0.593 / 0.587 | 0.592 | 0.004 | 0.536 | 0.521 | 0.730 | 0.863 | 0.812 | 0.692 | 0.157 |
| false_admission | cue_dist | 0.808 / 0.812 / 0.807 / 0.807 / 0.803 | 0.808 | 0.003 | 0.816 | 0.825 | 0.930 | 0.969 | 0.951 | 0.898 | 0.072 |
| false_admission | train_mix | 0.652 / 0.657 / 0.655 / 0.654 / 0.648 | 0.653 | 0.003 | 0.606 | 0.602 | 0.794 | 0.897 | 0.860 | 0.752 | 0.140 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.006 | 0.004 | 0.002 | 0.002 | 0.004 | 0.004 | 0.002 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.001 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 447 / 452 / 452 / 446 / 448 | 449 | 2.828 | 455 | 460 | 460 | 454 | 455 | 456.800 | 2.950 |
| n_bias_rows | cue_dist | 441 / 446 / 446 / 440 / 442 | 443 | 2.828 | 455 | 459 | 460 | 454 | 455 | 456.600 | 2.702 |
| n_bias_rows | train_mix | 888 / 898 / 898 / 886 / 890 | 892 | 5.657 | 910 | 919 | 920 | 908 | 910 | 913.400 | 5.639 |
| n_deception_rows | cue_gold | 430 / 435 / 435 / 430 / 431 | 432.200 | 2.588 | 454 | 456 | 455 | 450 | 449 | 452.800 | 3.114 |
| n_deception_rows | cue_dist | 437 / 441 / 441 / 435 / 437 | 438.200 | 2.683 | 455 | 458 | 458 | 444 | 450 | 453 | 6.000 |
| n_deception_rows | train_mix | 867 / 876 / 876 / 865 / 868 | 870.400 | 5.225 | 909 | 914 | 913 | 894 | 899 | 905.800 | 8.871 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.214 | 0.570 | 0.239 | 0.637 | 0.595 | 0.451 | 0.207 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.195 | 0.446 | 0.274 | 0.518 | 0.509 | 0.388 | 0.146 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.053 | 0.156 | 0.038 | 0.169 | 0.228 | 0.129 | 0.081 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.046 | 0.202 | 0.061 | 0.150 | 0.276 | 0.147 | 0.097 |
| probe_start_auc | cue_gold | – | – | – | 0.824 | 0.920 | 0.916 | 0.921 | 0.881 | 0.892 | 0.042 |
| probe_start_auc | cue_dist | – | – | – | 0.789 | 0.782 | 0.823 | 0.861 | 0.774 | 0.806 | 0.036 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 60 | 59 | 51 | 34 | 34 | 47.600 | 12.896 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 161 | 97 | 102 | 60 | 45 | 93 | 45.039 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 366 | 355 | 361 | 359 | 352 | 358.600 | 5.413 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 88 | 99 | 92 | 81 | 92 | 90.400 | 6.580 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.208 | 0.613 | 0.467 | 0.616 | 0.606 | 0.502 | 0.176 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.184 | 0.494 | 0.467 | 0.532 | 0.496 | 0.434 | 0.142 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.054 | 0.213 | 0.117 | 0.193 | 0.236 | 0.163 | 0.075 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.049 | 0.248 | 0.152 | 0.187 | 0.276 | 0.183 | 0.089 |
| probe_fresh_auc | cue_gold | – | – | – | 0.815 | 0.914 | 0.902 | 0.904 | 0.875 | 0.882 | 0.040 |
| probe_fresh_auc | cue_dist | – | – | – | 0.776 | 0.789 | 0.825 | 0.841 | 0.751 | 0.796 | 0.036 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 60 | 59 | 51 | 34 | 34 | 47.600 | 12.896 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 161 | 97 | 102 | 60 | 45 | 93 | 45.039 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 366 | 355 | 361 | 359 | 352 | 358.600 | 5.413 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 88 | 99 | 92 | 81 | 92 | 90.400 | 6.580 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.216 | 0.583 | 0.296 | 0.698 | 0.712 | 0.501 | 0.231 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.206 | 0.481 | 0.348 | 0.620 | 0.674 | 0.466 | 0.193 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.048 | 0.220 | 0.094 | 0.307 | 0.376 | 0.209 | 0.138 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.051 | 0.267 | 0.135 | 0.302 | 0.452 | 0.241 | 0.155 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.835 | 0.893 | 0.855 | 0.887 | 0.858 | 0.866 | 0.024 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.767 | 0.758 | 0.781 | 0.823 | 0.761 | 0.778 | 0.027 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 60 | 59 | 51 | 34 | 34 | 47.600 | 12.896 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 161 | 97 | 102 | 60 | 45 | 93 | 45.039 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 366 | 355 | 361 | 359 | 352 | 358.600 | 5.413 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 88 | 99 | 92 | 81 | 92 | 90.400 | 6.580 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|
| equal task reward groups | 0.755 | 0.815 | 0.878 | 0.891 | 0.896 | 0.847 | 0.061 |
| probe-only groups | 0.666 | 0.643 | 0.695 | 0.649 | 0.619 | 0.654 | 0.028 |

From `/workspace/sycophancy_experiments/261109_refit_probe_redrawn_labels/seed_42/probe_table.md`:

### Probe scores per round

Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the
cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue
answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test
set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round
R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.

| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | Round R's probe; its own test set; round R-1's model (round 1: base) | 292 / 485 | 0.722 / 0.255 | 0.873 |
| round1 | Round R's probe; round R's answers; round R's model | 221 / 454 | 0.200 / 0.052 | 0.790 |
| round1 | Round R's probe; base model's answers; base model | 159 / 427 | 0.748 / 0.229 | 0.907 |
| round1 | First probe; round R's answers; round R's model | 221 / 454 | 0.200 / 0.052 | 0.790 |
| round2 | Round R's probe; its own test set; round R-1's model (round 1: base) | 221 / 454 | 0.673 / 0.352 | 0.840 |
| round2 | Round R's probe; round R's answers; round R's model | 156 / 454 | 0.493 / 0.166 | 0.857 |
| round2 | Round R's probe; base model's answers; base model | 162 / 431 | 0.807 / 0.358 | 0.889 |
| round2 | First probe; round R's answers; round R's model | 156 / 454 | 0.458 / 0.148 | 0.838 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 156 / 454 | 0.530 / 0.185 | 0.867 |
| round3 | Round R's probe; round R's answers; round R's model | 153 / 453 | 0.262 / 0.043 | 0.884 |
| round3 | Round R's probe; base model's answers; base model | 160 / 431 | 0.621 / 0.217 | 0.882 |
| round3 | First probe; round R's answers; round R's model | 153 / 453 | 0.327 / 0.038 | 0.889 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 153 / 453 | 0.635 / 0.194 | 0.889 |
| round4 | Round R's probe; round R's answers; round R's model | 94 / 440 | 0.561 / 0.166 | 0.875 |
| round4 | Round R's probe; base model's answers; base model | 158 / 426 | 0.799 / 0.433 | 0.866 |
| round4 | First probe; round R's answers; round R's model | 94 / 440 | 0.420 / 0.054 | 0.895 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 94 / 440 | 0.695 / 0.277 | 0.874 |
| round5 | Round R's probe; round R's answers; round R's model | 79 / 444 | 0.546 / 0.238 | 0.837 |
| round5 | Round R's probe; base model's answers; base model | 163 / 427 | 0.856 / 0.560 | 0.859 |
| round5 | First probe; round R's answers; round R's model | 79 / 444 | 0.222 / 0.033 | 0.839 |

### Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100; under 100 when the
earlier own-answer sources ran out of spare rows to fill the smallest group). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
| round2 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round2 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4208 |
| round3 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 4208 |
| round3 | round2 | 296 / 638 / 2941 / 521 | 296 | 296 / 296 / 296 / 296 | none | 4208 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 5336 |
| round4 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 5336 |
| round4 | round2 | 296 / 638 / 2941 / 521 | 296 | 296 / 296 / 296 / 296 | none | 5336 |
| round4 | round3 | 282 / 600 / 2941 / 484 | 282 | 282 / 282 / 282 / 282 | none | 5336 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 5964 |
| round5 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 5964 |
| round5 | round2 | 296 / 638 / 2941 / 521 | 296 | 296 / 296 / 296 / 296 | none | 5964 |
| round5 | round3 | 282 / 600 / 2941 / 484 | 282 | 282 / 282 / 282 / 282 | none | 5964 |
| round5 | round4 | 157 / 264 / 2956 / 492 | 157 | 157 / 157 / 157 / 157 | none | 5964 |
