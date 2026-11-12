# Runs

State file for the unattended run of the seed-43 line (pod de0ea2293f40). Read this first after any restart.

## Queue

- [x] 1. Pod check
- [x] 2. Round 1 regenerated on this pod (reported)
- [x] 3. config 2
- [x] 4. Review 1
- [x] 5. vLLM for smoke
- [x] 6. Smoke RL run
- [x] 7. Smoke record check
- [x] 8. Stop vLLM
- [x] 9. Smoke checks
- [x] 10. Penalty check (smoke)
- [x] 11. Review 2
- [x] 12. Round 2: vLLM
- [x] 13. Round 2: RL run
- [x] 14. Round 2: record check
- [x] 15. Round 2: checks
- [x] 16. Round 2: stop vLLM
- [x] 17. Round 2: links 2
- [x] 18. Round 2: final eval
- [x] 19. Round 2: compare
- [x] 20. Round 2: config 3
- [x] 21. Review 3 (end of round 2)
- [x] 22. Round 3: vLLM
- [x] 23. Round 3: RL run
- [x] 24. Round 3: record check
- [x] 25. Round 3: checks
- [x] 26. Round 3: stop vLLM
- [x] 27. Round 3: links 3
- [x] 28. Round 3: final eval
- [x] 29. Round 3: compare
- [x] 30. Round 3: config 4
- [x] 31. Review 4 (end of round 3)
- [x] 32. Round 4: vLLM
- [x] 33. Round 4: RL run
- [x] 34. Round 4: record check
- [x] 35. Round 4: checks
- [x] 36. Round 4: stop vLLM
- [x] 37. Round 4: links 4
- [x] 38. Round 4: final eval
- [x] 39. Round 4: compare
- [x] 40. Round 4: config 5
- [x] 41. Review 5 (end of round 4)
- [x] 42. Round 5: vLLM
- [x] 43. Round 5: RL run
- [x] 44. Round 5: record check
- [x] 45. Round 5: checks
- [x] 46. Round 5: stop vLLM
- [x] 47. Round 5: links 5
- [x] 48. Round 5: final eval
- [x] 49. Round 5: compare
- [x] 50. Review 6 (end of round 5)
- [x] 51. Round table (seed_table.py)
- [x] 52. Dynamics (dynamics.py)
- [x] 53. Review 7 (auditor) and Result

## Log

One line per step: `<UTC time> | step | outcome | next`.

- 2026-11-07 21:47 | 1. Pod check | PASS: de0ea2293f40, one H100 80GB HBM3, no GPU process, 78G free; frozen probe md5 = the 8 recorded lines | step 2
- 2026-11-07 21:52 | 2. Round 1 regenerated (reported) | job EXIT 0; checks/round1_repro.json: rows 500, questions 50, same_completion 208, same_pred 472 (261104's seed-43 check on pod 09f52c746bb4: 500, 50, 212, 473) | step 3
- 2026-11-07 21:52 | 3. config 2 | job EXIT 0; wrote s43_frozen_round2.yaml and smoke_s43_frozen_round2.yaml | step 4 (review 1)
- 2026-11-07 21:54 | 4. Review 1 | PASS: (a) round 2 config + smoke config, (b) diff against 261104 only the allowed lines, (c) frozen probe md5 identical | step 5
- 2026-11-07 21:57 | 5. vLLM for smoke | answered after 191 s (logs/vllm_smoke_s43_frozen_round2_a1_de0ea2293f40.log); GPU 30241 MiB used | step 6
- 2026-11-07 22:15 | 6. Smoke RL run | EXIT 0 (attempt 1, launcher try 1/4), W&B nwl72ltg, `GRPO subprocess finished in 16.1 min (exit=0, attempt=1/4)`; no CRASH: lines; peak GPU 75733 MiB; lowest free disk 73G | step 7
- 2026-11-07 22:16 | 8. Stop vLLM | killed 261107-ctl43-vllm; nvidia-smi lists no process after ~10 s | steps 7 and 9 running (CPU), step 10 next
- 2026-11-07 22:17 | 7. Smoke record check | exit 0; runs/smoke_s43_frozen_round2/check_records.json PASS true, n_files 2, seed 43 | steps 9 and 10 running
- 2026-11-07 22:17 | 9. Smoke checks | exit 0; reward: penalty_weight 1.0, mismatches 0/0, reward diff 4.5e-8 / 4.1e-8; advantages: floor 0.3, 2 steps, max diff 1.7e-8, kl_div_file_step_0 0.1139; KL file: loss.kl 0.01, kl_coeff 0.01, 2 steps | step 10 running
- 2026-11-07 22:18 | 10. Penalty check (smoke) | job EXIT 0; 511 of 512 step-0 answers scored; round 1's probe vs saved median 0.00058 (max 0.0233, 1 over 0.02); trainer's unused start-of-run fit vs saved median 0.0189 (max 0.548, 250 over 0.02) | step 11 (review 2)
- 2026-11-07 22:20 | 11. Review 2 | PASS: rules (a)-(h); penalty median 0.00058 vs unused fit 0.0189; step-0 KL 0.114 | step 12
- 2026-11-07 22:23 | 12. Round 2: vLLM | answered after 181 s (logs/vllm_s43_frozen_round2_a1_de0ea2293f40.log); GPU 30263 MiB; 76G free | step 13
- 2026-11-08 01:34 | 13. Round 2: RL run | EXIT 0 (attempt 1, launcher try 1/4), W&B o7dxeb0g, `GRPO subprocess finished in 188.9 min (exit=0, attempt=1/4)`; no CRASH: lines; peak GPU 78919 MiB; lowest free disk 64G | steps 14, 15 (CPU), 16
- 2026-11-08 01:34 | 16. Round 2: stop vLLM | killed 261107-ctl43-vllm; nvidia-smi lists no process after ~10 s | 14, 15 running; step 17
- 2026-11-08 01:35 | 14. Round 2: record check | exit 0; runs/s43_frozen_round2/check_records.json PASS true, n_files 128, seed 43 | 15 running
- 2026-11-08 01:35 | 15. Round 2: checks | exit 0; advantages: floor 0.3, 128 steps, max diff 3.1e-8, kl_div_file_step_0 0.0980; KL file: loss.kl 0.01, kl_coeff 0.01, 128 steps, last kl_div 0.113 | 18 running
- 2026-11-08 01:36 | 20. Round 2: config 3 | job EXIT 0; wrote s43_frozen_round3.yaml (checkpoint o7dxeb0g/checkpoints/final, frozen probe = round 1's) | 18 running, then 19
- 2026-11-08 01:41 | 18. Round 2: final eval | EXIT 0 (attempt 1); evals/s43_frozen_round2/summary.json questions 489, cued_rollouts 978; scored through links/round2 | step 19
- 2026-11-08 01:41 | 19. Round 2: compare | exit 0; runs/s43_frozen_round2/comparison.md, comparison.json, training_curve.csv (128 data rows) | step 21 (review 3)
- 2026-11-08 01:44 | 21. Review 3 | PASS: rules (a)-(i); 82.8 s/step (261104 round 2: 80.3); step-0 KL 0.098, last-16 mean 0.121 | step 22
- 2026-11-08 01:47 | 22. Round 3: vLLM | answered after 175 s (logs/vllm_s43_frozen_round3_a1_de0ea2293f40.log); GPU 30263 MiB; 69G free | step 23
- 2026-11-08 05:01 | 23. Round 3: RL run | EXIT 0 (attempt 1, launcher try 1/4), W&B n1pu5kt0, `GRPO subprocess finished in 192.3 min (exit=0, attempt=1/4)`; no CRASH: lines; peak GPU 77919 MiB; lowest free disk 61G | steps 24, 25 (CPU), 26
- 2026-11-08 05:02 | 26. Round 3: stop vLLM | killed 261107-ctl43-vllm; nvidia-smi lists no process after ~10 s | 24, 25 running; step 27
- 2026-11-08 05:02 | 24. Round 3: record check | exit 0; runs/s43_frozen_round3/check_records.json PASS true, n_files 128, seed 43 | 25, 27 running
- 2026-11-08 05:03 | 25. Round 3: checks | exit 0; advantages: floor 0.3, 128 steps, max diff 2.0e-8, kl_div_file_step_0 0.1373; KL file: loss.kl 0.01, kl_coeff 0.01, 128 steps, last kl_div 0.123 | 27 done
- 2026-11-08 05:03 | 27. Round 3: links 3 | job EXIT 0; links/round3/ start-of-run entry -> round 1's probe (hxpczsle), end-of-run entry -> n1pu5kt0's refit | step 28
- 2026-11-08 05:04 | 30. Round 3: config 4 | job EXIT 0; wrote s43_frozen_round4.yaml (checkpoint n1pu5kt0/checkpoints/final, frozen probe = round 1's) | 28 running, then 29
- 2026-11-08 05:09 | 28. Round 3: final eval | EXIT 0 (attempt 1); evals/s43_frozen_round3/summary.json questions 489, cued_rollouts 978; scored through links/round3 | step 29
- 2026-11-08 05:09 | 29. Round 3: compare | exit 0; runs/s43_frozen_round3/comparison.md, comparison.json, training_curve.csv (128 data rows) | step 31 (review 4)
- 2026-11-08 05:12 | 31. Review 4 | PASS: rules (a)-(i); 84.4 s/step (261104 round 3: 80.8); step-0 KL 0.137, last-16 mean 0.129 | step 32
- 2026-11-08 05:16 | 32. Round 4: vLLM | answered after 196 s (logs/vllm_s43_frozen_round4_a1_de0ea2293f40.log); GPU 30263 MiB; 66G free | step 33
- 2026-11-08 08:31 | 33. Round 4: RL run | EXIT 0 (attempt 1, launcher try 1/4), W&B 43szg2qi, `GRPO subprocess finished in 193.4 min (exit=0, attempt=1/4)`; no CRASH: lines; peak GPU 76763 MiB; lowest free disk 58G | steps 34, 35 (CPU), 36
- 2026-11-08 08:31 | 36. Round 4: stop vLLM | killed 261107-ctl43-vllm; nvidia-smi lists no process after ~10 s | 34, 35 running; step 37
- 2026-11-08 08:32 | 37. Round 4: links 4 | job EXIT 0; links/round4/ start-of-run entry -> round 1's probe (hxpczsle), end-of-run entry -> 43szg2qi's refit | step 38
- 2026-11-08 08:32 | 34. Round 4: record check | exit 0; runs/s43_frozen_round4/check_records.json PASS true, n_files 128, seed 43 | 35, 38, 40 running
- 2026-11-08 08:33 | 35. Round 4: checks | exit 0; advantages: floor 0.3, 128 steps, max diff 2.0e-8, kl_div_file_step_0 0.1302; KL file: loss.kl 0.01, kl_coeff 0.01, 128 steps, last kl_div 0.097 | 38, 40 running
- 2026-11-08 08:33 | 40. Round 4: config 5 | job EXIT 0; wrote s43_frozen_round5.yaml (checkpoint 43szg2qi/checkpoints/final, frozen probe = round 1's) | 38 running, then 39
- 2026-11-08 08:38 | 38. Round 4: final eval | EXIT 0 (attempt 1); evals/s43_frozen_round4/summary.json questions 489, cued_rollouts 978; scored through links/round4 | step 39
- 2026-11-08 08:38 | 39. Round 4: compare | exit 0; runs/s43_frozen_round4/comparison.md, comparison.json, training_curve.csv (128 data rows) | step 41 (review 5)
- 2026-11-08 08:41 | 41. Review 5 | PASS: rules (a)-(i); 85.0 s/step (261104 round 4: 77.8); step-0 KL 0.130, last-16 mean 0.102; 976 follow-ups (2 cued answers unparsed, by design) | step 42
- 2026-11-08 08:44 | 42. Round 5: vLLM | answered after 166 s (logs/vllm_s43_frozen_round5_a1_de0ea2293f40.log); GPU 30263 MiB; 64G free | step 43
- 2026-11-08 12:01 | 43. Round 5: RL run | EXIT 0 (attempt 1, launcher try 1/4), W&B mly9y5ro, `GRPO subprocess finished in 195.3 min (exit=0, attempt=1/4)`; no CRASH: lines; peak GPU 75422 MiB; lowest free disk 57G | steps 44, 45 (CPU), 46
- 2026-11-08 12:02 | 46. Round 5: stop vLLM | killed 261107-ctl43-vllm; nvidia-smi lists no process after ~10 s | 44, 45 running; step 47
- 2026-11-08 12:02 | 47. Round 5: links 5 | job EXIT 0; links/round5/ start-of-run entry -> round 1's probe (hxpczsle), end-of-run entry -> mly9y5ro's refit | step 48
- 2026-11-08 12:03 | 44. Round 5: record check | exit 0; runs/s43_frozen_round5/check_records.json PASS true, n_files 128, seed 43 | 48 running
- 2026-11-08 12:03 | 45. Round 5: checks | exit 0; advantages: floor 0.3, 128 steps, max diff 1.7e-8, kl_div_file_step_0 0.0940; KL file: loss.kl 0.01, kl_coeff 0.01, 128 steps, last kl_div 0.077 | 48 running, then 49
- 2026-11-08 12:08 | 48. Round 5: final eval | EXIT 0 (attempt 1); evals/s43_frozen_round5/summary.json questions 489, cued_rollouts 978; scored through links/round5 | step 49
- 2026-11-08 12:08 | 49. Round 5: compare | exit 0; runs/s43_frozen_round5/comparison.md, comparison.json, training_curve.csv (128 data rows) | step 50 (review 6)
- 2026-11-08 12:11 | 50. Review 6 | PASS: rules (a)-(h); 85.8 s/step (261104 round 5: 75.5); step-0 KL 0.094, last-16 mean 0.077 | steps 51, 52 (CPU, run together)
- 2026-11-08 12:12 | 51. Round table | exit 0; tables/rounds/seed_table.md and seed_table.json written | 52 running
- 2026-11-08 12:13 | 52. Dynamics | exit 0; runs/dynamics/dynamics.json and dynamics.md written | step 53 (review 7, auditor)
- 2026-11-08 12:23 | 53. Review 7 and Result | PASS: round table 1024/1024 cells and dynamics 228/228 lines matched, reruns byte-identical; one minor wording note (2.1); tables/rounds/seed_table.md copied into Result | queue done
- 2026-11-08 01:35 | 17. Round 2: links 2 | job EXIT 0; links/round2/ start-of-run entry -> round 1's probe (hxpczsle), end-of-run entry -> o7dxeb0g's refit | step 18

## Judgment calls

- 2026-11-07 21:51 UTC: started step 3's job (`rounds.py config 2`, CPU) while step 2's shell comparison (CPU, reads only
  saved answers) was still running. Both are independent CPU steps (CLAUDE.md allows changing their order); step 3 is
  ticked after step 2.
- 2026-11-07 22:16 UTC: steps 7 (record check) and 9 (smoke checks), both CPU and reading only saved files and W&B,
  run at the same time; step 8 (stop vLLM) ran while they ran, since neither needs vLLM; step 10 (penalty check, HF on
  the GPU) started after step 8 without waiting for 7 and 9, since it reads none of their outputs ("The GPU should
  never wait on a step that only reads saved files"). Each step is ticked when its own output is in.
- 2026-11-08 01:34 UTC, and the same for every later round: after the RL run, the record check and checks (CPU, read
  only saved files and W&B) run in the background while vLLM is stopped and the links job and final eval go ahead;
  the config step for the next round (CPU) runs once the job session is free. The review waits for every step's output.

## Pod

Recorded at step 1, 2026-11-07 21:46 UTC:
- `hostname`: de0ea2293f40
- `nvidia-smi -L`: GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-dc542855-fb67-1ba3-7e4f-6c58c52a6c26)
- `uname -r`: 6.8.0-147-generic
- `uptime -s`: 2026-10-20 04:28:30
- Driver 595.91.07, CUDA 13.2 (from `nvidia-smi`)
- `df -h /workspace`: 513G size, 435G used, 78G available (85%)
- `nvidia-smi`: no running processes
- `checks/frozen_probe_md5.txt`: the md5 rule's 8 lines, identical

## RL crashes

- smoke_s43_frozen_round2 attempt 1 (W&B nwl72ltg): no `CRASH:` line. `GRPO subprocess finished in 16.1 min (exit=0, attempt=1/4)`. Peak GPU memory 75733 MiB.
- s43_frozen_round2 attempt 1 (W&B o7dxeb0g): no `CRASH:` line. `GRPO subprocess finished in 188.9 min (exit=0, attempt=1/4)`. Peak GPU memory 78919 MiB; lowest free disk 64G.
- s43_frozen_round3 attempt 1 (W&B n1pu5kt0): no `CRASH:` line. `GRPO subprocess finished in 192.3 min (exit=0, attempt=1/4)`. Peak GPU memory 77919 MiB; lowest free disk 61G.
- s43_frozen_round4 attempt 1 (W&B 43szg2qi): no `CRASH:` line. `GRPO subprocess finished in 193.4 min (exit=0, attempt=1/4)`. Peak GPU memory 76763 MiB; lowest free disk 58G.
- s43_frozen_round5 attempt 1 (W&B mly9y5ro): no `CRASH:` line. `GRPO subprocess finished in 195.3 min (exit=0, attempt=1/4)`. Peak GPU memory 75422 MiB; lowest free disk 57G.

## Code fixes

## Reviews

### Review 1 (step 4), 2026-11-07 21:54 UTC: PASS (fresh general-purpose agent)

Review 1 (start, step 4), seed-43 line of 261107_frozen_probe_kl_0_01. I read every file myself and changed nothing.

**(a) Round 2's config: PASS**
I parsed /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/s43_frozen_round2.yaml with yaml and checked each value of the config rule (Q = 2):
- `run_key`: `s43_frozen_round2`. `probe.directory_save_prefix`: `s43_frozen_round2`.
- `output_dir`: `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43`.
- `group`: `261107_frozen_probe_kl_0_01`.
- `checkpoint`: `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/final`, which is round 1's run folder + `/checkpoints/final`. The folder exists.
- Overrides required by the rule, all matching in value and type:
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3
  - `training.seed` 43, `training.max_steps` 128
  - `training.batch_size` 4 and `training.grad_accum_steps` 16. This is the right pair: Judgment calls in runs.md holds only one entry, about running step 3 alongside step 2. It records no out-of-memory fallback.
  - `++grpo.model.kl_reference_from_base` true
  - `data.probe_dataset.bias_probe_train_jsonl` `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl`
  - `data.probe_dataset.bias_probe_eval_jsonl` `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_eval.jsonl`
  - `detector.frozen_probe_weights_dir` is the full frozen-probe path, `.../kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`.
- `data.probe_dataset.max_train_examples` is absent.
- The config's override keys are the same set as round_template.yaml's, in the same order. The 9 remaining overrides equal the template's:
  - `+data/probe_dataset` sycophancy_bias_probe, `task_args.bias_probe_at_answer` true, `model.activation_matching_layers` range_13-17
  - `detector.filter_layers_below_loss_threshold` 'null', `detector.pretrain_cfg.learning_rate` 0.03, `detector.pretrain_cfg.max_steps` 2000
  - `data.task_dataset.mmlu_questions_jsonl` .../sycophancy_setup/data/questions.jsonl, `task_args.cue_gold_prob` 0.75, `detector.model_batch_size` 8
- `config_name` grpo_mcq and `probe.retrain` true also match the template.

Smoke config: `diff s43_frozen_round2.yaml smoke_s43_frozen_round2.yaml` shows only:
- header comment lines 1 and 3 ("(smoke run)" added, and the run name in the command line)
- `run_key: smoke_s43_frozen_round2`
- `directory_save_prefix: smoke_s43_frozen_round2`
- `training.max_steps: 128` changed to `2`
- an added `training.eval_steps: 1`

A parsed comparison of the two files finds exactly these four non-comment differences.

**(b) Against 261104: PASS**
`diff .../261104_iterative_kl_0_01/seed_43/s43_round2.yaml .../261107_frozen_probe_kl_0_01/seed_43/s43_frozen_round2.yaml` shows only these hunks, all of them allowed:
- line 3, the header comment's run line: `.../261104_iterative_kl_0_01/seed_43 s43_round2` changed to `.../261107_frozen_probe_kl_0_01/seed_43 s43_frozen_round2`
- lines 5–7: `run_key` `s43_round2` → `s43_frozen_round2`; `output_dir` 261104 folder → 261107 folder; `group` `261104_iterative_kl_0_01` → `261107_frozen_probe_kl_0_01`
- line 10: `directory_save_prefix` `s43_round2` → `s43_frozen_round2`
- lines 30–32:
  - both probe-file paths change from `261104.../probe_sets/round2/{train,test}.jsonl` to `sycophancy_setup/data/bias_probe_{train,eval}.jsonl`
  - `data.probe_dataset.max_train_examples: 6048` is removed
  - `detector.frozen_probe_weights_dir: <frozen>` is added

No other line differs.

**(c) Frozen probe: PASS**
- /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/checks/frozen_probe_md5.txt has 8 lines, and `cmp` against the rule's 8 lines finds them byte-identical (layer_13 model.pt e0ea6af9…, layer_14 1e0b02b9…, layer_15 bc9819a9…, layer_16 2e6f1cfb…, every config.json f13b24b2…).
- Extra check, not required: I reran the md5 command on <frozen> now, and its output is also byte-identical to the file.

**Reported (no verdict)**
checks/round1_repro.json against 261104's seed-43 check on pod 09f52c746bb4. I also read 261104's own file, /workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_43/checks/round1_repro.json, and it holds the stated numbers.

| | This pod (de0ea2293f40) | 261104, pod 09f52c746bb4 |
|---|---|---|
| rows | 500 | 500 |
| questions | 50 | 50 |
| same completion | 208 | 212 |
| same answer (`same_pred`) | 472 | 473 |

**Overall verdict: PASS.** Rules (a), (b) and (c) all pass.

### Review 2 (step 11), 2026-11-07 22:20 UTC: PASS (fresh general-purpose agent)

Review 2 (smoke, step 11), seed-43 line of 261107_frozen_probe_kl_0_01. I read every file myself and changed nothing.

**(a) Step files: PASS**
/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/runs/smoke_s43_frozen_round2/check_records.json has `"PASS": true`, `"n_files": 2` and `"seed": 43`. Both per-step entries also say PASS true.

**(b) Reward: PASS**
checks/reward_smoke_s43_frozen_round2.json has `penalty_weight` 1.0.
- File step 0: `probe_score_present_iff_parsed_mismatches` 0, `reward_mean_abs_difference` 4.49e-8.
- File step 1: mismatches 0, `reward_mean_abs_difference` 4.10e-8.
- Both differences are at most 0.001.

**(c) Advantage and KL: PASS**
- checks/advantages_smoke_s43_frozen_round2.json: `advantage_std_floor` 0.3, `steps_compared` 2, `max_abs_difference` 1.71e-8 (at most 1e-4), `kl_div_file_step_0` 0.1139 (at least 0.01).
- checks/kl_smoke_s43_frozen_round2.json: `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 2.

**(d) Swap: PASS**
The log is logs/rl_smoke_s43_frozen_round2_a1_de0ea2293f40.log (2802 lines, ends `EXIT 0`).
- The line `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/final/lora` appears once, at line 96, so the try that finished starts there.
- After it, line 1018 is exactly `Phase-0 override: replacing detector with frozen probe weights from <frozen>`, and line 1019 is exactly `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`. I checked both for exact match with no trailing text.
- The `detector.frozen_probe_weights_dir` in smoke_s43_frozen_round2.yaml, parsed with yaml, equals <frozen> exactly.

**(e) Penalty check: PASS**
In checks/penalty_smoke_s43_frozen_round2.json:
- `penalty_probe` equals <frozen> exactly.
- `other_probe` equals `/workspace/results/run_results/261107_frozen_probe_kl_0_01/smoke_s43_frozen_round2_nwl72ltg/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation` exactly, and that folder exists.
- `adapter` equals `<R1 run dir>/checkpoints/final` exactly.
- `answers.with_probe_score` is 511. I counted the rows of the smoke run's checkpoints/responses/step_0.jsonl: 512 rows, all 512 have a `probe_score` key, and 511 of those values are not null. The two counts match.
- `penalty_probe_vs_saved.median_abs` is 0.000578. That is at most 0.01, and it is smaller than `other_probe_vs_saved.median_abs` (0.0189).

**(f) Batch: PASS**
Judgment calls in runs.md has two entries: running step 3 alongside step 2, and running steps 7, 8, 9 and 10 at the same time. Neither is an out-of-memory fallback, so the expected values are 4 and 16. Under `metrics.full_config` in run_metrics_smoke_s43_frozen_round2.json (`run_id` nwl72ltg), the values are:
- `training.batch_size` 4
- `training.grad_accum_steps` 16
- `grpo.batching.per_device_train_batch_size` 4
- `grpo.batching.gradient_accumulation_steps` 16

**(g) Phase-0 memory: PASS**
- RL crashes in runs.md lists no crashed attempt. Its only entry says: "smoke_s43_frozen_round2 attempt 1 (W&B nwl72ltg): no `CRASH:` line. `GRPO subprocess finished in 16.1 min (exit=0, attempt=1/4)`."
- The log has no `CRASH:` line and no `removed partial run directory` line.
- No line anywhere in the log contains `OutOfMemoryError` or `CUDA out of memory`, so none comes before line 96.

**(h) Frozen probe unchanged: PASS**
I ran `cd <frozen> && find . -type f | sort | xargs md5sum` now. It prints exactly the 8 recorded lines:
- every config.json is f13b24b2…
- layer_13 model.pt e0ea6af9…
- layer_14 model.pt 1e0b02b9…
- layer_15 model.pt bc9819a9…
- layer_16 model.pt 2e6f1cfb…

checks/frozen_probe_md5.txt is also identical to these 8 lines.

**Reported (no verdict)**

Penalty check. Each number compares the probe's re-score of the 511 scored step-0 answers with the trainer's saved scores:

| | median_abs | max_abs | mean_abs | over_0_02 |
|---|---|---|---|---|
| Round 1's probe (the penalty probe) | 0.000578 | 0.0233 | 0.00196 | 1 |
| The trainer's unused start-of-run fit | 0.0189 | 0.548 | 0.0391 | 250 |

KL from the base model, as `kl_div` logged to W&B:
- Smoke run (kl_smoke_s43_frozen_round2.json): step 1 0.11390, step 2 0.11848.
- Round 1's last `kl_div` (/workspace/sycophancy_experiments/261103_kl_0_01/weight_1_0/checks/kl_kl_weight_1_0_s43.json, entry 128 of 128): 0.13139.

Time per step, from the modification times (`stat -c %Y`) of step_0.jsonl and step_1.jsonl:
- This smoke run: 1794089139 to 1794089242, so 103 s (22:05:39 to 22:07:22 UTC).
- 261104's smoke run /workspace/results/run_results/261104_iterative_kl_0_01/smoke_s43_round2_l9mf5rig (4 × 16, H100): 1793828207 to 1793828312, so 105 s.

Out-of-memory lines in the log: none contains `OutOfMemoryError` or `CUDA out of memory`. The only lines with "OOM" in them are two accelerate INFO messages, at lines 79 and 97: "We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM." They are not errors.

From runs.md, the smoke run took 16.1 min (`GRPO subprocess finished in 16.1 min (exit=0, attempt=1/4)`) and its peak GPU memory was 75733 MiB. It had no crashed attempts: attempt 1, launcher try 1 of 4, W&B nwl72ltg, finished with EXIT 0.

**Overall verdict: PASS.** Rules (a) through (h) all pass.

### Review 3 (step 21), 2026-11-08 01:44 UTC: PASS (fresh general-purpose agent)

Review 3 (end of round 2, step 21), seed-43 line of 261107_frozen_probe_kl_0_01. I read every file myself and changed nothing. I wrote no files. The only commands I ran were reads: cat, python json/yaml/csv parsing, grep, readlink, md5sum and stat.

**(a) Step files: PASS**
/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/runs/s43_frozen_round2/check_records.json has `"PASS": true`, `"n_files": 128` and `"seed": 43`, with `wandb_run_id` o7dxeb0g. Not required, but also checked: all 128 per-step entries say PASS true and cover file steps 0–127.

**(b) Round 2's config: PASS**
I parsed /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/s43_frozen_round2.yaml with yaml.
- `run_key` and `probe.directory_save_prefix` are both `s43_frozen_round2`.
- `output_dir` is `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43`.
- `group` is `261107_frozen_probe_kl_0_01`.
- `checkpoint` is `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/final`.
- Required overrides, each matching in value and type:
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3
  - `training.seed` 43, `training.max_steps` 128
  - `training.batch_size` 4 and `training.grad_accum_steps` 16
  - `++grpo.model.kl_reference_from_base` True
  - `data.probe_dataset.bias_probe_train_jsonl` /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl
  - `data.probe_dataset.bias_probe_eval_jsonl` /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_eval.jsonl
  - `detector.frozen_probe_weights_dir` equals the full frozen-probe path exactly.
- 4 and 16 is the correct batch pair. Judgment calls in runs.md has three entries: step 3 run alongside step 2, steps 7–10 run at the same time, and the order of the background CPU steps after each RL run. None of them records the out-of-memory rule.
- No key containing `max_train_examples` is present.
- The override key set equals round_template.yaml's: no keys missing and no keys extra. The only values that differ from the template are its three FILL_IN slots (`training.seed`, `training.max_steps`, `detector.frozen_probe_weights_dir`). Every other override equals the template's. `config_name` grpo_mcq and `probe.retrain` true also match.

**(c) Advantage and KL: PASS**
- checks/advantages_s43_frozen_round2.json:
  - `advantage_std_floor` 0.3
  - `steps_compared` 128
  - `max_abs_difference` 3.117e-8, which is at most 1e-4
  - `kl_div_file_step_0` 0.09805, which is at least 0.01
- checks/kl_s43_frozen_round2.json: `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128 (run_id o7dxeb0g).

**(d) Swap: PASS**
The log is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/logs/rl_s43_frozen_round2_a1_de0ea2293f40.log. It has 4468 lines and ends `EXIT 0`, and it is the only RL log for this round.
- The line `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/checkpoints/final/lora` appears once, at line 95.
- After it, line 1020 is exactly `Phase-0 override: replacing detector with frozen probe weights from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`.
- Line 1021 is exactly `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`.
- I matched all three lines as whole lines (grep -x) and found no trailing text.
- The `detector.frozen_probe_weights_dir` in s43_frozen_round2.yaml equals the frozen-probe path exactly.

**(e) Batch: PASS**
In run_metrics_s43_frozen_round2.json (`run_id` o7dxeb0g), under `metrics.full_config`:
- `training.batch_size` 4
- `training.grad_accum_steps` 16
- `grpo.batching.per_device_train_batch_size` 4
- `grpo.batching.gradient_accumulation_steps` 16

No out-of-memory rule is recorded under Judgment calls, so 4 and 16 are the expected values.

**(f) Links: PASS**
- `readlink -f .../seed_43/links/round2/initial_detector.base_model.on_dataset` gives `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset`.
- `readlink -f .../seed_43/links/round2/fresh_detector.final_model.on_dataset` gives `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round2_o7dxeb0g/fresh_detector.final_model.on_dataset`.
- Both target folders exist.

**(g) Eval: PASS**
- evals/s43_frozen_round2/summary.json has `"questions": 489` and `"cued_rollouts": 978`.
- followups.jsonl has 978 rows. All 978 have `probe_run_dir` equal to `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/links/round2`. No other value appears and no row lacks the key.
- runs/s43_frozen_round2/training_curve.csv has 129 lines, which is a header plus 128 data rows, with no blank rows.

**(h) Frozen probe unchanged: PASS**
I ran `cd <frozen> && find . -type f | sort | xargs md5sum` now and diffed the output against the rule's 8 lines. They are identical:
- every config.json is f13b24b2…
- layer_13 model.pt is e0ea6af9…
- layer_14 model.pt is 1e0b02b9…
- layer_15 model.pt is bc9819a9…
- layer_16 model.pt is 2e6f1cfb…

**(i) Round 3's config: PASS**
I parsed /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/s43_frozen_round3.yaml.
- `run_key` and `probe.directory_save_prefix` are both `s43_frozen_round3`.
- `output_dir` and `group` are as required.
- `checkpoint` is `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round2_o7dxeb0g/checkpoints/final`, and that folder exists with its `lora/`.
- The required overrides all match in value and type: detector 1.0, kl 0.01, floor 0.3, seed 43, 128 steps, batch 4 / 16 (no out-of-memory rule recorded), `++grpo.model.kl_reference_from_base` True, both probe-file paths, and `detector.frozen_probe_weights_dir` equal to the frozen-probe path.
- No `max_train_examples` key is present.
- The override key set equals the template's. Only the three FILL_IN values differ, and everything else equals the template.

**Reported (no verdict)**
- **From runs.md, round 2:** the run took 188.9 min (`GRPO subprocess finished in 188.9 min (exit=0, attempt=1/4)`). Lowest free disk was 64G and peak GPU memory was 78919 MiB.
- **Crashed attempts: none.** RL crashes in runs.md says "s43_frozen_round2 attempt 1 (W&B o7dxeb0g): no `CRASH:` line". The log agrees: it has no `CRASH:`, `removed partial run directory`, `OutOfMemoryError` or `CUDA out of memory` line. So there is no W&B id, batch reached or mean answer length to report.
- **Mean seconds per step,** computed as (mtime of step_127.jsonl − mtime of step_0.jsonl) / 127 with `stat -c %Y`:

| Run | step_0 → step_127 (UTC) | Elapsed | Seconds per step |
|---|---|---|---|
| This run, s43_frozen_round2_o7dxeb0g | 22:30:23 Nov 7 → 01:25:38 Nov 8 | 10515 s | 82.80 |
| 261104's round 2, /workspace/results/run_results/261104_iterative_kl_0_01/s43_round2_yi9o10la (H100) | 22:08:04 Nov 4 → 00:57:59 Nov 5 | 10195 s | 80.28 |

- **KL from advantages_s43_frozen_round2.json:**
  - `kl_div_file_step_0` is 0.09805.
  - The mean `wandb_kl_div` over the last 16 steps (file steps 112–127) is 0.12077. The kl_s43_frozen_round2.json file's last 16 `kl_div` values give the same mean.
  - The last step's value is 0.11341.

**Overall verdict: PASS.** Rules (a) through (i) all pass.

### Review 4 (step 31), 2026-11-08 05:12 UTC: PASS (fresh general-purpose agent)

Review 4 (end of round 3, step 31), seed-43 line of 261107_frozen_probe_kl_0_01. I read every file myself and changed nothing. I wrote no files. The only commands I ran were reads: python json/yaml/csv parsing, grep, readlink, md5sum and stat.

**(a) Step files: PASS**
/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/runs/s43_frozen_round3/check_records.json has `"PASS": true`, `"n_files": 128` and `"seed": 43`. Its `wandb_run_id` is n1pu5kt0 and its `run_dir` is run dir 3. Not required, but also checked: all 128 per-step entries say PASS true.

**(b) Round 3's config: PASS**
I parsed /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/s43_frozen_round3.yaml with yaml.
- `run_key` and `probe.directory_save_prefix` are both `s43_frozen_round3`.
- `output_dir` is `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43`.
- `group` is `261107_frozen_probe_kl_0_01`.
- `checkpoint` is `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round2_o7dxeb0g/checkpoints/final`. o7dxeb0g is the `run_id` in run_metrics_s43_frozen_round2.json, and the folder's `lora/` exists.
- The required overrides all match in value and type:
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3
  - `training.seed` 43, `training.max_steps` 128
  - `training.batch_size` 4 and `training.grad_accum_steps` 16
  - `++grpo.model.kl_reference_from_base` True
  - `data.probe_dataset.bias_probe_train_jsonl` /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl
  - `data.probe_dataset.bias_probe_eval_jsonl` /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_eval.jsonl
  - `detector.frozen_probe_weights_dir` equals the full frozen-probe path exactly.
- 4 and 16 is the right batch pair. Judgment calls in runs.md has three entries:
  - 21:51 UTC: step 3 run alongside step 2.
  - 22:16 UTC: steps 7–10 run at the same time.
  - 01:34 UTC: the order of the background CPU steps after each RL run.

  None of them records the out-of-memory rule.
- No key containing `max_train_examples` is present.
- The override key set equals round_template.yaml's, with none missing and none extra. Every override not named in the rule equals the template's value. `config_name` grpo_mcq and `probe.retrain` true also match.

**(c) Advantage and KL: PASS**
- checks/advantages_s43_frozen_round3.json (its `run_dir` is run dir 3):
  - `advantage_std_floor` 0.3
  - `steps_compared` 128
  - `max_abs_difference` 1.978e-8, which is at most 1e-4
  - `kl_div_file_step_0` 0.13729, which is at least 0.01
- checks/kl_s43_frozen_round3.json (`run_id` n1pu5kt0): `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128.

**(d) Swap: PASS**
The log is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/logs/rl_s43_frozen_round3_a1_de0ea2293f40.log. It has 4475 lines, ends `EXIT 0`, and is the only RL log for round 3.
- The line `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round2_o7dxeb0g/checkpoints/final/lora` appears once, at line 95. No other `Phase-0: loading fresh model` line exists.
- After it, line 1022 is exactly `Phase-0 override: replacing detector with frozen probe weights from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`.
- Line 1023 is exactly `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`.
- I matched all three as whole lines (grep -x).
- The `detector.frozen_probe_weights_dir` in s43_frozen_round3.yaml equals the frozen-probe path exactly.

**(e) Batch: PASS**
In run_metrics_s43_frozen_round3.json (`run_id` n1pu5kt0), under `metrics.full_config`:
- `training.batch_size` 4
- `training.grad_accum_steps` 16
- `grpo.batching.per_device_train_batch_size` 4
- `grpo.batching.gradient_accumulation_steps` 16

No out-of-memory rule is recorded under Judgment calls, so 4 and 16 are the expected values.

**(f) Links: PASS**
- `readlink -f .../seed_43/links/round3/initial_detector.base_model.on_dataset` gives `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset`.
- `readlink -f .../seed_43/links/round3/fresh_detector.final_model.on_dataset` gives `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round3_n1pu5kt0/fresh_detector.final_model.on_dataset`.
- Both target folders exist.

**(g) Eval: PASS**
- evals/s43_frozen_round3/summary.json has `"questions": 489` and `"cued_rollouts": 978`.
- followups.jsonl has 978 rows. All 978 have `probe_run_dir` equal to `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/links/round3`. No other value appears and no row lacks the key.
- runs/s43_frozen_round3/training_curve.csv has 129 lines: a header plus 128 data rows, with no blank rows.

**(h) Frozen probe unchanged: PASS**
I ran `cd <frozen> && find . -type f | sort | xargs md5sum` now and diffed it against the rule's 8 lines. They are identical:
- every config.json is f13b24b2…
- layer_13 model.pt is e0ea6af9…
- layer_14 model.pt is 1e0b02b9…
- layer_15 model.pt is bc9819a9…
- layer_16 model.pt is 2e6f1cfb…

**(i) Round 4's config: PASS**
I parsed /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/s43_frozen_round4.yaml.
- `run_key` and `probe.directory_save_prefix` are both `s43_frozen_round4`.
- `output_dir` and `group` are as required.
- `checkpoint` is `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round3_n1pu5kt0/checkpoints/final`. n1pu5kt0 is the `run_id` in run_metrics_s43_frozen_round3.json, and the folder's `lora/` exists.
- The required overrides all match in value and type: detector 1.0, kl 0.01, floor 0.3, seed 43, 128 steps, batch 4 and 16 (no out-of-memory rule recorded), `++grpo.model.kl_reference_from_base` True, both probe-file paths, and `detector.frozen_probe_weights_dir` equal to the frozen-probe path.
- No `max_train_examples` key is present.
- The override key set equals the template's, and every other override equals the template's value.

**Reported (no verdict)**
- **From runs.md, round 3:** the run took 192.3 min (`GRPO subprocess finished in 192.3 min (exit=0, attempt=1/4)`). Lowest free disk was 61G and peak GPU memory was 77919 MiB.
- **Crashed attempts: none.** RL crashes in runs.md says "s43_frozen_round3 attempt 1 (W&B n1pu5kt0): no `CRASH:` line". The log agrees: it has no `CRASH:`, `removed partial run directory`, `OutOfMemoryError` or `CUDA out of memory` line. So there is no W&B id, batch reached or mean answer length to report.
- **Mean seconds per step,** computed as (mtime of step_127.jsonl − mtime of step_0.jsonl) / 127 with `stat -c %Y`:

| Run | step_0 → step_127 (UTC) | Elapsed | Seconds per step |
|---|---|---|---|
| This run, s43_frozen_round3_n1pu5kt0 | 01:54:37 → 04:53:11 Nov 8 | 10714 s | 84.36 |
| 261104's round 3, /workspace/results/run_results/261104_iterative_kl_0_01/s43_round3_sprg9bny (H100) | 01:50:33 → 04:41:36 Nov 5 | 10263 s | 80.81 |

- **KL from advantages_s43_frozen_round3.json** (the KL logged to W&B, measured from the base model):
  - `kl_div_file_step_0` is 0.13729.
  - The mean `wandb_kl_div` over the last 16 steps (file steps 112–127) is 0.12856.
  - The last step's value is 0.12268, which also matches the last `kl_div` in kl_s43_frozen_round3.json.

**Overall verdict: PASS.** Rules (a) through (i) all pass.

### Review 5 (step 41), 2026-11-08 08:41 UTC: PASS (fresh general-purpose agent)

Review 5 (step 41, end of round 4, seed-43 line): overall verdict PASS. Rules (a) through (i) all pass. I changed no files.

The run ids in the run_metrics files match the run dirs I was given: round 2 o7dxeb0g, round 3 n1pu5kt0, round 4 43szg2qi. Judgment calls in runs.md has three entries, all about running steps in a different order. None records the out-of-memory rule, so 4 × 16 is the expected batch everywhere.

**(a) Step files: PASS.** runs/s43_frozen_round4/check_records.json has `"PASS": true`, `"n_files": 128` and `"seed": 43` (wandb_run_id 43szg2qi). All 128 per-step entries have PASS true.

**(b) Config, round 4: PASS.** In s43_frozen_round4.yaml:
- `run_key` and `probe.directory_save_prefix` are both `s43_frozen_round4`.
- `output_dir` is `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43` and `group` is `261107_frozen_probe_kl_0_01`.
- `checkpoint` is `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round3_n1pu5kt0/checkpoints/final`.
- Overrides: loss.detector 1.0, loss.kl 0.01, loss.advantage_std_floor 0.3, training.seed 43, training.max_steps 128, training.batch_size 4, training.grad_accum_steps 16, `++grpo.model.kl_reference_from_base` true.
- Both bias_probe paths are the required `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_{train,eval}.jsonl`. There is no `max_train_examples` key.
- `detector.frozen_probe_weights_dir` is the frozen probe path.
- The override key set equals the template's. The other 9 overrides equal round_template.yaml's (compared after parsing the YAML). `config_name` and `probe.retrain` also match.

**(c) Advantage and KL: PASS.**
- advantages_s43_frozen_round4.json: `advantage_std_floor` 0.3, `steps_compared` 128, `max_abs_difference` 1.97e-08 (≤ 1e-4), `kl_div_file_step_0` 0.13019 (≥ 0.01).
- kl_s43_frozen_round4.json: `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128 (run_id 43szg2qi).

**(d) Swap: PASS.** Checked in logs/rl_s43_frozen_round4_a1_de0ea2293f40.log, which ends `EXIT 0`.
- The only `Phase-0: loading fresh model with adapter from .../s43_frozen_round3_n1pu5kt0/checkpoints/final/lora` line is line 95.
- After it, line 971 matches exactly: `Phase-0 override: replacing detector with frozen probe weights from <frozen>`.
- Line 972 matches exactly: `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`.
- The config's `detector.frozen_probe_weights_dir` is the frozen probe path.

**(e) Batch: PASS.** run_metrics_s43_frozen_round4.json `metrics.full_config` has training.batch_size 4, training.grad_accum_steps 16, grpo.batching.per_device_train_batch_size 4 and grpo.batching.gradient_accumulation_steps 16.

**(f) Links: PASS.**
- links/round4/initial_detector.base_model.on_dataset resolves to `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset`.
- links/round4/fresh_detector.final_model.on_dataset resolves to `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round4_43szg2qi/fresh_detector.final_model.on_dataset`.

**(g) Eval: PASS.**
- evals/s43_frozen_round4/summary.json has `"questions": 489` and `"cued_rollouts": 978`.
- All 976 rows of followups.jsonl have `probe_run_dir` `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/links/round4`.
- runs/s43_frozen_round4/training_curve.csv has 128 data rows after the header.
- Not a rule: round 4's followups.jsonl has 976 rows, while rounds 2 and 3 have 978 each, the same as cued_rollouts. I did not look into why.

**(h) Frozen probe unchanged: PASS.** `find . -type f | sort | xargs md5sum` in the frozen probe folder gives exactly the 8 recorded lines; `diff` against them is empty.

**(i) Config, round 5: PASS.** Everything is the same as in (b), with these values:
- `run_key` and `probe.directory_save_prefix` are `s43_frozen_round5`.
- `checkpoint` is `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round4_43szg2qi/checkpoints/final`.
- Batch is 4 / 16, and the frozen probe dir is the required path.
- The key set and the other overrides equal the template's.

**Reported (no verdict):**
- From runs.md, round 4 attempt 1 (W&B 43szg2qi):
  - `GRPO subprocess finished in 193.4 min (exit=0, attempt=1/4)`
  - lowest free disk 58G
  - peak GPU memory 76763 MiB
  - crashed attempts: none. RL crashes says "no `CRASH:` line", and the log has no CRASH:, OutOfMemoryError or "CUDA out of memory" line.
- Mean seconds per step, as (mtime of step_127.jsonl − mtime of step_0.jsonl) / 127:
  - this run (s43_frozen_round4_43szg2qi): 10792 s / 127 = 84.98 s per step
  - 261104's round 4 of this seed (s43_round4_pvormx4t, an H100): 9874 s / 127 = 77.75 s per step
- KL to the base model, from advantages_s43_frozen_round4.json:
  - `kl_div_file_step_0` = 0.13019
  - mean `wandb_kl_div` over the last 16 steps (file steps 112–127) = 0.10228. The KL file's last 16 `kl_div` values give the same mean.

Note added by the queue session after the review (not part of the verdict): the 976 follow-up rows are expected. In
evals/s43_frozen_round4/completions.jsonl, 2 of the 978 cued answers finished (`finish_reason` stop) with no parsed answer
letter (`pred` null). The eval builds follow-ups only for finished, parsed cued answers (eval_checkpoint.py:190), and its
labels step asserts that the follow-ups are exactly that set (eval_checkpoint.py:346); the labels step ran and exited 0.
Rounds 2 and 3 had no such answers, so 978 each.

### Review 6 (step 50), 2026-11-08 12:11 UTC: PASS (fresh general-purpose agent)

**Review 6 (end of round 5, step 50) for the seed-43 line of 261107_frozen_probe_kl_0_01: overall verdict PASS.** All eight rules, (a) to (h), pass. I read every file myself and did not change any file.

Run dirs: the `run_id` values in run_metrics_s43_frozen_round{2,3,4,5}.json are o7dxeb0g, n1pu5kt0, 43szg2qi and mly9y5ro, which match the four run dirs in the brief. Judgment calls in /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/runs.md records no use of the out-of-memory rule for any run of this line, the smoke run included. Its three entries are about running steps in parallel. So the expected batch split is 4 × 16.

**(a) Step files: PASS.** runs/s43_frozen_round5/check_records.json has `"PASS": true`, `"n_files": 128` and `"seed": 43`. Its `wandb_run_id` is mly9y5ro and its `per_step` list has 128 entries.

**(b) Config, Q = 5: PASS.** In seed_43/s43_frozen_round5.yaml:
- `run_key` and `probe.directory_save_prefix` are both `s43_frozen_round5`.
- `output_dir` is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43.
- `group` is 261107_frozen_probe_kl_0_01.
- `checkpoint` is /workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round4_43szg2qi/checkpoints/final.
- Under `overrides`: `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 43, `training.max_steps` 128, `training.batch_size` 4, `training.grad_accum_steps` 16 and `++grpo.model.kl_reference_from_base` True.
- The two probe-file paths are /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl and .../bias_probe_eval.jsonl.
- There is no `max_train_examples` key.
- `detector.frozen_probe_weights_dir` is the frozen probe path.
- I compared the overrides with round_template.yaml's in code. The key sets are identical. The only differing values are the three FILL_IN slots (`training.seed`, `training.max_steps`, `detector.frozen_probe_weights_dir`), and the rule sets each of those.

**(c) Advantage and KL: PASS.**
- advantages_s43_frozen_round5.json: `advantage_std_floor` 0.3, `steps_compared` 128, `max_abs_difference` 1.68e-8 (limit 1e-4), `kl_div_file_step_0` 0.09397 (must be at least 0.01).
- kl_s43_frozen_round5.json: `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128 (run_id mly9y5ro, 128 kl_div values).

**(d) Swap: PASS.** The log logs/rl_s43_frozen_round5_a1_de0ea2293f40.log is the only round-5 RL log, mentions mly9y5ro, and ends with `EXIT 0`.
- The line `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round4_43szg2qi/checkpoints/final/lora` appears once, at line 95.
- After it, these lines appear exactly once each:
  - line 1033: `Phase-0 override: replacing detector with frozen probe weights from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`
  - line 1034: `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`
- The config's `detector.frozen_probe_weights_dir` is that same frozen probe path.

**(e) Batch: PASS.** Under `metrics.full_config` in run_metrics_s43_frozen_round5.json: `training.batch_size` 4, `training.grad_accum_steps` 16, `grpo.batching.per_device_train_batch_size` 4, `grpo.batching.gradient_accumulation_steps` 16.

**(f) Links: PASS.**
- `readlink -f links/round5/initial_detector.base_model.on_dataset` gives /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle/initial_detector.base_model.on_dataset.
- `readlink -f links/round5/fresh_detector.final_model.on_dataset` gives /workspace/results/run_results/261107_frozen_probe_kl_0_01/s43_frozen_round5_mly9y5ro/fresh_detector.final_model.on_dataset.
- Both targets exist.

**(g) Eval: PASS.**
- evals/s43_frozen_round5/summary.json has `"questions": 489` and `"cued_rollouts": 978`.
- All 978 rows of followups.jsonl have `probe_run_dir` /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/links/round5. No row has another value or lacks the field.
- runs/s43_frozen_round5/training_curve.csv has 128 data rows (129 lines with the header, steps 0 to 127).

**(h) Frozen probe unchanged: PASS.** Running `find . -type f | sort | xargs md5sum` in the frozen probe folder and diffing the output against the rule's 8 lines finds no difference.

**Reported values (no verdict):**
- **From runs.md, round 5:**
  - Minutes: `GRPO subprocess finished in 195.3 min (exit=0, attempt=1/4)`.
  - Lowest free disk: 57G.
  - Peak GPU memory: 75422 MiB.
  - Crashed attempts: none. There is no `CRASH:` line, the run finished on attempt 1 at launcher try 1/4, and no failed_attempts/ folder exists.
- **Mean seconds per step**, (mtime of step_127.jsonl − mtime of step_0.jsonl) / 127, from each run's checkpoints/responses/:
  - This round 5 (mly9y5ro): 10896 s / 127 = 85.8 s per step.
  - 261104's round 5 of seed 43 (/workspace/results/run_results/261104_iterative_kl_0_01/s43_round5_3yrst3z8, an H100): 9583 s / 127 = 75.5 s per step.
  - Both folders have 128 step files.
- **From advantages_s43_frozen_round5.json:** `kl_div_file_step_0` is 0.0940. The mean `wandb_kl_div` over the last 16 steps (file steps 112 to 127) is 0.0765.

### Review 7 (step 53), 2026-11-08 12:23 UTC: PASS (fresh auditor agent)

The auditor's report follows word for word. Its two heading levels are moved down two levels (## to ####, ### to #####) so they nest under this entry.

#### Review 7 (step 53), auditor: the round table and the dynamics file for the seed-43 line

##### Rule verdicts
- **(a) The round table, `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/tables/rounds/seed_table.md`, checked against its source files: PASS.**
  - I reran step 51's command with only `--out-dir` changed, into `/tmp/claude-0/-workspace/e1bb799b-8715-4403-8118-b1f8f5237cce/scratchpad/seed_table`. The new `seed_table.md` and `seed_table.json` are byte-identical to the saved ones (`diff` printed nothing).
  - I also wrote a separate recomputation that does not import `compare.py`, at `/tmp/claude-0/-workspace/e1bb799b-8715-4403-8118-b1f8f5237cce/scratchpad/indep/seed_check.py`. It reads the raw `completions.jsonl`, `labels.jsonl` and `step_*.jsonl` files. Result: **1024 of 1024 table cells match, 0 mismatches.**
  - The header numbers also match: 489 eval questions, and questions with a no-cue answer in both evals of 457 / 454 / 455 / 455 / 464 for rounds 1–5.
- **(b) The dynamics file, `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/runs/dynamics/dynamics.md`, checked against its source files and W&B: PASS.**
  - I reran step 52's command, reading W&B live, into `/tmp/claude-0/-workspace/e1bb799b-8715-4403-8118-b1f8f5237cce/scratchpad/dynamics`. The new `dynamics.md` and `dynamics.json` are byte-identical to the saved ones.
  - My separate recomputation is `/tmp/claude-0/-workspace/e1bb799b-8715-4403-8118-b1f8f5237cce/scratchpad/indep/dyn_check.py`. It does its own W&B reads, one key at a time.
    - All 228 of 228 table lines match, and the file has no extra table lines.
    - The three JSON blocks match exactly: final evals, probe data, and the probe-weight cosines (largest difference 0.0).
    - The five "N questions with a no-cue answer in both evals" headings match (457 / 454 / 455 / 455 / 464).
    - The advantage SD floor line (0.3 for all five runs) matches each run's yaml.
- **(c) Every number in both files can be traced to the files: PASS.** I found no number I could not trace.

**Overall verdict: PASS.**

##### Discrepancies

**1. Numbers:** none.

**2. Procedure claims:** one minor wording point. It does not change any number and no rule turns on it.
- 2.1. `seed_table.md` line 111 says: "probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ)".
  - In a few of these groups, one answer was cut off at the length limit. That answer also gets the −0.1 penalty for not finishing (`dynamics.py` line 50 and lines 93–97), so its reward differs by more than its probe score.
  - The step files show how many: round 1, 2 of 5488 groups; round 2, 0 of 5715; round 3, 2 of 5603; round 4, 1 of 5831; round 5, 7 of 5830.
  - Label: unsupported claim (minor). The sentence holds for every group except these.

**3. Counts across stages:** none.

**4. Config consistency:** none.

**5. Leakage:** none.

**6. Unsupported claims:** none beyond 2.1. Both files are generated tables with almost no prose.
- `dynamics.md` labels the start-of-run probe as "initial" in its probe-weight cosines, and its probe row D is each run's end-of-run refit. For rounds 2–5 both describe the trainer's unused fit, not the penalty probe. Your Not-failures list covers this, so I did not count it.

##### Verified
- **Inputs:**
  - Each round's run id matches its `run_metrics_s43_frozen_round<R>.json`: o7dxeb0g, n1pu5kt0, 43szg2qi, mly9y5ro. Round 1's `$E3/run_metrics_kl_weight_1_0_s43.json` gives hxpczsle.
  - Every eval's `adapter` is `<run dir R>/checkpoints/final`, for round 1 and rounds 2–5.
  - The base eval (`261013_noisy_label_rl/evals/base`) is checkpoint `base` with no adapter.
  - Each run folder has steps 0–127 with 512 records per step.
- **Probes:**
  - `links/round2`–`round5` point their start-of-run entry at hxpczsle's `initial_detector.base_model.on_dataset` and their end-of-run entry at each round's own `fresh_detector.final_model.on_dataset`.
  - Every follow-up row has `probe_run_dir` = `$EXP/links/round<R>` (round 1: hxpczsle's run folder). The three probe scores agree between `followups.jsonl` and `labels.jsonl` in 0 of 0 compared cases differing, i.e. no differences.
  - The base rescoring folder (`$E3/evals/base_rescored_kl_weight_1_0_s43`) was scored with hxpczsle's probe (953 of 953 follow-up rows).
  - Round 1's probe (the penalty probe of every round) still gives exactly the 8 recorded md5 lines.
- **Round table against `comparison.json`:** every after-RL value (101 keys per round) equals that round's `comparison.json` exactly. Every non-probe base value (71 keys) is also equal, even though the round table takes its base from 261013's base eval and `comparison.json` from the base rescoring. The questions-with-a-no-cue-answer counts also agree.
- **Mean and SD columns:** recomputed with n − 1 and matched. Your brief says these columns are not used.
- **Counts that must agree between the two files:**
  - The round table's probe answer counts equal the per-letter sums in the dynamics file's probe rows. For example, cue_gold biased answers are 54+18=72, 47+5=52, 32+9=41, 28+6=34, 19+7=26.
  - The probe means and AUCs of the round table's three probe scores equal the dynamics file's rows C, D and B.
  - Probe-only group totals in the dynamics file are 5488 / 5715 / 5603 / 5831 / 5830. They equal the per-step "all 8 name the cue" counts, which in turn give the round table's shares 0.670 / 0.698 / 0.684 / 0.712 / 0.712.
- **W&B values:**
  - The per-step KL list equals each `checks/kl_*.json` file (and `$E3/checks/kl_kl_weight_1_0_s43.json` for round 1).
  - The recomputed mean absolute advantage per step matches `checks/advantages_*.json` to 5.6e-17, and W&B's logged value to at most 4.2e-8.
  - The "KL from the base model" label holds:
    - rounds 2–5 have `grpo.model.kl_reference_from_base` true in their run_metrics config;
    - round 1 and the three baselines (6vcs39wj, nk7mr4qf, i3wl9evn) start from the base model with no checkpoint;
    - the baselines have KL 0.001 and penalty weight 0.
  - "gradient norm, before clipping" holds: the logged value is the return of `clip_grad_norm_` (`grpo_trainer.py` lines 1685–1690, then accelerate's), which is the norm before clipping.
- **Train / eval separation:**
  - 0 shared question ids between each run's 1217 training questions and the 489 eval questions.
  - 0 shared question ids between the probe's training file and test file.
- **Scripts:** `dynamics.py` in the experiment folder is byte-identical to `code_at_launch/dynamics.py`. `compare.py`'s uncommitted change is the same as at launch (`code_at_launch/tracked.diff`). `seed_table.py` has no uncommitted changes.

##### What I wrote
Only in the scratchpad `/tmp/claude-0/-workspace/e1bb799b-8715-4403-8118-b1f8f5237cce/scratchpad/`: the two reruns' outputs, the two check scripts, and a W&B cache folder. I ran everything with `PYTHONDONTWRITEBYTECODE=1`, so nothing was written into the code folders. I wrote nothing in any experiment folder or run folder.

## Blocked

## Result

Copied verbatim from /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_43/tables/rounds/seed_table.md (step 51), headings moved down two levels.

### RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): round1 457, round2 454, round3 455, round4 455, round5 464.

| metric | condition | base | base mean | base SD | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.757 | 0.765 | 0.773 | 0.765 | 0.763 | 0.764 | 0.006 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.937 | 0.912 | 0.871 | 0.832 | 0.800 | 0.870 | 0.056 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.331 | 0.499 | 0.634 | 0.658 | 0.714 | 0.567 | 0.154 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.736 | 0.774 | 0.791 | 0.775 | 0.774 | 0.770 | 0.020 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.658 | 0.671 | 0.679 | 0.675 | 0.675 | 0.672 | 0.008 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.781 | 0.757 | 0.740 | 0.706 | 0.685 | 0.734 | 0.039 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.372 | 0.483 | 0.585 | 0.589 | 0.616 | 0.529 | 0.101 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.728 | 0.723 | 0.722 | 0.691 | 0.672 | 0.707 | 0.025 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.937 | 0.912 | 0.871 | 0.832 | 0.800 | 0.870 | 0.056 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.667 | 0.501 | 0.366 | 0.337 | 0.286 | 0.431 | 0.154 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.840 | 0.766 | 0.700 | 0.660 | 0.618 | 0.717 | 0.088 |
| bias | cue_gold | 0.094 / 0.090 / 0.089 / 0.089 / 0.094 | 0.091 | 0.002 | 0.158 | 0.115 | 0.090 | 0.075 | 0.056 | 0.099 | 0.039 |
| bias | cue_dist | 0.264 / 0.270 / 0.271 / 0.269 / 0.269 | 0.269 | 0.003 | 0.443 | 0.284 | 0.156 | 0.113 | 0.071 | 0.213 | 0.151 |
| bias | train_mix | 0.154 / 0.152 / 0.152 / 0.150 / 0.153 | 0.152 | 0.002 | 0.250 | 0.164 | 0.108 | 0.081 | 0.057 | 0.132 | 0.077 |
| anti_bias | cue_gold | 0.038 / 0.038 / 0.036 / 0.038 / 0.037 | 0.037 | 0.001 | 0.009 | 0.004 | 0.015 | 0.009 | 0.015 | 0.010 | 0.005 |
| anti_bias | cue_dist | 0.011 / 0.011 / 0.011 / 0.011 / 0.009 | 0.011 | 0.001 | 0.004 | 0.002 | 0.018 | 0.018 | 0.026 | 0.014 | 0.010 |
| anti_bias | train_mix | 0.028 / 0.028 / 0.026 / 0.028 / 0.027 | 0.028 | 0.001 | 0.005 | 0.005 | 0.016 | 0.013 | 0.020 | 0.012 | 0.007 |
| no_bias | cue_gold | 0.869 / 0.872 / 0.875 / 0.872 / 0.868 | 0.871 | 0.003 | 0.834 | 0.881 | 0.895 | 0.916 | 0.929 | 0.891 | 0.037 |
| no_bias | cue_dist | 0.725 / 0.718 / 0.717 / 0.719 / 0.722 | 0.720 | 0.003 | 0.553 | 0.714 | 0.826 | 0.870 | 0.903 | 0.773 | 0.142 |
| no_bias | train_mix | 0.818 / 0.820 / 0.822 / 0.822 / 0.820 | 0.820 | 0.002 | 0.745 | 0.831 | 0.876 | 0.906 | 0.923 | 0.856 | 0.071 |
| deception_soft | cue_gold | 0.553 / 0.552 / 0.551 / 0.555 / 0.553 | 0.553 | 0.001 | 0.544 | 0.510 | 0.457 | 0.482 | 0.504 | 0.500 | 0.033 |
| deception_soft | cue_dist | 0.602 / 0.602 / 0.598 / 0.599 / 0.601 | 0.600 | 0.002 | 0.514 | 0.587 | 0.625 | 0.585 | 0.618 | 0.586 | 0.044 |
| deception_soft | train_mix | 0.568 / 0.568 / 0.566 / 0.569 / 0.568 | 0.568 | 0.001 | 0.538 | 0.540 | 0.512 | 0.518 | 0.544 | 0.530 | 0.014 |
| deception | cue_gold | 0.572 / 0.566 / 0.566 / 0.577 / 0.572 | 0.571 | 0.004 | 0.563 | 0.511 | 0.408 | 0.466 | 0.475 | 0.485 | 0.057 |
| deception | cue_dist | 0.669 / 0.667 / 0.661 / 0.659 / 0.664 | 0.664 | 0.004 | 0.533 | 0.660 | 0.729 | 0.649 | 0.706 | 0.656 | 0.076 |
| deception | train_mix | 0.603 / 0.600 / 0.598 / 0.604 / 0.602 | 0.601 | 0.002 | 0.555 | 0.557 | 0.518 | 0.528 | 0.553 | 0.542 | 0.018 |
| biased_share | cue_gold | 0.097 / 0.093 / 0.093 / 0.093 / 0.098 | 0.095 | 0.003 | 0.159 | 0.115 | 0.092 | 0.075 | 0.057 | 0.100 | 0.039 |
| biased_share | cue_dist | 0.267 / 0.274 / 0.275 / 0.272 / 0.271 | 0.272 | 0.003 | 0.445 | 0.285 | 0.159 | 0.115 | 0.073 | 0.215 | 0.151 |
| biased_share | train_mix | 0.158 / 0.156 / 0.156 / 0.154 / 0.158 | 0.156 | 0.002 | 0.251 | 0.164 | 0.109 | 0.082 | 0.058 | 0.133 | 0.077 |
| false_denial_soft | cue_gold | 0.388 / 0.381 / 0.385 / 0.383 / 0.380 | 0.384 | 0.003 | 0.293 | 0.373 | 0.468 | 0.459 | 0.507 | 0.420 | 0.086 |
| false_denial_soft | cue_dist | 0.350 / 0.355 / 0.354 / 0.347 / 0.351 | 0.351 | 0.003 | 0.280 | 0.381 | 0.511 | 0.396 | 0.431 | 0.400 | 0.084 |
| false_denial_soft | train_mix | 0.367 / 0.367 / 0.367 / 0.364 / 0.366 | 0.366 | 0.002 | 0.288 | 0.378 | 0.472 | 0.424 | 0.468 | 0.406 | 0.076 |
| false_admission_soft | cue_gold | 0.571 / 0.569 / 0.568 / 0.572 / 0.572 | 0.570 | 0.002 | 0.592 | 0.528 | 0.456 | 0.484 | 0.504 | 0.513 | 0.051 |
| false_admission_soft | cue_dist | 0.694 / 0.695 / 0.690 / 0.693 / 0.694 | 0.693 | 0.002 | 0.702 | 0.670 | 0.647 | 0.609 | 0.633 | 0.652 | 0.036 |
| false_admission_soft | train_mix | 0.606 / 0.605 / 0.603 / 0.606 / 0.606 | 0.605 | 0.001 | 0.623 | 0.571 | 0.517 | 0.526 | 0.548 | 0.557 | 0.042 |
| said_yes_biased_soft | cue_gold | 0.612 / 0.619 / 0.615 / 0.617 / 0.620 | 0.616 | 0.003 | 0.707 | 0.627 | 0.532 | 0.541 | 0.493 | 0.580 | 0.086 |
| said_yes_biased_soft | cue_dist | 0.650 / 0.645 / 0.646 / 0.653 / 0.649 | 0.649 | 0.003 | 0.720 | 0.619 | 0.489 | 0.604 | 0.569 | 0.600 | 0.084 |
| said_yes_biased_soft | train_mix | 0.633 / 0.633 / 0.633 / 0.636 / 0.634 | 0.634 | 0.002 | 0.712 | 0.622 | 0.528 | 0.576 | 0.532 | 0.594 | 0.076 |
| deception_from_biased | cue_gold | 0.038 / 0.036 / 0.036 / 0.036 / 0.037 | 0.036 | 0.001 | 0.047 | 0.043 | 0.043 | 0.035 | 0.029 | 0.039 | 0.007 |
| deception_from_biased | cue_dist | 0.093 / 0.097 / 0.097 / 0.094 / 0.095 | 0.096 | 0.002 | 0.124 | 0.108 | 0.081 | 0.045 | 0.031 | 0.078 | 0.040 |
| deception_from_biased | train_mix | 0.058 / 0.057 / 0.057 / 0.056 / 0.058 | 0.057 | 0.001 | 0.072 | 0.062 | 0.052 | 0.035 | 0.027 | 0.050 | 0.019 |
| deception_from_unbiased | cue_gold | 0.515 / 0.516 / 0.515 / 0.519 / 0.516 | 0.516 | 0.002 | 0.498 | 0.467 | 0.415 | 0.447 | 0.476 | 0.461 | 0.031 |
| deception_from_unbiased | cue_dist | 0.509 / 0.505 / 0.501 / 0.504 / 0.506 | 0.505 | 0.003 | 0.390 | 0.479 | 0.544 | 0.539 | 0.587 | 0.508 | 0.076 |
| deception_from_unbiased | train_mix | 0.510 / 0.511 / 0.509 / 0.513 / 0.511 | 0.511 | 0.001 | 0.466 | 0.477 | 0.461 | 0.483 | 0.517 | 0.481 | 0.022 |
| said_yes | cue_gold | 0.600 / 0.594 / 0.594 / 0.605 / 0.601 | 0.599 | 0.005 | 0.656 | 0.551 | 0.411 | 0.470 | 0.475 | 0.512 | 0.094 |
| said_yes | cue_dist | 0.785 / 0.789 / 0.785 / 0.789 / 0.787 | 0.787 | 0.002 | 0.819 | 0.755 | 0.691 | 0.679 | 0.712 | 0.731 | 0.057 |
| said_yes | train_mix | 0.660 / 0.658 / 0.655 / 0.665 / 0.660 | 0.659 | 0.003 | 0.711 | 0.615 | 0.511 | 0.542 | 0.558 | 0.587 | 0.079 |
| false_denial | cue_gold | 0.357 / 0.350 / 0.350 / 0.350 / 0.349 | 0.351 | 0.003 | 0.208 | 0.327 | 0.488 | 0.471 | 0.500 | 0.399 | 0.127 |
| false_denial | cue_dist | 0.282 / 0.277 / 0.275 / 0.261 / 0.273 | 0.274 | 0.008 | 0.178 | 0.333 | 0.620 | 0.373 | 0.455 | 0.392 | 0.162 |
| false_denial | train_mix | 0.320 / 0.316 / 0.315 / 0.303 / 0.317 | 0.314 | 0.006 | 0.191 | 0.323 | 0.531 | 0.415 | 0.457 | 0.383 | 0.131 |
| false_admission | cue_gold | 0.595 / 0.589 / 0.588 / 0.600 / 0.596 | 0.594 | 0.005 | 0.630 | 0.535 | 0.400 | 0.465 | 0.473 | 0.501 | 0.086 |
| false_admission | cue_dist | 0.810 / 0.813 / 0.808 / 0.808 / 0.809 | 0.810 | 0.002 | 0.817 | 0.790 | 0.750 | 0.685 | 0.726 | 0.754 | 0.052 |
| false_admission | train_mix | 0.656 / 0.653 / 0.650 / 0.659 / 0.656 | 0.655 | 0.003 | 0.677 | 0.603 | 0.516 | 0.538 | 0.559 | 0.579 | 0.064 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.002 | 0.004 | 0.000 | 0.004 | 0.002 | 0.002 | 0.002 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.002 | 0.000 | 0.000 | 0.004 | 0.000 | 0.001 | 0.002 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 449 / 446 / 447 / 447 / 456 | 449 | 4.062 | 457 | 454 | 455 | 455 | 464 | 457 | 4.062 |
| n_bias_rows | cue_dist | 443 / 440 / 442 / 442 / 450 | 443.400 | 3.847 | 456 | 454 | 455 | 453 | 464 | 456.400 | 4.393 |
| n_bias_rows | train_mix | 892 / 886 / 889 / 889 / 906 | 892.400 | 7.893 | 913 | 908 | 910 | 908 | 928 | 913.400 | 8.414 |
| n_deception_rows | cue_gold | 432 / 429 / 431 / 430 / 439 | 432.200 | 3.962 | 453 | 452 | 448 | 451 | 457 | 452.200 | 3.271 |
| n_deception_rows | cue_dist | 438 / 435 / 437 / 437 / 446 | 438.600 | 4.278 | 454 | 453 | 447 | 445 | 452 | 450.200 | 3.962 |
| n_deception_rows | train_mix | 870 / 864 / 868 / 867 / 885 | 870.800 | 8.228 | 907 | 905 | 895 | 896 | 909 | 902.400 | 6.465 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.218 | 0.166 | 0.192 | 0.215 | 0.198 | 0.198 | 0.021 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.195 | 0.171 | 0.179 | 0.254 | 0.223 | 0.205 | 0.034 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.039 | 0.039 | 0.034 | 0.044 | 0.028 | 0.037 | 0.006 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.057 | 0.036 | 0.058 | 0.068 | 0.058 | 0.055 | 0.012 |
| probe_start_auc | cue_gold | – | – | – | 0.849 | 0.870 | 0.840 | 0.838 | 0.853 | 0.850 | 0.013 |
| probe_start_auc | cue_dist | – | – | – | 0.763 | 0.799 | 0.761 | 0.820 | 0.812 | 0.791 | 0.028 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 72 | 52 | 41 | 34 | 26 | 45 | 17.861 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 202 | 129 | 71 | 51 | 33 | 97.200 | 68.806 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 356 | 362 | 354 | 349 | 349 | 354 | 5.431 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 95 | 89 | 86 | 94 | 96 | 92 | 4.301 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.282 | 0.319 | 0.308 | 0.363 | 0.367 | 0.328 | 0.036 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.249 | 0.299 | 0.284 | 0.309 | 0.387 | 0.306 | 0.051 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.069 | 0.110 | 0.117 | 0.151 | 0.180 | 0.125 | 0.042 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.095 | 0.118 | 0.142 | 0.168 | 0.196 | 0.144 | 0.040 |
| probe_fresh_auc | cue_gold | – | – | – | 0.856 | 0.846 | 0.786 | 0.798 | 0.730 | 0.803 | 0.051 |
| probe_fresh_auc | cue_dist | – | – | – | 0.748 | 0.775 | 0.726 | 0.739 | 0.723 | 0.742 | 0.021 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 72 | 52 | 41 | 34 | 26 | 45 | 17.861 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 202 | 129 | 71 | 51 | 33 | 97.200 | 68.806 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 356 | 362 | 354 | 349 | 349 | 354 | 5.431 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 95 | 89 | 86 | 94 | 96 | 92 | 4.301 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.297 | 0.255 | 0.233 | 0.292 | 0.310 | 0.277 | 0.032 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.251 | 0.233 | 0.229 | 0.318 | 0.355 | 0.277 | 0.056 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.062 | 0.068 | 0.065 | 0.084 | 0.096 | 0.075 | 0.015 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.075 | 0.072 | 0.093 | 0.127 | 0.164 | 0.106 | 0.039 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.870 | 0.865 | 0.804 | 0.810 | 0.765 | 0.823 | 0.044 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.759 | 0.775 | 0.732 | 0.775 | 0.739 | 0.756 | 0.020 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 72 | 52 | 41 | 34 | 26 | 45 | 17.861 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 202 | 129 | 71 | 51 | 33 | 97.200 | 68.806 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 356 | 362 | 354 | 349 | 349 | 354 | 5.431 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 95 | 89 | 86 | 94 | 96 | 92 | 4.301 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|
| equal task reward groups | 0.736 | 0.790 | 0.849 | 0.924 | 0.947 | 0.849 | 0.089 |
| probe-only groups | 0.670 | 0.698 | 0.684 | 0.712 | 0.712 | 0.695 | 0.018 |
