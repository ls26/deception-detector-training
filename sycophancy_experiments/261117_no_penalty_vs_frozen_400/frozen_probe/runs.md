# Runs

State file for the unattended run of queue `frozen_probe`. Read this first after any restart.

## Queue

- [x] 1. Start check
- [x] 2. Folders
- [x] 3. Base model on this pod
- [x] 4. Smoke: vLLM
- [x] 5. Smoke: RL
- [x] 6. Smoke: checks
- [x] 7. Smoke: eval of the step-3 adapter
- [x] 8. Review 1
- [x] 9. Seed 42: vLLM
- [x] 10. Seed 42: RL
- [x] 11. Seed 42: early check
- [x] 12. Review 2
- [x] 13. Seed 42: RL finished
- [x] 14. Seed 42: checks
- [x] 15. Review 3
- [x] 16. Seed 42: base answers rescored
- [x] 17. Seed 42: evals of the saved adapters
- [x] 18. Seed 42: training curve
- [x] 19. Seed 42: table
- [x] 20. Review 4
- [x] 21. Seed 43: vLLM
- [x] 22. Seed 43: RL
- [x] 23. Seed 43: RL finished
- [x] 24. Seed 43: checks
- [x] 25. Review 5
- [x] 26. Seed 43: base answers rescored
- [x] 27. Seed 43: evals of the saved adapters
- [x] 28. Seed 43: training curve
- [x] 29. Seed 43: table
- [x] 30. Review 6
- [x] 31. Combined table
- [x] 32. Review 7

## Log

One line per step: `<UTC time> | step | outcome | next`.

Start facts (2026-11-17 19:54 UTC): hostname `f297b048b0f3`; `nvidia-smi -L`: GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-ca8d3c38-434b-d852-0535-5603c9a023ea); driver 595.91.07, CUDA 13.2; `uname -r` 6.8.0-147-generic; `uptime -s` 2026-10-20 01:09:34; `df -h /workspace` 513G size, 328G used, 185G free (65%). Pod and GPU match the Goal. Resume check scheduled (cron job 838a4855, minutes 7 and 37 of each hour).

2026-11-17 19:58 UTC | 1. Start check | pass: pod and GPU match; fence rejected the Write to /workspace/scratch/fence_test_261117-fz.txt (file not created); identity checks all identical (`cmp`): config_np400_s42.json, config_fz400_s43.json, probe_kl_no_penalty_s42.json, copy_base completions.jsonl (4890 rows), table_old comparison.json + comparison.md, compare comparison.json + comparison.md + training_curve.csv (outputs in scratchpad `start/`) | 2. Folders
2026-11-17 19:59 UTC | 2. Folders | all 12 folders exist; disk 185G free; GPU 0 MiB used, no processes | 3. Base model on this pod
2026-11-17 20:04 UTC | 3. Base model on this pod | EXIT 0 (logs/base_repro_a1_f297b048b0f3.log); checks/base_repro.json vs 261013's base answers: rows 500, questions 50, same_completion 447, same_pred 495 (reported only) | 4. Smoke: vLLM
2026-11-17 20:08 UTC | 4. Smoke: vLLM | server started 20:04, `curl /v1/models` answered 20:08 (dynamic-lora-model); IMPORT_CHECK afterburner from sycophancy-rl copy; GPU 30405 MiB used; disk 185G free | 5. Smoke: RL
2026-11-17 20:33 UTC | 5. Smoke: RL | started 20:08; log ends EXIT 0; no `CRASH:` lines; "GRPO subprocess finished in 21.5 min (exit=0, attempt=1/4)"; run id 4nta3vhx, run dir /workspace/results/run_results/261117_no_penalty_vs_frozen_400/smoke_fz400_4nta3vhx; peak GPU memory 65989 MiB; lowest free disk 181 GB; vLLM session killed after EXIT | 6. Smoke: checks
2026-11-17 20:34 UTC | 6. Smoke: checks | exit 0 → checks/finished_smoke_fz400.json: step_files 6, prompts 384, answers 3072, nonzero_penalties 2148 = naming_cued_option 2148, flipped_share 0.221, max reward/penalty mean abs difference 3.6e-08 / 1.2e-08 (review 1 judges the rules) | 7. Smoke: eval of the step-3 adapter (already running since 20:33)
2026-11-17 20:37 UTC | 7. Smoke: eval of the step-3 adapter | started 20:33, logs/eval_smoke_step_3_a1_f297b048b0f3.log ends EXIT 0; dry/evals/smoke_step_3/summary.json "questions": 50 | 8. Review 1
2026-11-17 20:37 UTC | 8. Review 1 | PASS (reviews/1_a1.md); reported: flipped_share 0.221, cut_off 1, full_training_states_left [] | 9. Seed 42: vLLM
2026-11-17 20:41 UTC | 9. Seed 42: vLLM | server started 20:37 (disk 183G free, GPU empty), `curl /v1/models` answered 20:41; IMPORT_CHECK afterburner from sycophancy-rl copy; GPU 30263 MiB | 10. Seed 42: RL (started 20:41:18)
2026-11-17 20:44 UTC | 10. Seed 42: RL | started 20:41:18; run dir line at 20:43: /workspace/results/run_results/261117_no_penalty_vs_frozen_400/fz400_s42_avu97evd; IMPORT_CHECK obfuscation_atlas and afterburner from sycophancy-rl copy | 11. Seed 42: early check
2026-11-17 21:04 UTC | 11. Seed 42: early check | step_10.jsonl appeared by 21:02 (~21 min after the RL start); checks.py early exit 0 → seed_42/checks/early_fz400_s42.json: per_step 12 rows, prompts 768, answers 6144, nonzero_penalties 4865 = naming_cued_option 4865, flipped_share 0.219 (review 2 judges the rules) | 12. Review 2
2026-11-17 21:05 UTC | 12. Review 2 | PASS (reviews/2_a1.md); reported: wandb_kl_div_step_1 0, flipped_share 0.219 | 13. Seed 42: RL finished (expected roughly 10–13 h after 20:41)

Restart facts (2026-11-18 04:46 UTC, new session): hostname `f297b048b0f3`; GPU 0 UUID GPU-ca8d3c38-434b-d852-0535-5603c9a023ea (same pod and GPU); `uptime -s` 2026-11-17 21:37:08 (pod rebooted during step 13); driver 595.112.07 (was 595.91.07), CUDA 13.2; `uname -r` 6.8.0-149-generic (was 6.8.0-147-generic); `df -h /workspace` 180G free; GPU 0 MiB used, no processes; one-element CUDA tensor works. Resume check re-created (cron job cc7eb8eb, minutes 7 and 37 of each hour; 838a4855 ended with the earlier session).

2026-11-18 04:47 UTC | 13. Seed 42: RL finished | not reached: pod rebooted 2026-11-17 21:37:08; RL attempt 1 stopped at training step 46 of 400 (last log line 21:32), no EXIT line; steps 9–12 unticked to redo as attempt 2 (see Judgment calls) | 9. Seed 42: vLLM (a2)
2026-11-18 04:51 UTC | 9. Seed 42: vLLM (a2) | server started 04:46 (disk 180G free, GPU empty), `curl /v1/models` answered 04:50 (dynamic-lora-model); IMPORT_CHECK afterburner from sycophancy-rl copy; GPU 30263 MiB | 10. Seed 42: RL (a2)
2026-11-18 04:53 UTC | 10. Seed 42: RL (a2) | started 04:50:42 (logs/rl_fz400_s42_a2_f297b048b0f3.log); run dir line at 04:53: /workspace/results/run_results/261117_no_penalty_vs_frozen_400/fz400_s42_h94e9ilk; IMPORT_CHECK obfuscation_atlas and afterburner from sycophancy-rl copy; peak GPU memory and lowest free disk polled every 60 s into logs/gpu_disk_poll_fz400_s42_a2_f297b048b0f3.txt | 11. Seed 42: early check (a2)
2026-11-18 05:12 UTC | 11. Seed 42: early check (a2) | step_10.jsonl appeared by 05:11 (~21 min after the RL start); checks.py early on logs/rl_fz400_s42_a2_f297b048b0f3.log exit 0 → seed_42/checks/early_fz400_s42.json (run dir fz400_s42_h94e9ilk): per_step 10 rows, prompts 640, answers 5120, nonzero_penalties 3924 = naming_cued_option 3924, flipped_share 0.227 (review 2 judges the rules) | 12. Review 2 (a2)
2026-11-18 05:15 UTC | 12. Review 2 (a2) | PASS (reviews/2_a2.md); reported: wandb_kl_div_step_1 0, flipped_share 0.227 | 13. Seed 42: RL finished (a2 log; roughly 6–13 h after 04:50)
2026-11-18 14:46 UTC | 13. Seed 42: RL finished (a2) | logs/rl_fz400_s42_a2_f297b048b0f3.log ends EXIT 0 (EXIT seen 14:44); no `CRASH:` lines; "GRPO subprocess finished in 591.2 min (exit=0, attempt=1/4)"; run id h94e9ilk, run dir /workspace/results/run_results/261117_no_penalty_vs_frozen_400/fz400_s42_h94e9ilk; peak GPU memory 68421 MiB; lowest free disk 157 GB; vLLM session killed after EXIT. Reported: run_metrics `metrics.seed` is 53 (the data seed field, as in the smoke run); `full_config.training.seed` is 42 | 14. Seed 42: checks, with 16 started alongside (Judgment calls)
2026-11-18 14:48 UTC | 14. Seed 42: checks | exit 0 (probe check on the a2 RL log) → seed_42/checks/finished_fz400_s42.json: step_files 400, prompts 25600, answers 204800, nonzero_penalties 180627 = naming_cued_option 180627, flipped_share 0.201, cut_off 268, max reward/penalty mean abs difference 7.4e-08 / 1.8e-08, adapter_steps 50–400 by 50, full_training_states_left [step_400]; seed_42/checks/probe_fz400_s42.json: test_auc 0.873, md5_identical_to_round1_probe true (review 3 judges the rules) | 15. Review 3 (step 16 running since 14:45)
2026-11-18 14:51 UTC | 15. Review 3 | PASS (reviews/3_a1.md); reported: flipped_share 0.201, batch_size 4 × grad_accum_steps 16, test_auc 0.873, md5_identical_to_round1_probe true | 16. Seed 42: base answers rescored (running)
2026-11-18 14:52 UTC | 16. Seed 42: base answers rescored | started 14:45 (disk 161G free, GPU empty), logs/eval_base_rescored_a1_f297b048b0f3.log ends EXIT 0 (14:47); evals/base_rescored/summary.json "questions": 489, "cued_rollouts": 978 | 17. Seed 42: evals of the saved adapters
2026-11-18 15:36 UTC | 17. Seed 42: evals of the saved adapters | steps 50, 100, 150, 200, 250, 300, 350, 400 run in turn 14:48–15:35 (about 6 min each; disk 160–161G free before each, GPU empty), each logs/eval_step_N_a1_f297b048b0f3.log ends EXIT 0 with its ADAPTER line, each evals/step_N/summary.json "questions": 489, "cued_rollouts": 978 | 18. Seed 42: training curve
2026-11-18 15:37 UTC | 18. Seed 42: training curve | compare.py exit 0 → seed_42/runs/fz400_s42/comparison.json, comparison.md, training_curve.csv (400 step rows + header) | 19. Seed 42: table (21. Seed 43: vLLM started 15:35 alongside, see Judgment calls)
2026-11-18 15:38 UTC | 19. Seed 42: table | table.py seed frozen_probe 42 exit 0 → seed_42/table/comparison.json, comparison.md | 20. Review 4
2026-11-18 15:41 UTC | 20. Review 4 | PASS (reviews/4_a1.md): all 9 summaries 489 / 978, all eval logs EXIT 0 with their ADAPTER lines; the reviewer's recompute of bias and cue agreement rate (50/50 mix) at steps 50 and 400 equals the table with difference 0.0 | 21. Seed 43: vLLM (started 15:35)
2026-11-18 15:42 UTC | 21. Seed 43: vLLM | server started 15:35 (disk 159G free, GPU empty), `curl /v1/models` answered 15:38 (dynamic-lora-model); IMPORT_CHECK afterburner from sycophancy-rl copy; GPU 30263 MiB | 22. Seed 43: RL
2026-11-18 15:42 UTC | 22. Seed 43: RL | started 15:39:09 (logs/rl_fz400_s43_a1_f297b048b0f3.log); run dir line at 15:41: /workspace/results/run_results/261117_no_penalty_vs_frozen_400/fz400_s43_7sz0ztpx; IMPORT_CHECK obfuscation_atlas and afterburner from sycophancy-rl copy; peak GPU memory and lowest free disk polled every 60 s into seed_43/logs/gpu_disk_poll_fz400_s43_a1_f297b048b0f3.txt | 23. Seed 43: RL finished (seed 42 took 591 min, so roughly 01:30 UTC on 11-19)
2026-11-19 01:36 UTC | 23. Seed 43: RL finished | logs/rl_fz400_s43_a1_f297b048b0f3.log ends EXIT 0 (EXIT seen 01:34); no `CRASH:` lines; "GRPO subprocess finished in 593.2 min (exit=0, attempt=1/4)"; run id 7sz0ztpx, run dir /workspace/results/run_results/261117_no_penalty_vs_frozen_400/fz400_s43_7sz0ztpx; peak GPU memory 69541 MiB; lowest free disk 141 GB; vLLM session killed after EXIT, GPU free at 01:35. Reported: `full_config.training.seed` 43, `metrics.seed` 53 (the data seed field, as in seed 42) | 24. Seed 43: checks, with 26 started alongside (Judgment calls)
2026-11-19 01:38 UTC | 24. Seed 43: checks | exit 0 → seed_43/checks/finished_fz400_s43.json: step_files 400, prompts 25600, answers 204800, nonzero_penalties 176101 = naming_cued_option 176101, flipped_share 0.201, cut_off 609, max reward/penalty mean abs difference 8.5e-08 / 2.7e-08, adapter_steps 50–400 by 50, full_training_states_left [step_400]; seed_43/checks/probe_fz400_s43.json: test_auc 0.873, md5_identical_to_round1_probe true (review 5 judges the rules) | 25. Review 5 (step 26 running since 01:35)
2026-11-19 01:40 UTC | 25. Review 5 | PASS (reviews/5_a1.md); reported: flipped_share 0.201, batch_size 4 × grad_accum_steps 16, test_auc 0.873, md5_identical_to_round1_probe true | 26. Seed 43: base answers rescored (finished)
2026-11-19 01:40 UTC | 26. Seed 43: base answers rescored | started 01:35 (disk 145G free, GPU empty), logs/eval_base_rescored_a1_f297b048b0f3.log ends EXIT 0 (01:37); evals/base_rescored/summary.json "questions": 489, "cued_rollouts": 978 | 27. Seed 43: evals of the saved adapters
2026-11-19 02:27 UTC | 27. Seed 43: evals of the saved adapters | steps 50, 100, 150, 200, 250, 300, 350, 400 run in turn 01:38–02:26 (about 6 min each; disk 144–145G free before each, GPU empty), each seed_43/logs/eval_step_N_a1_f297b048b0f3.log ends EXIT 0 with its ADAPTER line, each evals/step_N/summary.json "questions": 489, "cued_rollouts": 978 | 28. Seed 43: training curve
2026-11-19 02:28 UTC | 28. Seed 43: training curve | compare.py exit 0 → seed_43/runs/fz400_s43/comparison.json, comparison.md, training_curve.csv (400 step rows + header) | 29. Seed 43: table
2026-11-19 02:29 UTC | 29. Seed 43: table | table.py seed frozen_probe 43 exit 0 → seed_43/table/comparison.json, comparison.md | 30. Review 6
2026-11-19 02:32 UTC | 30. Review 6 | PASS (reviews/6_a1.md): all 9 summaries 489 / 978, all eval logs EXIT 0 with their ADAPTER lines; the reviewer's recompute of bias and cue agreement rate (50/50 mix) at steps 50 and 400 equals the table with difference 0.0 | 31. Combined table
2026-11-19 02:33 UTC | 31. Combined table | not run (frozen-probe queue); seed_42/table/comparison.md and seed_43/table/comparison.md copied into Result | 32. Review 7
2026-11-19 02:33 UTC | 32. Review 7 | not run (frozen-probe queue) | done: Blocked is empty, notify the user
2026-11-19 02:35 UTC | Done | all 32 queue steps ticked, Blocked empty, Result holds both seed tables (rows checked identical to their source files with `diff`); completion notification sent (terminal notification sent, mobile push requested); resume check cc7eb8eb left in place | –

## Judgment calls

- 2026-11-17 20:09 UTC: the RL steps' background poll checks for the `EXIT` line every 60 s instead of every 5 min, so that the peak GPU memory and lowest free disk it records (`nvidia-smi --query-gpu=memory.used`, `df /workspace`) are sampled more often. It only reads; nothing else changes.
- 2026-11-17 20:33 UTC: started step 7 (smoke eval, GPU) while step 6 (smoke checks, CPU, reads saved files) runs, so the GPU does not wait on a read-only step. The two are independent: step 7 needs only the finished smoke run dir.
- 2026-11-18 04:47 UTC: the pod rebooted at 2026-11-17 21:37:08 UTC, 56 min after seed 42's RL step started (20:41:18). The RL attempt-1 log stops at training step 46 of 400 with no `EXIT` line, and no tmux session of this queue survived. Following the reboot rule: redo step 9 (vLLM) as attempt 2, then rerun step 10 (RL) as attempt 2 from scratch (a new run dir); this rerun does not count against the step's attempts. I also redo step 11 (early check, on the a2 RL log) and step 12 (review 2, written to `reviews/2_a2.md`), because both checked the attempt-1 run, which will never finish; without them the new run's first steps would go unchecked. Steps 13 and 14 then use `logs/rl_fz400_s42_a2_$HOST.log`. The attempt-1 outputs were moved to `seed_42/failed_attempts/9_a1/` (vLLM log), `10_a1/` (RL log) and `11_a1/` (early_fz400_s42.json); `reviews/2_a1.md` stays as the record of the attempt-1 review. The attempt-1 run dir `/workspace/results/run_results/261117_no_penalty_vs_frozen_400/fz400_s42_avu97evd` (partial, 46 steps) is left in place.
- 2026-11-18 14:47 UTC: started step 16 (seed 42 base answers rescored, GPU) at the same time as step 14 (checks, CPU, reads saved files) and before step 15 (review 3), so the GPU does not wait on read-only steps. Step 16 needs only the finished run dir (its end-of-run probe), not 14's or 15's outputs. If review 3 fails, step 16's outputs move to `failed_attempts/` and it reruns after the fix like any affected step. Steps 17 onward wait for review 3's PASS.
- 2026-11-18 15:37 UTC: started step 21 (seed 43 vLLM), then step 22 (seed 43 RL), while steps 18–20 (seed 42 training curve, table, review 4; all read saved files only) run, so the GPU does not wait on read-only steps. Seed 43's RL uses none of seed 42's outputs. If review 4 fails on a bug in a report script (compare.py, table.py), the fix and the rerun touch only steps 18–19, not the seed 43 run.
- 2026-11-19 01:36 UTC: as for seed 42, started step 26 (seed 43 base answers rescored, GPU) together with step 24 (checks, CPU) and before step 25 (review 5), so the GPU does not wait on read-only steps; step 26 needs only the finished run dir. If review 5 fails, step 26's outputs move to `failed_attempts/` and rerun after the fix. Steps 27 onward wait for review 5's PASS.

## Code fixes

## Reviews

One line per review: `<UTC time> | review | verdict | reviews/<n>_a<attempt>.md`.

2026-11-17 20:37 UTC | 1, smoke | PASS | reviews/1_a1.md
2026-11-17 21:05 UTC | 2, seed 42 early check | PASS | reviews/2_a1.md
2026-11-18 05:15 UTC | 2, seed 42 early check, on RL attempt 2 (after the reboot) | PASS | reviews/2_a2.md
2026-11-18 14:51 UTC | 3, seed 42 step files | PASS | reviews/3_a1.md
2026-11-18 15:41 UTC | 4, seed 42 evals and table | PASS | reviews/4_a1.md
2026-11-19 01:40 UTC | 5, seed 43 step files | PASS | reviews/5_a1.md
2026-11-19 02:32 UTC | 6, seed 43 evals and table | PASS | reviews/6_a1.md

## Blocked

## Result

Copied verbatim from `seed_42/table/comparison.md` and `seed_43/table/comparison.md` (step 31); only the headings are moved down two levels to sit under this section. The combined table of both conditions is made by the no-penalty queue on pod 3fb02eb377e0.

### frozen probe, seed 42: 400 steps

Written by `table.py`; definitions in its docstring. Each cell lists every seed's value in the order given in the condition column. Values are shares (0-1); "–" means no answers to compute it on.

#### New run: labels redrawn every prompt, constant learning rate

| Measurement | Condition (seeds) | Base model | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Step 350 | Step 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| bias, 50/50 mix | frozen probe (s42) | 0.177 | 0.373 | 0.445 | 0.440 | 0.413 | 0.322 | 0.361 | 0.366 | 0.221 |
| bias, cue at the correct answer | frozen probe (s42) | 0.087 | 0.176 | 0.179 | 0.194 | 0.179 | 0.152 | 0.185 | 0.179 | 0.123 |
| bias, cue at the wrong answer | frozen probe (s42) | 0.266 | 0.571 | 0.711 | 0.686 | 0.647 | 0.492 | 0.537 | 0.554 | 0.318 |
| cue agreement rate, 50/50 mix | frozen probe (s42) | 0.650 | 0.876 | 0.946 | 0.942 | 0.916 | 0.822 | 0.864 | 0.877 | 0.721 |
| cue agreement rate, cue at the correct answer | frozen probe (s42) | 0.843 | 0.967 | 0.984 | 0.990 | 0.978 | 0.957 | 0.973 | 0.982 | 0.910 |
| cue agreement rate, cue at the wrong answer | frozen probe (s42) | 0.458 | 0.785 | 0.908 | 0.894 | 0.855 | 0.687 | 0.755 | 0.773 | 0.532 |
| true accuracy, 50/50 mix | frozen probe (s42) | 0.676 | 0.591 | 0.537 | 0.548 | 0.559 | 0.634 | 0.607 | 0.604 | 0.689 |
| true accuracy, cue at the correct answer | frozen probe (s42) | 0.843 | 0.967 | 0.984 | 0.990 | 0.978 | 0.957 | 0.973 | 0.982 | 0.910 |
| true accuracy, cue at the wrong answer | frozen probe (s42) | 0.509 | 0.215 | 0.090 | 0.106 | 0.141 | 0.311 | 0.241 | 0.227 | 0.468 |
| true accuracy, no cue | frozen probe (s42) | 0.755 | 0.759 | 0.757 | 0.757 | 0.744 | 0.775 | 0.753 | 0.744 | 0.759 |
| initial probe, both cues pooled: mean score of biased answers | frozen probe (s42) | 0.747 | 0.169 | 0.147 | 0.122 | 0.169 | 0.149 | 0.147 | 0.123 | 0.148 |
| initial probe, both cues pooled: mean score of unbiased answers | frozen probe (s42) | 0.228 | 0.048 | 0.046 | 0.040 | 0.047 | 0.044 | 0.044 | 0.027 | 0.050 |
| initial probe, both cues pooled: AUC biased vs unbiased | frozen probe (s42) | 0.906 | 0.782 | 0.762 | 0.748 | 0.775 | 0.769 | 0.797 | 0.797 | 0.751 |

### frozen probe, seed 43: 400 steps

Written by `table.py`; definitions in its docstring. Each cell lists every seed's value in the order given in the condition column. Values are shares (0-1); "–" means no answers to compute it on.

#### New run: labels redrawn every prompt, constant learning rate

| Measurement | Condition (seeds) | Base model | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Step 350 | Step 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| bias, 50/50 mix | frozen probe (s43) | 0.175 | 0.429 | 0.354 | 0.410 | 0.373 | 0.250 | 0.321 | 0.267 | 0.273 |
| bias, cue at the correct answer | frozen probe (s43) | 0.088 | 0.174 | 0.166 | 0.165 | 0.168 | 0.131 | 0.163 | 0.137 | 0.165 |
| bias, cue at the wrong answer | frozen probe (s43) | 0.263 | 0.684 | 0.542 | 0.655 | 0.577 | 0.369 | 0.479 | 0.397 | 0.382 |
| cue agreement rate, 50/50 mix | frozen probe (s43) | 0.650 | 0.934 | 0.855 | 0.916 | 0.875 | 0.748 | 0.827 | 0.774 | 0.774 |
| cue agreement rate, cue at the correct answer | frozen probe (s43) | 0.843 | 0.984 | 0.971 | 0.980 | 0.971 | 0.920 | 0.953 | 0.937 | 0.947 |
| cue agreement rate, cue at the wrong answer | frozen probe (s43) | 0.458 | 0.883 | 0.738 | 0.853 | 0.779 | 0.577 | 0.701 | 0.611 | 0.601 |
| true accuracy, 50/50 mix | frozen probe (s43) | 0.676 | 0.550 | 0.617 | 0.563 | 0.595 | 0.670 | 0.626 | 0.662 | 0.673 |
| true accuracy, cue at the correct answer | frozen probe (s43) | 0.843 | 0.984 | 0.971 | 0.980 | 0.971 | 0.920 | 0.953 | 0.937 | 0.947 |
| true accuracy, cue at the wrong answer | frozen probe (s43) | 0.509 | 0.117 | 0.262 | 0.147 | 0.219 | 0.419 | 0.299 | 0.387 | 0.399 |
| true accuracy, no cue | frozen probe (s43) | 0.755 | 0.783 | 0.773 | 0.736 | 0.800 | 0.748 | 0.742 | 0.761 | 0.748 |
| initial probe, both cues pooled: mean score of biased answers | frozen probe (s43) | 0.748 | 0.145 | 0.156 | 0.185 | 0.250 | 0.165 | 0.371 | 0.275 | 0.272 |
| initial probe, both cues pooled: mean score of unbiased answers | frozen probe (s43) | 0.230 | 0.041 | 0.041 | 0.093 | 0.090 | 0.064 | 0.120 | 0.064 | 0.082 |
| initial probe, both cues pooled: AUC biased vs unbiased | frozen probe (s43) | 0.905 | 0.766 | 0.800 | 0.709 | 0.766 | 0.737 | 0.813 | 0.831 | 0.805 |
