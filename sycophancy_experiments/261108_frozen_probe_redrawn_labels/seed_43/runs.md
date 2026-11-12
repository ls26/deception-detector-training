# Runs

State file for the unattended run of queue `seed_43`. Read this first after any restart.

## Queue

- [x] 1. Start check
- [x] 2. Folders, frozen files, label known answer
- [x] 3. Round 1 regenerated on this pod
- [x] 4. Round 2: config
- [x] 5. Review 1
- [x] 6. Smoke: vLLM
- [x] 7. Smoke: RL
- [x] 8. Smoke: record check
- [x] 9. Smoke: stop vLLM
- [x] 10. Smoke: checks
- [x] 11. Smoke: penalty check
- [x] 12. Review 2
- [x] 13. Round 2: vLLM
- [x] 14. Round 2: RL
- [x] 15. Round 2: record check
- [x] 16. Round 2: checks
- [x] 17. Round 2: stop vLLM
- [x] 18. Round 2: links
- [x] 19. Round 2: eval
- [x] 20. Round 2: compare
- [x] 21. Round 3: config
- [x] 22. Review 3
- [x] 23. Round 3: vLLM
- [x] 24. Round 3: RL
- [x] 25. Round 3: record check
- [x] 26. Round 3: checks
- [x] 27. Round 3: stop vLLM
- [x] 28. Round 3: links
- [x] 29. Round 3: eval
- [x] 30. Round 3: compare
- [x] 31. Round 4: config
- [x] 32. Review 4
- [x] 33. Round 4: vLLM
- [x] 34. Round 4: RL
- [x] 35. Round 4: record check
- [x] 36. Round 4: checks
- [x] 37. Round 4: stop vLLM
- [x] 38. Round 4: links
- [x] 39. Round 4: eval
- [x] 40. Round 4: compare
- [x] 41. Round 5: config
- [x] 42. Review 5
- [x] 43. Round 5: vLLM
- [x] 44. Round 5: RL
- [x] 45. Round 5: record check
- [x] 46. Round 5: checks
- [x] 47. Round 5: stop vLLM
- [x] 48. Round 5: links
- [x] 49. Round 5: eval
- [x] 50. Round 5: compare
- [x] 51. Review 6
- [x] 52. Dynamics
- [x] 53. Seed table

## Log

One line per step: `<UTC time> | step | outcome | next`.

Start check system info (2026-11-09T00:01Z): `hostname` c43a39434be5; `nvidia-smi -L` GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-b38788d9-d352-7319-fe04-c9cc5b14e252); driver 595.91.07, CUDA 13.2; `uname -r` 6.8.0-147-generic; `uptime -s` 2026-10-20 04:28:30; `df -h /workspace` 513G size, 290G used, 223G available (57%). Matches the Goal's pod and GPU.

- 2026-11-09T00:04Z | 1. Start check | done: pod and GPU match; fence test rejected by the hook (no file created); identity checks all identical (checks.py smoke + advantages, check_records.py, compare.py 3 files, seed_table.py 2 files, dynamics.py 2 files; scratchpad `start/`) | next: 2
- 2026-11-09T00:05Z | 2. Folders, frozen files, label known answer | done: `checks/frozen_probe_md5.txt` 8 lines, identical to 261107 seed_42's list; `checks/labels_md5_check.txt` 4 × OK; `logs/labels_known_answer_a1_c43a39434be5.log` "Known answer PASS: seed 261013 reproduces .../questions.jsonl byte for byte (md5 226ed577ee96729ad02a47dbee2d918f)", EXIT 0 | next: 3
- 2026-11-09T00:10Z | 3. Round 1 regenerated on this pod | done (reported): `logs/round1_repro_a1_c43a39434be5.log` EXIT 0; `checks/round1_repro.json` rows 500, questions 50, same_completion 198, same_pred 468 (261107 on its H100: 500 / 50 / 208 / 472) | next: 4
- 2026-11-09T00:11Z | 4. Round 2: config | done: `logs/config_round2_a1_c43a39434be5.log` EXIT 0, wrote `s43_redrawn_round2.yaml` and `smoke_s43_redrawn_round2.yaml` | next: 5 (review 1)
- 2026-11-09T00:12Z | 5. Review 1 | PASS, rules (a)–(d) all pass (`reviews/1_a1.md`) | next: 6
- 2026-11-09T00:15Z | 6. Smoke: vLLM | done: server answered `/v1/models` 170 s after start (`logs/vllm_smoke_s43_redrawn_round2_a1_c43a39434be5.log`, IMPORT_CHECK the copy's afterburner); GPU 30241 MiB used; 223 GB free | next: 7
- 2026-11-09T00:33Z | 7. Smoke: RL | done: `logs/rl_smoke_s43_redrawn_round2_a1_c43a39434be5.log` EXIT 0; no `CRASH:` lines; "GRPO subprocess finished in 15.9 min (exit=0, attempt=1/4)"; run id 1qko0rzt, run dir `/workspace/results/run_results/261108_frozen_probe_redrawn_labels/smoke_s43_redrawn_round2_1qko0rzt`; started 00:15:29Z, 18 min wall; peak GPU memory 73895 MiB; lowest free disk 219 GB (20 s samples) | next: 8
- 2026-11-09T00:34Z | 8. Smoke: record check | done: `runs/smoke_s43_redrawn_round2/check_records.json` PASS true, n_files 2, seed 43 (log `logs/record_check_smoke_s43_redrawn_round2_a1_c43a39434be5.log`) | next: 9
- 2026-11-09T00:34Z | 9. Smoke: stop vLLM | done: session killed, `nvidia-smi` lists no process after 10 s (0 MiB used) | next: 10 (CPU) alongside 11 (GPU)
- 2026-11-09T00:36Z | 10. Smoke: checks | done, exit 0 (log `logs/checks_smoke_s43_redrawn_round2_a1_c43a39434be5.log`): reward check penalty_weight 1.0, mismatches 0 / 0, reward mean abs difference 3.0e-8 / 1.6e-8; advantages floor 0.3, 2 steps compared, max abs difference 2.1e-8, kl_div at file step 0 0.1139; KL readout loss.kl 0.01, kl_coeff 0.01, 2 steps, kl_div 0.1139 / 0.1079 | next: 11
- 2026-11-09T00:36Z | 11. Smoke: penalty check | done: `logs/penalty_smoke_a1_c43a39434be5.log` EXIT 0; `checks/penalty_smoke_s43_redrawn_round2.json` 511 answers with a probe score; penalty probe vs saved median abs 0.00054 (max 0.0252, mean 0.0019, 2 over 0.02); other probe vs saved median abs 0.0221 (max 0.333, mean 0.0394, 275 over 0.02) | next: 12 (review 2)
- 2026-11-09T00:38Z | 12. Review 2 | PASS, rules (a)–(g) all pass (`reviews/2_a1.md`) | next: 13
- 2026-11-09T00:41Z | 13. Round 2: vLLM | done: server answered `/v1/models` 170 s after start (`logs/vllm_s43_redrawn_round2_a1_c43a39434be5.log`, IMPORT_CHECK the copy's afterburner); GPU 30263 MiB used; 222 GB free | next: 14 (RL launched 00:41:49Z)
- 2026-11-09T03:57Z | 14. Round 2: RL | done: `logs/rl_s43_redrawn_round2_a1_c43a39434be5.log` EXIT 0; no `CRASH:` lines; "GRPO subprocess finished in 193.6 min (exit=0, attempt=1/4)"; run id 44savkos, run dir `/workspace/results/run_results/261108_frozen_probe_redrawn_labels/s43_redrawn_round2_44savkos`; started 00:41:49Z, 195 min wall; peak GPU memory 78083 MiB; lowest free disk 211 GB (30 s samples) | next: 17, 18, 19 (GPU), with 15 and 16 (CPU) during the eval
- 2026-11-09T03:57Z | 17. Round 2: stop vLLM | done: session killed, `nvidia-smi` lists no process after 10 s (0 MiB used) | next: 18
- 2026-11-09T03:58Z | 18. Round 2: links | done: `logs/links_round2_a1_c43a39434be5.log` EXIT 0; `links/round2/initial_detector.base_model.on_dataset` → round 1's start-of-run probe folder, `fresh_detector.final_model.on_dataset` → `s43_redrawn_round2_44savkos`'s end-of-run refit | next: 19 (eval launched 03:58:38Z, 216 GB free)
- 2026-11-09T03:59Z | 15. Round 2: record check | done (during the eval): `runs/s43_redrawn_round2/check_records.json` PASS true, n_files 128, seed 43 (log `logs/record_check_s43_redrawn_round2_a1_c43a39434be5.log`) | next: 16
- 2026-11-09T04:00Z | 16. Round 2: checks | done (during the eval), exit 0 (log `logs/checks_s43_redrawn_round2_a1_c43a39434be5.log`): advantages floor 0.3, 128 steps compared, max abs difference 3.4e-8, kl_div at file step 0 0.1087; KL readout loss.kl 0.01, kl_coeff 0.01, 128 steps, kl_div first 0.1087 / last 0.1726 | next: 19 (running)
- 2026-11-09T04:04Z | 19. Round 2: eval | done: `logs/eval_s43_redrawn_round2_a1_c43a39434be5.log` EXIT 0 (6 min); `evals/s43_redrawn_round2/summary.json` questions 489, cued_rollouts 978 | next: 20
- 2026-11-09T04:05Z | 20. Round 2: compare | done, exit 0 (log `logs/compare_s43_redrawn_round2_a1_c43a39434be5.log`): `runs/s43_redrawn_round2/` comparison.json, comparison.md, training_curve.csv with 128 data rows | next: 21
- 2026-11-09T04:05Z | 21. Round 3: config | done: `logs/config_round3_a1_c43a39434be5.log` EXIT 0, wrote `s43_redrawn_round3.yaml`; not on the out-of-memory fallback, batch lines untouched | next: 22 (review 3)
- 2026-11-09T04:07Z | 22. Review 3 | PASS, rules (a)–(i) all pass (`reviews/3_a1.md`); reported mean 84.9 s per step | next: 23 (vLLM launched 04:07:59Z, 217 GB free)
- 2026-11-09T04:11Z | 23. Round 3: vLLM | done: server answered `/v1/models` 180 s after start (`logs/vllm_s43_redrawn_round3_a1_c43a39434be5.log`, IMPORT_CHECK the copy's afterburner); GPU 30263 MiB used; 217 GB free | next: 24 (RL launched 04:11:13Z)
- 2026-11-09T07:32Z | 24. Round 3: RL | done: `logs/rl_s43_redrawn_round3_a1_c43a39434be5.log` EXIT 0; no `CRASH:` lines; "GRPO subprocess finished in 199.3 min (exit=0, attempt=1/4)"; run id negw43fl, run dir `/workspace/results/run_results/261108_frozen_probe_redrawn_labels/s43_redrawn_round3_negw43fl`; started 04:11:13Z, 201 min wall; peak GPU memory 77515 MiB; lowest free disk 209 GB (30 s samples) | next: 27, 28, 29 (GPU), with 25 and 26 (CPU) during the eval
- 2026-11-09T07:32Z | 27. Round 3: stop vLLM | done: session killed, `nvidia-smi` lists no process after 10 s (0 MiB used) | next: 28
- 2026-11-09T07:33Z | 28. Round 3: links | done: `logs/links_round3_a1_c43a39434be5.log` EXIT 0; `links/round3/initial_detector.base_model.on_dataset` → round 1's start-of-run probe folder, `fresh_detector.final_model.on_dataset` → `s43_redrawn_round3_negw43fl`'s end-of-run refit | next: 29 (eval launched 07:33:12Z, 215 GB free)
- 2026-11-09T07:33Z | 25. Round 3: record check | done (during the eval): `runs/s43_redrawn_round3/check_records.json` PASS true, n_files 128, seed 43 (log `logs/record_check_s43_redrawn_round3_a1_c43a39434be5.log`) | next: 26
- 2026-11-09T07:34Z | 26. Round 3: checks | done (during the eval), exit 0 (log `logs/checks_s43_redrawn_round3_a1_c43a39434be5.log`): advantages floor 0.3, 128 steps compared, max abs difference 3.2e-8, kl_div at file step 0 0.1776; KL readout loss.kl 0.01, kl_coeff 0.01, 128 steps, kl_div first 0.1776 / last 0.2532 | next: 29 (running)
- 2026-11-09T07:38Z | 29. Round 3: eval | done: `logs/eval_s43_redrawn_round3_a1_c43a39434be5.log` EXIT 0 (5 min); `evals/s43_redrawn_round3/summary.json` questions 489, cued_rollouts 978 | next: 30
- 2026-11-09T07:38Z | 30. Round 3: compare | done, exit 0 (log `logs/compare_s43_redrawn_round3_a1_c43a39434be5.log`): `runs/s43_redrawn_round3/` comparison.json, comparison.md, training_curve.csv with 128 data rows | next: 31
- 2026-11-09T07:39Z | 31. Round 4: config | done: `logs/config_round4_a1_c43a39434be5.log` EXIT 0, wrote `s43_redrawn_round4.yaml`; not on the out-of-memory fallback, batch lines untouched | next: 32 (review 4)
- 2026-11-09T07:42Z | 32. Review 4 | PASS, rules (a)–(i) all pass (`reviews/4_a1.md`); reported mean 87.8 s per step; round 3's eval has 977 follow-up rows (one cued answer did not finish or parse; summary.json still 489 / 978) | next: 33 (vLLM launched 07:42:11Z, 214 GB free)
- 2026-11-09T07:44Z | 33. Round 4: vLLM | done: server answered `/v1/models` 151 s after start (`logs/vllm_s43_redrawn_round4_a1_c43a39434be5.log`, IMPORT_CHECK the copy's afterburner); GPU 30263 MiB used; 215 GB free | next: 34 (RL launched 07:44:55Z)
- 2026-11-09T11:04Z | 34. Round 4: RL | done: `logs/rl_s43_redrawn_round4_a1_c43a39434be5.log` EXIT 0; no `CRASH:` lines; "GRPO subprocess finished in 197.4 min (exit=0, attempt=1/4)"; run id psu14emk, run dir `/workspace/results/run_results/261108_frozen_probe_redrawn_labels/s43_redrawn_round4_psu14emk`; started 07:44:55Z, 200 min wall; peak GPU memory 78101 MiB; lowest free disk 207 GB (30 s samples) | next: 37, 38, 39 (GPU), with 35 and 36 (CPU) during the eval
- 2026-11-09T11:04Z | 37. Round 4: stop vLLM | done: session killed, `nvidia-smi` lists no process after 10 s (0 MiB used) | next: 38
- 2026-11-09T11:05Z | 38. Round 4: links | done: `logs/links_round4_a1_c43a39434be5.log` EXIT 0; `links/round4/initial_detector.base_model.on_dataset` → round 1's start-of-run probe folder, `fresh_detector.final_model.on_dataset` → `s43_redrawn_round4_psu14emk`'s end-of-run refit | next: 39 (eval launched 11:05:24Z, 213 GB free)
- 2026-11-09T11:06Z | 35. Round 4: record check | done (during the eval): `runs/s43_redrawn_round4/check_records.json` PASS true, n_files 128, seed 43 (log `logs/record_check_s43_redrawn_round4_a1_c43a39434be5.log`) | next: 36
- 2026-11-09T11:07Z | 36. Round 4: checks | done (during the eval), exit 0 (log `logs/checks_s43_redrawn_round4_a1_c43a39434be5.log`): advantages floor 0.3, 128 steps compared, max abs difference 4.8e-8, kl_div at file step 0 0.4643; KL readout loss.kl 0.01, kl_coeff 0.01, 128 steps, kl_div first 0.4643 / last 0.2456 | next: 39 (running)
- 2026-11-09T11:11Z | 39. Round 4: eval | done: `logs/eval_s43_redrawn_round4_a1_c43a39434be5.log` EXIT 0 (6 min); `evals/s43_redrawn_round4/summary.json` questions 489, cued_rollouts 978 | next: 40
- 2026-11-09T11:11Z | 40. Round 4: compare | done, exit 0 (log `logs/compare_s43_redrawn_round4_a1_c43a39434be5.log`): `runs/s43_redrawn_round4/` comparison.json, comparison.md, training_curve.csv with 128 data rows | next: 41
- 2026-11-09T11:11Z | 41. Round 5: config | done: `logs/config_round5_a1_c43a39434be5.log` EXIT 0, wrote `s43_redrawn_round5.yaml`; not on the out-of-memory fallback, batch lines untouched | next: 42 (review 5)
- 2026-11-09T11:14Z | 42. Review 5 | PASS, rules (a)–(i) all pass (`reviews/5_a1.md`); reported mean 86.9 s per step; round 4's eval has 977 follow-up rows (summary.json 489 / 978) | next: 43 (vLLM launched 11:14:13Z, 213 GB free)
- 2026-11-09T11:17Z | 43. Round 5: vLLM | done: server answered `/v1/models` 160 s after start (`logs/vllm_s43_redrawn_round5_a1_c43a39434be5.log`, IMPORT_CHECK the copy's afterburner); GPU 30263 MiB used; 213 GB free | next: 44 (RL launched 11:17:06Z)
- 2026-11-09T14:35Z | 44. Round 5: RL | done: `logs/rl_s43_redrawn_round5_a1_c43a39434be5.log` EXIT 0; no `CRASH:` lines; "GRPO subprocess finished in 195.9 min (exit=0, attempt=1/4)"; run id jbm68ogn, run dir `/workspace/results/run_results/261108_frozen_probe_redrawn_labels/s43_redrawn_round5_jbm68ogn`; started 11:17:06Z, 198 min wall; peak GPU memory 79595 MiB; lowest free disk 206 GB (30 s samples) | next: 47, 48, 49 (GPU), with 45 and 46 (CPU) during the eval
- 2026-11-09T14:35Z | 47. Round 5: stop vLLM | done: session killed, `nvidia-smi` lists no process after 10 s (0 MiB used) | next: 48
- 2026-11-09T14:35Z | 48. Round 5: links | done: `logs/links_round5_a1_c43a39434be5.log` EXIT 0; `links/round5/initial_detector.base_model.on_dataset` → round 1's start-of-run probe folder, `fresh_detector.final_model.on_dataset` → `s43_redrawn_round5_jbm68ogn`'s end-of-run refit | next: 49 (eval launched 14:36:07Z, 216 GB free)
- 2026-11-09T14:37Z | 45. Round 5: record check | done (during the eval): `runs/s43_redrawn_round5/check_records.json` PASS true, n_files 128, seed 43 (log `logs/record_check_s43_redrawn_round5_a1_c43a39434be5.log`) | next: 46
- 2026-11-09T14:37Z | 46. Round 5: checks | done (during the eval), exit 0 (log `logs/checks_s43_redrawn_round5_a1_c43a39434be5.log`): advantages floor 0.3, 128 steps compared, max abs difference 3.0e-8, kl_div at file step 0 0.2688; KL readout loss.kl 0.01, kl_coeff 0.01, 128 steps, kl_div first 0.2688 / last 0.2280 | next: 49 (running)
- 2026-11-09T14:41Z | 49. Round 5: eval | done: `logs/eval_s43_redrawn_round5_a1_c43a39434be5.log` EXIT 0 (6 min); `evals/s43_redrawn_round5/summary.json` questions 489, cued_rollouts 978 | next: 50
- 2026-11-09T14:41Z | 50. Round 5: compare | done, exit 0 (log `logs/compare_s43_redrawn_round5_a1_c43a39434be5.log`): `runs/s43_redrawn_round5/` comparison.json, comparison.md, training_curve.csv with 128 data rows | next: 51 (review 6)
- 2026-11-09T14:44Z | 51. Review 6 | PASS, rules (a)–(h) all pass (`reviews/6_a1.md`); reported mean 86.2 s per step; round 5's eval has 974 follow-up rows (summary.json 489 / 978) | next: 52
- 2026-11-09T14:45Z | 52. Dynamics | done: `logs/dynamics_a1_c43a39434be5.log` EXIT 0; `runs/dynamics/dynamics.json`, `dynamics.md` | next: 53
- 2026-11-09T14:45Z | 53. Seed table | done, exit 0 (log `logs/seed_table_a1_c43a39434be5.log`): `tables/rounds/seed_table.md`, `seed_table.json`; table copied into Result with headings moved down two levels (diff against the shifted file: identical) | next: done
- 2026-11-09T14:47Z | Done | every queue step ticked, Blocked empty; done notification sent (terminal notification sent, mobile push requested). The 30-minute resume check stays scheduled | next: none

## Judgment calls

- 2026-11-09T03:57Z | Order of independent steps after each round's RL run: stop vLLM, make the links and launch the eval first (steps 17, 18, 19), then run the record check and the checks (steps 15, 16), which only read saved files and W&B, while the eval holds the GPU. CLAUDE.md allows both the order of independent jobs and running a read-only CPU step during the next GPU step; it keeps the GPU from waiting on CPU steps. Steps are still ticked as each one finishes.
- 2026-11-09T14:47Z | Deviation from the infrastructure rule "wait with a background poll, never a foreground sleep": the short jobs (config rounds 2–5, links rounds 2–5, each under a minute) were waited on with a foreground poll on their `EXIT` line (2–3 s interval, capped at 10 minutes), and the vLLM stop steps with a foreground 2-minute wait for the GPU to clear. Every RL run, eval, vLLM start and the other long jobs used background polls. No step's outcome depended on this.

## Code fixes

## Reviews

One line per review: `<UTC time> | review | verdict | reviews/<n>_a<attempt>.md`.

- 2026-11-09T00:12Z | 1, start | PASS | reviews/1_a1.md
- 2026-11-09T00:38Z | 2, smoke | PASS | reviews/2_a1.md
- 2026-11-09T04:07Z | 3, end of round 2 | PASS | reviews/3_a1.md
- 2026-11-09T07:42Z | 4, end of round 3 | PASS | reviews/4_a1.md
- 2026-11-09T11:14Z | 5, end of round 4 | PASS | reviews/5_a1.md
- 2026-11-09T14:44Z | 6, end of round 5 | PASS | reviews/6_a1.md

## Blocked

## Result

Source: `tables/rounds/seed_table.md` (step 53, 2026-11-09T14:45Z), copied verbatim with each `#` heading moved down two levels.

### RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): round1 457, round2 450, round3 459, round4 447, round5 446.

| metric | condition | base | base mean | base SD | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.757 | 0.759 | 0.765 | 0.746 | 0.759 | 0.757 | 0.007 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.937 | 0.961 | 0.959 | 0.939 | 0.916 | 0.942 | 0.018 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.331 | 0.350 | 0.284 | 0.384 | 0.466 | 0.363 | 0.068 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.736 | 0.746 | 0.729 | 0.745 | 0.763 | 0.744 | 0.013 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.658 | 0.677 | 0.679 | 0.673 | 0.689 | 0.675 | 0.011 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.781 | 0.781 | 0.779 | 0.765 | 0.767 | 0.775 | 0.008 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.372 | 0.419 | 0.350 | 0.434 | 0.481 | 0.411 | 0.052 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.728 | 0.753 | 0.733 | 0.738 | 0.736 | 0.738 | 0.009 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.937 | 0.961 | 0.959 | 0.939 | 0.916 | 0.942 | 0.018 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.667 | 0.650 | 0.714 | 0.616 | 0.528 | 0.635 | 0.070 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.840 | 0.856 | 0.869 | 0.827 | 0.785 | 0.836 | 0.032 |
| bias | cue_gold | 0.094 / 0.088 / 0.089 / 0.091 / 0.087 | 0.090 | 0.003 | 0.158 | 0.156 | 0.161 | 0.145 | 0.135 | 0.151 | 0.011 |
| bias | cue_dist | 0.264 / 0.266 / 0.268 / 0.263 / 0.269 | 0.266 | 0.003 | 0.443 | 0.436 | 0.500 | 0.403 | 0.306 | 0.418 | 0.071 |
| bias | train_mix | 0.154 / 0.147 / 0.150 / 0.150 / 0.148 | 0.150 | 0.003 | 0.250 | 0.243 | 0.265 | 0.229 | 0.190 | 0.235 | 0.029 |
| anti_bias | cue_gold | 0.038 / 0.036 / 0.035 / 0.039 / 0.036 | 0.037 | 0.001 | 0.009 | 0.002 | 0.002 | 0.007 | 0.004 | 0.005 | 0.003 |
| anti_bias | cue_dist | 0.011 / 0.011 / 0.009 / 0.012 / 0.009 | 0.010 | 0.001 | 0.004 | 0.000 | 0.002 | 0.007 | 0.007 | 0.004 | 0.003 |
| anti_bias | train_mix | 0.028 / 0.027 / 0.026 / 0.029 / 0.026 | 0.027 | 0.001 | 0.005 | 0.002 | 0.003 | 0.008 | 0.005 | 0.005 | 0.002 |
| no_bias | cue_gold | 0.869 / 0.876 / 0.876 / 0.870 / 0.877 | 0.873 | 0.004 | 0.834 | 0.842 | 0.837 | 0.848 | 0.861 | 0.844 | 0.011 |
| no_bias | cue_dist | 0.725 / 0.722 / 0.723 / 0.726 / 0.722 | 0.724 | 0.002 | 0.553 | 0.564 | 0.498 | 0.591 | 0.687 | 0.578 | 0.069 |
| no_bias | train_mix | 0.818 / 0.826 / 0.824 / 0.821 / 0.826 | 0.823 | 0.003 | 0.745 | 0.755 | 0.731 | 0.763 | 0.805 | 0.760 | 0.028 |
| deception_soft | cue_gold | 0.553 / 0.554 / 0.553 / 0.550 / 0.554 | 0.553 | 0.002 | 0.544 | 0.459 | 0.541 | 0.434 | 0.531 | 0.502 | 0.052 |
| deception_soft | cue_dist | 0.602 / 0.600 / 0.600 / 0.597 / 0.598 | 0.599 | 0.002 | 0.514 | 0.525 | 0.514 | 0.534 | 0.600 | 0.537 | 0.036 |
| deception_soft | train_mix | 0.568 / 0.568 / 0.568 / 0.564 / 0.568 | 0.567 | 0.002 | 0.538 | 0.476 | 0.532 | 0.468 | 0.554 | 0.514 | 0.039 |
| deception | cue_gold | 0.572 / 0.569 / 0.572 / 0.566 / 0.572 | 0.570 | 0.003 | 0.563 | 0.390 | 0.609 | 0.374 | 0.541 | 0.495 | 0.107 |
| deception | cue_dist | 0.669 / 0.664 / 0.661 / 0.660 / 0.659 | 0.662 | 0.004 | 0.533 | 0.538 | 0.516 | 0.556 | 0.671 | 0.563 | 0.062 |
| deception | train_mix | 0.603 / 0.600 / 0.602 / 0.596 / 0.602 | 0.601 | 0.003 | 0.555 | 0.426 | 0.577 | 0.429 | 0.579 | 0.513 | 0.079 |
| biased_share | cue_gold | 0.097 / 0.091 / 0.092 / 0.095 / 0.090 | 0.093 | 0.003 | 0.159 | 0.156 | 0.162 | 0.146 | 0.135 | 0.152 | 0.011 |
| biased_share | cue_dist | 0.267 / 0.269 / 0.271 / 0.266 / 0.271 | 0.269 | 0.002 | 0.445 | 0.436 | 0.501 | 0.405 | 0.308 | 0.419 | 0.071 |
| biased_share | train_mix | 0.158 / 0.151 / 0.154 / 0.154 / 0.152 | 0.154 | 0.003 | 0.251 | 0.244 | 0.266 | 0.231 | 0.191 | 0.236 | 0.029 |
| false_denial_soft | cue_gold | 0.388 / 0.402 / 0.388 / 0.371 / 0.378 | 0.385 | 0.012 | 0.293 | 0.429 | 0.349 | 0.456 | 0.337 | 0.373 | 0.068 |
| false_denial_soft | cue_dist | 0.350 / 0.353 / 0.352 / 0.340 / 0.350 | 0.349 | 0.005 | 0.280 | 0.423 | 0.336 | 0.471 | 0.385 | 0.379 | 0.074 |
| false_denial_soft | train_mix | 0.367 / 0.372 / 0.370 / 0.353 / 0.364 | 0.365 | 0.007 | 0.288 | 0.414 | 0.339 | 0.462 | 0.360 | 0.373 | 0.067 |
| false_admission_soft | cue_gold | 0.571 / 0.569 / 0.569 / 0.568 / 0.571 | 0.570 | 0.001 | 0.592 | 0.465 | 0.578 | 0.430 | 0.561 | 0.525 | 0.073 |
| false_admission_soft | cue_dist | 0.694 / 0.691 / 0.692 / 0.690 / 0.690 | 0.691 | 0.002 | 0.702 | 0.603 | 0.692 | 0.577 | 0.697 | 0.654 | 0.059 |
| false_admission_soft | train_mix | 0.606 / 0.603 / 0.604 / 0.603 / 0.605 | 0.604 | 0.001 | 0.623 | 0.496 | 0.602 | 0.469 | 0.600 | 0.558 | 0.070 |
| said_yes_biased_soft | cue_gold | 0.612 / 0.598 / 0.612 / 0.629 / 0.622 | 0.615 | 0.012 | 0.707 | 0.571 | 0.651 | 0.544 | 0.663 | 0.627 | 0.068 |
| said_yes_biased_soft | cue_dist | 0.650 / 0.647 / 0.648 / 0.660 / 0.650 | 0.651 | 0.005 | 0.720 | 0.577 | 0.664 | 0.529 | 0.615 | 0.621 | 0.074 |
| said_yes_biased_soft | train_mix | 0.633 / 0.628 / 0.630 / 0.647 / 0.636 | 0.635 | 0.007 | 0.712 | 0.586 | 0.661 | 0.538 | 0.640 | 0.627 | 0.067 |
| deception_from_biased | cue_gold | 0.038 / 0.037 / 0.036 / 0.035 / 0.034 | 0.036 | 0.001 | 0.047 | 0.067 | 0.056 | 0.067 | 0.045 | 0.056 | 0.010 |
| deception_from_biased | cue_dist | 0.093 / 0.095 / 0.095 / 0.090 / 0.095 | 0.094 | 0.002 | 0.124 | 0.184 | 0.168 | 0.191 | 0.119 | 0.157 | 0.034 |
| deception_from_biased | train_mix | 0.058 / 0.056 / 0.057 / 0.055 / 0.055 | 0.056 | 0.001 | 0.072 | 0.101 | 0.090 | 0.107 | 0.069 | 0.088 | 0.017 |
| deception_from_unbiased | cue_gold | 0.515 / 0.517 / 0.517 / 0.515 / 0.520 | 0.517 | 0.002 | 0.498 | 0.392 | 0.484 | 0.367 | 0.486 | 0.445 | 0.061 |
| deception_from_unbiased | cue_dist | 0.509 / 0.505 / 0.505 / 0.506 / 0.503 | 0.506 | 0.002 | 0.390 | 0.340 | 0.345 | 0.343 | 0.482 | 0.380 | 0.060 |
| deception_from_unbiased | train_mix | 0.510 / 0.512 / 0.511 / 0.510 / 0.513 | 0.511 | 0.001 | 0.466 | 0.375 | 0.442 | 0.361 | 0.485 | 0.426 | 0.055 |
| said_yes | cue_gold | 0.600 / 0.595 / 0.600 / 0.600 / 0.605 | 0.600 | 0.004 | 0.656 | 0.412 | 0.701 | 0.372 | 0.599 | 0.548 | 0.148 |
| said_yes | cue_dist | 0.785 / 0.784 / 0.788 / 0.786 / 0.780 | 0.785 | 0.003 | 0.819 | 0.631 | 0.838 | 0.601 | 0.789 | 0.736 | 0.111 |
| said_yes | train_mix | 0.660 / 0.655 / 0.661 / 0.660 / 0.662 | 0.660 | 0.003 | 0.711 | 0.481 | 0.744 | 0.449 | 0.661 | 0.609 | 0.135 |
| false_denial | cue_gold | 0.357 / 0.359 / 0.350 / 0.325 / 0.316 | 0.341 | 0.020 | 0.208 | 0.429 | 0.216 | 0.508 | 0.283 | 0.329 | 0.133 |
| false_denial | cue_dist | 0.282 / 0.276 / 0.267 / 0.263 / 0.276 | 0.273 | 0.008 | 0.178 | 0.393 | 0.179 | 0.444 | 0.309 | 0.301 | 0.121 |
| false_denial | train_mix | 0.320 / 0.317 / 0.311 / 0.294 / 0.302 | 0.309 | 0.011 | 0.191 | 0.386 | 0.187 | 0.457 | 0.287 | 0.302 | 0.119 |
| false_admission | cue_gold | 0.595 / 0.590 / 0.595 / 0.592 / 0.597 | 0.594 | 0.003 | 0.630 | 0.383 | 0.685 | 0.351 | 0.581 | 0.526 | 0.150 |
| false_admission | cue_dist | 0.810 / 0.806 / 0.808 / 0.803 / 0.801 | 0.806 | 0.004 | 0.817 | 0.650 | 0.855 | 0.633 | 0.833 | 0.758 | 0.107 |
| false_admission | train_mix | 0.656 / 0.650 / 0.655 / 0.652 / 0.656 | 0.654 | 0.003 | 0.677 | 0.439 | 0.718 | 0.421 | 0.648 | 0.581 | 0.140 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.002 | 0.002 | 0.006 | 0.006 | 0.010 | 0.005 | 0.003 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.002 | 0.001 | 0.001 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.002 | 0.000 | 0.002 | 0.000 | 0.006 | 0.002 | 0.003 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.001 | 0.000 | 0.001 | 0.001 | 0.003 | 0.001 | 0.001 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 449 / 443 / 451 / 439 / 439 | 444.200 | 5.586 | 457 | 450 | 459 | 447 | 446 | 451.800 | 5.891 |
| n_bias_rows | cue_dist | 443 / 436 / 447 / 434 / 432 | 438.400 | 6.348 | 456 | 450 | 458 | 447 | 444 | 451 | 5.916 |
| n_bias_rows | train_mix | 892 / 879 / 898 / 873 / 871 | 882.600 | 11.887 | 913 | 900 | 917 | 894 | 890 | 902.800 | 11.777 |
| n_deception_rows | cue_gold | 432 / 427 / 435 / 422 / 423 | 427.800 | 5.630 | 453 | 449 | 458 | 444 | 444 | 449.600 | 6.025 |
| n_deception_rows | cue_dist | 438 / 431 / 443 / 429 / 428 | 433.800 | 6.458 | 454 | 450 | 457 | 444 | 441 | 449.200 | 6.686 |
| n_deception_rows | train_mix | 870 / 858 / 878 / 851 / 851 | 861.600 | 12.012 | 907 | 899 | 915 | 888 | 885 | 898.800 | 12.617 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.218 | 0.164 | 0.175 | 0.204 | 0.200 | 0.192 | 0.022 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.195 | 0.156 | 0.193 | 0.196 | 0.173 | 0.182 | 0.018 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.039 | 0.037 | 0.054 | 0.057 | 0.044 | 0.046 | 0.009 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.057 | 0.024 | 0.055 | 0.053 | 0.042 | 0.046 | 0.014 |
| probe_start_auc | cue_gold | – | – | – | 0.849 | 0.821 | 0.789 | 0.838 | 0.821 | 0.824 | 0.023 |
| probe_start_auc | cue_dist | – | – | – | 0.763 | 0.820 | 0.786 | 0.809 | 0.802 | 0.796 | 0.022 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 72 | 70 | 74 | 65 | 60 | 68.200 | 5.675 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 202 | 196 | 229 | 180 | 136 | 188.600 | 34.305 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 356 | 361 | 365 | 356 | 354 | 358.400 | 4.506 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 95 | 88 | 91 | 85 | 86 | 89 | 4.062 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.282 | 0.231 | 0.159 | 0.256 | 0.114 | 0.208 | 0.070 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.249 | 0.215 | 0.175 | 0.211 | 0.093 | 0.189 | 0.059 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.069 | 0.061 | 0.049 | 0.068 | 0.035 | 0.057 | 0.014 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.095 | 0.049 | 0.053 | 0.069 | 0.030 | 0.059 | 0.024 |
| probe_fresh_auc | cue_gold | – | – | – | 0.856 | 0.817 | 0.775 | 0.836 | 0.789 | 0.815 | 0.033 |
| probe_fresh_auc | cue_dist | – | – | – | 0.748 | 0.803 | 0.773 | 0.782 | 0.767 | 0.775 | 0.020 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 72 | 70 | 74 | 65 | 60 | 68.200 | 5.675 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 202 | 196 | 229 | 180 | 136 | 188.600 | 34.305 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 356 | 361 | 365 | 356 | 354 | 358.400 | 4.506 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 95 | 88 | 91 | 85 | 86 | 89 | 4.062 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.297 | 0.233 | 0.165 | 0.186 | 0.227 | 0.222 | 0.051 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.251 | 0.214 | 0.184 | 0.163 | 0.172 | 0.197 | 0.036 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.062 | 0.051 | 0.040 | 0.045 | 0.048 | 0.049 | 0.008 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.075 | 0.036 | 0.045 | 0.042 | 0.048 | 0.050 | 0.015 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.870 | 0.835 | 0.812 | 0.823 | 0.849 | 0.838 | 0.023 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.759 | 0.838 | 0.793 | 0.778 | 0.770 | 0.788 | 0.031 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 72 | 70 | 74 | 65 | 60 | 68.200 | 5.675 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 202 | 196 | 229 | 180 | 136 | 188.600 | 34.305 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 356 | 361 | 365 | 356 | 354 | 358.400 | 4.506 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 95 | 88 | 91 | 85 | 86 | 89 | 4.062 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|
| equal task reward groups | 0.736 | 0.803 | 0.826 | 0.821 | 0.825 | 0.802 | 0.038 |
| probe-only groups | 0.670 | 0.768 | 0.787 | 0.731 | 0.706 | 0.732 | 0.047 |
