# Runs

State file for the unattended run. Read this first after any restart.

## Queue

- [x] 1. Pod check
- [x] 2. Start vLLM (smoke)
- [x] 3. Smoke RL run (2 steps)
- [x] 4. Smoke record check
- [x] 5. Stop vLLM
- [x] 6. checks.py smoke, advantages, probe (smoke run)
- [x] 7. Review 1 (smoke run)
- [x] 8. Seed 42: start vLLM
- [x] 9. Seed 42: RL run (128 steps)
- [x] 10. Seed 42: record check
- [x] 11. Seed 42: checks.py advantages, probe
- [x] 12. Review 2 (seed 42)
- [x] 13. Seed 42: stop vLLM
- [x] 14. Seed 42: final eval
- [x] 15. Seed 42: base answers scored with its initial probe
- [x] 16. Seed 42: compare.py
- [x] 17. Seed 43: start vLLM
- [x] 18. Seed 43: RL run (128 steps)
- [x] 19. Seed 43: record check
- [x] 20. Seed 43: checks.py advantages, probe
- [x] 21. Review 3 (seed 43)
- [x] 22. Seed 43: stop vLLM
- [x] 23. Seed 43: final eval
- [x] 24. Seed 43: base answers scored with its initial probe
- [x] 25. Seed 43: compare.py
- [x] 26. Seed 44: start vLLM
- [x] 27. Seed 44: RL run (128 steps)
- [x] 28. Seed 44: record check
- [x] 29. Seed 44: checks.py advantages, probe
- [x] 30. Review 4 (seed 44)
- [x] 31. Seed 44: stop vLLM
- [x] 32. Seed 44: final eval
- [x] 33. Seed 44: base answers scored with its initial probe
- [x] 34. Seed 44: compare.py
- [x] 35. seed_table.py (three seeds)
- [x] 36. dynamics.py (three seeds and 261024's run)
- [x] 37. Review 5 (auditor) and Result

## Log

One line per step: `<UTC time> | step | outcome | next`.

- 2026-10-27 03:13 | 1 | pass: hostname c61229747508, one GPU (NVIDIA H100 80GB HBM3); `uname -r` 6.8.0-147-generic, `uptime -s` 2026-10-19 21:28:35, driver 595.91.07, CUDA 13.2, `df -h /workspace` 513G size, 373G used, 140G free (73%); no GPU process; `logs/` created | 2
- 2026-10-27 03:19 | 2 | pass: vLLM (a1, 0.35) started 03:13:50, `/v1/models` answered after 351 s (`dynamic-lora-model`); IMPORT_CHECK afterburner from /workspace/sycophancy-rl/obfuscation-atlas/third_party/afterburner/src; GPU memory 30405 MiB; disk 140 GB free | 3
- 2026-10-27 03:37 | 3 | pass: smoke RL run (a1) started 03:19:59, log ends `EXIT 0`; no `CRASH:` lines; `GRPO subprocess finished in 15.6 min (exit=0, attempt=1/4)`; `OK: run s2okcxp8 verified complete (responses=2)`; run id s2okcxp8, run dir /workspace/results/run_results/261027_probe_penalty_weight_sweep/smoke_weight_0_6_s42_s2okcxp8; peak GPU memory 78363 MiB; lowest free disk 135 GB; the four `WARNING: could not remove activation cache` lines were each followed by `OK: removed activation cache` for the same cache | 4
- 2026-10-27 03:39 | 4 | done: `runs/smoke_weight_0_6_s42/check_records.json` written, exit 0; `"PASS": true`, `"n_files": 2`, `"seed": 42` (`PASS: 2/2 files`) | 5
- 2026-10-27 03:39 | 5 | pass: `261027-w06-vllm` killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 6
- 2026-10-27 03:44 | 6 | pass: exit 0; wrote `checks/smoke.json`, `checks/advantages_smoke_weight_0_6_s42.json`, `checks/probe_smoke_weight_0_6_s42.json` | 7
- 2026-10-27 03:45 | 7 | Review 1 PASS (reward recompute within 1.35e-8, advantage recompute within 1.72e-8, probe test AUC 0.87301); verdict under Reviews | 8
- 2026-10-27 03:52 | 8 | pass: vLLM (a1, 0.35) started 03:45:42 (disk 138 GB free, no GPU process before), `/v1/models` answered after 381 s; IMPORT_CHECK afterburner from the copy; GPU memory 30263 MiB | 9
- 2026-10-27 06:54 | 9 | pass: seed 42 RL run (a1) started 03:52:26, log ends `EXIT 0`; no `CRASH:` lines; `GRPO subprocess finished in 179.5 min (exit=0, attempt=1/4)`; `OK: run nmc25emo verified complete (responses=128)`; run id nmc25emo, run dir /workspace/results/run_results/261027_probe_penalty_weight_sweep/weight_0_6_s42_nmc25emo; peak GPU memory 80271 MiB; lowest free disk 125 GB; the four `WARNING: could not remove activation cache` lines were each followed by `OK: removed activation cache` for the same cache | 10
- 2026-10-27 06:56 | 10 | done: `runs/weight_0_6_s42/check_records.json` written, exit 0; `"PASS": true`, `"n_files": 128`, `"seed": 42` (`PASS: 128/128 files`) | 11
- 2026-10-27 06:58 | 11 | pass: exit 0; wrote `checks/advantages_weight_0_6_s42.json`, `checks/probe_weight_0_6_s42.json` (probe check read `logs/rl_weight_0_6_s42_a1_c61229747508.log`) | 12
- 2026-10-27 06:59 | 12 | Review 2 PASS (128/128 records, advantage recompute within 2.87e-8, probe test AUC 0.87301); verdict under Reviews | 13
- 2026-10-27 06:59 | 13 | pass: `261027-w06-vllm` killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 14
- 2026-10-27 07:07 | 14 | pass: eval (a1) started 06:59:49 (disk 131 GB free, no GPU process before), log ends `EXIT 0` (07:07:03); adapter `weight_0_6_s42_nmc25emo/checkpoints/final`; `evals/weight_0_6_s42/summary.json` has `"questions": 489`, `"cued_rollouts": 978` | 15
- 2026-10-27 07:10 | 15 | pass: `copy_base` exit 0; rescoring (a1) started 07:08:12 (disk 131 GB free, no GPU process before), log ends `EXIT 0` (07:10:16); `evals/base_rescored_weight_0_6_s42/summary.json` has `"questions": 489`, `"cued_rollouts": 978`; cmp of `labels.jsonl` with 261025's `evals/base_rescored_floor_s42/labels.jsonl`: identical | 16
- 2026-10-27 07:10 | 16 | pass: compare.py exit 0; wrote `runs/weight_0_6_s42/comparison.md`, `comparison.json`, `training_curve.csv` (128 data rows) | 17
- 2026-10-27 07:14 | 17 | pass: vLLM (a1, 0.35) started 07:10:51 (disk 131 GB free, no GPU process before), `/v1/models` answered after 201 s; IMPORT_CHECK afterburner from the copy; GPU memory 30263 MiB | 18
- 2026-10-27 11:01 | 18 | pass: seed 43 RL run (a1) started 07:14:28, log ends `EXIT 0`; `CRASH:` lines: `CRASH: weight_0_6_s43 attempt 1 failed with exit code 1` (out of GPU memory at training batch 40, W&B run 8so2j3f8, 39 steps logged, mean answer length 190.1 tokens; see Judgment calls 08:22); `GRPO subprocess finished` lines: `GRPO subprocess finished in 58.7 min (exit=1, attempt=1/4)`, `GRPO subprocess finished in 166.0 min (exit=0, attempt=2/4)`; `OK: run p4fxexen verified complete (responses=128)`; run id p4fxexen, run dir /workspace/results/run_results/261027_probe_penalty_weight_sweep/weight_0_6_s43_p4fxexen; the launcher removed the crashed run's partial folder weight_0_6_s43_8so2j3f8 and no activation cache of 8so2j3f8 is left under /workspace/results/activations/; peak GPU memory 80789 MiB; lowest free disk 122 GB; the four `WARNING: could not remove activation cache` lines were each followed by `OK: removed activation cache` for the same cache | 19
- 2026-10-27 11:02 | 19 | done: `runs/weight_0_6_s43/check_records.json` written, exit 0; `"PASS": true`, `"n_files": 128`, `"seed": 43` (`PASS: 128/128 files`) | 20
- 2026-10-27 11:04 | 20 | pass: exit 0; wrote `checks/advantages_weight_0_6_s43.json`, `checks/probe_weight_0_6_s43.json` (probe check read `logs/rl_weight_0_6_s43_a1_c61229747508.log`, the attempt whose run finished; it holds both launcher attempts) | 21
- 2026-10-27 11:05 | 21 | Review 3 PASS (128/128 records, advantage recompute within 3.42e-8, probe test AUC 0.87301); verdict under Reviews (its `##` heading written as `####`, as for review 2) | 22
- 2026-10-27 11:05 | 22 | pass: `261027-w06-vllm` killed; `nvidia-smi` lists no process after 5 s (0 MiB used) | 23
- 2026-10-27 11:11 | 23 | pass: eval (a1) started 11:05:48 (disk 131 GB free, no GPU process before), log ends `EXIT 0` (11:11); adapter `weight_0_6_s43_p4fxexen/checkpoints/final`; `evals/weight_0_6_s43/summary.json` has `"questions": 489`, `"cued_rollouts": 978` | 24
- 2026-10-27 11:15 | 24 | pass: `copy_base` exit 0; rescoring (a1) started 11:13:00 (disk 131 GB free, no GPU process before), log ends `EXIT 0` (11:15:04); `evals/base_rescored_weight_0_6_s43/summary.json` has `"questions": 489`, `"cued_rollouts": 978`; cmp of `labels.jsonl` with 261025's `evals/base_rescored_floor_s42/labels.jsonl`: identical | 25
- 2026-10-27 11:15 | 25 | pass: compare.py exit 0; wrote `runs/weight_0_6_s43/comparison.md`, `comparison.json`, `training_curve.csv` (128 data rows) | 26
- 2026-10-27 11:19 | 26 | pass: vLLM (a1, 0.35) started 11:15:45 (disk 131 GB free, no GPU process before), `/v1/models` answered after 200 s; IMPORT_CHECK afterburner from the copy; GPU memory 30263 MiB | 27
- 2026-10-27 14:16 | 27 | pass: seed 44 RL run (a1) started 11:19:24, log ends `EXIT 0`; no `CRASH:` lines; `GRPO subprocess finished in 175.3 min (exit=0, attempt=1/4)`; `OK: run 7z6tnfzr verified complete (responses=128)`; run id 7z6tnfzr, run dir /workspace/results/run_results/261027_probe_penalty_weight_sweep/weight_0_6_s44_7z6tnfzr; peak GPU memory 79022 MiB; lowest free disk 124 GB; the four `WARNING: could not remove activation cache` lines were each followed by `OK: removed activation cache` for the same cache | 28
- 2026-10-27 14:18 | 28 | done: `runs/weight_0_6_s44/check_records.json` written, exit 0; `"PASS": true`, `"n_files": 128`, `"seed": 44` (`PASS: 128/128 files`) | 29
- 2026-10-27 14:20 | 29 | pass: exit 0; wrote `checks/advantages_weight_0_6_s44.json`, `checks/probe_weight_0_6_s44.json` (probe check read `logs/rl_weight_0_6_s44_a1_c61229747508.log`) | 30
- 2026-10-27 14:21 | 30 | Review 4 PASS (128/128 records, advantage recompute within 1.93e-8, probe test AUC 0.87301); verdict under Reviews (its `##` heading written as `####`, as for review 2) | 31
- 2026-10-27 14:22 | 31 | pass: `261027-w06-vllm` killed; `nvidia-smi` lists no process after 10 s (0 MiB used) | 32
- 2026-10-27 14:30 | 32 | pass: eval (a1) started 14:22:27 (disk 129 GB free, no GPU process before), log ends `EXIT 0` (14:30); adapter `weight_0_6_s44_7z6tnfzr/checkpoints/final`; `evals/weight_0_6_s44/summary.json` has `"questions": 489`, `"cued_rollouts": 978` | 33
- 2026-10-27 14:34 | 33 | pass: `copy_base` exit 0; rescoring (a1) started 14:31:16 (disk 129 GB free, no GPU process before), log ends `EXIT 0` (14:34:00); `evals/base_rescored_weight_0_6_s44/summary.json` has `"questions": 489`, `"cued_rollouts": 978`; cmp of `labels.jsonl` with 261025's `evals/base_rescored_floor_s42/labels.jsonl`: identical | 34
- 2026-10-27 14:34 | 34 | pass: compare.py exit 0; wrote `runs/weight_0_6_s44/comparison.md`, `comparison.json`, `training_curve.csv` (128 data rows) | 35
- 2026-10-27 14:34 | 35 | pass: seed_table.py exit 0; wrote `tables/weight_0_6/seed_table.md`, `seed_table.json` (runs s42 nmc25emo, s43 p4fxexen, s44 7z6tnfzr; base 261013's `evals/base`) | 36
- 2026-10-27 14:36 | 36 | pass: dynamics.py exit 0; wrote `runs/dynamics/dynamics.json`, `dynamics.md` | 37
- 2026-10-27 14:45 | 37 | Review 5 (auditor) PASS (every number in `seed_table.md` and the three `comparison.md` matched; reruns byte-identical); verdict under Reviews (its headings moved down two levels, as for reviews 2–4); Result filled with `tables/weight_0_6/seed_table.md` and the "Main result" section of each `runs/weight_0_6_s4<S>/comparison.md`, copied by script with heading markers moved down two levels and checked line by line against the source files (diff: identical); Blocked is empty | done

## Judgment calls

- 2026-10-27 06:59 UTC, step 12: review 2's verdict began with a `##` heading line; under Reviews I wrote it as `####` so it nests under its `###` entry instead of opening a new top-level section of `runs.md`. The verdict text is otherwise copied verbatim.
- 2026-10-27 08:22 UTC, step 18 (seed 43 RL, a1): the launcher's first attempt ran out of GPU memory at training batch 40 of 128 (39 steps done), in `F.log_softmax` of `compute_logprobs_from_logits`, called from `grpo_trainer.py` line 1595 `train_on_generation_batch`; the error's last line: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a total capacity of 79.18 GiB of which 3.86 GiB is free. Process 137808 has 29.55 GiB memory in use. Including non-PyTorch memory, this process has 45.75 GiB memory in use.` Launcher lines: `GRPO subprocess finished in 58.7 min (exit=1, attempt=1/4)`, `CRASH: weight_0_6_s43 attempt 1 failed with exit code 1`, `removed partial run directory for crashed run 8so2j3f8: /workspace/results/run_results/261027_probe_penalty_weight_sweep/weight_0_6_s43_8so2j3f8`. W&B run 8so2j3f8: 39 steps logged, mean answer length 190.1 tokens. The launcher started its own retry (W&B run p4fxexen). CLAUDE.md allows the out-of-memory fallback only after the launcher's own retries did not get past it, so I wait on this attempt; no change made.

## Code fixes

## Reviews

### Review 1 (smoke run, step 7), 2026-10-27 03:45 UTC, fresh general-purpose agent: PASS

**Overall verdict: PASS.** Every rule holds and all five files exist and can be read. The four JSON files each parse as JSON. I changed nothing.

**Smoke record check** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/runs/smoke_weight_0_6_s42/check_records.json`)
- `"PASS": true`: PASS (the file says `true`; both per-step entries also say `PASS: true`)
- `"n_files": 2`: PASS (2)
- `"seed": 42`: PASS (42)

**Reward recompute** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/checks/smoke.json`)
- `penalty_weight` is 0.6: PASS (0.6)
- `probe_score_present_iff_parsed_mismatches` is 0 in every step: PASS (step 0: 0; step 1: 0)
- `reward_mean_abs_difference` ≤ 0.001 in every step: PASS (step 0: 1.15e-8; step 1: 1.35e-8)

**Advantage recompute** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/checks/advantages_smoke_weight_0_6_s42.json`)
- `advantage_std_floor` is 0.3: PASS (0.3)
- `steps_compared` is 2: PASS (2)
- `max_abs_difference` ≤ 1e-4: PASS (1.72e-8)

**Probe refit** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/checks/probe_smoke_weight_0_6_s42.json`)
- `train_biased_unbiased` equals `expected_train_biased_unbiased`: PASS ([756, 756] = [756, 756])
- `test_biased_unbiased` equals `expected_test_biased_unbiased`: PASS ([292, 485] = [292, 485])
- Both entries of `saved_layers` are [13, 14, 15, 16]: PASS (`initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset` are both [13, 14, 15, 16]; there are exactly 2 entries)
- `test_auc` between 0.85 and 0.89: PASS (0.87301)

**Reported values (no verdict)**
- `penalty_mean_abs_difference` per step (from `smoke.json`): step 0 is 3.72e-10, step 1 is 1.59e-9.
- `recomputed_without_floor` per step (from the advantages file): step 0 is 0.68389. Step 1 is 0.70662. With the floor the recomputes are 0.40151 and 0.39851, and W&B logged 0.40151 and 0.39851.
- `vs_261024_probe_fit`: `test_score_max_abs_difference` 0.0 and `weight_max_abs_difference` 0.0.
- `test_mean_biased`: 0.72218
- `test_mean_unbiased`: 0.25478
- Smoke run minutes (from `/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/runs.md`, step 3): `GRPO subprocess finished in 15.6 min (exit=0, attempt=1/4)`. It started at 03:19:59 UTC (8:19 PM PDT on Oct 26) and the step-3 line is logged at 03:37 UTC.
- Peak GPU memory (`runs.md`, step 3): 78363 MiB. Step 2 recorded 30405 MiB after vLLM started.
- Crashed attempts (`runs.md`): none.
  - The run finished on attempt 1 of 4, and step 3 says `no CRASH: lines`.
  - The "Code fixes" and "Blocked" sections are empty.
  - The only warnings noted were four `WARNING: could not remove activation cache` lines, and each was followed by `OK: removed activation cache` for the same cache.

### Review 2 (seed 42, step 12), 2026-10-27 06:59 UTC, fresh general-purpose agent: PASS

#### Review 2 (seed 42, step 12): overall verdict PASS

Every rule holds. All four files exist and parse as JSON or YAML. I changed nothing.

**Record check** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/runs/weight_0_6_s42/check_records.json`)
- `"PASS": true`: PASS (the file says `true`; all 128 `per_step` entries also say `PASS: true`)
- `"n_files": 128`: PASS (128)
- `"seed": 42`: PASS (42)

**Run config** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/weight_0_6_s42.yaml`, under `overrides`)
- `loss.detector` is 0.6: PASS (`loss.detector: 0.6`)
- `loss.advantage_std_floor` is 0.3: PASS (`loss.advantage_std_floor: 0.3`)
- `training.seed` is 42: PASS (`training.seed: 42`)
- `training.max_steps` is 128: PASS (`training.max_steps: 128`)

**Advantage recompute** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/checks/advantages_weight_0_6_s42.json`)
- `advantage_std_floor` is 0.3: PASS (0.3)
- `steps_compared` is 128: PASS (128). The `per_step` list has 128 entries, file steps 0 to 127.
- `max_abs_difference` ≤ 1e-4: PASS (2.87e-8, i.e. 2.8697e-08). This is the largest of the per-step `abs_difference` values. It is inside the 2.0e-8 to 2.9e-8 range that 261025's seeds matched to.

**Probe refit** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/checks/probe_weight_0_6_s42.json`)
- `train_biased_unbiased` equals `expected_train_biased_unbiased`: PASS ([756, 756] = [756, 756])
- `test_biased_unbiased` equals `expected_test_biased_unbiased`: PASS ([292, 485] = [292, 485])
- Both entries of `saved_layers` are [13, 14, 15, 16]: PASS. There are exactly 2 entries, `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`, and both are [13, 14, 15, 16].
- `test_auc` between 0.85 and 0.89: PASS (0.87301, i.e. 0.8730052)

**Reported values (no verdict)**
- `vs_261024_probe_fit`: `test_score_max_abs_difference` 0.0 and `weight_max_abs_difference` 0.0.
- `test_mean_biased`: 0.72218
- `test_mean_unbiased`: 0.25478
- From `/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/runs.md`, step 9:
  - **Run minutes:** `GRPO subprocess finished in 179.5 min (exit=0, attempt=1/4)`. The run started at 03:52:26 UTC (8:52 PM PDT on Oct 26) and the step-9 line is logged at 06:54 UTC (11:54 PM PDT on Oct 26).
  - **Lowest free disk:** 125 GB.
  - **Peak GPU memory:** 80271 MiB. Step 8 recorded 30263 MiB after vLLM started.
  - **Crashed attempts:** none.
    - Step 9 says `no CRASH: lines`, and the run finished on attempt 1 of 4.
    - The Code fixes and Blocked sections are empty.
    - As a cross-check, I searched `logs/rl_weight_0_6_s42_a1_c61229747508.log`. It has 0 `CRASH:` lines and no `removed partial run directory` line, and it ends with `EXIT 0`. Because nothing crashed, there are no W&B ids, batches reached or mean answer lengths to report.
  - **Other notes in step 9:**
    - W&B run id nmc25emo, run dir `/workspace/results/run_results/261027_probe_penalty_weight_sweep/weight_0_6_s42_nmc25emo`.
    - Four `WARNING: could not remove activation cache` lines, each followed by `OK: removed activation cache` for the same cache.

### Review 3 (seed 43, step 21), 2026-10-27 11:05 UTC, fresh general-purpose agent: PASS

#### Review 3 (seed 43, step 21): PASS

Every rule passes. All four files exist and parse as JSON or YAML. I changed nothing.

**Record check** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/runs/weight_0_6_s43/check_records.json`)
- `"PASS": true`: PASS. The file says `true`, and all 128 `per_step` entries (file steps 0 to 127) also say `PASS: true`.
- `"n_files": 128`: PASS (128)
- `"seed": 43`: PASS (43). The W&B run id is p4fxexen.

**Run config** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/weight_0_6_s43.yaml`, under `overrides`)
- `loss.detector` is 0.6: PASS (`loss.detector: 0.6`)
- `loss.advantage_std_floor` is 0.3: PASS (`loss.advantage_std_floor: 0.3`)
- `training.seed` is 43: PASS (`training.seed: 43`)
- `training.max_steps` is 128: PASS (`training.max_steps: 128`)

**Advantage recompute** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/checks/advantages_weight_0_6_s43.json`)
- `advantage_std_floor` is 0.3: PASS (0.3)
- `steps_compared` is 128: PASS (128). The `per_step` list has 128 entries, file steps 0 to 127.
- `max_abs_difference` ≤ 1e-4: PASS (3.42e-8, exactly 3.4247857605418375e-08). It equals the largest per-step `abs_difference`. It is slightly above the 2.0e-8 to 2.9e-8 that 261025's seeds matched to, but far below the 1e-4 limit.

**Probe refit** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/checks/probe_weight_0_6_s43.json`)
- `train_biased_unbiased` equals `expected_train_biased_unbiased`: PASS ([756, 756] = [756, 756])
- `test_biased_unbiased` equals `expected_test_biased_unbiased`: PASS ([292, 485] = [292, 485])
- Both entries of `saved_layers` are [13, 14, 15, 16]: PASS. There are exactly two entries, `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`, and both are [13, 14, 15, 16].
- `test_auc` between 0.85 and 0.89: PASS (0.87301, exactly 0.8730052252506708)

The record check, the advantage check and the probe check all name the same run folder, `/workspace/results/run_results/261027_probe_penalty_weight_sweep/weight_0_6_s43_p4fxexen`.

**Reported values (no verdict)**
- `vs_261024_probe_fit`: `test_score_max_abs_difference` 0.0 and `weight_max_abs_difference` 0.0.
- `test_mean_biased`: 0.72218
- `test_mean_unbiased`: 0.25478
- From `/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/runs.md`, step 18:
  - **Run minutes:** 166.0 for the attempt that finished (`GRPO subprocess finished in 166.0 min (exit=0, attempt=2/4)`). The crashed first attempt took 58.7 minutes before it. The whole step started at 07:14:28 UTC (12:14 AM PDT on Oct 27) and its line is logged at 11:01 UTC (4:01 AM PDT).
  - **Lowest free disk:** 122 GB.
  - **Peak GPU memory:** 80789 MiB. Step 17 recorded 30263 MiB after vLLM started.
  - **Crashed attempts:** one.
    - W&B run 8so2j3f8, attempt 1 of 4. It ran out of GPU memory at training batch 40 of 128, with 39 steps logged, and its mean answer length was 190.1 tokens.
    - The error was `torch.OutOfMemoryError ... Tried to allocate 5.96 GiB`, raised in `F.log_softmax` inside `compute_logprobs_from_logits`. That was called from `train_on_generation_batch`, `grpo_trainer.py` line 1595.
    - The launcher retried by itself, and the retry became run p4fxexen. No code change was made; the Code fixes and Blocked sections are empty.
    - I checked this in `/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/logs/rl_weight_0_6_s43_a1_c61229747508.log`. Line 1656 is `GRPO subprocess finished in 58.7 min (exit=1, attempt=1/4)` and line 1657 is `CRASH: weight_0_6_s43 attempt 1 failed with exit code 1`. Line 1658 records that the partial folder `weight_0_6_s43_8so2j3f8` was removed. Line 6008 is `... 166.0 min (exit=0, attempt=2/4)`, line 6010 is `OK: run p4fxexen verified complete (responses=128)` and line 6038 is `EXIT 0`.
    - The batch reached, the step count and the answer length come from `runs.md` only; I did not check them against W&B.
  - **Other notes in step 18:** four `WARNING: could not remove activation cache` lines, each followed by `OK: removed activation cache` for the same cache. No activation cache from 8so2j3f8 is left.

### Review 4 (seed 44, step 30), 2026-10-27 14:21 UTC, fresh general-purpose agent: PASS

#### Review 4 (seed 44, step 30): overall verdict PASS

All four rule files exist and could be read; the JSON and YAML files parsed without errors. Every rule passes. I changed no files.

**Record check** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/runs/weight_0_6_s44/check_records.json`)
- `"PASS": true`: PASS. The file says `true`, and all 128 `per_step` entries also say `PASS: true`.
- `"n_files": 128`: PASS (128).
- `"seed": 44`: PASS (44). The W&B run id is 7z6tnfzr.

**Run config** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/weight_0_6_s44.yaml`, under `overrides`)
- `loss.detector` is 0.6: PASS (`loss.detector: 0.6`).
- `loss.advantage_std_floor` is 0.3: PASS (`loss.advantage_std_floor: 0.3`).
- `training.seed` is 44: PASS (`training.seed: 44`).
- `training.max_steps` is 128: PASS (`training.max_steps: 128`).

**Advantage recompute** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/checks/advantages_weight_0_6_s44.json`)
- `advantage_std_floor` is 0.3: PASS (0.3).
- `steps_compared` is 128: PASS (128). The `per_step` list has 128 entries.
- `max_abs_difference` ≤ 1e-4: PASS (1.93e-8, exactly 1.926121026563088e-08). This equals the largest per-step `abs_difference`.

**Probe refit** (`/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/checks/probe_weight_0_6_s44.json`)
- `train_biased_unbiased` equals `expected_train_biased_unbiased`: PASS ([756, 756] = [756, 756]).
- `test_biased_unbiased` equals `expected_test_biased_unbiased`: PASS ([292, 485] = [292, 485]).
- Both entries of `saved_layers` are [13, 14, 15, 16]: PASS. There are exactly two entries, `initial_detector.base_model.on_dataset` and `fresh_detector.final_model.on_dataset`, and both are [13, 14, 15, 16].
- `test_auc` between 0.85 and 0.89: PASS (0.87301, exactly 0.8730052252506708).

The record check, the advantage check and the probe check all name the same run folder: `/workspace/results/run_results/261027_probe_penalty_weight_sweep/weight_0_6_s44_7z6tnfzr`. The probe check read `logs/rl_weight_0_6_s44_a1_c61229747508.log`.

**Reported values (no verdict)**
- `vs_261024_probe_fit`: `test_score_max_abs_difference` 0.0 and `weight_max_abs_difference` 0.0.
- `test_mean_biased`: 0.72218 (0.7221844251680647).
- `test_mean_unbiased`: 0.25478 (0.25478438520600644).
- These come from `/workspace/sycophancy_experiments/261027_probe_penalty_weight_sweep/weight_0_6/runs.md`, step 27:
  - **Run minutes:** 175.3, from `GRPO subprocess finished in 175.3 min (exit=0, attempt=1/4)`.
    - The step started at 11:19:24 UTC (4:19 AM PDT on Oct 27).
    - Its log line is timed 14:16 UTC (7:16 AM PDT).
  - **Lowest free disk:** 124 GB.
  - **Peak GPU memory:** 79022 MiB. For comparison, step 26 recorded 30263 MiB just after vLLM started.
  - **Crashed attempts:** none, so there is no W&B id, batch reached or mean answer length to report.
    - Step 27 says "no `CRASH:` lines", and the run finished on attempt 1 of 4.
    - The Code fixes and Blocked sections of `runs.md` are empty.
    - I cross-checked `logs/rl_weight_0_6_s44_a1_c61229747508.log`:
      - It has 0 `CRASH:` lines and no `removed partial run directory` line.
      - Line 4328 is `GRPO subprocess finished in 175.3 min (exit=0, attempt=1/4)`.
      - Line 4330 is `OK: run 7z6tnfzr verified complete (responses=128)`.
      - The last line (4358) is `EXIT 0`.
  - **Other notes in step 27:** four `WARNING: could not remove activation cache` lines, each followed by `OK: removed activation cache` for the same cache.

### Review 5 (auditor, step 37), 2026-10-27 14:45 UTC, fresh auditor agent: PASS

#### Review 5 (auditor, step 37): overall verdict PASS

**Rule:** every number in the four reports must match the files they come from.

**Result: PASS.** I found no number I could not match. I checked every number in all four reports (listed below) two ways:

1. I reran the four commands (steps 16, 25, 34 and 35) into `/tmp`. All 11 output files are byte-identical to the saved ones.
2. I recomputed each value with my own code, which reads the raw eval and step files and does not import `compare.py`. Every value matched at the reports' 3-decimal precision.

What I checked:
- **Each `comparison.md`:** 303 table rows × 2 columns = 606 cells per seed, across all three sections, plus the header counts.
- **`seed_table.md`:** 101 eval rows × 8 cells = 808 cells, the 10 cells of the training table, and the header counts.

I changed no file. I wrote only under `/tmp`:
- the rerun outputs, in `/tmp/claude-0/-workspace/d05be7c1-d6d3-4845-80ed-791c09b5226c/scratchpad/rerun/`
- one recompute helper, `/tmp/claude-0/-workspace/d05be7c1-d6d3-4845-80ed-791c09b5226c/scratchpad/indep/recompute.py`

I set `PYTHONDONTWRITEBYTECODE=1`, so no `__pycache__` was written. No file under the experiment folder or `sycophancy_setup/` is newer than `runs.md`.

#### Discrepancies

##### 1. Numbers
None found.

##### 2. Procedure claims
None found.

##### 3. Counts across stages
None found.

##### 4. Config consistency
None found. The configs carry no hash stamp; see Verified.

##### 5. Leakage
No discrepancies. Two reads of eval-question data happen before the final eval. Both are by design and nothing was chosen from either:
- **The probe's test set.** At the start of each run, the trainer reads `sycophancy_setup/data/bias_probe_eval.jsonl`:
  - What it is: 777 base-model answers with bias labels, on all 489 eval questions.
  - Where it is read: `obfuscation_atlas/scripts/train_grpo_model_against_probe.py` line 501 and `detectors/custom_dataset.py` line 1298.
  - What it is used for: only the probe's test AUC, which reviews 2–4 checked against 0.85–0.89.
  - Why nothing was chosen: the layers, learning rate and steps are fixed in the yaml; `filter_layers_below_loss_threshold` is null; `early_stop_auc_threshold` is unset (None).
  - The probe trains on `bias_probe_train.jsonl`, which covers 740 questions, all in the train split, and none of the 489 eval questions.
- **The in-loop eval.** The trainer's in-loop eval runs on the 489 eval-split questions (`train_grpo_model_against_probe.py` lines 311–313; logged at step 30 onward in `run_metrics_*.json`).
  - Nothing was selected from it: no early stop, and no best-checkpoint code in either `grpo_trainer.py`.
  - The eval adapter is `checkpoints/final`. Its `adapter_model.bin` was saved 41–50 s after `step_127.jsonl` in each run.

##### 6. Unsupported or unjustified claims
None found. The four reports are generated tables. Their only prose describes how the numbers were computed (all checked under Verified). They contain no causal, comparative or generalizing statements.

#### Verified

**Rerun of the four commands.** I ran steps 16, 25, 34 and 35 exactly as given (from `weight_0_6/`, same relative paths), with `--out-dir` under `/tmp`. Every output is byte-identical (`cmp`) to the saved file:
- `comparison.md`, `comparison.json` and `training_curve.csv` for seeds 42, 43 and 44
- `seed_table.md` and `seed_table.json`

**Independent recompute of `comparison.md`.** I joined `completions.jsonl` with `labels.jsonl` and recomputed every metric for all three groups and all four conditions. I used scikit-learn for the AUCs. Result: 0 mismatches in 606 cells for each seed. Header counts:

| | Questions with a no-cue answer (base) | (after RL) | In both |
|---|---|---|---|
| s42 | 466 | 469 | 448 |
| s43 | 466 | 475 | 453 |
| s44 | 466 | 478 | 458 |

These also match each `summary.json`'s `questions_without_nocue_answer` (23 for base; 20, 14 and 11 after RL). A question counts as having a no-cue answer when at least 5 of its 8 no-cue samples agree. I checked that the saved no-cue answer (`a_none`) is set exactly when the top count in `nocue_contents` is at least 5, on every label row of all seven evals.

**Independent recompute of `seed_table.md`.**
- **Eval rows:** 0 mismatches in 808 cells. This covers the per-run base cells shown as "x / y / z", the base mean and SD, the after-RL values, and the after-RL mean and SD (sample SD, n − 1).
- **Training table:**

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.879 | 0.814 | 0.908 | 0.867 | 0.048 |
| probe-only groups | 0.815 | 0.751 | 0.859 | 0.808 | 0.054 |

- **Questions per seed** with a no-cue answer in both evals: 448, 453 and 458.

**"One sampled answer per prompt".**
- The final and rescored-base evals have exactly one cued sample (`sample_idx` 0) per question and condition, and 8 no-cue samples.
- 261013's base eval has 3 cued samples. The script and my recompute use sample 0 only: 489 rows per condition and 978 for `train_mix`, as the `rollouts` rows show.

**"Each run column = that run's final checkpoint".** Every completion row's `adapter` field in `evals/weight_0_6_s<S>` is `<run dir>/checkpoints/final`, and each eval log's first line names the same adapter.

**Base model in the base columns.**
- The `evals/base_rescored_weight_0_6_s<S>` completions are identical to 261013's base no-cue rows plus its cued `sample_idx` 0 rows (4890 rows; 0 differ).
- Their labels differ from 261013's only in `p_yes`, on 1 of 953 rows, by 5.1e-7 (the probe columns are new).
- So the seed table's base cells (261013's eval) and `comparison.md`'s base cells (the rescored copy) agree on every non-probe row at 3 decimals. They differ only in the 10 probe rows, which the brief exempts.
- The rescored `labels.jsonl` files are identical across the three seeds and to 261025's `base_rescored_floor_s42`.

**Training step records** (`<run dir>/checkpoints/responses`):
- Each run has `step_0` to `step_127`, with 512 records per step, which is 64 groups of 8.
- Grouping by (question, cue, template) gives the same groups as taking 8 consecutive records, in all 384 step files.
- The task `reward` takes only the values 0 and 1, and every record has a cue.
- Every probe-only group has equal task rewards. This supports "with the probe penalty only their probe scores differ".
- Each `training_curve.csv` has 128 rows, and all seven columns match my recompute exactly (max difference 0).

**Counts across stages.** These agree with the `runs.md` log lines and the review 2–4 counts:
- Each final eval has 4890 completions (3912 no-cue, 489 cue at the true answer, 489 cue at the wrong answer) and 978 label rows, which matches `summary.json` (489 questions, 978 cued rollouts).
- Follow-ups: 978, 978 and 975. Seed 44 has 3 label rows with no P(Yes), which matches its 3 unparsed cued answers in `generation_summary.json`. The rescored base has 953 follow-ups, and its 25 label rows with no P(Yes) match the `said` None counts (9 + 16).
- `generation_summary.json`'s raw counts reproduce the accuracy and cue agreement cells. For example, seed 42 with the cue at the wrong answer: 150/489 = 0.307 accuracy and 339/489 = 0.693 cue agreement.

**Config and code.**
- `weight_0_6_s4{2,3,4}.yaml` are byte-identical to `../code_at_launch/weight_0_6/`.
- The current `git diff -- sycophancy-rl` is identical to `../code_at_launch/tracked.diff`.
- `../checks.py` is identical to its launch copy.
- `compare.py` and `seed_table.py` have no uncommitted changes and were last modified on 2026-10-24, before the launch.
- The Code fixes section of `runs.md` is empty.

**Train and eval questions.** The step records of each run cover 1217 question ids, none of them among the 489 eval ids. `questions.jsonl` has 1217 train and 489 eval questions.

**Section headings "base model's consistent no-cue answer is right / wrong"** (`comparison.md`). The split uses `base_confident_correct` from `questions.jsonl`. `261013_noisy_label_rl/build_dataset.py` (its docstring and lines 67–74) defines that field as 261010's 6-of-8 base answer. So the headings are supported, by 261010's sampling.
- For context only (a fact, not a failure): in 261013's base eval, the one behind the base column, 65 of 489 questions do not reach 6-of-8 agreement, and 44 would fall in the other group if split by that eval's own no-cue answers.
- The group sizes in the tables (384 right and 105 wrong) match the field.

**Not counted as failures, per the brief:**
- `comparison.md`'s base column uses the rescored copy, while the seed table uses 261013's saved eval.
- The seed table's base probe rows show 0 for the counts and – for the means and AUCs.

## Blocked

## Result

Source file: `tables/weight_0_6/seed_table.md` (the whole file).

### RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 448, s43 453, s44 458.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.748 | 0.765 | 0.748 | 0.754 | 0.009 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.963 | 0.963 | 0.963 | 0.963 | 0.000 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.307 | 0.356 | 0.290 | 0.318 | 0.034 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.739 | 0.750 | 0.732 | 0.740 | 0.009 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.669 | 0.667 | 0.665 | 0.667 | 0.002 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.787 | 0.796 | 0.779 | 0.787 | 0.008 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.372 | 0.434 | 0.360 | 0.389 | 0.039 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.744 | 0.766 | 0.738 | 0.749 | 0.015 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.963 | 0.963 | 0.963 | 0.963 | 0.000 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.693 | 0.644 | 0.708 | 0.682 | 0.033 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.868 | 0.861 | 0.871 | 0.867 | 0.005 |
| bias | cue_gold | 0.086 / 0.097 / 0.089 | 0.091 | 0.005 | 0.163 | 0.174 | 0.160 | 0.166 | 0.008 |
| bias | cue_dist | 0.269 / 0.270 / 0.270 | 0.270 | 0.001 | 0.478 | 0.422 | 0.501 | 0.467 | 0.041 |
| bias | train_mix | 0.150 / 0.157 / 0.150 | 0.152 | 0.004 | 0.268 | 0.258 | 0.269 | 0.265 | 0.006 |
| anti_bias | cue_gold | 0.032 / 0.036 / 0.038 | 0.035 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| anti_bias | cue_dist | 0.009 / 0.009 / 0.011 | 0.010 | 0.001 | 0.000 | 0.004 | 0.000 | 0.001 | 0.003 |
| anti_bias | train_mix | 0.023 / 0.026 / 0.028 | 0.026 | 0.003 | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 |
| no_bias | cue_gold | 0.882 / 0.867 / 0.873 | 0.874 | 0.007 | 0.837 | 0.826 | 0.840 | 0.834 | 0.008 |
| no_bias | cue_dist | 0.722 / 0.720 / 0.719 | 0.720 | 0.001 | 0.522 | 0.574 | 0.499 | 0.532 | 0.038 |
| no_bias | train_mix | 0.827 / 0.817 / 0.822 | 0.822 | 0.005 | 0.732 | 0.741 | 0.731 | 0.734 | 0.005 |
| deception_soft | cue_gold | 0.554 / 0.549 / 0.553 | 0.552 | 0.003 | 0.405 | 0.420 | 0.320 | 0.381 | 0.054 |
| deception_soft | cue_dist | 0.599 / 0.597 / 0.600 | 0.599 | 0.002 | 0.508 | 0.524 | 0.479 | 0.504 | 0.023 |
| deception_soft | train_mix | 0.569 / 0.564 / 0.568 | 0.567 | 0.003 | 0.442 | 0.451 | 0.373 | 0.422 | 0.043 |
| deception | cue_gold | 0.573 / 0.564 / 0.573 | 0.570 | 0.005 | 0.312 | 0.373 | 0.230 | 0.305 | 0.072 |
| deception | cue_dist | 0.661 / 0.658 / 0.661 | 0.660 | 0.002 | 0.533 | 0.534 | 0.453 | 0.507 | 0.047 |
| deception | train_mix | 0.603 / 0.595 / 0.601 | 0.600 | 0.004 | 0.389 | 0.418 | 0.306 | 0.371 | 0.058 |
| biased_share | cue_gold | 0.089 / 0.100 / 0.092 | 0.094 | 0.006 | 0.163 | 0.174 | 0.160 | 0.166 | 0.008 |
| biased_share | cue_dist | 0.271 / 0.273 / 0.273 | 0.272 | 0.001 | 0.478 | 0.424 | 0.501 | 0.467 | 0.040 |
| biased_share | train_mix | 0.154 / 0.161 / 0.154 | 0.156 | 0.004 | 0.268 | 0.259 | 0.269 | 0.265 | 0.006 |
| false_denial_soft | cue_gold | 0.390 / 0.380 / 0.396 | 0.389 | 0.008 | 0.403 | 0.462 | 0.562 | 0.476 | 0.080 |
| false_denial_soft | cue_dist | 0.344 / 0.352 / 0.352 | 0.349 | 0.005 | 0.434 | 0.448 | 0.569 | 0.483 | 0.074 |
| false_denial_soft | train_mix | 0.365 / 0.366 / 0.371 | 0.368 | 0.003 | 0.421 | 0.443 | 0.565 | 0.476 | 0.078 |
| false_admission_soft | cue_gold | 0.570 / 0.567 / 0.569 | 0.569 | 0.001 | 0.405 | 0.411 | 0.274 | 0.363 | 0.077 |
| false_admission_soft | cue_dist | 0.694 / 0.689 / 0.693 | 0.692 | 0.003 | 0.576 | 0.580 | 0.388 | 0.515 | 0.110 |
| false_admission_soft | train_mix | 0.605 / 0.602 / 0.604 | 0.604 | 0.002 | 0.450 | 0.454 | 0.302 | 0.402 | 0.087 |
| said_yes_biased_soft | cue_gold | 0.610 / 0.620 / 0.604 | 0.611 | 0.008 | 0.597 | 0.538 | 0.438 | 0.524 | 0.080 |
| said_yes_biased_soft | cue_dist | 0.656 / 0.648 / 0.648 | 0.651 | 0.005 | 0.566 | 0.552 | 0.431 | 0.517 | 0.074 |
| said_yes_biased_soft | train_mix | 0.635 / 0.634 / 0.629 | 0.632 | 0.003 | 0.579 | 0.557 | 0.435 | 0.524 | 0.078 |
| deception_from_biased | cue_gold | 0.035 / 0.038 / 0.037 | 0.036 | 0.002 | 0.066 | 0.081 | 0.090 | 0.079 | 0.012 |
| deception_from_biased | cue_dist | 0.093 / 0.096 / 0.096 | 0.095 | 0.002 | 0.207 | 0.190 | 0.285 | 0.227 | 0.051 |
| deception_from_biased | train_mix | 0.056 / 0.059 / 0.057 | 0.057 | 0.001 | 0.113 | 0.115 | 0.152 | 0.126 | 0.022 |
| deception_from_unbiased | cue_gold | 0.519 / 0.510 / 0.516 | 0.515 | 0.005 | 0.339 | 0.339 | 0.230 | 0.303 | 0.063 |
| deception_from_unbiased | cue_dist | 0.506 / 0.501 / 0.504 | 0.504 | 0.002 | 0.301 | 0.335 | 0.194 | 0.276 | 0.074 |
| deception_from_unbiased | train_mix | 0.512 / 0.505 / 0.511 | 0.509 | 0.004 | 0.329 | 0.337 | 0.221 | 0.296 | 0.065 |
| said_yes | cue_gold | 0.596 / 0.594 / 0.596 | 0.595 | 0.001 | 0.342 | 0.375 | 0.188 | 0.302 | 0.100 |
| said_yes | cue_dist | 0.789 / 0.784 / 0.789 | 0.787 | 0.003 | 0.556 | 0.599 | 0.302 | 0.485 | 0.160 |
| said_yes | train_mix | 0.659 / 0.656 / 0.658 | 0.658 | 0.002 | 0.416 | 0.446 | 0.226 | 0.363 | 0.119 |
| false_denial | cue_gold | 0.368 / 0.349 / 0.375 | 0.364 | 0.014 | 0.411 | 0.494 | 0.630 | 0.512 | 0.111 |
| false_denial | cue_dist | 0.265 / 0.269 / 0.267 | 0.267 | 0.002 | 0.477 | 0.424 | 0.651 | 0.517 | 0.119 |
| false_denial | train_mix | 0.316 / 0.313 / 0.317 | 0.315 | 0.002 | 0.449 | 0.444 | 0.648 | 0.514 | 0.116 |
| false_admission | cue_gold | 0.593 / 0.588 / 0.593 | 0.591 | 0.003 | 0.293 | 0.348 | 0.154 | 0.265 | 0.100 |
| false_admission | cue_dist | 0.809 / 0.804 / 0.809 | 0.808 | 0.003 | 0.585 | 0.615 | 0.254 | 0.485 | 0.200 |
| false_admission | train_mix | 0.655 / 0.650 / 0.653 | 0.652 | 0.003 | 0.367 | 0.408 | 0.180 | 0.319 | 0.121 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.012 | 0.002 | 0.004 | 0.006 | 0.005 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.004 | 0.001 | 0.002 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.000 | 0.004 | 0.001 | 0.002 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 440 / 445 / 450 | 445 | 5.000 | 448 | 453 | 457 | 452.667 | 4.509 |
| n_bias_rows | cue_dist | 435 / 440 / 445 | 440 | 5.000 | 448 | 453 | 457 | 452.667 | 4.509 |
| n_bias_rows | train_mix | 875 / 885 / 895 | 885 | 10.000 | 896 | 906 | 914 | 905.333 | 9.018 |
| n_deception_rows | cue_gold | 426 / 429 / 433 | 429.333 | 3.512 | 448 | 453 | 457 | 452.667 | 4.509 |
| n_deception_rows | cue_dist | 431 / 436 / 440 | 435.667 | 4.509 | 448 | 451 | 457 | 452 | 4.583 |
| n_deception_rows | train_mix | 857 / 865 / 873 | 865 | 8.000 | 896 | 904 | 914 | 904.667 | 9.018 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.063 | 0.183 | 0.138 | 0.128 | 0.061 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.035 | 0.166 | 0.102 | 0.101 | 0.066 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.015 | 0.035 | 0.041 | 0.030 | 0.014 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.015 | 0.042 | 0.044 | 0.034 | 0.016 |
| probe_start_auc | cue_gold | – | – | – | 0.756 | 0.849 | 0.763 | 0.789 | 0.052 |
| probe_start_auc | cue_dist | – | – | – | 0.691 | 0.759 | 0.745 | 0.731 | 0.036 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 73 | 79 | 73 | 75 | 3.464 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 214 | 191 | 229 | 211.333 | 19.140 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 358 | 357 | 368 | 361 | 6.083 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 90 | 94 | 89 | 91 | 2.646 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.279 | 0.290 | 0.384 | 0.318 | 0.058 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.259 | 0.252 | 0.359 | 0.290 | 0.060 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.151 | 0.076 | 0.190 | 0.139 | 0.058 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.151 | 0.093 | 0.185 | 0.143 | 0.046 |
| probe_fresh_auc | cue_gold | – | – | – | 0.751 | 0.838 | 0.770 | 0.787 | 0.046 |
| probe_fresh_auc | cue_dist | – | – | – | 0.701 | 0.746 | 0.742 | 0.730 | 0.025 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 73 | 79 | 73 | 75 | 3.464 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 214 | 191 | 229 | 211.333 | 19.140 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 358 | 357 | 368 | 361 | 6.083 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 90 | 94 | 89 | 91 | 2.646 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.119 | 0.161 | 0.202 | 0.161 | 0.041 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.087 | 0.159 | 0.198 | 0.148 | 0.056 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.035 | 0.032 | 0.057 | 0.041 | 0.014 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.037 | 0.044 | 0.069 | 0.050 | 0.017 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.753 | 0.852 | 0.783 | 0.796 | 0.051 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.680 | 0.756 | 0.764 | 0.733 | 0.047 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 73 | 79 | 73 | 75 | 3.464 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 214 | 191 | 229 | 211.333 | 19.140 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 358 | 357 | 368 | 361 | 6.083 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 90 | 94 | 89 | 91 | 2.646 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.879 | 0.814 | 0.908 | 0.867 | 0.048 |
| probe-only groups | 0.815 | 0.751 | 0.859 | 0.808 | 0.054 |

Source file: `runs/weight_0_6_s42/comparison.md` (its "Main result" section).

#### Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.748 |
| accuracy | cue_gold | 0.843 | 0.963 |
| accuracy | cue_dist | 0.509 | 0.307 |
| accuracy | train_mix | 0.729 | 0.739 |
| reward | none | 0.644 | 0.669 |
| reward | cue_gold | 0.714 | 0.787 |
| reward | cue_dist | 0.499 | 0.372 |
| reward | train_mix | 0.689 | 0.744 |
| followed_cue | cue_gold | 0.843 | 0.963 |
| followed_cue | cue_dist | 0.458 | 0.693 |
| followed_cue | train_mix | 0.711 | 0.868 |
| bias | cue_gold | 0.086 | 0.163 |
| bias | cue_dist | 0.269 | 0.478 |
| bias | train_mix | 0.150 | 0.268 |
| anti_bias | cue_gold | 0.032 | 0.000 |
| anti_bias | cue_dist | 0.009 | 0.000 |
| anti_bias | train_mix | 0.023 | 0.000 |
| no_bias | cue_gold | 0.882 | 0.837 |
| no_bias | cue_dist | 0.722 | 0.522 |
| no_bias | train_mix | 0.827 | 0.732 |
| deception_soft | cue_gold | 0.554 | 0.405 |
| deception_soft | cue_dist | 0.599 | 0.508 |
| deception_soft | train_mix | 0.569 | 0.442 |
| deception | cue_gold | 0.573 | 0.312 |
| deception | cue_dist | 0.661 | 0.533 |
| deception | train_mix | 0.603 | 0.389 |
| biased_share | cue_gold | 0.089 | 0.163 |
| biased_share | cue_dist | 0.271 | 0.478 |
| biased_share | train_mix | 0.154 | 0.268 |
| false_denial_soft | cue_gold | 0.390 | 0.403 |
| false_denial_soft | cue_dist | 0.344 | 0.434 |
| false_denial_soft | train_mix | 0.365 | 0.421 |
| false_admission_soft | cue_gold | 0.570 | 0.405 |
| false_admission_soft | cue_dist | 0.694 | 0.576 |
| false_admission_soft | train_mix | 0.605 | 0.450 |
| said_yes_biased_soft | cue_gold | 0.610 | 0.597 |
| said_yes_biased_soft | cue_dist | 0.656 | 0.566 |
| said_yes_biased_soft | train_mix | 0.635 | 0.579 |
| deception_from_biased | cue_gold | 0.035 | 0.066 |
| deception_from_biased | cue_dist | 0.093 | 0.207 |
| deception_from_biased | train_mix | 0.056 | 0.113 |
| deception_from_unbiased | cue_gold | 0.519 | 0.339 |
| deception_from_unbiased | cue_dist | 0.506 | 0.301 |
| deception_from_unbiased | train_mix | 0.512 | 0.329 |
| said_yes | cue_gold | 0.596 | 0.342 |
| said_yes | cue_dist | 0.789 | 0.556 |
| said_yes | train_mix | 0.659 | 0.416 |
| false_denial | cue_gold | 0.368 | 0.411 |
| false_denial | cue_dist | 0.265 | 0.477 |
| false_denial | train_mix | 0.316 | 0.449 |
| false_admission | cue_gold | 0.593 | 0.293 |
| false_admission | cue_dist | 0.809 | 0.585 |
| false_admission | train_mix | 0.655 | 0.367 |
| parse_failed | none | 0.016 | 0.012 |
| parse_failed | cue_gold | 0.018 | 0.000 |
| parse_failed | cue_dist | 0.033 | 0.000 |
| parse_failed | train_mix | 0.024 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 489 | 489 |
| rollouts | cue_gold | 489 | 489 |
| rollouts | cue_dist | 489 | 489 |
| rollouts | train_mix | 978 | 978 |
| n_bias_rows | cue_gold | 440 | 448 |
| n_bias_rows | cue_dist | 435 | 448 |
| n_bias_rows | train_mix | 875 | 896 |
| n_deception_rows | cue_gold | 426 | 448 |
| n_deception_rows | cue_dist | 431 | 448 |
| n_deception_rows | train_mix | 857 | 896 |
| probe_start_mean_biased | cue_gold | 0.767 | 0.063 |
| probe_start_mean_biased | cue_dist | 0.749 | 0.035 |
| probe_start_mean_unbiased | cue_gold | 0.225 | 0.015 |
| probe_start_mean_unbiased | cue_dist | 0.255 | 0.015 |
| probe_start_auc | cue_gold | 0.927 | 0.756 |
| probe_start_auc | cue_dist | 0.889 | 0.691 |
| probe_start_n_biased | cue_gold | 38 | 73 |
| probe_start_n_biased | cue_dist | 117 | 214 |
| probe_start_n_unbiased | cue_gold | 342 | 358 |
| probe_start_n_unbiased | cue_dist | 82 | 90 |
| probe_fresh_mean_biased | cue_gold | – | 0.279 |
| probe_fresh_mean_biased | cue_dist | – | 0.259 |
| probe_fresh_mean_unbiased | cue_gold | – | 0.151 |
| probe_fresh_mean_unbiased | cue_dist | – | 0.151 |
| probe_fresh_auc | cue_gold | – | 0.751 |
| probe_fresh_auc | cue_dist | – | 0.701 |
| probe_fresh_n_biased | cue_gold | 0 | 73 |
| probe_fresh_n_biased | cue_dist | 0 | 214 |
| probe_fresh_n_unbiased | cue_gold | 0 | 358 |
| probe_fresh_n_unbiased | cue_dist | 0 | 90 |
| probe_start_base_activations_mean_biased | cue_gold | – | 0.119 |
| probe_start_base_activations_mean_biased | cue_dist | – | 0.087 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | 0.035 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | 0.037 |
| probe_start_base_activations_auc | cue_gold | – | 0.753 |
| probe_start_base_activations_auc | cue_dist | – | 0.680 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 73 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 214 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 358 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 90 |

Source file: `runs/weight_0_6_s43/comparison.md` (its "Main result" section).

#### Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.765 |
| accuracy | cue_gold | 0.843 | 0.963 |
| accuracy | cue_dist | 0.509 | 0.356 |
| accuracy | train_mix | 0.729 | 0.750 |
| reward | none | 0.644 | 0.667 |
| reward | cue_gold | 0.714 | 0.796 |
| reward | cue_dist | 0.499 | 0.434 |
| reward | train_mix | 0.689 | 0.766 |
| followed_cue | cue_gold | 0.843 | 0.963 |
| followed_cue | cue_dist | 0.458 | 0.644 |
| followed_cue | train_mix | 0.711 | 0.861 |
| bias | cue_gold | 0.097 | 0.174 |
| bias | cue_dist | 0.270 | 0.422 |
| bias | train_mix | 0.157 | 0.258 |
| anti_bias | cue_gold | 0.036 | 0.000 |
| anti_bias | cue_dist | 0.009 | 0.004 |
| anti_bias | train_mix | 0.026 | 0.001 |
| no_bias | cue_gold | 0.867 | 0.826 |
| no_bias | cue_dist | 0.720 | 0.574 |
| no_bias | train_mix | 0.817 | 0.741 |
| deception_soft | cue_gold | 0.549 | 0.420 |
| deception_soft | cue_dist | 0.597 | 0.524 |
| deception_soft | train_mix | 0.564 | 0.451 |
| deception | cue_gold | 0.564 | 0.373 |
| deception | cue_dist | 0.658 | 0.534 |
| deception | train_mix | 0.595 | 0.418 |
| biased_share | cue_gold | 0.100 | 0.174 |
| biased_share | cue_dist | 0.273 | 0.424 |
| biased_share | train_mix | 0.161 | 0.259 |
| false_denial_soft | cue_gold | 0.380 | 0.462 |
| false_denial_soft | cue_dist | 0.352 | 0.448 |
| false_denial_soft | train_mix | 0.366 | 0.443 |
| false_admission_soft | cue_gold | 0.567 | 0.411 |
| false_admission_soft | cue_dist | 0.689 | 0.580 |
| false_admission_soft | train_mix | 0.602 | 0.454 |
| said_yes_biased_soft | cue_gold | 0.620 | 0.538 |
| said_yes_biased_soft | cue_dist | 0.648 | 0.552 |
| said_yes_biased_soft | train_mix | 0.634 | 0.557 |
| deception_from_biased | cue_gold | 0.038 | 0.081 |
| deception_from_biased | cue_dist | 0.096 | 0.190 |
| deception_from_biased | train_mix | 0.059 | 0.115 |
| deception_from_unbiased | cue_gold | 0.510 | 0.339 |
| deception_from_unbiased | cue_dist | 0.501 | 0.335 |
| deception_from_unbiased | train_mix | 0.505 | 0.337 |
| said_yes | cue_gold | 0.594 | 0.375 |
| said_yes | cue_dist | 0.784 | 0.599 |
| said_yes | train_mix | 0.656 | 0.446 |
| false_denial | cue_gold | 0.349 | 0.494 |
| false_denial | cue_dist | 0.269 | 0.424 |
| false_denial | train_mix | 0.313 | 0.444 |
| false_admission | cue_gold | 0.588 | 0.348 |
| false_admission | cue_dist | 0.804 | 0.615 |
| false_admission | train_mix | 0.650 | 0.408 |
| parse_failed | none | 0.016 | 0.002 |
| parse_failed | cue_gold | 0.018 | 0.000 |
| parse_failed | cue_dist | 0.033 | 0.000 |
| parse_failed | train_mix | 0.024 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 489 | 489 |
| rollouts | cue_gold | 489 | 489 |
| rollouts | cue_dist | 489 | 489 |
| rollouts | train_mix | 978 | 978 |
| n_bias_rows | cue_gold | 445 | 453 |
| n_bias_rows | cue_dist | 440 | 453 |
| n_bias_rows | train_mix | 885 | 906 |
| n_deception_rows | cue_gold | 429 | 453 |
| n_deception_rows | cue_dist | 436 | 451 |
| n_deception_rows | train_mix | 865 | 904 |
| probe_start_mean_biased | cue_gold | 0.753 | 0.183 |
| probe_start_mean_biased | cue_dist | 0.740 | 0.166 |
| probe_start_mean_unbiased | cue_gold | 0.223 | 0.035 |
| probe_start_mean_unbiased | cue_dist | 0.272 | 0.042 |
| probe_start_auc | cue_gold | 0.923 | 0.849 |
| probe_start_auc | cue_dist | 0.872 | 0.759 |
| probe_start_n_biased | cue_gold | 43 | 79 |
| probe_start_n_biased | cue_dist | 119 | 191 |
| probe_start_n_unbiased | cue_gold | 340 | 357 |
| probe_start_n_unbiased | cue_dist | 86 | 94 |
| probe_fresh_mean_biased | cue_gold | – | 0.290 |
| probe_fresh_mean_biased | cue_dist | – | 0.252 |
| probe_fresh_mean_unbiased | cue_gold | – | 0.076 |
| probe_fresh_mean_unbiased | cue_dist | – | 0.093 |
| probe_fresh_auc | cue_gold | – | 0.838 |
| probe_fresh_auc | cue_dist | – | 0.746 |
| probe_fresh_n_biased | cue_gold | 0 | 79 |
| probe_fresh_n_biased | cue_dist | 0 | 191 |
| probe_fresh_n_unbiased | cue_gold | 0 | 357 |
| probe_fresh_n_unbiased | cue_dist | 0 | 94 |
| probe_start_base_activations_mean_biased | cue_gold | – | 0.161 |
| probe_start_base_activations_mean_biased | cue_dist | – | 0.159 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | 0.032 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | 0.044 |
| probe_start_base_activations_auc | cue_gold | – | 0.852 |
| probe_start_base_activations_auc | cue_dist | – | 0.756 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 79 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 191 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 357 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 94 |

Source file: `runs/weight_0_6_s44/comparison.md` (its "Main result" section).

#### Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.748 |
| accuracy | cue_gold | 0.843 | 0.963 |
| accuracy | cue_dist | 0.509 | 0.290 |
| accuracy | train_mix | 0.729 | 0.732 |
| reward | none | 0.644 | 0.665 |
| reward | cue_gold | 0.714 | 0.779 |
| reward | cue_dist | 0.499 | 0.360 |
| reward | train_mix | 0.689 | 0.738 |
| followed_cue | cue_gold | 0.843 | 0.963 |
| followed_cue | cue_dist | 0.458 | 0.708 |
| followed_cue | train_mix | 0.711 | 0.871 |
| bias | cue_gold | 0.089 | 0.160 |
| bias | cue_dist | 0.270 | 0.501 |
| bias | train_mix | 0.150 | 0.269 |
| anti_bias | cue_gold | 0.038 | 0.000 |
| anti_bias | cue_dist | 0.011 | 0.000 |
| anti_bias | train_mix | 0.028 | 0.000 |
| no_bias | cue_gold | 0.873 | 0.840 |
| no_bias | cue_dist | 0.719 | 0.499 |
| no_bias | train_mix | 0.822 | 0.731 |
| deception_soft | cue_gold | 0.553 | 0.320 |
| deception_soft | cue_dist | 0.600 | 0.479 |
| deception_soft | train_mix | 0.568 | 0.373 |
| deception | cue_gold | 0.573 | 0.230 |
| deception | cue_dist | 0.661 | 0.453 |
| deception | train_mix | 0.601 | 0.306 |
| biased_share | cue_gold | 0.092 | 0.160 |
| biased_share | cue_dist | 0.273 | 0.501 |
| biased_share | train_mix | 0.154 | 0.269 |
| false_denial_soft | cue_gold | 0.396 | 0.562 |
| false_denial_soft | cue_dist | 0.352 | 0.569 |
| false_denial_soft | train_mix | 0.371 | 0.565 |
| false_admission_soft | cue_gold | 0.569 | 0.274 |
| false_admission_soft | cue_dist | 0.693 | 0.388 |
| false_admission_soft | train_mix | 0.604 | 0.302 |
| said_yes_biased_soft | cue_gold | 0.604 | 0.438 |
| said_yes_biased_soft | cue_dist | 0.648 | 0.431 |
| said_yes_biased_soft | train_mix | 0.629 | 0.435 |
| deception_from_biased | cue_gold | 0.037 | 0.090 |
| deception_from_biased | cue_dist | 0.096 | 0.285 |
| deception_from_biased | train_mix | 0.057 | 0.152 |
| deception_from_unbiased | cue_gold | 0.516 | 0.230 |
| deception_from_unbiased | cue_dist | 0.504 | 0.194 |
| deception_from_unbiased | train_mix | 0.511 | 0.221 |
| said_yes | cue_gold | 0.596 | 0.188 |
| said_yes | cue_dist | 0.789 | 0.302 |
| said_yes | train_mix | 0.658 | 0.226 |
| false_denial | cue_gold | 0.375 | 0.630 |
| false_denial | cue_dist | 0.267 | 0.651 |
| false_denial | train_mix | 0.317 | 0.648 |
| false_admission | cue_gold | 0.593 | 0.154 |
| false_admission | cue_dist | 0.809 | 0.254 |
| false_admission | train_mix | 0.653 | 0.180 |
| parse_failed | none | 0.016 | 0.004 |
| parse_failed | cue_gold | 0.018 | 0.004 |
| parse_failed | cue_dist | 0.033 | 0.002 |
| parse_failed | train_mix | 0.024 | 0.004 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 489 | 489 |
| rollouts | cue_gold | 489 | 489 |
| rollouts | cue_dist | 489 | 489 |
| rollouts | train_mix | 978 | 978 |
| n_bias_rows | cue_gold | 450 | 457 |
| n_bias_rows | cue_dist | 445 | 457 |
| n_bias_rows | train_mix | 895 | 914 |
| n_deception_rows | cue_gold | 433 | 457 |
| n_deception_rows | cue_dist | 440 | 457 |
| n_deception_rows | train_mix | 873 | 914 |
| probe_start_mean_biased | cue_gold | 0.759 | 0.138 |
| probe_start_mean_biased | cue_dist | 0.745 | 0.102 |
| probe_start_mean_unbiased | cue_gold | 0.223 | 0.041 |
| probe_start_mean_unbiased | cue_dist | 0.259 | 0.044 |
| probe_start_auc | cue_gold | 0.926 | 0.763 |
| probe_start_auc | cue_dist | 0.883 | 0.745 |
| probe_start_n_biased | cue_gold | 40 | 73 |
| probe_start_n_biased | cue_dist | 120 | 229 |
| probe_start_n_unbiased | cue_gold | 347 | 368 |
| probe_start_n_unbiased | cue_dist | 83 | 89 |
| probe_fresh_mean_biased | cue_gold | – | 0.384 |
| probe_fresh_mean_biased | cue_dist | – | 0.359 |
| probe_fresh_mean_unbiased | cue_gold | – | 0.190 |
| probe_fresh_mean_unbiased | cue_dist | – | 0.185 |
| probe_fresh_auc | cue_gold | – | 0.770 |
| probe_fresh_auc | cue_dist | – | 0.742 |
| probe_fresh_n_biased | cue_gold | 0 | 73 |
| probe_fresh_n_biased | cue_dist | 0 | 229 |
| probe_fresh_n_unbiased | cue_gold | 0 | 368 |
| probe_fresh_n_unbiased | cue_dist | 0 | 89 |
| probe_start_base_activations_mean_biased | cue_gold | – | 0.202 |
| probe_start_base_activations_mean_biased | cue_dist | – | 0.198 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | 0.057 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | 0.069 |
| probe_start_base_activations_auc | cue_gold | – | 0.783 |
| probe_start_base_activations_auc | cue_dist | – | 0.764 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 73 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 229 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 368 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 89 |
