# Runs

State file for the unattended run of the seed-42 line (pod da70f8166885). Read this first after any restart.

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

- 2026-11-07 21:43 | 1. Pod check | pass: da70f8166885, one H200, no GPU process, 78G free; checks/frozen_probe_md5.txt = the md5 rule's 8 lines | step 2
- 2026-11-07 21:47 | 2. Round 1 regenerated (reported) | EXIT 0 (generation 21:43–21:46); checks/round1_repro.json: rows 500, questions 50, same_completion 199, same_pred 467 (of 500, against 261103's seed-42 eval from pod 33f36c5bf32d) | step 3
- 2026-11-07 21:48 | 3. config 2 | EXIT 0; wrote s42_frozen_round2.yaml and smoke_s42_frozen_round2.yaml | step 4 (review 1)
- 2026-11-07 21:49 | 4. Review 1 | PASS ((a) config rule and smoke config, (b) diff against 261104's round 2, (c) md5 all pass) | step 5
- 2026-11-07 21:55 | 5. vLLM for smoke | started 21:49:40, answered after 301 s (52018 MiB held by VLLM::EngineCore) | step 6
- 2026-11-07 22:18 | 6. Smoke RL run | EXIT 0, run id 45f2so6j; started 21:55:00; `GRPO subprocess finished in 20.4 min (exit=0, attempt=1/4)`; no `CRASH:` line; no out-of-memory line; peak GPU memory 97095 MiB (vLLM + trainer), lowest free disk 73G | step 7
- 2026-11-07 22:19 | 7. Smoke record check | exit 0; runs/smoke_s42_frozen_round2/check_records.json: PASS true, n_files 2, seed 42 | step 8
- 2026-11-07 22:19 | 8. Stop vLLM | session killed; nvidia-smi lists no process after ~10 s (0 MiB) | step 9
- 2026-11-07 22:22 | 9. Smoke checks | exit 0; reward: penalty_weight 1.0, mismatches 0 and 0, reward_mean_abs_difference 6.4e-9 and 1.5e-8; advantages: floor 0.3, 2 steps, max_abs_difference 1.4e-8, kl_div_file_step_0 0.172; KL file: loss.kl 0.01, kl_coeff 0.01, 2 steps, kl_div 0.172, 0.120 | step 10
- 2026-11-07 22:24 | 10. Penalty check (smoke) | EXIT 0 (22:21:45–22:23:50); 512 of 512 step-0 answers scored; round 1's probe vs saved: median 0.00060, max 0.023, mean 0.0021, 1 over 0.02; trainer's unused start-of-run probe vs saved: median 0.0071, max 0.238, mean 0.0198, 129 over 0.02 | step 11 (review 2)
- 2026-11-07 22:26 | 11. Review 2 | PASS (rules (a)–(h) all pass) | step 12
- 2026-11-07 22:32 | 12. Round 2: vLLM | started 22:26:54, answered after 286 s (52018 MiB) | step 13
- 2026-11-08 02:47 | 13. Round 2: RL run | EXIT 0, run id z3pe3lcj; started 2026-11-07 22:31:57; `GRPO subprocess finished in 252.6 min (exit=0, attempt=1/4)`; no `CRASH:` line; no out-of-memory line; peak GPU memory 97288 MiB (vLLM + trainer), lowest free disk 62G | step 14
- 2026-11-08 02:48 | 14. Round 2: record check | exit 0; runs/s42_frozen_round2/check_records.json: PASS true, n_files 128, seed 42 | step 15
- 2026-11-08 02:50 | 15. Round 2: checks | exit 0; advantages: floor 0.3, 128 steps, max_abs_difference 1.8e-8, kl_div_file_step_0 0.152; KL file: loss.kl 0.01, kl_coeff 0.01, 128 steps, kl_div last 0.114, mean of last 16 0.120 | step 16
- 2026-11-08 02:50 | 16. Round 2: stop vLLM | session killed; nvidia-smi lists no process after ~10 s (0 MiB) | step 17
- 2026-11-08 02:51 | 17. Round 2: links 2 | EXIT 0; links/round2: start-of-run entry -> round 1's (am1g59lu) probe, end-of-run entry -> z3pe3lcj's own refit | step 18
- 2026-11-08 02:59 | 18. Round 2: final eval | EXIT 0 (02:51–02:59); summary.json: questions 489, cued_rollouts 978 | step 19
- 2026-11-08 03:00 | 19. Round 2: compare | exit 0; runs/s42_frozen_round2/comparison.md, comparison.json, training_curve.csv (128 data rows) | step 20
- 2026-11-08 03:01 | 20. Round 2: config 3 | EXIT 0; wrote s42_frozen_round3.yaml (checkpoint z3pe3lcj's final adapter) | step 21 (review 3)
- 2026-11-08 03:03 | 21. Review 3 (end of round 2) | PASS (rules (a)–(i) all pass) | step 22
- 2026-11-08 03:08 | 22. Round 3: vLLM | started 03:03:56, answered after 256 s (52018 MiB) | step 23
- 2026-11-08 07:21 | 23. Round 3: RL run | EXIT 0, run id gh04frmi; started 03:08:25; `GRPO subprocess finished in 250.2 min (exit=0, attempt=1/4)`; no `CRASH:` line; no out-of-memory line; peak GPU memory 100388 MiB (vLLM + trainer), lowest free disk 60G | step 24
- 2026-11-08 07:22 | 24. Round 3: record check | exit 0; runs/s42_frozen_round3/check_records.json: PASS true, n_files 128, seed 42 | step 25
- 2026-11-08 07:24 | 25. Round 3: checks | exit 0; advantages: floor 0.3, 128 steps, max_abs_difference 1.8e-8, kl_div_file_step_0 0.121; KL file: loss.kl 0.01, kl_coeff 0.01, 128 steps, kl_div last 0.099, mean of last 16 0.100 | step 26
- 2026-11-08 07:24 | 26. Round 3: stop vLLM | session killed; nvidia-smi lists no process after ~10 s (0 MiB) | step 27
- 2026-11-08 07:25 | 27. Round 3: links 3 | EXIT 0; links/round3: start-of-run entry -> round 1's (am1g59lu) probe, end-of-run entry -> gh04frmi's own refit | step 28
- 2026-11-08 07:33 | 28. Round 3: final eval | EXIT 0 (07:25–07:32); summary.json: questions 489, cued_rollouts 978 | step 29
- 2026-11-08 07:33 | 29. Round 3: compare | exit 0; runs/s42_frozen_round3/comparison.md, comparison.json, training_curve.csv (128 data rows) | step 30
- 2026-11-08 07:34 | 30. Round 3: config 4 | EXIT 0; wrote s42_frozen_round4.yaml (checkpoint gh04frmi's final adapter) | step 31 (review 4)
- 2026-11-08 07:37 | 31. Review 4 (end of round 3) | PASS (rules (a)–(i) all pass) | step 32
- 2026-11-08 07:41 | 32. Round 4: vLLM | started 07:36:55, answered after 256 s (52018 MiB) | step 33
- 2026-11-08 11:54 | 33. Round 4: RL run | EXIT 0, run id 8aiyurfx; started 07:41:25; `GRPO subprocess finished in 250.3 min (exit=0, attempt=1/4)`; no `CRASH:` line; no out-of-memory line; peak GPU memory 98828 MiB (vLLM + trainer), lowest free disk 57G | step 34
- 2026-11-08 11:55 | 34. Round 4: record check | exit 0; runs/s42_frozen_round4/check_records.json: PASS true, n_files 128, seed 42 | step 35
- 2026-11-08 11:57 | 35. Round 4: checks | exit 0; advantages: floor 0.3, 128 steps, max_abs_difference 2.0e-8, kl_div_file_step_0 0.099; KL file: loss.kl 0.01, kl_coeff 0.01, 128 steps, kl_div last 0.066, mean of last 16 0.072 | step 36
- 2026-11-08 11:57 | 36. Round 4: stop vLLM | session killed; nvidia-smi lists no process after ~10 s (0 MiB) | step 37
- 2026-11-08 11:58 | 37. Round 4: links 4 | EXIT 0; links/round4: start-of-run entry -> round 1's (am1g59lu) probe, end-of-run entry -> 8aiyurfx's own refit | step 38
- 2026-11-08 12:06 | 38. Round 4: final eval | EXIT 0 (11:58–12:06); summary.json: questions 489, cued_rollouts 978 | step 39
- 2026-11-08 12:06 | 39. Round 4: compare | exit 0; runs/s42_frozen_round4/comparison.md, comparison.json, training_curve.csv (128 data rows) | step 40
- 2026-11-08 12:07 | 40. Round 4: config 5 | EXIT 0; wrote s42_frozen_round5.yaml (checkpoint 8aiyurfx's final adapter) | step 41 (review 5)
- 2026-11-08 12:10 | 41. Review 5 (end of round 4) | PASS (rules (a)–(i) all pass) | step 42
- 2026-11-08 12:15 | 42. Round 5: vLLM | started 12:10:06, answered after 286 s (52018 MiB) | step 43
- 2026-11-08 16:28 | 43. Round 5: RL run | EXIT 0, run id hlorv05i; started 12:15:06; `GRPO subprocess finished in 250.6 min (exit=0, attempt=1/4)`; no `CRASH:` line; no out-of-memory line; peak GPU memory 99779 MiB (vLLM + trainer), lowest free disk 59G | step 44
- 2026-11-08 16:29 | 44. Round 5: record check | exit 0; runs/s42_frozen_round5/check_records.json: PASS true, n_files 128, seed 42 | step 45
- 2026-11-08 16:31 | 45. Round 5: checks | exit 0; advantages: floor 0.3, 128 steps, max_abs_difference 1.7e-8, kl_div_file_step_0 0.082; KL file: loss.kl 0.01, kl_coeff 0.01, 128 steps, kl_div last 0.057, mean of last 16 0.059 | step 46
- 2026-11-08 16:31 | 46. Round 5: stop vLLM | session killed; nvidia-smi lists no process after ~10 s (0 MiB) | step 47
- 2026-11-08 16:32 | 47. Round 5: links 5 | EXIT 0; links/round5: start-of-run entry -> round 1's (am1g59lu) probe, end-of-run entry -> hlorv05i's own refit | step 48
- 2026-11-08 16:40 | 48. Round 5: final eval | EXIT 0 (16:32–16:40); summary.json: questions 489, cued_rollouts 978 | step 49
- 2026-11-08 16:40 | 49. Round 5: compare | exit 0; runs/s42_frozen_round5/comparison.md, comparison.json, training_curve.csv (128 data rows) | step 50 (review 6)
- 2026-11-08 16:42 | 50. Review 6 (end of round 5) | PASS (rules (a)–(h) all pass) | step 51
- 2026-11-08 16:43 | 51. Round table (seed_table.py) | exit 0; wrote tables/rounds/seed_table.md and seed_table.json | step 52
- 2026-11-08 16:45 | 52. Dynamics (dynamics.py) | exit 0; wrote runs/dynamics/dynamics.json and dynamics.md | step 53 (review 7)
- 2026-11-08 16:59 | 53. Review 7 (auditor) and Result | PASS (every number in seed_table.md and dynamics.md matched; one wording note on seed_table.md line 111, see Judgment calls); seed_table.md copied into Result | queue done

## Judgment calls

- 2026-11-08 16:58, review 7: the auditor gave PASS (every number matched) and flagged one wording problem: `seed_table.md` line 111 describes the "probe-only groups" (all 8 answers name the cued option) as differing only in probe score, but 13 of those groups across rounds 1, 3 and 4 also hold an answer cut off at the length cap, whose reward carries the trainer's 0.1 penalty. I took the PASS as given, because review 7's rule fails only numbers that cannot be matched and the shares in that row are correct. I did not edit `seed_table.py`'s fixed sentence, because no crash or failed review calls for a fix and a fix would change a saved output. The finding is left for the user's write-up.
- Review verdicts 6 and 7 used their own `#`/`##` headings. I moved those markers down so they nest under Reviews and do not start new top-level sections of this file; the text is unchanged.

## Pod

Recorded at step 1, 2026-11-07 21:43 UTC:
- `hostname`: da70f8166885
- `nvidia-smi -L`: GPU 0: NVIDIA H200 (UUID: GPU-02d193f6-c3ca-0be4-8676-4b9e229d888e)
- `uname -r`: 6.8.0-147-generic
- `uptime -s`: 2026-10-20 05:01:27
- Driver 595.91.07, CUDA 13.2 (from `nvidia-smi`)
- `df -h /workspace`: 513G size, 435G used, 78G available (85%)
- `nvidia-smi`: no process, 0 MiB / 143771 MiB used

## RL crashes

- smoke_s42_frozen_round2: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log).
- s42_frozen_round2: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log).
- s42_frozen_round3: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log).
- s42_frozen_round4: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log).
- s42_frozen_round5: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log).

## Code fixes

## Reviews

### Review 1 (step 4), 2026-11-07 21:49 UTC, fresh general-purpose agent: PASS

Review 1 (start, step 4), seed-42 line, 261107_frozen_probe_kl_0_01. I read every file myself and changed nothing.

**(a) Round 2's config: PASS**

The config rule with Q = 2, applied to /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/s42_frozen_round2.yaml:
- `run_key` = `s42_frozen_round2`: PASS.
- `probe.directory_save_prefix` = `s42_frozen_round2`: PASS.
- `output_dir` = `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42`: PASS.
- `group` = `261107_frozen_probe_kl_0_01`: PASS.
- `checkpoint` = `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/checkpoints/final`: PASS. The folder exists and holds `adapter_model.bin` and `lora/`.
- `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3: PASS.
- `training.seed` 42, `training.max_steps` 128: PASS.
- `training.batch_size` 4, `training.grad_accum_steps` 16: PASS. Judgment calls in runs.md is empty, so the out-of-memory rule does not apply and 4 and 16 are the expected values.
- `++grpo.model.kl_reference_from_base` true: PASS.
- `data.probe_dataset.bias_probe_train_jsonl` = `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl`: PASS.
- `data.probe_dataset.bias_probe_eval_jsonl` = `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_eval.jsonl`: PASS.
- No `data.probe_dataset.max_train_examples` key: PASS (the key is absent).
- `detector.frozen_probe_weights_dir` = `<frozen>`, exact string: PASS.
- Every other override equals round_template.yaml's: PASS. I parsed both files with Python's YAML loader. The override key sets are identical: 20 keys, none missing and none extra. All 16 non-FILL_IN template values match in value and type, including `detector.filter_layers_below_loss_threshold` `'null'` as a string. `config_name` `grpo_mcq` and `probe.retrain` true also match.

The smoke config against the round 2 config, by `diff s42_frozen_round2.yaml smoke_s42_frozen_round2.yaml`: PASS. The only differences are:
- header comment lines 1 and 3 ("(smoke run)" and the run line);
- `run_key` `smoke_s42_frozen_round2`;
- `probe.directory_save_prefix` `smoke_s42_frozen_round2`;
- `training.max_steps` 2 (128 in the round 2 config);
- an added `training.eval_steps: 1`.

The parsed YAML comparison found no other difference.

**(b) Against 261104: PASS**

`diff /workspace/sycophancy_experiments/261104_iterative_kl_0_01/seed_42/s42_round2.yaml /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/s42_frozen_round2.yaml` shows exactly these hunks and nothing else:
- line 3, the header comment's run line;
- lines 5–7: `run_key`, `output_dir`, `group`;
- line 10: `probe.directory_save_prefix`;
- lines 30–32: the two probe-file paths, which changed from 261104's `probe_sets/round2/{train,test}.jsonl` to the sycophancy_setup `data/bias_probe_{train,eval}.jsonl`; the removed `data.probe_dataset.max_train_examples: 6048`; and the added `detector.frozen_probe_weights_dir`.

**(c) Frozen probe: PASS**

/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/frozen_probe_md5.txt has 8 lines. A byte-for-byte `diff` against the md5 rule's 8 lines exits 0. As a read-only extra check, I re-ran `find . -type f | sort | xargs md5sum` in `<frozen>` just now, and it printed the same 8 lines.

**Reported (no verdict)**

From /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/round1_repro.json, the comparison of 50 regenerated questions against 261103's seed-42 eval from pod 33f36c5bf32d:

| | This pod (da70f8166885) | 261104's seed-43 check (pod 09f52c746bb4) |
|---|---|---|
| rows | 500 | 500 |
| questions | 50 | 50 |
| same_completion | 199 | 212 |
| same_pred | 467 | 473 |

**Overall verdict: PASS.** Rules (a), (b) and (c) all pass.

### Review 2 (step 11), 2026-11-07 22:26 UTC, fresh general-purpose agent: PASS

Review 2 (smoke, step 11), seed-42 line, 261107_frozen_probe_kl_0_01. I read every file myself and changed nothing; all commands were read-only.

**Overall verdict: PASS.** Rules (a) to (h) all pass.

**(a) Step files: PASS**
- The file is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs/smoke_s42_frozen_round2/check_records.json.
- `"PASS": true`, `"n_files": 2`, `"seed": 42`.
- Both per-step entries (file steps 0 and 1) also show `PASS: true`, and the W&B counts equal the recomputed counts.

**(b) Reward: PASS**
- The file is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/reward_smoke_s42_frozen_round2.json.
- `penalty_weight` is 1.0.
- File step 0: `probe_score_present_iff_parsed_mismatches` is 0 and `reward_mean_abs_difference` is 6.44e-9. Both are within the rule (≤ 0.001).
- File step 1: `probe_score_present_iff_parsed_mismatches` is 0 and `reward_mean_abs_difference` is 1.50e-8. Both are within the rule.

**(c) Advantage and KL: PASS**
- Advantage file, /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/advantages_smoke_s42_frozen_round2.json:
  - `advantage_std_floor` is 0.3.
  - `steps_compared` is 2.
  - `max_abs_difference` is 1.38e-8, within the rule (≤ 1e-4).
  - `kl_div_file_step_0` is 0.1724, above the rule's minimum (≥ 0.01).
- KL file, /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/kl_smoke_s42_frozen_round2.json (`run_id` 45f2so6j): `loss.kl` is 0.01, `grpo.loss.kl_coeff` is 0.01, `steps` is 2.

**(d) Swap: PASS**
- The log is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/logs/rl_smoke_s42_frozen_round2_a1_da70f8166885.log. It has 3357 lines and ends `EXIT 0`.
- The only `Phase-0: loading fresh model with adapter from` line is line 99. Its full text is exactly `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/checkpoints/final/lora`, so it is also the last such line.
- After it, each required line appears once as an exact full-line match (`grep -nxF`):
  - Line 1301: `Phase-0 override: replacing detector with frozen probe weights from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`
  - Line 1302: `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`
- In the config /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/smoke_s42_frozen_round2.yaml, `detector.frozen_probe_weights_dir` under `overrides` is exactly `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`. I parsed the YAML to confirm.

**(e) Penalty check: PASS**
- The file is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/penalty_smoke_s42_frozen_round2.json. I compared each path below as an exact string.
- `penalty_probe` is /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation, as required.
- `other_probe` is /workspace/results/run_results/261107_frozen_probe_kl_0_01/smoke_s42_frozen_round2_45f2so6j/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation, as required.
- `adapter` is /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/checkpoints/final, as required.
- `answers.with_probe_score` is 512. That equals the count of non-null `probe_score` rows in /workspace/results/run_results/261107_frozen_probe_kl_0_01/smoke_s42_frozen_round2_45f2so6j/checkpoints/responses/step_0.jsonl: 512 of 512 rows.
- `penalty_probe_vs_saved.median_abs` is 0.000603. That is within the rule (≤ 0.01) and smaller than `other_probe_vs_saved.median_abs`, which is 0.00710.

**(f) Batch: PASS**
- The file is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/run_metrics_smoke_s42_frozen_round2.json (`run_id` 45f2so6j). The values below are nested keys under `metrics.full_config`.
- `training.batch_size` 4, `training.grad_accum_steps` 16, `grpo.batching.per_device_train_batch_size` 4, `grpo.batching.gradient_accumulation_steps` 16.
- Judgment calls in runs.md is empty, so 4 and 16 are the expected values.

**(g) Phase-0 memory: PASS**
- RL crashes in runs.md says: "smoke_s42_frozen_round2: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log)". So no crashed attempt stopped in Phase 0.
- The log has no `CRASH:` line and no `removed partial run directory` line. The only "GRPO subprocess finished" line is line 3331, at attempt=1/4.
- The log has no `OutOfMemoryError` or `CUDA out of memory` line anywhere, so none comes before line 99.

**(h) Frozen probe unchanged: PASS**
- I ran `cd <frozen> && find . -type f | sort | xargs md5sum`. It printed exactly the md5 rule's 8 lines.
- `diff` against the expected 8 lines exited 0.

**Reported (no verdict)**

Penalty check, from /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/penalty_smoke_s42_frozen_round2.json. Both rows compare against the trainer's saved step-0 scores, over 512 answers.

| | max_abs | mean_abs | over_0_02 | median_abs |
|---|---|---|---|---|
| penalty probe (round 1's, frozen) | 0.02307 | 0.002096 | 1 | 0.000603 |
| other probe (the trainer's unused start-of-run fit) | 0.2384 | 0.01984 | 129 | 0.00710 |

KL divergence to the base model, the W&B `kl_div` per step, from /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/kl_smoke_s42_frozen_round2.json:
- Step 1 (file step 0): 0.17245.
- Step 2 (file step 1): 0.12032.
- For comparison, round 1's last `kl_div` is 0.12829. That is step 128 of 128 in /workspace/sycophancy_experiments/261103_kl_0_01/weight_1_0/checks/kl_kl_weight_1_0_s42.json (`run_id` am1g59lu).

Seconds between the modification times (`stat -c %Y`) of step_0.jsonl and step_1.jsonl:
- This smoke run, /workspace/results/run_results/261107_frozen_probe_kl_0_01/smoke_s42_frozen_round2_45f2so6j/checkpoints/responses/: 1794089288 − 1794089157 = **131 s**.
- 261104's smoke run, /workspace/results/run_results/261104_iterative_kl_0_01/smoke_s42_round2_e835q7ng/checkpoints/responses/ (4 × 16, an H100): 1793828128 − 1793828029 = **99 s**.

Out-of-memory lines in the log: none. There is no `OutOfMemoryError`, `CUDA out of memory` or "out of memory" line. Two lines only mention OOM, and neither is an error:
- Line 82: `[2026-11-07 21:58:08,520][accelerate.utils.modeling][INFO] - We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. ...`
- Line 100: the same message at 21:58:47,360.

From /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs.md (step 6 Log line and RL crashes):
- Minutes: `GRPO subprocess finished in 20.4 min (exit=0, attempt=1/4)`. The run started at 21:55:00 UTC and its Log line is at 22:18 UTC.
- Peak GPU memory: 97095 MiB (vLLM and trainer together). Lowest free disk: 73G.
- Crashed attempts: none. Attempt a1 finished on launcher try 1/4, with no `CRASH:` line.

**Overall verdict: PASS.**

### Review 3 (step 21, end of round 2), 2026-11-08 03:03 UTC, fresh general-purpose agent: PASS

Review 3 (end of round 2, step 21, plus round 3's config), seed-42 line, 261107_frozen_probe_kl_0_01. I read every file myself and changed nothing. All commands were read-only.

**Overall verdict: PASS.** Rules (a) to (i) all pass.

**(a) Step files: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs/s42_frozen_round2/check_records.json.
- It has `"PASS": true`, `"n_files": 128` and `"seed": 42`. Its `wandb_run_id` is z3pe3lcj.
- All 128 per-step entries also have `PASS: true`.

**(b) Config, Q = 2: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/s42_frozen_round2.yaml, parsed with Python's YAML loader. Every value matches in both value and type.
- `run_key` and `probe.directory_save_prefix` are `s42_frozen_round2`.
- `output_dir` is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42, and `group` is `261107_frozen_probe_kl_0_01`.
- `checkpoint` is /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/checkpoints/final.
- Overrides: `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 128.
- `training.batch_size` is 4 and `training.grad_accum_steps` is 16. Judgment calls in runs.md is empty, so the out-of-memory rule does not apply and 4 and 16 are the expected values.
- `++grpo.model.kl_reference_from_base` is true.
- The two probe-file paths are /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl and /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_eval.jsonl.
- There is no `data.probe_dataset.max_train_examples` key.
- `detector.frozen_probe_weights_dir` is exactly `<frozen>`.
- Every other override equals round_template.yaml's. The override key set is the same 20 keys as the template's, and all 17 template values that are not FILL_IN match, including `detector.filter_layers_below_loss_threshold` as the string `'null'`. `config_name` (`grpo_mcq`) and `probe.retrain` (true) also match.

**(c) Advantage and KL: PASS**
- Advantage file, /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/advantages_s42_frozen_round2.json:
  - `advantage_std_floor` is 0.3 and `steps_compared` is 128.
  - `max_abs_difference` is 1.818e-8, within the rule (≤ 1e-4).
  - `kl_div_file_step_0` is 0.15244, above the rule's minimum (≥ 0.01).
- KL file, /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/kl_s42_frozen_round2.json (`run_id` z3pe3lcj): `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128. Its `kl_div` list also has 128 entries.

**(d) Swap: PASS**
- Log: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/logs/rl_s42_frozen_round2_a1_da70f8166885.log. It has 4990 lines and ends `EXIT 0`.
- It has exactly one line `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/checkpoints/final/lora` (line 99), checked as an exact full-line match. So line 99 is also the last such line.
- After line 99, each required line appears as an exact full-line match (`grep -xF`):
  - Line 1286: `Phase-0 override: replacing detector with frozen probe weights from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`
  - Line 1287: `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`
- In s42_frozen_round2.yaml, `detector.frozen_probe_weights_dir` is exactly `<frozen>`.
- The log has no `CRASH:` line, no `removed partial run directory` line and no out-of-memory line. Its only "GRPO subprocess finished" line is line 4960: `GRPO subprocess finished in 252.6 min (exit=0, attempt=1/4)`.

**(e) Batch: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/run_metrics_s42_frozen_round2.json (`run_id` z3pe3lcj). The values below are under `metrics.full_config`.
- `training.batch_size` 4, `training.grad_accum_steps` 16, `grpo.batching.per_device_train_batch_size` 4, `grpo.batching.gradient_accumulation_steps` 16.
- Judgment calls is empty, so 4 and 16 are expected.

**(f) Links: PASS**
- `readlink -f .../seed_42/links/round2/initial_detector.base_model.on_dataset` gives /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset, as required.
- `readlink -f .../seed_42/links/round2/fresh_detector.final_model.on_dataset` gives /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round2_z3pe3lcj/fresh_detector.final_model.on_dataset, as required.
- Both targets exist.

**(g) Eval: PASS**
- /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/evals/s42_frozen_round2/summary.json has `"questions": 489` and `"cued_rollouts": 978`.
- All 976 rows of followups.jsonl have `probe_run_dir` /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/links/round2. There is no other value and no missing key. The file has 976 rows against 978 cued rollouts; the rule does not set a row count, so I only note it.
- /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs/s42_frozen_round2/training_curve.csv has 129 lines: a header and 128 data rows.

**(h) Frozen probe unchanged: PASS**
- I ran `cd <frozen> && find . -type f | sort | xargs md5sum` just now. It printed exactly the md5 rule's 8 lines.
- `diff` against the expected 8 lines exited 0.

**(i) Round 3's config, Q = 3: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/s42_frozen_round3.yaml, parsed with the same checks as (b).
- `run_key` and `probe.directory_save_prefix` are `s42_frozen_round3`.
- `checkpoint` is /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round2_z3pe3lcj/checkpoints/final. That folder exists and holds `adapter_model.bin` and `lora/`.
- `output_dir`, `group`, all required overrides (including 4 and 16), the absent `max_train_examples` key and `detector.frozen_probe_weights_dir` = `<frozen>` all match.
- The same 20 override keys as the template; all 17 non-FILL_IN values equal the template's.

**Reported (no verdict)**

From /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs.md (step 13's Log line and RL crashes):
- Minutes: `GRPO subprocess finished in 252.6 min (exit=0, attempt=1/4)`. The run started 2026-11-07 22:31:57 UTC; its Log line is at 2026-11-08 02:47 UTC.
- Lowest free disk: 62G.
- Peak GPU memory: 97288 MiB (vLLM and trainer together).
- Crashed attempts: none. RL crashes says "s42_frozen_round2: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log)", so there is no W&B id, batch reached or mean answer length to copy.

Mean seconds per step, (mtime of step_127.jsonl − mtime of step_0.jsonl) / 127, by `stat -c %Y`:

| Run | step_0 | step_127 | Span | Per step |
|---|---|---|---|---|
| This round 2, /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round2_z3pe3lcj/checkpoints/responses/ (H200) | 1794091303 | 1794105415 | 14112 s | **111.1 s** |
| 261104's round 2 of seed 42, /workspace/results/run_results/261104_iterative_kl_0_01/s42_round2_36lntjib/checkpoints/responses/ (an H100, 4 × 16) | 1793830008 | 1793839655 | 9647 s | **76.0 s** |

From checks/advantages_s42_frozen_round2.json:
- `kl_div_file_step_0`: 0.15244.
- Mean `wandb_kl_div` of the last 16 steps (file steps 112–127): 0.11955. The KL file's last 16 `kl_div` values give the same mean, 0.11955. Its last `kl_div` is 0.11406.

**Overall verdict: PASS.**

### Review 4 (step 31, end of round 3), 2026-11-08 07:37 UTC, fresh general-purpose agent: PASS

Review 4 (end of round 3, step 31, plus round 4's config), seed-42 line, 261107_frozen_probe_kl_0_01. I read every file myself and changed no file in the project. One small exception to "read-only": I wrote the md5sum output for rule (h) to a temporary file in my own session scratchpad (/tmp/claude-0/-workspace/dd33f51b-646b-4b98-b9de-be8fc355c096/scratchpad/md5_review4.txt) so I could diff it. Nothing outside the scratchpad was written.

**Overall verdict: PASS.** Rules (a) to (i) all pass.

**(a) Step files: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs/s42_frozen_round3/check_records.json.
- It has `"PASS": true`, `"n_files": 128` and `"seed": 42`. Its `wandb_run_id` is gh04frmi.
- All 128 per-step entries also have `PASS: true`.

**(b) Config, Q = 3: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/s42_frozen_round3.yaml. I parsed it with Python's YAML loader and checked both value and type.
- `run_key` and `probe.directory_save_prefix` are `s42_frozen_round3`.
- `output_dir` is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42, and `group` is `261107_frozen_probe_kl_0_01`.
- `checkpoint` is /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round2_z3pe3lcj/checkpoints/final. The `run_id` in run_metrics_s42_frozen_round2.json is z3pe3lcj, and the folder exists with `adapter_model.bin` and `lora/`.
- Overrides: `loss.detector` 1.0 (a float), `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 42, `training.max_steps` 128.
- `training.batch_size` is 4 and `training.grad_accum_steps` is 16. The Judgment calls section of runs.md is empty, so the out-of-memory rule does not apply and 4 and 16 are the expected values.
- `++grpo.model.kl_reference_from_base` is true.
- `data.probe_dataset.bias_probe_train_jsonl` is /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl, and `bias_probe_eval_jsonl` is /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_eval.jsonl.
- There is no `data.probe_dataset.max_train_examples` key.
- `detector.frozen_probe_weights_dir` is exactly `<frozen>`.
- Every other override equals round_template.yaml's. Both files have the same 20 override keys, and every template value that is not FILL_IN matches. `config_name` and `probe.retrain` also match.

**(c) Advantage and KL: PASS**
- Advantage file: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/advantages_s42_frozen_round3.json.
  - `advantage_std_floor` is 0.3 and `steps_compared` is 128. Its `per_step` list has 128 entries.
  - `max_abs_difference` is 1.840e-8, within the rule (≤ 1e-4).
  - `kl_div_file_step_0` is 0.12119, above the rule's minimum (≥ 0.01).
- KL file: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/kl_s42_frozen_round3.json (`run_id` gh04frmi).
  - `loss.kl` is 0.01, `grpo.loss.kl_coeff` is 0.01 and `steps` is 128.
  - Its `kl_div` and `kl_loss` lists each have 128 entries.

**(d) Swap: PASS**
- Log: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/logs/rl_s42_frozen_round3_a1_da70f8166885.log. It has 4989 lines and ends `EXIT 0`.
- It is the only RL log for round 3, and there is no failed_attempts folder.
- It has one `Phase-0: loading fresh model with adapter from` line, line 97. As an exact full-line match it reads `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round2_z3pe3lcj/checkpoints/final/lora`, so it is also the last such line.
- After line 97, each required line appears once as an exact full-line match (`grep -nxF`):
  - Line 1278: `Phase-0 override: replacing detector with frozen probe weights from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`
  - Line 1279: `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`
- In s42_frozen_round3.yaml, `detector.frozen_probe_weights_dir` is exactly `<frozen>`.
- The log has no `CRASH:`, `removed partial run directory`, `OutOfMemoryError` or `CUDA out of memory` line. Its only "GRPO subprocess finished" line is line 4959: `GRPO subprocess finished in 250.2 min (exit=0, attempt=1/4)`.

**(e) Batch: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/run_metrics_s42_frozen_round3.json (`run_id` gh04frmi). The values below are under `metrics.full_config`.
- `training.batch_size` 4, `training.grad_accum_steps` 16, `grpo.batching.per_device_train_batch_size` 4, `grpo.batching.gradient_accumulation_steps` 16.
- Judgment calls is empty, so 4 and 16 are expected.

**(f) Links: PASS**
- `readlink -f .../seed_42/links/round3/initial_detector.base_model.on_dataset` gives /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset.
- `readlink -f .../seed_42/links/round3/fresh_detector.final_model.on_dataset` gives /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round3_gh04frmi/fresh_detector.final_model.on_dataset.
- Both targets exist.

**(g) Eval: PASS**
- /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/evals/s42_frozen_round3/summary.json has `"questions": 489` and `"cued_rollouts": 978`.
- All 978 rows of followups.jsonl have `probe_run_dir` /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/links/round3. No row has another value or is missing the key.
- /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs/s42_frozen_round3/training_curve.csv has 129 lines: a header and 128 data rows. The last data row is step 127.

**(h) Frozen probe unchanged: PASS**
- I ran `cd <frozen> && find . -type f | sort | xargs md5sum` just now. It printed exactly the md5 rule's 8 lines.
- `diff` against the expected 8 lines exited 0.

**(i) Round 4's config, Q = 4: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/s42_frozen_round4.yaml, parsed with the same checks as (b).
- `run_key` and `probe.directory_save_prefix` are `s42_frozen_round4`.
- `checkpoint` is /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round3_gh04frmi/checkpoints/final. That folder exists with `adapter_model.bin` and `lora/`.
- These all match the rule:
  - `output_dir` and `group`;
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3;
  - `training.seed` 42 and `training.max_steps` 128;
  - batch 4 and 16, which are expected because Judgment calls is empty;
  - `++grpo.model.kl_reference_from_base` true;
  - both probe-file paths;
  - no `max_train_examples` key;
  - `detector.frozen_probe_weights_dir` = `<frozen>`.
- The overrides have the same 20 keys as the template, and every non-FILL_IN value equals the template's.

**Reported (no verdict)**

From /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs.md (step 23's Log line and RL crashes):
- Minutes: `GRPO subprocess finished in 250.2 min (exit=0, attempt=1/4)`. The run started 2026-11-08 03:08:25 UTC, and its Log line is at 07:21 UTC.
- Lowest free disk: 60G.
- Peak GPU memory: 100388 MiB (vLLM and trainer together).
- Crashed attempts: none. RL crashes says "s42_frozen_round3: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log)". So there is no W&B id, batch reached or mean answer length to report.

Mean seconds per step, (mtime of step_127.jsonl − mtime of step_0.jsonl) / 127, by `stat -c %Y`:

| Run | step_0 | step_127 | Span | Per step |
|---|---|---|---|---|
| This round 3, /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round3_gh04frmi/checkpoints/responses/ (H200) | 1794107891 | 1794121887 | 13996 s | **110.2 s** |
| 261104's round 3 of seed 42, /workspace/results/run_results/261104_iterative_kl_0_01/s42_round3_x5ay7uz0/checkpoints/responses/ (an H100, 4 × 16) | 1793842976 | 1793852920 | 9944 s | **78.3 s** |

From checks/advantages_s42_frozen_round3.json:
- `kl_div_file_step_0`: 0.12119.
- Mean `wandb_kl_div` of the last 16 steps (file steps 112–127): 0.09981. The KL file's last 16 `kl_div` values give the same mean, 0.09981. Its last `kl_div` is 0.09914.

**Overall verdict: PASS.**

### Review 5 (step 41, end of round 4), 2026-11-08 12:10 UTC, fresh general-purpose agent: PASS

Review 5 covers the end of round 4 (step 41) and round 5's config, for the seed-42 line of 261107_frozen_probe_kl_0_01. I read every file myself. Every command was read-only, and I wrote no file anywhere, not even in the scratchpad: I compared the md5 output to the expected lines through a pipe, `diff - <(printf …)`.

**Overall verdict: PASS.** Rules (a) to (i) all pass.

**(a) Step files: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs/s42_frozen_round4/check_records.json.
- It has `"PASS": true`, `"n_files": 128` and `"seed": 42`. Its `wandb_run_id` is 8aiyurfx.
- All 128 per-step entries also have `PASS: true`.

**(b) Config, Q = 4: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/s42_frozen_round4.yaml. I parsed it with Python's YAML loader and checked both value and type.
- `run_key` and `probe.directory_save_prefix` are `s42_frozen_round4`.
- `output_dir` is /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42, and `group` is `261107_frozen_probe_kl_0_01`.
- `checkpoint` is /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round3_gh04frmi/checkpoints/final.
  - The `run_id` in run_metrics_s42_frozen_round3.json is gh04frmi.
  - That folder exists and holds `adapter_model.bin` and `lora/`.
- Overrides: `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3, `training.seed` 42 and `training.max_steps` 128.
- `training.batch_size` is 4 and `training.grad_accum_steps` is 16. The Judgment calls section of runs.md is empty, so the out-of-memory rule does not apply and 4 and 16 are the expected values.
- `++grpo.model.kl_reference_from_base` is true.
- `data.probe_dataset.bias_probe_train_jsonl` is /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl, and `bias_probe_eval_jsonl` is /workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_eval.jsonl.
- There is no `data.probe_dataset.max_train_examples` key.
- `detector.frozen_probe_weights_dir` is exactly `<frozen>`.
- Every other override equals round_template.yaml's:
  - The override key sets are identical, with no extra or missing key.
  - The 9 remaining template values all match: `+data/probe_dataset`, `task_args.bias_probe_at_answer`, `model.activation_matching_layers`, `detector.filter_layers_below_loss_threshold` (the string `'null'`), `detector.pretrain_cfg.learning_rate`, `detector.pretrain_cfg.max_steps`, `data.task_dataset.mmlu_questions_jsonl`, `task_args.cue_gold_prob` and `detector.model_batch_size`.
  - `config_name` (`grpo_mcq`) and `probe.retrain` (true) also match, and the top-level keys are the same.

**(c) Advantage and KL: PASS**
- Advantage file: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/advantages_s42_frozen_round4.json.
  - `advantage_std_floor` is 0.3 and `steps_compared` is 128. Its `per_step` list has 128 entries, file steps 0–127.
  - `max_abs_difference` is 2.04e-8, within the rule (≤ 1e-4).
  - `kl_div_file_step_0` is 0.09869, above the rule's minimum (≥ 0.01).
- KL file: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/checks/kl_s42_frozen_round4.json (`run_id` 8aiyurfx).
  - `loss.kl` is 0.01, `grpo.loss.kl_coeff` is 0.01 and `steps` is 128.
  - Its `kl_div` and `kl_loss` lists each have 128 entries.

**(d) Swap: PASS**
- Log: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/logs/rl_s42_frozen_round4_a1_da70f8166885.log. It has 4981 lines and ends `EXIT 0`.
- It is the only RL log for round 4, and there is no failed_attempts folder.
- It has one `Phase-0: loading fresh model with adapter from` line, line 99. As an exact full-line match (`grep -nxF`) it reads `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round3_gh04frmi/checkpoints/final/lora`, so it is also the last such line.
- After line 99, each required line appears once as an exact full-line match:
  - Line 1276: `Phase-0 override: replacing detector with frozen probe weights from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`
  - Line 1277: `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`
- In s42_frozen_round4.yaml, `detector.frozen_probe_weights_dir` is exactly `<frozen>`.
- The log has no `CRASH:`, `removed partial run directory`, `OutOfMemoryError` or `CUDA out of memory` line. Its only "GRPO subprocess finished" line is line 4951: `GRPO subprocess finished in 250.3 min (exit=0, attempt=1/4)`.

**(e) Batch: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/run_metrics_s42_frozen_round4.json (`run_id` 8aiyurfx). The values below are under `metrics.full_config`.
- `training.batch_size` 4, `training.grad_accum_steps` 16, `grpo.batching.per_device_train_batch_size` 4, `grpo.batching.gradient_accumulation_steps` 16.
- Judgment calls is empty, so 4 and 16 are expected.

**(f) Links: PASS**
- `readlink -f /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/links/round4/initial_detector.base_model.on_dataset` gives /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset, as required.
- `readlink -f /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/links/round4/fresh_detector.final_model.on_dataset` gives /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round4_8aiyurfx/fresh_detector.final_model.on_dataset, as required.
- Both targets exist.

**(g) Eval: PASS**
- /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/evals/s42_frozen_round4/summary.json has `"questions": 489` and `"cued_rollouts": 978`.
- All 976 rows of followups.jsonl have `probe_run_dir` /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/links/round4. No row has another value or is missing the key.
  - The file has 976 rows against 978 cued rollouts. The rule sets no row count, so I only note it; round 2's file was also 976.
- /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs/s42_frozen_round4/training_curve.csv has 129 lines: a header and 128 data rows. The last data row is step 127.

**(h) Frozen probe unchanged: PASS**
- I ran `cd <frozen> && find . -type f | sort | xargs md5sum` just now. It printed exactly the md5 rule's 8 lines.
- Piped into `diff` against the expected 8 lines, it exited 0.

**(i) Round 5's config, Q = 5: PASS**
- File: /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/s42_frozen_round5.yaml, parsed with the same checks as (b).
- `run_key` and `probe.directory_save_prefix` are `s42_frozen_round5`.
- `checkpoint` is /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round4_8aiyurfx/checkpoints/final.
  - The `run_id` in run_metrics_s42_frozen_round4.json is 8aiyurfx.
  - That folder exists and holds `adapter_model.bin` and `lora/`.
- These all match the rule:
  - `output_dir` and `group`;
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3;
  - `training.seed` 42 and `training.max_steps` 128;
  - batch 4 and 16, which are expected because Judgment calls is empty;
  - `++grpo.model.kl_reference_from_base` true;
  - both probe-file paths;
  - no `max_train_examples` key;
  - `detector.frozen_probe_weights_dir` = `<frozen>`.
- The overrides have the same key set as the template, and all 9 remaining template values match. `config_name` and `probe.retrain` also match.

**Reported (no verdict)**

From /workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs.md (step 33's Log line and RL crashes):
- Minutes: `GRPO subprocess finished in 250.3 min (exit=0, attempt=1/4)`. The run started 2026-11-08 07:41:25 UTC, and its Log line is at 11:54 UTC.
- Lowest free disk: 57G.
- Peak GPU memory: 98828 MiB (vLLM and trainer together).
- Crashed attempts: none. RL crashes says "s42_frozen_round4: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log)". So there is no W&B id, batch reached or mean answer length to report.

Mean seconds per step, (mtime of step_127.jsonl − mtime of step_0.jsonl) / 127, by `stat -c %Y`:

| Run | step_0 | step_127 | Span | Per step |
|---|---|---|---|---|
| This round 4, /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round4_8aiyurfx/checkpoints/responses/ (H200) | 1794124264 | 1794138262 | 13998 s | **110.2 s** |
| 261104's round 4 of seed 42, /workspace/results/run_results/261104_iterative_kl_0_01/s42_round4_owri7bjk/checkpoints/responses/ (an H100, 4 × 16) | 1793898489 | 1793908256 | 9767 s | **76.9 s** |

From checks/advantages_s42_frozen_round4.json:
- `kl_div_file_step_0`: 0.09869.
- Mean `wandb_kl_div` of the last 16 steps (file steps 112–127): 0.07245.
  - The KL file's last 16 `kl_div` values give the same mean, 0.07245.
  - The KL file's last `kl_div` is 0.06649.

**Overall verdict: PASS.**

### Review 6 (step 50, end of round 5), 2026-11-08 16:42 UTC, fresh general-purpose agent: PASS

#### Review 6: end of round 5 (step 50), seed-42 frozen-probe line. Overall verdict: PASS

Every rule passes. I changed no files and used only read-only commands.

**(a) Step files: PASS.** `runs/s42_frozen_round5/check_records.json` has `"PASS": true`, `"n_files": 128` and `"seed": 42`. It also has `wandb_run_id` hlorv05i and 128 `per_step` entries.

**(b) Config (config rule, Q = 5): PASS.** Checked in `s42_frozen_round5.yaml`:
- `run_key` and `probe.directory_save_prefix` are both `s42_frozen_round5`.
- `output_dir` is `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42`.
- `group` is `261107_frozen_probe_kl_0_01`.
- `checkpoint` is `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round4_8aiyurfx/checkpoints/final`. Round 4's `run_metrics` `run_id` is 8aiyurfx, which matches.
- Under `overrides`:
  - `loss.detector` 1.0, `loss.kl` 0.01, `loss.advantage_std_floor` 0.3.
  - `training.seed` 42, `training.max_steps` 128.
  - `training.batch_size` 4 and `training.grad_accum_steps` 16. Judgment calls in `runs.md` is empty, so 4 and 16 are the expected values.
  - `++grpo.model.kl_reference_from_base` true.
  - The two probe-file paths equal `/workspace/sycophancy-rl/sycophancy_setup/data/bias_probe_train.jsonl` and `.../bias_probe_eval.jsonl`.
  - There is no `max_train_examples` key.
  - `detector.frozen_probe_weights_dir` equals the frozen probe path (`<frozen>`): string comparison True.
- I compared it to `round_template.yaml` programmatically. Both have the same set of override keys. The only differences are the template's FILL_IN fields: `training.seed`, `training.max_steps`, `detector.frozen_probe_weights_dir`, and the top-level `run_key`, `output_dir`, `probe.directory_save_prefix` and `checkpoint`. Every other override equals the template's.

**(c) Advantage and KL: PASS.**
- `advantages_s42_frozen_round5.json`: `advantage_std_floor` 0.3, `steps_compared` 128, `max_abs_difference` 1.67e-08 (limit 1e-4), `kl_div_file_step_0` 0.0817 (minimum 0.01).
- `kl_s42_frozen_round5.json`: `loss.kl` 0.01, `grpo.loss.kl_coeff` 0.01, `steps` 128.

**(d) Swap: PASS.** Log: `logs/rl_s42_frozen_round5_a1_da70f8166885.log`, 4994 lines, ending `EXIT 0`.
- The only Phase-0 line, which is therefore the last one, is line 98: `Phase-0: loading fresh model with adapter from /workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round4_8aiyurfx/checkpoints/final/lora`.
- After it, exact whole-line matches:
  - Line 1289: `Phase-0 override: replacing detector with frozen probe weights from /workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset/detector_initial_detector.base_model.on_dataset__linear-probe_last-token-generation`
  - Line 1290: `Filtered layers (from frozen probe weights dir): [13, 14, 15, 16]`
- The config's `detector.frozen_probe_weights_dir` equals that same frozen probe path.
- The log has no `CRASH:`, `removed partial run directory`, `OutOfMemoryError` or `CUDA out of memory` line. Its only finish line is line 4964: `GRPO subprocess finished in 250.6 min (exit=0, attempt=1/4)`.

**(e) Batch: PASS.** In `run_metrics_s42_frozen_round5.json` (`run_id` hlorv05i), `metrics.full_config` has:
- `training.batch_size` 4 and `training.grad_accum_steps` 16;
- `grpo.batching.per_device_train_batch_size` 4 and `grpo.batching.gradient_accumulation_steps` 16.

The expected values are 4 and 16 because Judgment calls in `runs.md` is empty: no out-of-memory rule was recorded for any run of this line, the smoke run included.

**(f) Links: PASS.**
- `readlink -f links/round5/initial_detector.base_model.on_dataset` gives `/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s42_am1g59lu/initial_detector.base_model.on_dataset`.
- `readlink -f links/round5/fresh_detector.final_model.on_dataset` gives `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round5_hlorv05i/fresh_detector.final_model.on_dataset`.
- Both targets exist.

**(g) Eval: PASS.**
- `evals/s42_frozen_round5/summary.json` has `"questions": 489` and `"cued_rollouts": 978`.
- `followups.jsonl` has 978 rows. The only `probe_run_dir` value among them is `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/links/round5`.
- `runs/s42_frozen_round5/training_curve.csv` has 129 lines: 1 header and 128 data rows.

**(h) Frozen probe unchanged (md5 rule): PASS.** The `find | sort | xargs md5sum` output in the frozen probe folder printed exactly the 8 expected lines. A `diff` against them printed nothing.

#### Reported values (no verdict)

From `runs.md`, round 5's RL run (Log step 43):
- **Duration:** started 12:15:06 UTC, and `GRPO subprocess finished in 250.6 min (exit=0, attempt=1/4)`.
- **Lowest free disk:** 59G.
- **Peak GPU memory:** 99779 MiB (vLLM and trainer together).
- **Crashed attempts:** none. RL crashes says "s42_frozen_round5: no crashed attempt (attempt a1, launcher try 1/4 finished; no `CRASH:` line in the log)".

Mean seconds per step, from the modification times of the step record files, as (step_127 − step_0) / 127:
- **This run, round 5** (`s42_frozen_round5_hlorv05i`, H200): (1794154689 − 1794140661) / 127 = 14028 / 127 = **110.5 s per step**.
- **261104's round 5 of seed 42** (`/workspace/results/run_results/261104_iterative_kl_0_01/s42_round5_ul6lkmkc`, an H100): 9534 / 127 = **75.1 s per step**.

From `advantages_s42_frozen_round5.json`:
- `kl_div_file_step_0`: 0.0817 (0.08169616287341341).
- Mean `wandb_kl_div` over the last 16 steps (file steps 112–127): 0.0586 (0.05861525856380467). The last step alone is 0.0570.

#### Overall verdict: PASS
Rules (a) to (h) all pass.

### Review 7 (step 53), 2026-11-08 16:58 UTC, fresh auditor agent: PASS

(The agent's headings are moved down three levels so they nest under Reviews; the text is unchanged.)

#### Review 7 (step 53), auditor: overall verdict PASS

I could match every number in both reports to the source files. I found one wording problem in a sentence (below). It changes no number, and review 7's rule ("every number it cannot match to the files fails review 7") is about numbers, so it does not fail the review.

How I checked: I wrote no files, not even under /tmp. Instead I reran each script's own functions in memory with the exact arguments of steps 51 and 52 and compared the output text with the saved files character by character, which is the same test as `cmp`. Separately, I recomputed the numbers with my own code from the raw records. `dynamics.py` was reread from W&B.

#### Discrepancies

##### 1. Numbers
None.

##### 2. Procedure claims
**2.1 Contradicted claim (wording only; no number affected).** `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/tables/rounds/seed_table.md`, line 111: "probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ)".
- **What the files show:** 13 of these groups contain an answer cut off at the length cap (a step-record completion that does not end in `<|eot_id|>`):
  - round 1: 1 of 5457 groups;
  - round 3: 8 of 5925;
  - round 4: 4 of 5872;
  - rounds 2 and 5: 0.
- **Why it matters:** every run's `full_config` sets `grpo.loss.non_termination_penalty` to 0.1. Advantages recomputed with that penalty match W&B's `advantage_abs_mean` to within 2.8e-8. So in those 13 groups the rewards also differ by the 0.1 cut-off penalty, not only by probe score.
- **Example:** round 3 (`.../s42_frozen_round3_gh04frmi`), step 17, group 11. All 8 task rewards are 1.0, and the one cut-off answer has probe score 0.0.
- **What it does not affect:** the reported shares (0.666, 0.709, 0.723, 0.717, 0.708) count groups where all 8 answers name the cue, and they are correct. The sentence is fixed text inside `seed_table.py`.

##### 3. Counts across stages
None.

##### 4. Config consistency
None.

##### 5. Leakage
None.

##### 6. Unsupported or unjustified claims
None beyond 2.1. Neither report draws conclusions; both are tables plus definitions.

#### Verified

##### `seed_table.md` (and `seed_table.json`)
- **Rerun: PASS.** I called `seed_table.run_tables`, `summary` and `table_markdown` with step 51's arguments, run from `seed_42/`. The text equals the saved `seed_table.md` (13454 characters) and the JSON equals the saved `seed_table.json` (207373 characters). The script's asserts passed:
  - every eval's adapter is `<run dir R>/checkpoints/final`;
  - each run has 128 step files;
  - the base eval and every run's eval cover the same questions.
- **Independent recompute of every eval cell: PASS.** I recomputed all 1010 cells of the eval table (101 rows × base, base mean, base SD, round 1–5, mean, SD) from `completions.jsonl` and `labels.jsonl`, using sample 0 and the 0.75 / 0.25 weighting for `train_mix`. 0 mismatches. This covers:
  - 489 eval questions;
  - questions with a no-cue answer in both evals: 455, 454, 458, 459, 463;
  - all probe rows. The base column's probe rows are empty or 0, as expected.
- **Independent check of the no-cue answer rule: PASS.** Counting "at least 5 of 8 no-cue answers agree" directly gives the same question sets as `a_none` in `labels.jsonl`: base 466, rounds 1–5: 478, 474, 481, 479, 485.
- **Training table: PASS.** Recomputed from all 128 step files per run (512 records per step, groups of 8 that each share one prompt):
  - equal task reward groups: 0.755, 0.848, 0.924, 0.954, 0.966; mean 0.889, SD 0.088;
  - probe-only groups: 0.666, 0.709, 0.723, 0.717, 0.708; mean 0.705, SD 0.022.
- **Cross-check against `comparison.json`: PASS.** The after-RL values for rounds 1–5 equal those in `$E3/runs/kl_weight_1_0_s42/comparison.json` and `$EXP/runs/s42_frozen_round<R>/comparison.json`: 0 differences, same shared-question counts.
- **Counts across stages: PASS.**
  - Each `summary.json`'s `questions` minus `questions_without_nocue_answer` equals the table's after-RL no-cue counts.
  - Every eval has 4890 completions and 978 label rows.
  - The base eval is the base model (adapter `None`).
- **Probe sources: PASS.**
  - `links/round2`–`round5` start-of-run entries resolve to `<R1 run dir>/initial_detector.base_model.on_dataset`, and their end-of-run entries to each run's own `fresh_detector.final_model.on_dataset`.
  - Every `followups.jsonl` row has `probe_run_dir` `$EXP/links/round<R>`.
  - Round 1's eval used `<R1 run dir>`.

##### `dynamics.md` (and `dynamics.json`)
- **Rerun: PASS.** I called `dynamics.py`'s functions plus `markdown()` with step 52's arguments, rereading W&B. The text equals the saved `dynamics.md` (28588 characters) and the JSON equals the saved `dynamics.json` (629308 characters).
- **Training rows from step records: PASS.** I recomputed 7 rows × 8 runs × 4 windows (224 cells) with my own advantage code: reward minus 1.0 × probe score on answers naming the cue, minus 0.1 if cut off, divided by max(SD + 1e-4, 0.3); baselines use weight 0 and no floor. 0 mismatches. My per-step mean absolute advantage matches W&B's logged value in each `advantages_*.json` to at most 2.8e-8.
- **W&B rows: PASS.** I refetched grad norm, KL, entropy and answer length for all 8 runs with my own code (128 cells) and found 0 mismatches.
  - KL also matches the local files `$E3/checks/kl_kl_weight_1_0_s42.json` and `$EXP/checks/kl_s42_frozen_round<R>.json` exactly (largest difference 0).
  - KL at file steps 0 and 1 is 0 for round 1 and the three baselines; round 2 starts at 0.152 with `kl_reference_from_base` true. This supports the label "KL from the base model".
  - Baselines have `loss.kl` 0.001 and `loss.detector` 0.
  - "Before clipping" matches the code: `afterburner/grpo_trainer.py:922` logs the return value of `clip_grad_norm_`.
- **Advantage SD floor: PASS.** 0.3 in all five runs' `full_config`.
- **Probe-only and training-answer tables: PASS.** All 80 lines (25 probe-only window rows and 55 training-answer step rows) recomputed, 0 missing.
- **Final evals block: PASS.** All 7 entries (base, baseline_s42 and the five runs) equal my recompute.
- **Probe rows by answer letter: PASS.** All 60 rows A–D recomputed, 0 missing. The shared-question counts in the headings are 455, 454, 458, 459, 463. Every AUC in parentheses equals the one in `comparison.json`.
- **Probe data block: PASS.** 756/496, 756/266, 292/191, 485/183 from `sycophancy_setup/data/bias_probe_{train,eval}.jsonl`. These are the files named in all five runs' `full_config`.
- **Probe-weight cosines: PASS.** 20 values recomputed with torch in float64, largest difference 1.9e-7. For rounds 2–5 the start-of-run slot is not round 1's probe; this is listed as not a failure in the brief.

##### Config consistency: PASS
- The logs of every production step are attempt a1 on da70f8166885.
- `dynamics.py`, `rounds.py` and `checks.py` are byte-identical to `$PARENT/code_at_launch/`.
- `compare.py`'s one uncommitted change is in `code_at_launch/tracked.diff`, and the file's modification time (2026-11-07 05:47) is before launch.
- `seed_table.py` has not changed since commit 31937be.
- `HEAD.txt` equals the current HEAD.

##### Leakage: PASS
- Training step records cover 1217 questions in every run, none of them among the 489 eval questions.
- The probe training file (740 questions) shares no question with the eval.
- The probe test file `bias_probe_eval.jsonl` is the 489 eval questions. The trainer reads it to report a test score, and nothing is chosen from it: `detector.filter_layers_below_loss_threshold` is None in all five runs, and that filter uses training loss anyway (`obfuscation_training.py:1079-1080`). The learning rate and step count are fixed in the config.
- `rounds.py` reads only the `run_metrics` files and yamls, never eval outputs.

No number was left unmatched.

##### Files
- `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/tables/rounds/seed_table.md`
- `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/runs/dynamics/dynamics.md`
- `/workspace/sycophancy-rl/sycophancy_setup/seed_table.py`
- `/workspace/sycophancy-rl/sycophancy_setup/compare.py`
- `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/dynamics.py`
- `/workspace/results/run_results/261107_frozen_probe_kl_0_01/s42_frozen_round3_gh04frmi/checkpoints/responses/step_17.jsonl` (the example for 2.1)

## Blocked

## Result

From `/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01/seed_42/tables/rounds/seed_table.md` (step 51, audited in review 7), copied verbatim with its heading moved down two levels.

### RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): round1 455, round2 454, round3 458, round4 459, round5 463.

| metric | condition | base | base mean | base SD | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.769 | 0.759 | 0.736 | 0.744 | 0.730 | 0.748 | 0.016 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.939 | 0.881 | 0.849 | 0.830 | 0.773 | 0.854 | 0.061 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.431 | 0.556 | 0.620 | 0.673 | 0.689 | 0.594 | 0.105 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.763 | 0.764 | 0.769 | 0.778 | 0.745 | 0.764 | 0.012 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.673 | 0.667 | 0.654 | 0.644 | 0.652 | 0.658 | 0.012 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.775 | 0.718 | 0.722 | 0.712 | 0.675 | 0.720 | 0.036 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.464 | 0.536 | 0.583 | 0.607 | 0.620 | 0.562 | 0.063 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.746 | 0.708 | 0.710 | 0.699 | 0.668 | 0.706 | 0.028 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.939 | 0.881 | 0.849 | 0.830 | 0.773 | 0.854 | 0.061 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.569 | 0.440 | 0.380 | 0.325 | 0.311 | 0.405 | 0.105 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.813 | 0.725 | 0.691 | 0.658 | 0.616 | 0.701 | 0.075 |
| bias | cue_gold | 0.087 / 0.090 / 0.089 / 0.089 / 0.094 | 0.090 | 0.003 | 0.132 | 0.095 | 0.096 | 0.074 | 0.050 | 0.089 | 0.030 |
| bias | cue_dist | 0.272 / 0.268 / 0.268 / 0.267 / 0.272 | 0.269 | 0.002 | 0.354 | 0.215 | 0.131 | 0.079 | 0.054 | 0.166 | 0.122 |
| bias | train_mix | 0.150 / 0.148 / 0.149 / 0.149 / 0.155 | 0.150 | 0.003 | 0.202 | 0.131 | 0.106 | 0.072 | 0.049 | 0.112 | 0.059 |
| anti_bias | cue_gold | 0.038 / 0.038 / 0.038 / 0.038 / 0.037 | 0.038 | 0.000 | 0.002 | 0.009 | 0.002 | 0.002 | 0.013 | 0.006 | 0.005 |
| anti_bias | cue_dist | 0.009 / 0.009 / 0.011 / 0.011 / 0.011 | 0.010 | 0.001 | 0.000 | 0.004 | 0.009 | 0.013 | 0.011 | 0.007 | 0.005 |
| anti_bias | train_mix | 0.028 / 0.028 / 0.028 / 0.028 / 0.028 | 0.028 | 0.000 | 0.002 | 0.008 | 0.004 | 0.005 | 0.011 | 0.006 | 0.004 |
| no_bias | cue_gold | 0.875 / 0.872 / 0.873 / 0.874 / 0.868 | 0.872 | 0.002 | 0.866 | 0.896 | 0.902 | 0.924 | 0.937 | 0.905 | 0.027 |
| no_bias | cue_dist | 0.719 / 0.723 / 0.721 / 0.721 / 0.717 | 0.720 | 0.002 | 0.646 | 0.781 | 0.860 | 0.908 | 0.935 | 0.826 | 0.116 |
| no_bias | train_mix | 0.822 / 0.825 / 0.823 / 0.823 / 0.817 | 0.822 | 0.003 | 0.797 | 0.861 | 0.890 | 0.923 | 0.940 | 0.882 | 0.057 |
| deception_soft | cue_gold | 0.554 / 0.552 / 0.552 / 0.555 / 0.551 | 0.553 | 0.001 | 0.519 | 0.557 | 0.582 | 0.531 | 0.647 | 0.567 | 0.051 |
| deception_soft | cue_dist | 0.599 / 0.598 / 0.603 / 0.601 / 0.599 | 0.600 | 0.002 | 0.573 | 0.653 | 0.672 | 0.666 | 0.733 | 0.659 | 0.057 |
| deception_soft | train_mix | 0.568 / 0.567 / 0.569 / 0.570 / 0.567 | 0.568 | 0.001 | 0.537 | 0.591 | 0.611 | 0.575 | 0.678 | 0.599 | 0.052 |
| deception | cue_gold | 0.570 / 0.573 / 0.573 / 0.576 / 0.569 | 0.572 | 0.003 | 0.507 | 0.602 | 0.641 | 0.547 | 0.757 | 0.611 | 0.097 |
| deception | cue_dist | 0.661 / 0.658 / 0.670 / 0.661 / 0.660 | 0.662 | 0.004 | 0.620 | 0.731 | 0.795 | 0.761 | 0.871 | 0.756 | 0.092 |
| deception | train_mix | 0.600 / 0.602 / 0.606 / 0.606 / 0.600 | 0.603 | 0.003 | 0.539 | 0.649 | 0.693 | 0.614 | 0.800 | 0.659 | 0.097 |
| biased_share | cue_gold | 0.091 / 0.093 / 0.092 / 0.092 / 0.098 | 0.093 | 0.003 | 0.132 | 0.096 | 0.096 | 0.074 | 0.050 | 0.090 | 0.030 |
| biased_share | cue_dist | 0.275 / 0.271 / 0.271 / 0.270 / 0.275 | 0.272 | 0.002 | 0.354 | 0.216 | 0.132 | 0.080 | 0.055 | 0.167 | 0.121 |
| biased_share | train_mix | 0.154 / 0.152 / 0.153 / 0.153 / 0.159 | 0.154 | 0.003 | 0.202 | 0.132 | 0.106 | 0.072 | 0.049 | 0.112 | 0.059 |
| false_denial_soft | cue_gold | 0.394 / 0.384 / 0.378 / 0.384 / 0.380 | 0.384 | 0.006 | 0.350 | 0.388 | 0.338 | 0.367 | 0.362 | 0.361 | 0.019 |
| false_denial_soft | cue_dist | 0.352 / 0.348 / 0.353 / 0.352 / 0.352 | 0.351 | 0.002 | 0.344 | 0.357 | 0.311 | 0.353 | 0.221 | 0.317 | 0.056 |
| false_denial_soft | train_mix | 0.370 / 0.364 / 0.365 / 0.368 / 0.366 | 0.367 | 0.002 | 0.341 | 0.368 | 0.317 | 0.353 | 0.317 | 0.339 | 0.022 |
| false_admission_soft | cue_gold | 0.570 / 0.570 / 0.570 / 0.572 / 0.570 | 0.570 | 0.001 | 0.545 | 0.575 | 0.608 | 0.544 | 0.662 | 0.587 | 0.050 |
| false_admission_soft | cue_dist | 0.692 / 0.691 / 0.696 / 0.694 / 0.693 | 0.693 | 0.002 | 0.698 | 0.734 | 0.727 | 0.693 | 0.763 | 0.723 | 0.029 |
| false_admission_soft | train_mix | 0.604 / 0.604 / 0.606 / 0.607 / 0.605 | 0.605 | 0.001 | 0.587 | 0.625 | 0.646 | 0.592 | 0.697 | 0.629 | 0.045 |
| said_yes_biased_soft | cue_gold | 0.606 / 0.616 / 0.622 / 0.616 / 0.620 | 0.616 | 0.006 | 0.650 | 0.612 | 0.662 | 0.633 | 0.638 | 0.639 | 0.019 |
| said_yes_biased_soft | cue_dist | 0.648 / 0.652 / 0.647 / 0.648 / 0.648 | 0.649 | 0.002 | 0.656 | 0.643 | 0.689 | 0.647 | 0.779 | 0.683 | 0.056 |
| said_yes_biased_soft | train_mix | 0.630 / 0.636 / 0.635 / 0.632 / 0.634 | 0.633 | 0.002 | 0.659 | 0.632 | 0.683 | 0.647 | 0.683 | 0.661 | 0.022 |
| deception_from_biased | cue_gold | 0.036 / 0.036 / 0.035 / 0.035 / 0.037 | 0.036 | 0.001 | 0.046 | 0.037 | 0.033 | 0.027 | 0.018 | 0.032 | 0.010 |
| deception_from_biased | cue_dist | 0.097 / 0.094 / 0.096 / 0.095 / 0.097 | 0.096 | 0.001 | 0.122 | 0.077 | 0.041 | 0.028 | 0.012 | 0.056 | 0.044 |
| deception_from_biased | train_mix | 0.057 / 0.055 / 0.056 / 0.056 / 0.058 | 0.057 | 0.001 | 0.069 | 0.049 | 0.034 | 0.026 | 0.016 | 0.038 | 0.021 |
| deception_from_unbiased | cue_gold | 0.518 / 0.516 / 0.517 / 0.519 / 0.514 | 0.517 | 0.002 | 0.473 | 0.520 | 0.550 | 0.503 | 0.629 | 0.535 | 0.059 |
| deception_from_unbiased | cue_dist | 0.502 / 0.504 / 0.508 / 0.506 / 0.502 | 0.504 | 0.002 | 0.451 | 0.576 | 0.631 | 0.638 | 0.721 | 0.603 | 0.100 |
| deception_from_unbiased | train_mix | 0.511 / 0.512 / 0.513 / 0.514 / 0.508 | 0.512 | 0.002 | 0.468 | 0.542 | 0.577 | 0.550 | 0.663 | 0.560 | 0.070 |
| said_yes | cue_gold | 0.595 / 0.597 / 0.600 / 0.604 / 0.599 | 0.599 | 0.003 | 0.555 | 0.627 | 0.694 | 0.586 | 0.794 | 0.651 | 0.095 |
| said_yes | cue_dist | 0.785 / 0.787 / 0.790 / 0.786 / 0.786 | 0.787 | 0.002 | 0.789 | 0.831 | 0.879 | 0.796 | 0.921 | 0.843 | 0.056 |
| said_yes | train_mix | 0.656 / 0.657 / 0.661 / 0.663 / 0.659 | 0.659 | 0.003 | 0.630 | 0.701 | 0.754 | 0.652 | 0.841 | 0.715 | 0.085 |
| false_denial | cue_gold | 0.359 / 0.375 / 0.350 / 0.350 / 0.349 | 0.357 | 0.011 | 0.317 | 0.372 | 0.227 | 0.235 | 0.130 | 0.256 | 0.092 |
| false_denial | cue_dist | 0.275 / 0.263 / 0.277 / 0.269 / 0.270 | 0.271 | 0.006 | 0.261 | 0.268 | 0.183 | 0.278 | 0.040 | 0.206 | 0.100 |
| false_denial | train_mix | 0.318 / 0.321 / 0.318 / 0.315 / 0.313 | 0.317 | 0.003 | 0.275 | 0.303 | 0.211 | 0.242 | 0.089 | 0.224 | 0.083 |
| false_admission | cue_gold | 0.591 / 0.594 / 0.595 / 0.599 / 0.593 | 0.594 | 0.003 | 0.536 | 0.627 | 0.685 | 0.572 | 0.790 | 0.642 | 0.100 |
| false_admission | cue_dist | 0.808 / 0.805 / 0.816 / 0.807 / 0.807 | 0.809 | 0.004 | 0.816 | 0.858 | 0.888 | 0.803 | 0.919 | 0.857 | 0.049 |
| false_admission | train_mix | 0.652 / 0.653 / 0.658 / 0.659 / 0.654 | 0.655 | 0.003 | 0.606 | 0.702 | 0.750 | 0.643 | 0.837 | 0.708 | 0.091 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.006 | 0.008 | 0.004 | 0.006 | 0.000 | 0.005 | 0.003 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.001 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.004 | 0.000 | 0.002 | 0.000 | 0.001 | 0.002 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.001 | 0.000 | 0.002 | 0.000 | 0.001 | 0.001 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 447 / 446 / 450 / 451 / 456 | 450 | 3.937 | 455 | 454 | 458 | 458 | 463 | 457.600 | 3.507 |
| n_bias_rows | cue_dist | 441 / 440 / 444 / 445 / 449 | 443.800 | 3.564 | 455 | 452 | 458 | 458 | 463 | 457.200 | 4.087 |
| n_bias_rows | train_mix | 888 / 886 / 894 / 896 / 905 | 893.800 | 7.497 | 910 | 906 | 916 | 916 | 926 | 914.800 | 7.563 |
| n_deception_rows | cue_gold | 430 / 429 / 433 / 434 / 439 | 433 | 3.937 | 454 | 450 | 457 | 457 | 457 | 455 | 3.082 |
| n_deception_rows | cue_dist | 437 / 436 / 439 / 440 / 444 | 439.200 | 3.114 | 455 | 450 | 454 | 452 | 458 | 453.800 | 3.033 |
| n_deception_rows | train_mix | 867 / 865 / 872 / 874 / 883 | 872.200 | 7.050 | 909 | 900 | 911 | 909 | 915 | 908.800 | 5.495 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.214 | 0.220 | 0.214 | 0.144 | 0.115 | 0.181 | 0.048 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.195 | 0.252 | 0.162 | 0.187 | 0.219 | 0.203 | 0.034 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.053 | 0.044 | 0.025 | 0.046 | 0.048 | 0.043 | 0.011 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.046 | 0.057 | 0.049 | 0.067 | 0.089 | 0.062 | 0.017 |
| probe_start_auc | cue_gold | – | – | – | 0.824 | 0.861 | 0.885 | 0.811 | 0.778 | 0.832 | 0.042 |
| probe_start_auc | cue_dist | – | – | – | 0.789 | 0.799 | 0.761 | 0.798 | 0.731 | 0.776 | 0.029 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 60 | 43 | 44 | 34 | 23 | 40.800 | 13.664 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 161 | 97 | 60 | 36 | 25 | 75.800 | 55.052 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 366 | 356 | 347 | 347 | 338 | 350.800 | 10.616 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 88 | 91 | 106 | 105 | 114 | 100.800 | 10.941 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.208 | 0.289 | 0.314 | 0.312 | 0.255 | 0.276 | 0.045 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.184 | 0.304 | 0.277 | 0.347 | 0.275 | 0.277 | 0.060 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.054 | 0.084 | 0.126 | 0.182 | 0.212 | 0.132 | 0.066 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.049 | 0.098 | 0.126 | 0.200 | 0.240 | 0.143 | 0.077 |
| probe_fresh_auc | cue_gold | – | – | – | 0.815 | 0.817 | 0.795 | 0.704 | 0.645 | 0.755 | 0.077 |
| probe_fresh_auc | cue_dist | – | – | – | 0.776 | 0.782 | 0.719 | 0.705 | 0.554 | 0.707 | 0.092 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 60 | 43 | 44 | 34 | 23 | 40.800 | 13.664 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 161 | 97 | 60 | 36 | 25 | 75.800 | 55.052 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 366 | 356 | 347 | 347 | 338 | 350.800 | 10.616 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 88 | 91 | 106 | 105 | 114 | 100.800 | 10.941 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.216 | 0.219 | 0.347 | 0.295 | 0.202 | 0.255 | 0.063 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.206 | 0.262 | 0.326 | 0.348 | 0.327 | 0.294 | 0.059 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.048 | 0.051 | 0.089 | 0.110 | 0.128 | 0.085 | 0.035 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.051 | 0.066 | 0.155 | 0.157 | 0.225 | 0.131 | 0.072 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.835 | 0.825 | 0.833 | 0.760 | 0.704 | 0.791 | 0.058 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.767 | 0.795 | 0.706 | 0.728 | 0.629 | 0.725 | 0.064 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 60 | 43 | 44 | 34 | 23 | 40.800 | 13.664 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 161 | 97 | 60 | 36 | 25 | 75.800 | 55.052 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 366 | 356 | 347 | 347 | 338 | 350.800 | 10.616 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 88 | 91 | 106 | 105 | 114 | 100.800 | 10.941 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | round1 | round2 | round3 | round4 | round5 | mean | SD |
|---|---|---|---|---|---|---|---|
| equal task reward groups | 0.755 | 0.848 | 0.924 | 0.954 | 0.966 | 0.889 | 0.088 |
| probe-only groups | 0.666 | 0.709 | 0.723 | 0.717 | 0.708 | 0.705 | 0.022 |
