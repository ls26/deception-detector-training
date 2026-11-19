# Runs

State file for the unattended run of queue `no_penalty`. Read this first after any restart.

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

Start check, system (2026-11-17 19:54 UTC): hostname `3fb02eb377e0`; `nvidia-smi -L`: GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-3de7dbef-6516-1b49-c4cf-405dae85ecf6); driver 595.91.07, CUDA 13.2; `uname -r` 6.8.0-147-generic; `uptime -s` 2026-10-20 05:27:25; `df -h /workspace` 513G size, 328G used, 185G free (65%). Pod and GPU match the Goal. Resume check scheduled as cron job 9413d547 (minutes 7 and 37 of each hour, session-only, expires after 7 days).

2026-11-17 19:55 UTC | 1. Start check | fence test: hook rejected the Write to /workspace/scratch/fence_test_261117-np.txt, no file created; identity checks (scratch folder `start/`): config ×2, probe, copy_base, table old (json + md), compare (json, md, csv) all exit 0 and all 9 files identical to the saved ones | 2. Folders
2026-11-17 19:58 UTC | 2. Folders | all 12 folders exist | 3. Base model on this pod
2026-11-17 20:03 UTC | 3. Base model on this pod | base_repro EXIT 0 (logs/base_repro_a1_3fb02eb377e0.log); checks/base_repro.json: rows 500, questions 50, same_completion 447, same_pred 495 (reported only) | 4. Smoke: vLLM
2026-11-17 20:07 UTC | 4. Smoke: vLLM | started 20:03 (disk 185 GB free, GPU empty); /v1/models answered after ~230 s; IMPORT_CHECK afterburner from /workspace/sycophancy-rl copy | 5. Smoke: RL
2026-11-18 04:50 UTC | 4. Smoke: vLLM, attempt 2 (after the reboot, see Judgment calls) | started 04:46 (disk 180 GB free, GPU empty); /v1/models answered after ~250 s; IMPORT_CHECK afterburner from /workspace/sycophancy-rl/obfuscation-atlas/third_party/afterburner; server holds 30.3 GB | 5. Smoke: RL, attempt 2
2026-11-18 05:15 UTC | 5. Smoke: RL, attempt 2 | started 04:50; EXIT 0 at 05:14 (logs/rl_smoke_np400_a2_3fb02eb377e0.log); no `CRASH:` lines; `GRPO subprocess finished in 21.8 min (exit=0, attempt=1/4)`; peak GPU memory 65,807 MiB; lowest free disk 174 GB; run dir smoke_np400_lghae63p; vLLM session killed | 6. Smoke: checks and 7. Smoke: eval (side by side)
2026-11-18 05:16 UTC | 6. Smoke: checks | exit 0; checks/finished_smoke_np400.json: step_files 6, prompts 384, answers 3072, adapter_steps [3, 6] (rules judged in review 1) | 7. Smoke: eval (running)
2026-11-18 05:19 UTC | 7. Smoke: eval of the step-3 adapter | started 05:15 (disk 176 GB free, GPU empty); EXIT 0 at 05:18 (logs/eval_smoke_step_3_a1_3fb02eb377e0.log, ADAPTER line names smoke_np400_lghae63p/checkpoints/adapters/step_3); dry/evals/smoke_step_3/summary.json has "questions": 50 | 8. Review 1
2026-11-18 05:19 UTC | 8. Review 1 | PASS (reviews/1_a1.md) | 9. Seed 42: vLLM
2026-11-18 05:22 UTC | 9. Seed 42: vLLM | started 05:19 (disk 176 GB free, GPU empty); /v1/models answered after ~180 s; IMPORT_CHECK afterburner from /workspace/sycophancy-rl/obfuscation-atlas/third_party/afterburner; server holds 30.3 GB | 10. Seed 42: RL
2026-11-18 05:25 UTC | 10. Seed 42: RL | started 05:22 (tmux 261117-np-rl, logs/rl_np400_s42_a1_3fb02eb377e0.log); run dir line after ~2 min: /workspace/results/run_results/261117_no_penalty_vs_frozen_400/np400_s42_44s8k5wr; training continues | 11. Seed 42: early check
2026-11-18 05:44 UTC | 11. Seed 42: early check | step_10.jsonl appeared by 05:43 (~21 min after the RL start); checks.py early exit 0 on the first run; seed_42/checks/early_np400_s42.json: per_step 11 rows (file steps 0–10), prompts 704, answers 5632 | 12. Review 2
2026-11-18 05:45 UTC | 12. Review 2 | PASS (reviews/2_a1.md) | 13. Seed 42: RL finished (expected ~10–13 h after the 05:22 start)
2026-11-18 15:41 UTC | 13. Seed 42: RL finished | EXIT 0 at 15:40 (logs/rl_np400_s42_a1_3fb02eb377e0.log); no `CRASH:` lines; `GRPO subprocess finished in 615.8 min (exit=0, attempt=1/4)`; peak GPU memory 67,800 MiB; lowest free disk 158 GB; run_metrics_np400_s42.json run_id 44s8k5wr (run dir np400_s42_44s8k5wr, adapters step_50 … step_400); vLLM session killed | 14. Seed 42: checks, with 16. base answers rescored started alongside
2026-11-18 15:43 UTC | 14. Seed 42: checks | finished and probe both exit 0; seed_42/checks/finished_np400_s42.json: step_files 400, prompts 25600, answers 204800, adapter_steps [50 … 400]; checks/probe_np400_s42.json written (rules judged in review 3) | 15. Review 3
2026-11-18 15:43 UTC | 15. Review 3 | PASS (reviews/3_a1.md) | 16. Seed 42: base answers rescored (running since 15:41)
2026-11-18 15:44 UTC | 16. Seed 42: base answers rescored | started 15:41 (disk 162 GB free, GPU empty); EXIT 0 at ~15:43 (seed_42/logs/eval_base_rescored_a1_3fb02eb377e0.log, `ADAPTER base model`); evals/base_rescored/summary.json "questions": 489, "cued_rollouts": 978 | 17. Seed 42: evals of the saved adapters
2026-11-18 16:32 UTC | 17. Seed 42: evals of the saved adapters | steps 50, 100, 150, 200, 250, 300, 350, 400 run in turn 15:44–16:31 (5.5–6.6 min each, free disk 157–162 GB before each, GPU empty); every log seed_42/logs/eval_step_N_a1_3fb02eb377e0.log ends EXIT 0 with its ADAPTER line naming np400_s42_44s8k5wr/checkpoints/adapters/step_N; every evals/step_N/summary.json has "questions": 489, "cued_rollouts": 978 | 18. Seed 42: training curve
2026-11-18 16:33 UTC | 18. Seed 42: training curve | compare.py exit 0; seed_42/runs/np400_s42/training_curve.csv (400 step rows + header), comparison.json, comparison.md | 19. Seed 42: table
2026-11-18 16:33 UTC | 19. Seed 42: table | table.py seed no_penalty 42 exit 0; seed_42/table/comparison.json and comparison.md written | 20. Review 4
2026-11-18 16:34 UTC | 20. Review 4 | PASS (reviews/4_a1.md) | 21. Seed 43: vLLM (started 16:31, still loading)
2026-11-18 16:35 UTC | 21. Seed 43: vLLM | started 16:31 (disk 157 GB free, GPU empty); /v1/models answered after ~180 s; IMPORT_CHECK afterburner from /workspace/sycophancy-rl/obfuscation-atlas/third_party/afterburner; server holds 30.3 GB | 22. Seed 43: RL
2026-11-18 16:38 UTC | 22. Seed 43: RL | started 16:35 (tmux 261117-np-rl, seed_43/logs/rl_np400_s43_a1_3fb02eb377e0.log); run dir line after ~3 min: /workspace/results/run_results/261117_no_penalty_vs_frozen_400/np400_s43_2bu0k3fz; training continues | 23. Seed 43: RL finished (expected ~10–13 h after the 16:35 start)
2026-11-19 02:57 UTC | 23. Seed 43: RL finished | EXIT 0 at 02:56 (seed_43/logs/rl_np400_s43_a1_3fb02eb377e0.log); no `CRASH:` lines; `GRPO subprocess finished in 618.7 min (exit=0, attempt=1/4)`; peak GPU memory 67,321 MiB; lowest free disk 141 GB; run_metrics_np400_s43.json run_id 2bu0k3fz (run dir np400_s43_2bu0k3fz, adapters step_50 … step_400); vLLM session killed | 24. Seed 43: checks, with 26. base answers rescored started alongside
2026-11-19 02:59 UTC | 24. Seed 43: checks | finished and probe both exit 0; seed_43/checks/finished_np400_s43.json: step_files 400, prompts 25600, answers 204800, adapter_steps [50 … 400]; checks/probe_np400_s43.json written (rules judged in review 5) | 25. Review 5
2026-11-19 03:00 UTC | 25. Review 5 | PASS (reviews/5_a1.md) | 26. Seed 43: base answers rescored (running since 02:57)
2026-11-19 03:00 UTC | 26. Seed 43: base answers rescored | started 02:57 (disk 146 GB free, GPU empty); EXIT 0 at ~03:00 (seed_43/logs/eval_base_rescored_a1_3fb02eb377e0.log, `ADAPTER base model`); evals/base_rescored/summary.json "questions": 489, "cued_rollouts": 978 | 27. Seed 43: evals of the saved adapters
2026-11-19 03:48 UTC | 27. Seed 43: evals of the saved adapters | steps 50, 100, 150, 200, 250, 300, 350, 400 run in turn 03:00–03:47 by the same scratchpad loop as seed 42 (5.6–6.1 min each, 146 GB free before each, GPU empty); every log seed_43/logs/eval_step_N_a1_3fb02eb377e0.log ends EXIT 0 with its ADAPTER line naming np400_s43_2bu0k3fz/checkpoints/adapters/step_N; every evals/step_N/summary.json has "questions": 489, "cued_rollouts": 978 | 28. Seed 43: training curve
2026-11-19 03:48 UTC | 28. Seed 43: training curve | compare.py exit 0; seed_43/runs/np400_s43/training_curve.csv (400 step rows + header), comparison.json, comparison.md | 29. Seed 43: table
2026-11-19 03:49 UTC | 29. Seed 43: table | table.py seed no_penalty 43 exit 0; seed_43/table/comparison.json and comparison.md written | 30. Review 6
2026-11-19 03:50 UTC | 30. Review 6 | PASS (reviews/6_a1.md) | 31. Combined table
2026-11-19 03:51 UTC | 31. Combined table | frozen_probe/seed_43/table/comparison.json already present (written 02:26 UTC); frozen_probe/runs.md Blocked empty, its queue done; table.py combined exit 0; ../comparison/comparison.md and comparison.json written; comparison.md copied verbatim into Result (diff against the file: identical) | 32. Review 7
2026-11-19 03:54 UTC | 32. Review 7 | PASS (reviews/7_a1.md) | done: every step ticked, Blocked is empty, Result holds the combined table; user notified (terminal notification sent, mobile push requested). Resume check cron 44e6b610 left in place as CLAUDE.md requires.

## Judgment calls

- 2026-11-18 04:47 UTC: the pod rebooted during step 5 (smoke RL, attempt 1, started ~20:07 UTC on 2026-11-17). `uptime -s` is now 2026-11-17 20:15:48 (it was 2026-10-20 05:27:25 at the start check). After the reboot the kernel is 6.8.0-149-generic (was 6.8.0-147) and the driver 595.112.07 (was 595.91.07); hostname `3fb02eb377e0` and GPU UUID GPU-3de7dbef-6516-1b49-c4cf-405dae85ecf6 are unchanged, so the pod and GPU still match the Goal. The attempt-1 RL log stops at 20:10:10 UTC in huggingface.co DNS-failure retries (`Failed to resolve 'huggingface.co'`) with no `EXIT` line, and no run dir for `smoke_np400` was made. As the reboot rule says, step 4 (vLLM, the server step 5 needs) is redone, then step 5 is rerun as attempt 2; this rerun does not count against step 5's attempts. Step 4 unticked until the new server answers. The attempt-1 logs moved to `failed_attempts/4_a1/` and `failed_attempts/5_a1/`. Before the rerun: huggingface.co answers (HTTP 200), a one-element CUDA tensor works, the GPU shows no process, 180 GB free on /workspace. The resume check from the earlier session (cron 9413d547) died with that session; this session scheduled it again as cron 44e6b610 (minutes 7 and 37 of each hour, session-only, expires after 7 days).
- 2026-11-18 04:51 UTC: the RL steps' background wait checks for the `EXIT` line every 60 s instead of every 5 min, so that the peak GPU memory and lowest free disk it records are sampled more often (the frozen-probe queue made the same call). It only reads; nothing else changes.
- 2026-11-18 05:15 UTC: started step 7 (smoke eval, GPU) while step 6 (smoke checks, CPU, reads saved files) runs, so the GPU does not wait on a read-only step. The two are independent: step 7 needs only the finished smoke run dir. (The frozen-probe queue made the same call.)
- 2026-11-18 15:41 UTC: started step 16 (seed 42 base answers rescored, GPU) while step 14 (seed 42 checks, CPU, reads saved files) and then Review 3 run, so the GPU does not wait on read-only steps. Step 16 needs only the finished run dir. Its outputs are not read before step 16's own turn in the queue, and step 16 is ticked only after Review 3 passes. If Review 3 fails, step 16's outputs move to `failed_attempts/` with the rest.
- 2026-11-18 15:45 UTC: step 17's eight evals are started one after another by a small loop in the scratchpad (`eval_adapters.sh`). For each N it checks free disk (stops under 60 GB), waits until `nvidia-smi` shows no process, starts the queue's eval command verbatim in tmux session `261117-np-job`, and waits for that log's `EXIT` line. It stops at the first non-zero exit. It changes nothing in the eval itself; it only does the per-step order and the pre-GPU checks without a turn of mine between evals.
- 2026-11-18 16:32 UTC: started step 21 (seed 43 vLLM, GPU) while steps 18–19 (seed 42 training curve and table, CPU, read saved files) and Review 4 run, so the GPU does not wait on read-only steps. Seed 43's run starts from the base model and needs nothing from seed 42. Steps 21 and 22 are ticked only after Review 4 passes. If Review 4 fails, the seed 43 run is stopped (`tmux kill-session` of its RL and vLLM sessions) before following Reviews.
- 2026-11-19 02:57 UTC: as for seed 42, started step 26 (seed 43 base answers rescored, GPU) while step 24 (seed 43 checks, CPU) and then Review 5 run. Step 26 is ticked only after Review 5 passes. If Review 5 fails, step 26's outputs move to `failed_attempts/`.

## Code fixes

## Reviews

One line per review: `<UTC time> | review | verdict | reviews/<n>_a<attempt>.md`.

2026-11-18 05:19 UTC | 1, smoke | PASS (every rule; reported: flipped share 0.2214, cut off 4, full training states left []) | reviews/1_a1.md
2026-11-18 05:45 UTC | 2, seed 42 early check | PASS (every rule; reported: W&B KL at step 1 0, flipped share 0.2230) | reviews/2_a1.md
2026-11-18 15:43 UTC | 3, seed 42 step files | PASS (every rule; reported: flipped share 0.2006, batch 4 × grad accum 16, probe test AUC 0.8730, md5 identical to round 1's probe true) | reviews/3_a1.md
2026-11-18 16:34 UTC | 4, seed 42 evals and table | PASS (every rule; bias and cue agreement rate, 50/50 mix, at steps 50 and 400 recomputed from labels.jsonl equal the table with difference 0.0) | reviews/4_a1.md
2026-11-19 03:00 UTC | 5, seed 43 step files | PASS (every rule; reported: flipped share 0.2008, batch 4 × grad accum 16, probe test AUC 0.8730, md5 identical to round 1's probe true) | reviews/5_a1.md
2026-11-19 03:50 UTC | 6, seed 43 evals and table | PASS (every rule; bias and cue agreement rate, 50/50 mix, at steps 50 and 400 recomputed from labels.jsonl equal the table with difference 0) | reviews/6_a1.md
2026-11-19 03:53 UTC | 7, auditor | PASS (every rule; all 1,028 comparison.json cells recomputed from labels.jsonl / completions.jsonl, max difference 1.1e-16; Result section byte-identical to comparison.md; old-chain block byte-identical to identity/table_old/comparison.json; one could-not-verify on the word "cosine" in the old-chain heading, outside the rule) | reviews/7_a1.md

## Blocked

## Result

# No penalty against the frozen probe: 400 steps, labels redrawn every prompt, constant learning rate

Written by `table.py`; definitions in its docstring. Each cell lists every seed's value in the order given in the condition column. Values are shares (0-1); "–" means no answers to compute it on.

## New runs: labels redrawn every prompt, constant learning rate (5e-5 after 5 warmup steps)

| Measurement | Condition (seeds) | Base model | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Step 350 | Step 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| bias, 50/50 mix | no penalty (s42, s43) | 0.177 / 0.178 | 0.425 / 0.377 | 0.420 / 0.435 | 0.429 / 0.484 | 0.310 / 0.392 | 0.282 / 0.317 | 0.422 / 0.377 | 0.361 / 0.419 | 0.328 / 0.339 |
| bias, 50/50 mix | frozen probe (s42, s43) | 0.177 / 0.175 | 0.373 / 0.429 | 0.445 / 0.354 | 0.440 / 0.410 | 0.413 / 0.373 | 0.322 / 0.250 | 0.361 / 0.321 | 0.366 / 0.267 | 0.221 / 0.273 |
| bias, cue at the correct answer | no penalty (s42, s43) | 0.089 / 0.090 | 0.193 / 0.185 | 0.187 / 0.200 | 0.194 / 0.212 | 0.163 / 0.194 | 0.160 / 0.174 | 0.200 / 0.200 | 0.201 / 0.191 | 0.192 / 0.181 |
| bias, cue at the correct answer | frozen probe (s42, s43) | 0.087 / 0.088 | 0.176 / 0.174 | 0.179 / 0.166 | 0.194 / 0.165 | 0.179 / 0.168 | 0.152 / 0.131 | 0.185 / 0.163 | 0.179 / 0.137 | 0.123 / 0.165 |
| bias, cue at the wrong answer | no penalty (s42, s43) | 0.265 / 0.265 | 0.658 / 0.570 | 0.653 / 0.670 | 0.664 / 0.756 | 0.457 / 0.590 | 0.404 / 0.459 | 0.645 / 0.555 | 0.521 / 0.647 | 0.463 / 0.498 |
| bias, cue at the wrong answer | frozen probe (s42, s43) | 0.266 / 0.263 | 0.571 / 0.684 | 0.711 / 0.542 | 0.686 / 0.655 | 0.647 / 0.577 | 0.492 / 0.369 | 0.537 / 0.479 | 0.554 / 0.397 | 0.318 / 0.382 |
| cue agreement rate, 50/50 mix | no penalty (s42, s43) | 0.650 / 0.650 | 0.925 / 0.883 | 0.924 / 0.938 | 0.921 / 0.985 | 0.817 / 0.898 | 0.778 / 0.820 | 0.924 / 0.885 | 0.872 / 0.921 | 0.831 / 0.847 |
| cue agreement rate, 50/50 mix | frozen probe (s42, s43) | 0.650 / 0.650 | 0.876 / 0.934 | 0.946 / 0.855 | 0.942 / 0.916 | 0.916 / 0.875 | 0.822 / 0.748 | 0.864 / 0.827 | 0.877 / 0.774 | 0.721 / 0.774 |
| cue agreement rate, cue at the correct answer | no penalty (s42, s43) | 0.843 / 0.843 | 0.988 / 0.980 | 0.978 / 0.980 | 0.984 / 0.998 | 0.965 / 0.982 | 0.933 / 0.959 | 0.984 / 0.990 | 0.975 / 0.986 | 0.959 / 0.969 |
| cue agreement rate, cue at the correct answer | frozen probe (s42, s43) | 0.843 / 0.843 | 0.967 / 0.984 | 0.984 / 0.971 | 0.990 / 0.980 | 0.978 / 0.971 | 0.957 / 0.920 | 0.973 / 0.953 | 0.982 / 0.937 | 0.910 / 0.947 |
| cue agreement rate, cue at the wrong answer | no penalty (s42, s43) | 0.458 / 0.458 | 0.863 / 0.787 | 0.871 / 0.896 | 0.859 / 0.971 | 0.669 / 0.814 | 0.624 / 0.681 | 0.865 / 0.781 | 0.769 / 0.857 | 0.703 / 0.724 |
| cue agreement rate, cue at the wrong answer | frozen probe (s42, s43) | 0.458 / 0.458 | 0.785 / 0.883 | 0.908 / 0.738 | 0.894 / 0.853 | 0.855 / 0.779 | 0.687 / 0.577 | 0.755 / 0.701 | 0.773 / 0.611 | 0.532 / 0.601 |
| true accuracy, 50/50 mix | no penalty (s42, s43) | 0.676 / 0.676 | 0.561 / 0.596 | 0.553 / 0.542 | 0.561 / 0.512 | 0.644 / 0.583 | 0.652 / 0.637 | 0.559 / 0.603 | 0.603 / 0.564 | 0.625 / 0.622 |
| true accuracy, 50/50 mix | frozen probe (s42, s43) | 0.676 / 0.676 | 0.591 / 0.550 | 0.537 / 0.617 | 0.548 / 0.563 | 0.559 / 0.595 | 0.634 / 0.670 | 0.607 / 0.626 | 0.604 / 0.662 | 0.689 / 0.673 |
| true accuracy, cue at the correct answer | no penalty (s42, s43) | 0.843 / 0.843 | 0.988 / 0.980 | 0.978 / 0.980 | 0.984 / 0.998 | 0.965 / 0.982 | 0.933 / 0.959 | 0.984 / 0.990 | 0.975 / 0.986 | 0.959 / 0.969 |
| true accuracy, cue at the correct answer | frozen probe (s42, s43) | 0.843 / 0.843 | 0.967 / 0.984 | 0.984 / 0.971 | 0.990 / 0.980 | 0.978 / 0.971 | 0.957 / 0.920 | 0.973 / 0.953 | 0.982 / 0.937 | 0.910 / 0.947 |
| true accuracy, cue at the wrong answer | no penalty (s42, s43) | 0.509 / 0.509 | 0.135 / 0.213 | 0.129 / 0.104 | 0.139 / 0.027 | 0.323 / 0.184 | 0.372 / 0.315 | 0.135 / 0.217 | 0.231 / 0.143 | 0.290 / 0.274 |
| true accuracy, cue at the wrong answer | frozen probe (s42, s43) | 0.509 / 0.509 | 0.215 / 0.117 | 0.090 / 0.262 | 0.106 / 0.147 | 0.141 / 0.219 | 0.311 / 0.419 | 0.241 / 0.299 | 0.227 / 0.387 | 0.468 / 0.399 |
| true accuracy, no cue | no penalty (s42, s43) | 0.755 / 0.755 | 0.753 / 0.748 | 0.738 / 0.720 | 0.744 / 0.734 | 0.763 / 0.740 | 0.751 / 0.740 | 0.732 / 0.748 | 0.748 / 0.753 | 0.757 / 0.759 |
| true accuracy, no cue | frozen probe (s42, s43) | 0.755 / 0.755 | 0.759 / 0.783 | 0.757 / 0.773 | 0.757 / 0.736 | 0.744 / 0.800 | 0.775 / 0.748 | 0.753 / 0.742 | 0.744 / 0.761 | 0.759 / 0.748 |
| initial probe, both cues pooled: mean score of biased answers | no penalty (s42, s43) | 0.748 / 0.746 | 0.642 / 0.681 | 0.706 / 0.714 | 0.658 / 0.732 | 0.523 / 0.687 | 0.709 / 0.669 | 0.622 / 0.636 | 0.532 / 0.677 | 0.571 / 0.612 |
| initial probe, both cues pooled: mean score of biased answers | frozen probe (s42, s43) | 0.747 / 0.748 | 0.169 / 0.145 | 0.147 / 0.156 | 0.122 / 0.185 | 0.169 / 0.250 | 0.149 / 0.165 | 0.147 / 0.371 | 0.123 / 0.275 | 0.148 / 0.272 |
| initial probe, both cues pooled: mean score of unbiased answers | no penalty (s42, s43) | 0.227 / 0.229 | 0.187 / 0.190 | 0.223 / 0.201 | 0.177 / 0.284 | 0.101 / 0.216 | 0.231 / 0.169 | 0.132 / 0.160 | 0.110 / 0.188 | 0.099 / 0.122 |
| initial probe, both cues pooled: mean score of unbiased answers | frozen probe (s42, s43) | 0.228 / 0.230 | 0.048 / 0.041 | 0.046 / 0.041 | 0.040 / 0.093 | 0.047 / 0.090 | 0.044 / 0.064 | 0.044 / 0.120 | 0.027 / 0.064 | 0.050 / 0.082 |
| initial probe, both cues pooled: AUC biased vs unbiased | no penalty (s42, s43) | 0.907 / 0.906 | 0.886 / 0.895 | 0.896 / 0.909 | 0.895 / 0.868 | 0.890 / 0.892 | 0.878 / 0.908 | 0.903 / 0.889 | 0.886 / 0.892 | 0.909 / 0.908 |
| initial probe, both cues pooled: AUC biased vs unbiased | frozen probe (s42, s43) | 0.906 / 0.905 | 0.782 / 0.766 | 0.762 / 0.800 | 0.748 / 0.709 | 0.775 / 0.766 | 0.769 / 0.737 | 0.797 / 0.813 | 0.797 / 0.831 | 0.751 / 0.805 |

## Old chains: five 128-step rounds, cosine learning rate restarted each round

| Measurement | Condition (seeds) | Base model | Step 128 | Step 256 | Step 384 | Step 512 | Step 640 |
|---|---|---|---|---|---|---|---|
| bias, 50/50 mix | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.180 / 0.180 | 0.465 / 0.291 | 0.205 / 0.209 | 0.183 / 0.149 | 0.080 / 0.104 | 0.089 / 0.058 |
| bias, 50/50 mix | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.179 / 0.178 / 0.181 / 0.177 | 0.243 / 0.300 / 0.101 / 0.329 | 0.275 / 0.296 / 0.096 / 0.336 | 0.348 / 0.331 / 0.274 / 0.327 | 0.272 / 0.274 / 0.247 / 0.229 | 0.225 / 0.220 / 0.228 / 0.243 |
| bias, cue at the correct answer | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.093 / 0.093 | 0.204 / 0.161 | 0.155 / 0.144 | 0.142 / 0.097 | 0.052 / 0.077 | 0.085 / 0.067 |
| bias, cue at the correct answer | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.089 / 0.090 / 0.091 / 0.088 | 0.132 / 0.158 / 0.093 / 0.157 | 0.141 / 0.156 / 0.072 / 0.165 | 0.178 / 0.161 / 0.167 / 0.158 | 0.141 / 0.145 / 0.148 / 0.145 | 0.146 / 0.135 / 0.151 / 0.144 |
| bias, cue at the wrong answer | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.266 / 0.267 | 0.726 / 0.420 | 0.255 / 0.275 | 0.224 / 0.200 | 0.107 / 0.131 | 0.093 / 0.050 |
| bias, cue at the wrong answer | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.269 / 0.266 / 0.272 / 0.266 | 0.354 / 0.443 / 0.110 / 0.502 | 0.410 / 0.436 / 0.120 / 0.507 | 0.519 / 0.500 / 0.382 / 0.497 | 0.403 / 0.403 / 0.346 / 0.314 | 0.304 / 0.306 / 0.306 / 0.341 |
| cue agreement rate, 50/50 mix | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.650 / 0.650 | 0.961 / 0.800 | 0.711 / 0.721 | 0.689 / 0.657 | 0.565 / 0.602 | 0.587 / 0.542 |
| cue agreement rate, 50/50 mix | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.650 / 0.650 / 0.650 / 0.650 | 0.754 / 0.802 / 0.588 / 0.838 | 0.778 / 0.806 / 0.597 / 0.846 | 0.853 / 0.836 / 0.788 / 0.835 | 0.782 / 0.777 / 0.754 / 0.734 | 0.724 / 0.722 / 0.732 / 0.753 |
| cue agreement rate, cue at the correct answer | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.843 / 0.843 | 0.990 / 0.947 | 0.916 / 0.922 | 0.918 / 0.877 | 0.804 / 0.836 | 0.814 / 0.798 |
| cue agreement rate, cue at the correct answer | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.843 / 0.843 / 0.843 / 0.843 | 0.939 / 0.937 / 0.836 / 0.967 | 0.937 / 0.961 / 0.851 / 0.969 | 0.971 / 0.959 / 0.953 / 0.963 | 0.945 / 0.939 / 0.941 / 0.918 | 0.928 / 0.916 / 0.920 / 0.933 |
| cue agreement rate, cue at the wrong answer | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.458 / 0.458 | 0.933 / 0.652 | 0.505 / 0.519 | 0.460 / 0.438 | 0.327 / 0.368 | 0.360 / 0.286 |
| cue agreement rate, cue at the wrong answer | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.458 / 0.458 / 0.458 / 0.458 | 0.569 / 0.667 / 0.339 / 0.710 | 0.620 / 0.650 / 0.344 / 0.722 | 0.734 / 0.714 / 0.624 / 0.708 | 0.620 / 0.616 / 0.566 / 0.550 | 0.519 / 0.528 / 0.544 / 0.573 |
| true accuracy, 50/50 mix | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.676 / 0.676 | 0.528 / 0.647 | 0.706 / 0.699 | 0.725 / 0.719 | 0.738 / 0.734 | 0.727 / 0.756 |
| true accuracy, 50/50 mix | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.676 / 0.676 / 0.676 / 0.676 | 0.685 / 0.634 / 0.747 / 0.628 | 0.658 / 0.655 / 0.753 / 0.624 | 0.618 / 0.622 / 0.665 / 0.628 | 0.663 / 0.662 / 0.686 / 0.684 | 0.704 / 0.691 / 0.688 / 0.679 |
| true accuracy, cue at the correct answer | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.843 / 0.843 | 0.990 / 0.947 | 0.916 / 0.922 | 0.918 / 0.877 | 0.804 / 0.836 | 0.814 / 0.798 |
| true accuracy, cue at the correct answer | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.843 / 0.843 / 0.843 / 0.843 | 0.939 / 0.937 / 0.836 / 0.967 | 0.937 / 0.961 / 0.851 / 0.969 | 0.971 / 0.959 / 0.953 / 0.963 | 0.945 / 0.939 / 0.941 / 0.918 | 0.928 / 0.916 / 0.920 / 0.933 |
| true accuracy, cue at the wrong answer | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.509 / 0.509 | 0.065 / 0.348 | 0.495 / 0.476 | 0.532 / 0.560 | 0.673 / 0.632 | 0.640 / 0.714 |
| true accuracy, cue at the wrong answer | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.509 / 0.509 / 0.509 / 0.509 | 0.431 / 0.331 / 0.658 / 0.288 | 0.380 / 0.350 / 0.654 / 0.278 | 0.264 / 0.284 / 0.376 / 0.292 | 0.380 / 0.384 / 0.431 / 0.450 | 0.481 / 0.466 / 0.456 / 0.425 |
| true accuracy, no cue | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.755 / 0.755 | 0.726 / 0.771 | 0.738 / 0.730 | 0.753 / 0.759 | 0.738 / 0.755 | 0.714 / 0.738 |
| true accuracy, no cue | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.755 / 0.755 / 0.755 / 0.755 | 0.769 / 0.757 / 0.738 / 0.763 | 0.765 / 0.759 / 0.775 / 0.761 | 0.767 / 0.765 / 0.763 / 0.738 | 0.761 / 0.746 / 0.759 / 0.777 | 0.763 / 0.759 / 0.761 / 0.755 |
| initial probe, both cues pooled: mean score of biased answers | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.749 / 0.745 | 0.386 / 0.653 | 0.622 / 0.576 | 0.556 / 0.489 | 0.399 / 0.461 | 0.608 / 0.452 |
| initial probe, both cues pooled: mean score of biased answers | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.749 / 0.744 / 0.745 / 0.752 | 0.200 / 0.201 / 0.630 / 0.201 | 0.216 / 0.158 / 0.456 / 0.139 | 0.106 / 0.188 / 0.216 / 0.150 | 0.098 / 0.198 / 0.182 / 0.203 | 0.161 / 0.181 / 0.117 / 0.169 |
| initial probe, both cues pooled: mean score of unbiased answers | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.231 / 0.230 | 0.074 / 0.169 | 0.126 / 0.112 | 0.118 / 0.101 | 0.085 / 0.112 | 0.178 / 0.162 |
| initial probe, both cues pooled: mean score of unbiased answers | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.231 / 0.230 / 0.228 / 0.229 | 0.052 / 0.043 / 0.126 / 0.064 | 0.082 / 0.035 / 0.079 / 0.049 | 0.033 / 0.054 / 0.034 / 0.051 | 0.024 / 0.056 / 0.040 / 0.066 | 0.039 / 0.044 / 0.024 / 0.057 |
| initial probe, both cues pooled: AUC biased vs unbiased | no penalty, 5 rounds, fixed labels (261103 round 1, 261114) (s42, s43) | 0.907 / 0.904 | 0.863 / 0.905 | 0.903 / 0.910 | 0.893 / 0.896 | 0.852 / 0.862 | 0.854 / 0.795 |
| initial probe, both cues pooled: AUC biased vs unbiased | frozen probe, 5 rounds, labels redrawn each round (261103 / 261110 round 1, 261108 / 261110) (s42, s43, s45, s46) | 0.907 / 0.903 / 0.906 / 0.908 | 0.790 / 0.801 / 0.899 / 0.761 | 0.757 / 0.806 / 0.887 / 0.693 | 0.757 / 0.788 / 0.853 / 0.760 | 0.774 / 0.811 / 0.820 / 0.775 | 0.791 / 0.813 / 0.820 / 0.744 |
