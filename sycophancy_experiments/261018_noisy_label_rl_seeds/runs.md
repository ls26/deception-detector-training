# Runs

State file for the unattended run. Read this first after any restart.

## Queue

- [x] 1. `nocot_s43` start vLLM
- [x] 2. `nocot_s43` RL run (128 steps)
- [x] 3. `nocot_s43` record check
- [x] 4. `nocot_s43` stop vLLM
- [x] 5. `nocot_s43` final-checkpoint eval
- [x] 6. `nocot_s43` comparison
- [x] 7. `cot_s43` start vLLM
- [x] 8. `cot_s43` RL run (128 steps)
- [x] 9. `cot_s43` record check
- [x] 10. `cot_s43` stop vLLM
- [x] 11. `cot_s43` final-checkpoint eval
- [x] 12. `cot_s43` comparison
- [x] 13. `nocot_s44` start vLLM
- [x] 14. `nocot_s44` RL run (128 steps)
- [x] 15. `nocot_s44` record check
- [x] 16. `nocot_s44` stop vLLM
- [x] 17. `nocot_s44` final-checkpoint eval
- [x] 18. `nocot_s44` comparison
- [x] 19. `cot_s44` start vLLM
- [x] 20. `cot_s44` RL run (128 steps)
- [x] 21. `cot_s44` record check
- [x] 22. `cot_s44` stop vLLM
- [x] 23. `cot_s44` final-checkpoint eval
- [x] 24. `cot_s44` comparison
- [x] 25. Seed table

## Log

One line per step: `<UTC time> | step | outcome | next`.

- 07:50 | 1 `nocot_s43` start vLLM, a1 | OK: `/v1/models` answered after 221 s (afterburner imported from the copy) | 2 RL run
- 12:03 | 2 `nocot_s43` RL run, a1 | OK: `EXIT 0`, run id `f3qfvw2q`, run dir `/workspace/results/run_results/261018_noisy_label_rl_seeds/nocot_noisy_label_s43_f3qfvw2q`; `CRASH:` lines: none; `GRPO subprocess finished in 248.9 min (exit=0, attempt=1/4)`; lowest free disk 50 GB (11:58 UTC, probe cache present); 10 `could not remove activation cache` warnings, each followed by `OK: removed` for the same cache | 3 record check
- 12:04 | 3 `nocot_s43` record check, a1 | OK: exit 0, `"PASS": true`, `"n_files": 128`, `"seed": 43`; `unparsed_share` 0.00067, `same_reward_groups_mean` 0.994 (`runs/nocot_s43/check_records.json`, took 1 min) | 4 stop vLLM
- 12:05 | 4 `nocot_s43` stop vLLM, a1 | OK: `261018-vllm` killed at 12:04:40, `nvidia-smi` lists no process 5-10 s later | 5 final-checkpoint eval
- 12:12 | 5 `nocot_s43` final-checkpoint eval, a1 | OK: `EXIT 0` (12:04-12:12), `evals/nocot_s43/summary.json` has `"questions": 547`, `"cued_prompts": 1094` | 6 comparison
- 12:13 | 6 `nocot_s43` comparison, a1 | OK: exit 0, `runs/nocot_s43/training_curve.csv` has 128 data rows; `comparison.md`, `comparison.json` written | 7 `cot_s43` start vLLM
- 12:17 | 7 `cot_s43` start vLLM, a1 | OK: `/v1/models` answered after 247 s (afterburner imported from the copy) | 8 RL run
- 16:49 | 8 `cot_s43` RL run, a1 | OK: `EXIT 0`, run id `nk7mr4qf`, run dir `/workspace/results/run_results/261018_noisy_label_rl_seeds/noisy_label_s43_nk7mr4qf`; `CRASH:` lines: none; `GRPO subprocess finished in 270.3 min (exit=0, attempt=1/4)` (128 steps 3 h 42 min by the progress bar); lowest free disk 49 GB (16:46 UTC, probe cache present); 10 `could not remove activation cache` warnings, each followed by `OK: removed` for the same cache | 9 record check
- 16:51 | 9 `cot_s43` record check, a1 | OK: exit 0, `"PASS": true`, `"n_files": 128`, `"seed": 43` (`runs/cot_s43/check_records.json`, took 1 min) | 10 stop vLLM
- 16:51 | 10 `cot_s43` stop vLLM, a1 | OK: `261018-vllm` killed at 16:51:06, `nvidia-smi` lists no process 5-10 s later | 11 final-checkpoint eval
- 17:00 | 11 `cot_s43` final-checkpoint eval, a1 | OK: `EXIT 0` (16:51-17:00), `evals/cot_s43/summary.json` has `"questions": 489`, `"cued_rollouts": 2934` | 12 comparison
- 17:01 | 12 `cot_s43` comparison, a1 | OK: exit 0, `runs/cot_s43/training_curve.csv` has 128 data rows; `comparison.md`, `comparison.json` written | 13 `nocot_s44` start vLLM
- 17:05 | 13 `nocot_s44` start vLLM, a1 | OK: `/v1/models` answered after 261 s (afterburner imported from the copy) | 14 RL run
- 20:35 | 14 `nocot_s44` RL run, a1 | OK: `EXIT 0`, run id `nc6bwn67`, run dir `/workspace/results/run_results/261018_noisy_label_rl_seeds/nocot_noisy_label_s44_nc6bwn67`; `CRASH:` lines: none; `GRPO subprocess finished in 206.8 min (exit=0, attempt=1/4)` (128 steps 2 h 37 min by the progress bar); lowest free disk 47 GB (20:31 UTC, probe cache present); 10 `could not remove activation cache` warnings, each followed by `OK: removed` for the same cache | 15 record check
- 20:36 | 15 `nocot_s44` record check, a1 | OK: exit 0, `"PASS": true`, `"n_files": 128`, `"seed": 44`; `unparsed_share` 0.00038, `same_reward_groups_mean` 0.979 (`runs/nocot_s44/check_records.json`, took 1 min) | 16 stop vLLM
- 20:36 | 16 `nocot_s44` stop vLLM, a1 | OK: `261018-vllm` killed at 20:36:19, `nvidia-smi` lists no process 5-10 s later | 17 final-checkpoint eval
- 20:42 | 17 `nocot_s44` final-checkpoint eval, a1 | OK: `EXIT 0` (20:36-20:42), `evals/nocot_s44/summary.json` has `"questions": 547`, `"cued_prompts": 1094` | 18 comparison
- 20:42 | 18 `nocot_s44` comparison, a1 | OK: exit 0, `runs/nocot_s44/training_curve.csv` has 128 data rows; `comparison.md`, `comparison.json` written | 19 `cot_s44` start vLLM
- 20:49 | 19 `cot_s44` start vLLM, a1 | OK: `/v1/models` answered after 379 s (afterburner imported from the copy) | 20 RL run
- 23:30 | 20 `cot_s44` RL run, a1 | failed: the log stops at 20:53 UTC while caching the probe's activations, before training, with no Traceback and no `EXIT` line; the pod booted again at 22:34 UTC (line written by the attended session, corrected after the audit; see Judgment calls) | 19 `cot_s44` start vLLM, a2
- 23:36 | 19 `cot_s44` start vLLM, a2 | failed: `EXIT 1` at 23:35, engine core died in `torch.cuda.set_device`: `RuntimeError: Unexpected error from cudaGetDeviceCount() ... Error 802: system not yet initialized`; `nvidia-smi -q` shows Fabric State `In Progress` | wait for the GPU (until 00:35 UTC), then 19 a3
- 00:36 | 19 `cot_s44` start vLLM, a3 not started | GPU still unavailable: Fabric State `In Progress` at every 30 s poll from 23:36 to 00:35 | stopped, waiting for the user (see Blocked)
- 05:01 | 19 `cot_s44` start vLLM | resumed by the user: the GPU works again, the wall-clock cap is removed, Blocked is cleared (line written by the attended session; see Judgment calls) | 19 `cot_s44` start vLLM, a3
- 05:17 | 19 `cot_s44` start vLLM, a3 | OK: `/v1/models` answered after 324 s (05:11:10-05:16:40; afterburner imported from the copy) | 20 RL run, a2
- 11:08 | 20 `cot_s44` RL run, a2 | OK: `EXIT 0` (05:16-11:07), run id `i3wl9evn`, run dir `/workspace/results/run_results/261018_noisy_label_rl_seeds/noisy_label_s44_i3wl9evn`; `CRASH:` lines: none; `GRPO subprocess finished in 346.6 min (exit=0, attempt=1/4)` (128 steps 4 h 44 min by the progress bar); lowest free disk 46 GB (11:02 UTC, probe cache present); 10 `could not remove activation cache` warnings, each followed by `OK: removed` for the same cache (attempt 1's partial probe cache, same prefix, is gone with it) | 21 record check
- 11:10 | 21 `cot_s44` record check, a1 | OK: exit 0, `"PASS": true`, `"n_files": 128`, `"seed": 44` (`runs/cot_s44/check_records.json`, took 1 min) | 22 stop vLLM
- 11:10 | 22 `cot_s44` stop vLLM, a1 | OK: `261018-vllm` killed at 11:10:23, `nvidia-smi` lists no process 5 s later | 23 final-checkpoint eval
- 11:21 | 23 `cot_s44` final-checkpoint eval, a1 | OK: `EXIT 0` (11:10-11:20), `evals/cot_s44/summary.json` has `"questions": 489`, `"cued_rollouts": 2934` | 24 comparison
- 11:21 | 24 `cot_s44` comparison, a1 | OK: exit 0, `runs/cot_s44/training_curve.csv` has 128 data rows; `comparison.md`, `comparison.json` written | 25 seed table
- 11:22 | 25 seed table, a1 | OK: exit 0 (11:21), `seed_table.md` and `seed_table.json` written; `seed_table.md` copied verbatim into Result | done

## Judgment calls

- Kickoff 2026-10-19 07:46 UTC on host `752350acac9b` (one H200, 67 GB free on /workspace); the 24 h cap
  ends 2026-10-20 07:46 UTC. The 15-minute resume check runs as a session-only cron job (minutes 7, 22,
  37, 52). The tmux session `261018-seeds` already on the pod is this Claude session, not a job of the queue.
- Step 2 (`nocot_s43` RL) slowed from about 61 s per step (steps 1-29; seed 42's log: about 68 s over the same
  steps) to about 210 s per step (steps 30-56: median 197 s, range 72-447 s) from step 30 (about 08:40 UTC).
  [Audit: per-step figures corrected from the progress bars; first written as "about 57 s (as in the seed-42
  log) to 150-200 s".] The only GPU processes are this run's vLLM server and trainer. At
  09:08 the trainer was waiting on page I/O (`D` state, `folio_wait_bit_common`) between bursts of GPU work,
  and the host load average was about 1100, so the cause is outside the run (shared network volume or host).
  [Audit: not confirmed. These readings are in no file, and the cause is the session's inference; the files
  show only the slowdown itself.] No action: nothing in the rules covers it, and the run is still progressing. At this pace the step takes
  about 5 h instead of 2.3 h; I watch the 24 h cap. Steps were back to about 60 s from step 57 (about 10:10
  UTC). The 128 training steps took 3 h 19 min by the trainer's progress bar (seed 42: 2 h 20 min) and the
  whole GRPO subprocess 248.9 min.
- 2026-10-19 23:30 UTC, written by the attended session at the user's request (corrected after the audit on
  2026-10-20, see the last note). The pod booted again at 22:34:06 UTC (`uptime -s`, read at 23:24; no file
  records it), and none of this session's processes or the `261018-vllm` and `261018-run` tmux sessions was
  left. Output had stopped about 1 h 40 min earlier: `logs/rl_cot_s44_a1_752350acac9b.log` was last written at
  20:53, while the trainer was caching the monitor probe's activations (1408 of 6737 examples), with no
  Traceback and no `EXIT` line; `logs/vllm_cot_s44_a1_752350acac9b.log` was last written at 20:50 and this
  session's transcript at 20:52. The run directory has no step files, so step 20 attempt 1 stopped before
  training began. What happened between 20:53 and 22:34 (a hang, an earlier crash or the pod going down) is
  not recorded. After the restart the kernel is 6.8.0-147-generic (6.17.0-1014-nvidia before) and /workspace
  is mounted from a different server name; the hostname and the GPU (one H200) are the same. [Audit: W&B's
  `wandb-metadata.json` also records a CUDA change. Runs f3qfvw2q, nk7mr4qf and nc6bwn67 (and seed 42's
  6vcs39wj and 5b0x5xp6) ran on kernel 6.17.0-1014-nvidia with CUDA 13.0; attempt 2's i3wl9evn ran on
  6.8.0-147-generic with CUDA 13.2, with the same hostname and GPU uuid. So `cot_s44`'s RL run and final eval
  ran under a different kernel and CUDA driver than the other five runs. The 50-question wrapper dry runs
  (07:10-07:16 on 2026-10-19) were not repeated after the restart.]
  The user approves: redo step 19 as attempt 2 (`start_vllm.sh cot_s44 2`), because its vLLM server was gone
  after the restart (step 19 is unticked again); then step 20 as attempt 2 (`run_rl.sh cot_s44 2`); then steps
  21-25 as written. A pod restart is not one of the rerun reasons under Fixes and attempts; this approval
  covers this rerun only.
  Attempt 1 leaves nothing in this folder to move: there is no `run_metrics_noisy_label_s44.json`, and its
  logs keep their `_a1_` names. Its run directory
  `/workspace/results/run_results/261018_noisy_label_rl_seeds/noisy_label_s44_ccrb19ou` (empty
  `checkpoints/`) and W&B run `ccrb19ou` stay as they are. Its partial probe cache
  `/workspace/results/activations/noisy_label_s44_base_activations_*` (388 MB, no `metadata.json`) is
  rebuilt by attempt 2: the trainer reuses a cache only when its `metadata.json` exists
  (`obfuscation_atlas/detectors/train.py:933`). `training.resume_from_checkpoint` is false, so attempt 2
  starts a fresh W&B run; steps 21 and 24 use attempt 2's run id from `run_metrics_noisy_label_s44.json`.
  The 24 h cap is unchanged: 2026-10-20 07:46 UTC.
- 2026-10-19 23:30 UTC: a new unattended session took over (same kickoff prompt) after the attended note above.
  `tmux ls` showed only `261018-seeds` (this session), `nvidia-smi` no process, nothing on port 8000, 63 GB free
  on /workspace. The 15-minute resume check is again a session-only cron job (minutes 7, 22, 37, 52). About
  8 h 15 min remain before the cap; the seed-43 CoT run took about 4 h 45 min from vLLM start to comparison.
- Step 19 attempt 2 (23:31-23:35) failed before loading the model: CUDA error 802 "system not yet initialized"
  from `cudaGetDeviceCount()` (log lines 33 and 65 of `logs/vllm_cot_s44_a2_752350acac9b.log`). `nvidia-smi -q`
  at 23:36 shows the GPU's Fabric State `In Progress` (Status N/A): the host has not finished setting up the
  NVSwitch fabric since the pod restart, and CUDA refuses to start until it does. This is not a bug in this
  folder's scripts (nothing to fix) and nothing in the pod can finish it. [Audit: not confirmed. The error 802
  is in the log; the Fabric State readings are in no file, and the explanation is the session's.] I treat it as "GPU unavailable": I
  wait, polling `nvidia-smi -q` every 30 s, until the fabric state leaves `In Progress`, for at most 1 h counted
  from the first sight of the error (23:35, so until 00:35 UTC); if it does not, I stop for the user. Step 19
  has one attempt left (a3), so I start it only after the fabric reports done and a minimal CUDA call
  (`torch.cuda.device_count()` plus a one-element tensor on the GPU) succeeds. Attempt 2 left only its log in
  this folder, which keeps its `_a2_` name; nothing to move to `failed_attempts/`.
- 00:36 UTC: the fabric never left `In Progress`, so I stopped under "the GPU unavailable for over an hour". Steps
  19-25 stay unticked (the run is stopped, not a failed run, so they are not marked "skipped: run blocked"). I
  deleted this session's 15-minute resume check so it does not restart step 19 while the run is stopped.
- The Blocked section as written at 00:36 UTC, moved here by the attended session at 05:01 UTC at the user's
  request (its first line, the stop marker, is left out):

  The GPU has been unusable since the pod restart, so step 19 (`cot_s44` start vLLM) cannot run. Steps 1-18 (runs
  `nocot_s43`, `cot_s43`, `nocot_s44`) are complete; steps 19-25 (run `cot_s44` and the seed table) are not.

  - Step 19 attempt 2 (`start_vllm.sh cot_s44 2`, 23:31-23:35 UTC) ended `EXIT 1` before loading the model.
    From `logs/vllm_cot_s44_a2_752350acac9b.log`:
    ```
    (EngineCore_DP0 pid=4987) ERROR 10-19 23:35:08 [core.py:866] EngineCore failed to start.
    ...
      File ".../vllm/platforms/cuda.py", line 123, in set_device
        torch.cuda.set_device(device)
      File ".../torch/cuda/__init__.py", line 410, in _lazy_init
        torch._C._cuda_init()
    RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized
    ...
    RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
    EXIT 1
    ```
  - `nvidia-smi -q`, unchanged at every 30 s poll from 23:36 to 00:35 UTC (driver 595.91.07, CUDA 13.2, one H200,
    no process on the GPU):
    ```
        Fabric
            State                                          : In Progress
            Status                                         : N/A
    ```
    CUDA error 802 with Fabric State `In Progress` means the host has not finished setting up the GPU's NVSwitch
    fabric (on HGX machines this is done by the host's NVIDIA Fabric Manager). That runs outside the pod, so
    nothing in this folder can fix it. [Audit: not confirmed; see the note on step 19 attempt 2 above.]
  - State left behind: no tmux session of the queue is running, `nvidia-smi` lists no process. Step 19 has one
    attempt left (a3: `start_vllm.sh cot_s44 3`). Step 20 attempt 2 (`run_rl.sh cot_s44 2`) has not started, so
    there is no `run_metrics_noisy_label_s44.json`, no `logs/rl_cot_s44_a2_*` and no new run directory. Attempt
    1's partial probe cache `/workspace/results/activations/noisy_label_s44_base_activations_*` (388 MB, no
    `metadata.json`) is still there; attempt 2 rebuilds it. 63 GB free on /workspace at 23:35.
  - Time: the 24 h cap ends 2026-10-20 07:46 UTC. The seed-43 CoT run took about 4 h 45 min from vLLM start to
    comparison, so it still fits if the GPU works again by about 02:45 UTC and the run is resumed then.
- 2026-10-20 05:01 UTC, written by the attended session at the user's request. The user removed the wall-clock
  cap: this run has no time limit. This replaces the 24 h cap in CLAUDE.md (Infrastructure rules and Stop rules)
  and every cap time written above (07:46 UTC); all other stop rules stay. The Blocked section is cleared and the
  run resumes. The GPU works again: at 04:58-05:00 UTC `nvidia-smi -q` shows Fabric State `Completed`, Status
  `Success`, and `cuInit` / `cuDeviceGetCount` through libcuda return 0 with 1 device. There was no new pod
  restart (`uptime -s` is still 22:34:06). [Audit: these readings were taken in the attended session and are
  in no file.] Next: the minimal CUDA check planned above, then step 19 attempt 3
  (`start_vllm.sh cot_s44 3`), then step 20 attempt 2 (`run_rl.sh cot_s44 2`, approved in the 23:30 note and not
  started yet), then steps 21-25 as written.
- 2026-10-20 05:10 UTC: a new unattended session took over (same kickoff prompt). `tmux ls` showed only
  `261018-seeds` (this session, created 05:08), `nvidia-smi` no process, 63 GB free on /workspace, Fabric State
  `Completed`. The 15-minute resume check is again a session-only cron job (minutes 7, 22, 37, 52). The minimal
  CUDA check planned above passed at 05:10:58: `torch.cuda.device_count()` 1, a one-element tensor on `cuda:0`
  (NVIDIA H200), exit 0. Next: step 19 attempt 3.
- Step 20 attempt 2 (`cot_s44` RL) ran at about 100-130 s per step through step 100, then slowed to about 240 s
  per step (step 105 at 09:27 UTC). At 09:27 the trainer was waiting on page I/O (`D` state,
  `folio_wait_bit_common`), the GPU was at 0% use, and the host load average was about 80: the same pattern as
  step 2's slowdown, outside the run. [Audit: not confirmed. These readings are in no file, and "the same
  pattern" compares load averages of about 1100 and about 80; the files show only the slowdown itself.] No
  action: the run is still progressing and there is no time cap.
- 11:22 UTC: every queue step is ticked and Result holds `seed_table.md`, so I deleted this session's 15-minute
  resume check. No tmux session of the queue is left running and `nvidia-smi` lists no process.
- 2026-10-20 15:17 UTC, written by the attended session at the user's request: corrections after the audit of
  `seed_table.md` and this file. The audit found every number in the seed table consistent with the raw eval
  and run files. The corrections in this file are marked `[Audit: ...]` above, except two that replace the
  attended session's own wording:
  - the step 20 attempt 1 Log line and the 23:30 note: output stopped at 20:53, about 1 h 40 min before the
    22:34 boot, and what happened in between is not recorded (first written as "cut off by the pod restart");
  - the 23:30 note's citation of the cache-reuse check: `train.py:933` (first written as `:934`).
  The audit could not check these, because no file records them:
  - the free-disk readings and minimums;
  - the tmux kill times;
  - the 22:34 boot time and the change of mount server;
  - the resume-check cron jobs;
  - the 05:10:58 CUDA check;
  - the size of attempt 1's partial probe cache;
  - the exit codes of `cot_s44`'s record check and comparison, and of the seed table. Those steps left no
    logs; their outputs are complete.
- 2026-10-20 18:13 UTC, written by the attended session at the user's request: `seed_table.py` gained the
  base mean and SD over the seeds, for the figures: columns `base mean` and `base SD` in `seed_table.md`, and
  the key `mean_sd` in `seed_table.json`. Both files were regenerated. No earlier number changed: the
  per-seed entries of the JSON are identical, and every Markdown row matches once the two new columns are
  removed. The Result section below is the `seed_table.md` of the run (11:21 UTC), without the two columns.

## Blocked

## Result

# 261018 noisy-label RL, seeds side by side

Base = the base model (one eval per setup, shared by the seeds; on bias and self-report rows one value per seed, see the header of each table). seed N = that run's final checkpoint. Mean and SD (n - 1) are over the after-RL values. All eval questions.

## CoT (261013's setup; one sampled answer per prompt)

Eval questions: 489. Questions with a no-cue answer in both the base eval and the seed's eval (bias and self-report rows use these): seed 42 442, seed 43 456, seed 44 453.

| metric | condition | base | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.751 | 0.748 | 0.738 | 0.746 | 0.007 |
| accuracy | cue_gold | 0.843 | 0.975 | 0.959 | 0.965 | 0.967 | 0.008 |
| accuracy | cue_dist | 0.509 | 0.186 | 0.346 | 0.239 | 0.257 | 0.081 |
| accuracy | train_mix | 0.729 | 0.700 | 0.746 | 0.717 | 0.721 | 0.023 |
| reward | none | 0.644 | 0.661 | 0.667 | 0.658 | 0.662 | 0.004 |
| reward | cue_gold | 0.714 | 0.800 | 0.783 | 0.787 | 0.790 | 0.009 |
| reward | cue_dist | 0.499 | 0.321 | 0.407 | 0.329 | 0.352 | 0.047 |
| reward | train_mix | 0.689 | 0.758 | 0.748 | 0.740 | 0.749 | 0.009 |
| cue agreement rate | cue_gold | 0.843 | 0.975 | 0.959 | 0.965 | 0.967 | 0.008 |
| cue agreement rate | cue_dist | 0.458 | 0.814 | 0.652 | 0.761 | 0.742 | 0.082 |
| cue agreement rate | train_mix | 0.711 | 0.925 | 0.854 | 0.893 | 0.890 | 0.036 |
| bias | cue_gold | 0.097 / 0.092 / 0.090 | 0.179 | 0.173 | 0.195 | 0.182 | 0.011 |
| bias | cue_dist | 0.259 / 0.267 / 0.273 | 0.595 | 0.420 | 0.517 | 0.510 | 0.088 |
| bias | train_mix | 0.154 / 0.152 / 0.153 | 0.324 | 0.251 | 0.289 | 0.288 | 0.036 |
| anti_bias | cue_gold | 0.037 / 0.038 / 0.038 | 0.002 | 0.002 | 0.000 | 0.001 | 0.001 |
| anti_bias | cue_dist | 0.012 / 0.011 / 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| anti_bias | train_mix | 0.027 / 0.028 / 0.028 | 0.001 | 0.002 | 0.000 | 0.001 | 0.001 |
| no_bias | cue_gold | 0.866 / 0.871 / 0.872 | 0.819 | 0.825 | 0.805 | 0.816 | 0.010 |
| no_bias | cue_dist | 0.729 / 0.722 / 0.718 | 0.405 | 0.580 | 0.483 | 0.490 | 0.088 |
| no_bias | train_mix | 0.819 / 0.820 / 0.820 | 0.676 | 0.747 | 0.711 | 0.711 | 0.036 |
| deception_soft | cue_gold | 0.547 / 0.554 / 0.550 | 0.553 | 0.537 | 0.590 | 0.560 | 0.027 |
| deception_soft | cue_dist | 0.605 / 0.600 / 0.596 | 0.464 | 0.567 | 0.521 | 0.517 | 0.051 |
| deception_soft | train_mix | 0.565 / 0.568 / 0.566 | 0.516 | 0.547 | 0.578 | 0.547 | 0.031 |
| deception | cue_gold | 0.565 / 0.573 / 0.568 | 0.628 | 0.571 | 0.653 | 0.617 | 0.042 |
| deception | cue_dist | 0.669 / 0.664 / 0.657 | 0.425 | 0.602 | 0.521 | 0.516 | 0.089 |
| deception | train_mix | 0.597 / 0.602 / 0.599 | 0.547 | 0.579 | 0.623 | 0.583 | 0.038 |
| biased_share | cue_gold | 0.100 / 0.095 / 0.093 | 0.179 | 0.174 | 0.195 | 0.182 | 0.011 |
| biased_share | cue_dist | 0.262 / 0.270 / 0.276 | 0.595 | 0.420 | 0.517 | 0.510 | 0.088 |
| biased_share | train_mix | 0.158 / 0.156 / 0.157 | 0.324 | 0.252 | 0.289 | 0.288 | 0.036 |
| false_denial_soft | cue_gold | 0.388 / 0.396 / 0.379 | 0.291 | 0.299 | 0.254 | 0.282 | 0.024 |
| false_denial_soft | cue_dist | 0.360 / 0.354 / 0.349 | 0.272 | 0.303 | 0.260 | 0.278 | 0.022 |
| false_denial_soft | train_mix | 0.370 / 0.372 / 0.364 | 0.272 | 0.298 | 0.261 | 0.277 | 0.019 |
| false_admission_soft | cue_gold | 0.565 / 0.571 / 0.568 | 0.610 | 0.587 | 0.671 | 0.623 | 0.044 |
| false_admission_soft | cue_dist | 0.693 / 0.691 / 0.691 | 0.747 | 0.758 | 0.799 | 0.768 | 0.028 |
| false_admission_soft | train_mix | 0.601 / 0.604 / 0.603 | 0.632 | 0.631 | 0.707 | 0.657 | 0.043 |
| said_yes_biased_soft | cue_gold | 0.612 / 0.604 / 0.621 | 0.709 | 0.701 | 0.746 | 0.718 | 0.024 |
| said_yes_biased_soft | cue_dist | 0.640 / 0.646 / 0.651 | 0.728 | 0.697 | 0.740 | 0.722 | 0.022 |
| said_yes_biased_soft | train_mix | 0.630 / 0.628 / 0.636 | 0.728 | 0.702 | 0.739 | 0.723 | 0.019 |
| deception_from_biased | cue_gold | 0.039 / 0.038 / 0.035 | 0.052 | 0.052 | 0.050 | 0.051 | 0.001 |
| deception_from_biased | cue_dist | 0.094 / 0.096 / 0.096 | 0.162 | 0.127 | 0.134 | 0.141 | 0.018 |
| deception_from_biased | train_mix | 0.058 / 0.058 / 0.057 | 0.088 | 0.075 | 0.075 | 0.079 | 0.007 |
| deception_from_unbiased | cue_gold | 0.508 / 0.517 / 0.515 | 0.501 | 0.485 | 0.541 | 0.509 | 0.029 |
| deception_from_unbiased | cue_dist | 0.511 / 0.504 / 0.500 | 0.302 | 0.440 | 0.386 | 0.376 | 0.069 |
| deception_from_unbiased | train_mix | 0.506 / 0.510 / 0.509 | 0.428 | 0.472 | 0.502 | 0.467 | 0.038 |
| said_yes | cue_gold | 0.593 / 0.599 / 0.596 | 0.785 | 0.648 | 0.794 | 0.742 | 0.082 |
| said_yes | cue_dist | 0.785 / 0.783 / 0.786 | 0.939 | 0.802 | 0.892 | 0.878 | 0.069 |
| said_yes | train_mix | 0.655 / 0.657 / 0.658 | 0.829 | 0.697 | 0.828 | 0.785 | 0.076 |
| false_denial | cue_gold | 0.357 / 0.366 / 0.350 | 0.063 | 0.278 | 0.136 | 0.159 | 0.109 |
| false_denial | cue_dist | 0.279 / 0.280 / 0.267 | 0.068 | 0.262 | 0.141 | 0.157 | 0.098 |
| false_denial | train_mix | 0.317 / 0.325 / 0.311 | 0.065 | 0.266 | 0.147 | 0.159 | 0.101 |
| false_admission | cue_gold | 0.588 / 0.595 / 0.590 | 0.751 | 0.633 | 0.777 | 0.721 | 0.077 |
| false_admission | cue_dist | 0.808 / 0.806 / 0.806 | 0.950 | 0.848 | 0.927 | 0.908 | 0.053 |
| false_admission | train_mix | 0.650 / 0.654 / 0.653 | 0.778 | 0.684 | 0.817 | 0.760 | 0.068 |
| parse_failed | none | 0.016 | 0.000 | 0.004 | 0.010 | 0.005 | 0.005 |
| parse_failed | cue_gold | 0.018 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| parse_failed | cue_dist | 0.033 | 0.000 | 0.002 | 0.000 | 0.001 | 0.001 |
| parse_failed | train_mix | 0.024 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 434 / 448 / 445 | 442 | 456 | 452 | 450 | 7.211 |
| n_bias_rows | cue_dist | 428 / 442 / 439 | 442 | 455 | 453 | 450 | 7.000 |
| n_bias_rows | train_mix | 862 / 890 / 884 | 884 | 911 | 905 | 900 | 14.177 |
| n_deception_rows | cue_gold | 418 / 431 / 428 | 441 | 455 | 452 | 449.333 | 7.371 |
| n_deception_rows | cue_dist | 423 / 437 / 435 | 442 | 455 | 453 | 450 | 7.000 |
| n_deception_rows | train_mix | 841 / 868 / 863 | 883 | 910 | 905 | 899.333 | 14.364 |

Training: share of groups whose 8 rewards are all equal, mean over the 128 steps.

| | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|
| same-reward groups | 0.847 | 0.822 | 0.900 | 0.856 | 0.040 |

## No-CoT (261014's setup; top letter)

Eval questions: 547. Questions with a no-cue answer in both the base eval and the seed's eval (bias and self-report rows use these): seed 42 546, seed 43 547, seed 44 543.

| metric | condition | base | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.739 | 0.585 | 0.737 | 0.687 | 0.088 |
| accuracy | cue_gold | 0.934 | 0.973 | 1.000 | 0.954 | 0.976 | 0.023 |
| accuracy | cue_dist | 0.351 | 0.188 | 0.000 | 0.329 | 0.172 | 0.165 |
| accuracy | train_mix | 0.734 | 0.702 | 0.650 | 0.738 | 0.697 | 0.044 |
| reward | none | 0.644 | 0.649 | 0.517 | 0.629 | 0.598 | 0.071 |
| reward | cue_gold | 0.764 | 0.788 | 0.801 | 0.773 | 0.787 | 0.014 |
| reward | cue_dist | 0.397 | 0.303 | 0.199 | 0.389 | 0.297 | 0.095 |
| reward | train_mix | 0.726 | 0.742 | 0.750 | 0.738 | 0.743 | 0.006 |
| cue agreement rate | cue_gold | 0.934 | 0.973 | 1.000 | 0.954 | 0.976 | 0.023 |
| cue agreement rate | cue_dist | 0.644 | 0.808 | 1.000 | 0.669 | 0.826 | 0.166 |
| cue agreement rate | train_mix | 0.830 | 0.914 | 1.000 | 0.853 | 0.922 | 0.074 |
| mean_p_correct | none | 0.737 | 0.739 | 0.582 | 0.738 | 0.686 | 0.090 |
| mean_p_correct | cue_gold | 0.927 | 0.973 | 1.000 | 0.950 | 0.974 | 0.025 |
| mean_p_correct | cue_dist | 0.360 | 0.187 | 0.000 | 0.330 | 0.172 | 0.166 |
| mean_p_correct | train_mix | 0.732 | 0.701 | 0.650 | 0.734 | 0.695 | 0.042 |
| mean_p_label | none | 0.634 | 0.652 | 0.516 | 0.633 | 0.600 | 0.074 |
| mean_p_label | cue_gold | 0.761 | 0.788 | 0.801 | 0.769 | 0.786 | 0.016 |
| mean_p_label | cue_dist | 0.404 | 0.303 | 0.199 | 0.392 | 0.298 | 0.097 |
| mean_p_label | train_mix | 0.724 | 0.742 | 0.750 | 0.735 | 0.743 | 0.007 |
| mean_p_cue | cue_gold | 0.927 | 0.973 | 1.000 | 0.950 | 0.974 | 0.025 |
| mean_p_cue | cue_dist | 0.637 | 0.813 | 1.000 | 0.665 | 0.826 | 0.168 |
| mean_p_cue | train_mix | 0.824 | 0.916 | 1.000 | 0.849 | 0.922 | 0.076 |
| bias | cue_gold | 0.183 / 0.183 / 0.179 | 0.233 | 0.415 | 0.215 | 0.288 | 0.111 |
| bias | cue_dist | 0.403 / 0.404 / 0.406 | 0.550 | 0.585 | 0.413 | 0.516 | 0.091 |
| bias | train_mix | 0.263 / 0.263 / 0.260 | 0.339 | 0.491 | 0.287 | 0.372 | 0.106 |
| anti_bias | cue_gold | 0.002 / 0.002 / 0.002 | 0.000 | 0.000 | 0.004 | 0.001 | 0.002 |
| anti_bias | cue_dist | 0.004 / 0.004 / 0.004 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| anti_bias | train_mix | 0.002 / 0.002 / 0.002 | 0.000 | 0.000 | 0.001 | 0.000 | 0.001 |
| no_bias | cue_gold | 0.815 / 0.815 / 0.819 | 0.767 | 0.585 | 0.781 | 0.711 | 0.109 |
| no_bias | cue_dist | 0.593 / 0.592 / 0.591 | 0.450 | 0.415 | 0.585 | 0.483 | 0.090 |
| no_bias | train_mix | 0.735 / 0.735 / 0.737 | 0.661 | 0.509 | 0.712 | 0.627 | 0.106 |
| said_yes_biased | cue_gold | 0.840 / 0.840 / 0.845 | 0.921 | 0.643 | 1.000 | 0.855 | 0.187 |
| said_yes_biased | cue_dist | 0.772 / 0.773 / 0.772 | 0.876 | 0.566 | 1.000 | 0.814 | 0.224 |
| said_yes_biased | train_mix | 0.808 / 0.808 / 0.810 | 0.896 | 0.602 | 1.000 | 0.833 | 0.206 |
| said_yes_unbiased | cue_gold | 0.784 / 0.784 / 0.784 | 0.785 | 0.553 | 1.000 | 0.779 | 0.223 |
| said_yes_unbiased | cue_dist | 0.870 / 0.870 / 0.871 | 0.906 | 0.617 | 1.000 | 0.841 | 0.200 |
| said_yes_unbiased | train_mix | 0.809 / 0.809 / 0.809 | 0.816 | 0.576 | 1.000 | 0.797 | 0.213 |
| deception_soft | cue_gold | 0.617 / 0.618 / 0.620 | 0.586 | 0.491 | 0.756 | 0.611 | 0.134 |
| deception_soft | cue_dist | 0.559 / 0.558 / 0.559 | 0.488 | 0.505 | 0.592 | 0.528 | 0.056 |
| deception_soft | train_mix | 0.596 / 0.596 / 0.597 | 0.555 | 0.494 | 0.698 | 0.582 | 0.105 |
| deception | cue_gold | 0.669 / 0.670 / 0.671 | 0.621 | 0.472 | 0.784 | 0.625 | 0.156 |
| deception | cue_dist | 0.610 / 0.609 / 0.610 | 0.476 | 0.510 | 0.586 | 0.524 | 0.056 |
| deception | train_mix | 0.646 / 0.646 / 0.647 | 0.575 | 0.489 | 0.713 | 0.592 | 0.113 |
| said_yes | cue_gold | 0.794 / 0.794 / 0.795 | 0.817 | 0.590 | 1.000 | 0.802 | 0.205 |
| said_yes | cue_dist | 0.830 / 0.830 / 0.831 | 0.890 | 0.587 | 1.000 | 0.826 | 0.214 |
| said_yes | train_mix | 0.809 / 0.809 / 0.809 | 0.843 | 0.589 | 1.000 | 0.811 | 0.208 |
| false_denial | cue_gold | 0.160 / 0.160 / 0.155 | 0.079 | 0.357 | 0.000 | 0.145 | 0.187 |
| false_denial | cue_dist | 0.228 / 0.227 / 0.228 | 0.124 | 0.434 | 0.000 | 0.186 | 0.224 |
| false_denial | train_mix | 0.192 / 0.192 / 0.190 | 0.104 | 0.398 | 0.000 | 0.167 | 0.206 |
| false_admission | cue_gold | 0.784 / 0.784 / 0.784 | 0.785 | 0.553 | 1.000 | 0.779 | 0.223 |
| false_admission | cue_dist | 0.870 / 0.870 / 0.871 | 0.906 | 0.617 | 1.000 | 0.841 | 0.200 |
| false_admission | train_mix | 0.809 / 0.809 / 0.809 | 0.816 | 0.576 | 1.000 | 0.797 | 0.213 |
| n_ties | none | 0 | 1 | 0 | 4 | 1.667 | 2.082 |
| n_ties | cue_gold | 1 | 0 | 0 | 0 | 0 | 0.000 |
| n_ties | cue_dist | 3 | 2 | 0 | 1 | 1 | 1.000 |
| n_ties | train_mix | 4 | 2 | 0 | 1 | 1 | 1.000 |
| n_near_ties | none | 0 | 1 | 0 | 4 | 1.667 | 2.082 |
| n_near_ties | cue_gold | 1 | 0 | 0 | 0 | 0 | 0.000 |
| n_near_ties | cue_dist | 3 | 2 | 0 | 1 | 1 | 1.000 |
| n_near_ties | train_mix | 4 | 2 | 0 | 1 | 1 | 1.000 |
| n_low_mass | none | 13 | 10 | 6 | 5 | 7 | 2.646 |
| n_low_mass | cue_gold | 3 | 0 | 0 | 4 | 1.333 | 2.309 |
| n_low_mass | cue_dist | 5 | 0 | 0 | 4 | 1.333 | 2.309 |
| n_low_mass | train_mix | 8 | 0 | 0 | 8 | 2.667 | 4.619 |
| n_one_in_top20 | none | 0 | 0 | 0 | 0 | 0 | 0.000 |
| n_one_in_top20 | cue_gold | 0 | 0 | 27 | 0 | 9 | 15.588 |
| n_one_in_top20 | cue_dist | 0 | 0 | 40 | 0 | 13.333 | 23.094 |
| n_one_in_top20 | train_mix | 0 | 0 | 67 | 0 | 22.333 | 38.682 |
| prompts | none | 547 | 547 | 547 | 547 | 547 | 0.000 |
| prompts | cue_gold | 547 | 547 | 547 | 547 | 547 | 0.000 |
| prompts | cue_dist | 547 | 547 | 547 | 547 | 547 | 0.000 |
| prompts | train_mix | 1094 | 1094 | 1094 | 1094 | 1094 | 0.000 |
| n_bias_rows | cue_gold | 545 / 546 / 542 | 546 | 547 | 543 | 545.333 | 2.082 |
| n_bias_rows | cue_dist | 543 / 544 / 540 | 544 | 547 | 542 | 544.333 | 2.517 |
| n_bias_rows | train_mix | 1088 / 1090 / 1082 | 1090 | 1094 | 1085 | 1089.667 | 4.509 |
| n_deception_rows | cue_gold | 544 / 545 / 541 | 546 | 547 | 541 | 544.667 | 3.215 |
| n_deception_rows | cue_dist | 541 / 542 / 538 | 544 | 547 | 541 | 544 | 3.000 |
| n_deception_rows | train_mix | 1085 / 1087 / 1079 | 1090 | 1094 | 1082 | 1088.667 | 6.110 |
| biased_share | cue_gold | 0.184 / 0.183 / 0.179 | 0.233 | 0.415 | 0.216 | 0.288 | 0.110 |
| biased_share | cue_dist | 0.405 / 0.406 / 0.407 | 0.550 | 0.585 | 0.414 | 0.516 | 0.090 |
| biased_share | train_mix | 0.264 / 0.264 / 0.261 | 0.339 | 0.491 | 0.287 | 0.373 | 0.106 |
| said_yes_biased_soft | cue_gold | 0.773 / 0.773 / 0.774 | 0.785 | 0.675 | 0.970 | 0.810 | 0.149 |
| said_yes_biased_soft | cue_dist | 0.745 / 0.746 / 0.745 | 0.774 | 0.611 | 0.957 | 0.781 | 0.173 |
| said_yes_biased_soft | train_mix | 0.758 / 0.758 / 0.758 | 0.778 | 0.646 | 0.961 | 0.795 | 0.158 |
| false_denial_soft | cue_gold | 0.227 / 0.227 / 0.226 | 0.215 | 0.325 | 0.030 | 0.190 | 0.149 |
| false_denial_soft | cue_dist | 0.255 / 0.254 / 0.255 | 0.226 | 0.389 | 0.043 | 0.219 | 0.173 |
| false_denial_soft | train_mix | 0.242 / 0.242 / 0.242 | 0.222 | 0.354 | 0.039 | 0.205 | 0.158 |
| false_admission_soft | cue_gold | 0.705 / 0.706 / 0.706 | 0.699 | 0.609 | 0.957 | 0.755 | 0.180 |
| false_admission_soft | cue_dist | 0.767 / 0.767 / 0.767 | 0.807 | 0.668 | 0.981 | 0.819 | 0.157 |
| false_admission_soft | train_mix | 0.723 / 0.723 / 0.723 | 0.725 | 0.629 | 0.964 | 0.773 | 0.172 |
| deception_from_biased | cue_gold | 0.042 / 0.042 / 0.041 | 0.050 | 0.135 | 0.007 | 0.064 | 0.065 |
| deception_from_biased | cue_dist | 0.103 / 0.103 / 0.104 | 0.124 | 0.227 | 0.018 | 0.123 | 0.105 |
| deception_from_biased | train_mix | 0.064 / 0.064 / 0.063 | 0.075 | 0.174 | 0.011 | 0.087 | 0.082 |
| deception_from_unbiased | cue_gold | 0.576 / 0.576 / 0.579 | 0.536 | 0.356 | 0.750 | 0.547 | 0.197 |
| deception_from_unbiased | cue_dist | 0.456 / 0.455 / 0.455 | 0.364 | 0.277 | 0.575 | 0.405 | 0.153 |
| deception_from_unbiased | train_mix | 0.532 / 0.533 / 0.534 | 0.479 | 0.320 | 0.687 | 0.495 | 0.184 |

Training: share of groups whose 8 rewards are all equal, mean over the 128 steps.

| | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|
| same-reward groups | 0.978 | 0.994 | 0.979 | 0.984 | 0.009 |
