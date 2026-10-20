# Queue for the unattended run (input for /unattended-setup)

Goal: two more seeds (43, 44) of each of the two noisy-label RL runs, in the exact settings of the seed-42
runs: the CoT run `261013_noisy_label_rl` (1217 / 489 confident questions) and the no-CoT run
`261014_nocot_noisy_label_rl` (1304 / 547 no-cue-confident questions, the assistant prefilled with
`The answer is (`). In both, the reward is agreement with a noisy training label (20% of questions flipped),
the training cue points at that label 75% of the time, there is no probe penalty, 128 steps, 8 answers per
question. Only the training seed, the names, the W&B group and the probe cache prefix differ from seed 42
(the four `<run>.yaml`). Each final checkpoint is evaluated with its setup's own `eval_checkpoint.py` and
compared with the setup's existing base-model eval by its setup's own `compare.py` (both unchanged, called by
absolute path); `seed_table.py` then puts seeds 42, 43 and 44 side by side. Plan:
`/workspace/.claude-global/plans/i-m-working-on-the-linked-dragon.md`.

Runs, in this order: `nocot_s43`, `cot_s43`, `nocot_s44`, `cot_s44`. For run `<run>`:
- `<setup>` is `/workspace/sycophancy_experiments/261013_noisy_label_rl` for `cot_*` and
  `/workspace/sycophancy_experiments/261014_nocot_noisy_label_rl` for `nocot_*`;
- `<run_key>` is `run_key` in `<run>.yaml` (`noisy_label_s43`, `noisy_label_s44`, `nocot_noisy_label_s43`,
  `nocot_noisy_label_s44`); `<run_id>` is `run_id` in `run_metrics_<run_key>.json` (written by the RL step);
- `RL_DIR` is `/workspace/results/run_results/261018_noisy_label_rl_seeds/<run_key>_<run_id>`;
- the seed is the number after `_s`.
Write the full paths in the commands.

CPU steps run from this folder after `source run_env.sh` (sets PYTHONPATH to the code copy, HF_HOME, W&B
secrets, `$PY`, `$RUN_DIR`, `$HOST`); run each with `run_in_background` (importing torch from the network
volume can take up to ~10 min). The GPU steps run through the wrapper scripts, each in its own tmux session,
with exactly the tmux command given; the wrapper sources run_env.sh itself and writes
`logs/<kind>_<run>_a<attempt>_$HOST.log` ending with `EXIT <code>` (the vLLM log only when the server stops
by itself). Every command below is attempt 1.

| # | Step (per run, steps 1-6 for `nocot_s43`, 7-12 `cot_s43`, 13-18 `nocot_s44`, 19-24 `cot_s44`) | Command | Writes | Check |
|---|---|---|---|---|
| a | Start vLLM | `tmux new-session -d -s 261018-vllm 'bash /workspace/sycophancy_experiments/261018_noisy_label_rl_seeds/start_vllm.sh <run> 1'` | `logs/vllm_<run>_a1_$HOST.log` | within 30 min `curl -sf http://127.0.0.1:8000/v1/models` answers (the first start on this pod took 24 min with cold caches; 4 min on the seed-42 pod) |
| b | RL run (128 steps) | `tmux new-session -d -s 261018-run 'bash /workspace/sycophancy_experiments/261018_noisy_label_rl_seeds/run_rl.sh <run> 1'` | `logs/rl_<run>_a1_$HOST.log`, `run_metrics_<run_key>.json` | log ends `EXIT 0`; copy every `CRASH:` line and every `GRPO subprocess finished in ... min` line into runs.md; record the lowest free disk seen while waiting (poll `df -h /workspace` with the `EXIT` wait) |
| c | Record check | `mkdir -p runs/<run>`, then `$PY <setup>/check_records.py --run-dir RL_DIR --wandb-run-id <run_id> --out runs/<run>/check_records.json` | `runs/<run>/check_records.json` | exit 0, `"PASS": true`, `"n_files": 128`, `"seed"` equals the run's seed; for `nocot_*` record `unparsed_share` and `same_reward_groups_mean` |
| d | Stop vLLM | `tmux kill-session -t 261018-vllm` | – | `nvidia-smi` lists no process within 2 min |
| e | Final-checkpoint eval | `tmux new-session -d -s 261018-eval 'bash /workspace/sycophancy_experiments/261018_noisy_label_rl_seeds/run_eval.sh <run> 1'` | `logs/eval_<run>_a1_$HOST.log`, `evals/<run>/` | log ends `EXIT 0`; `evals/<run>/summary.json` has `"questions": 489` and `"cued_rollouts": 2934` (`cot_*`) or `"questions": 547` and `"cued_prompts": 1094` (`nocot_*`) |
| f | Comparison | `$PY <setup>/compare.py --base <setup>/evals/base --final evals/<run> --run-dir RL_DIR --out-dir runs/<run>` | `runs/<run>/comparison.md`, `comparison.json`, `training_curve.csv` | exit 0; `runs/<run>/training_curve.csv` has 128 data rows |
| 25 | Seed table (only if steps 1-24 all passed) | `$PY seed_table.py` | `seed_table.md`, `seed_table.json` | exit 0; copy `seed_table.md` verbatim into the Result section of runs.md |

Experiment-specific rules for the protocol:
- **What may be fixed.** Only a bug in this folder's own scripts (`start_vllm.sh`, `run_rl.sh`, `run_eval.sh`,
  `run_env.sh`, `seed_table.py`). A bug is a traceback, or a check failing because one of these scripts
  computes something other than what its docstring or the plan says. Never edit the training code
  (`/workspace/sycophancy-rl`), the two setup folders (their `eval_checkpoint.py`, `check_records.py`,
  `compare.py` and `dataset/` are used unchanged), or the vendored GRPO framework. If a fix would need any
  of them, stop.
- **What may not be changed.** Any definition: prompt text, the prefill, the parser, the reward, the cue
  rate, the flips, seeds, question files, the training settings in the four `<run>.yaml` (steps, answers per
  question, learning rate, KL, temperature, probe settings), label rules, metrics, eval questions and
  conditions, the run order. Never loosen a check or edit it to match the data.
- **The one allowed setting change.** After an out-of-memory crash in the monitor probe's work (before RL
  starts, or the probe evals after it) that the launcher's own retries did not get past, add
  `detector.model_batch_size: 8` to the overrides of that run's `<run>.yaml` only (the probe's batching
  only, 261012's fix on an H100) and rerun its RL step as attempt 2. Record it under Judgment calls; it
  appears in the README as a difference from seed 42.
- **Attempts.** An RL step gets one attempt: the launcher already retries a crashed run up to 3 more times
  inside it (copy every `CRASH:` line). It reruns only after the batch-size change above, or after a fix to
  this folder's scripts when it failed before the trainer started. Every other step gets at most 3
  attempts (after a fix to this folder's scripts, or for a transient reason such as a W&B API error).
  Before a rerun, move the failed attempt's outputs in this folder into `failed_attempts/<step>_a<N>/` (an
  eval output folder, a `run_metrics_*.json`, `runs/<run>/`); never delete them. A failed RL attempt's run
  directory stays where the launcher left it; record its path.
- **A failed run.** If a run's step still fails after its attempts, mark that run Blocked in runs.md with
  the failing output, kill `261018-vllm` if it is running, skip the run's remaining steps and go on with the
  next run. Step 25 runs only if all four runs passed every step; otherwise stop after the last run.
- **Stop for the user** (record under Blocked, "stopped, waiting for the user", kill the `261018-vllm` tmux
  session if you started it, end the turn) when: a fix would change a definition or touch anything outside
  this folder; you cannot tell a bug from a definition question; disk under 50 GB free before a GPU step;
  the GPU is unavailable for over an hour; vLLM does not answer within 30 min on two attempts; the 24 h cap.
- **Writes outside this folder that the run makes on purpose** (none is a reason to stop):
  - the RL run directories under `/workspace/results/run_results/261018_noisy_label_rl_seeds/`;
  - the trainer's shared `/workspace/results/run_results/wandb_resume_map.json` (+ `.lock`);
  - activation caches under `/workspace/results/activations/` (prefixes `noisy_label_s43`, `noisy_label_s44`,
    `nocot_noisy_label_s43`, `nocot_noisy_label_s44`, `activations_<run_id>`; the launcher removes them after
    each run);
  - in `/workspace/sycophancy-rl/obfuscation-atlas/`: Hydra `outputs/`, `wandb/`, empty
    `checkpoints/run_<time>/` folders and the export's `artifacts/` (all git-ignored or empty);
  - vLLM's compile cache under `/root/.cache/vllm/`;
  - W&B runs in group `261018_noisy_label_rl_seeds`.
- **Read-only inputs:** `/workspace/sycophancy-rl`, `/workspace/obfuscation-atlas` (its venv runs
  everything), `/workspace/probes-as-training-signals`, and `sycophancy_experiments/` folders
  `260819_domain_baseline`, `260823_domain_baseline_followup_wordings`, `260824_rescore`,
  `261007_format_label_cue_ablation`, `261010_confident_slice`, `261010_soft_bias_labels`,
  `261013_noisy_label_rl` (its scripts, question file, base eval, seed-42 run), `261014_nocot_noisy_label_rl`
  (the same).
- **Not failures:**
  - The in-loop eval logged at step N scores the policy from the previous eval point (the first row scores
    the starting model).
  - The launcher's closing printout calls `final_eval/rewards/task_success` and the in-loop `task_success`
    "accuracy"; here they are the reward (agreement with the noisy training label). True accuracy is
    `rewards_metrics/true_correct`.
  - No-CoT runs: every training answer is one token, so `response_length_*` metrics are 1 and
    `response_length_at_max_fraction` is 1; `logprob_corr` warnings about zero spread when a group's 8
    answers are the same letter; eval `finish_reason` "length".
  - `WARNING: could not remove activation cache ... (non-fatal)` followed later by
    `OK: removed activation cache` for the same cache.
  - The monitor probe's evals pair the no-CoT runs' one-letter answers with 260819's chain-of-thought
    completions; its numbers are not used.
  - Free disk below 50 GB during an RL step while its probe cache exists (the 50 GB rule is a check before
    each GPU step; seed 42's no-CoT run dipped from 70 to 51 GB).
  - `runs/nocot_s4x/comparison.md` contains a section on the seed-42 CoT run: 261014's `compare.py` always
    reads it from a fixed folder. `runs/cot_s4x/comparison.md` counts all CoT samples; the seed table counts
    one per prompt.
  - A CPU step that prints nothing for up to ~10 min while importing torch.
  - A vLLM log with no `EXIT` line after its tmux session is killed.
  - Python cannot write `__pycache__` into the read-only setup folders.
- **Time** (cap 24 h), from the seed-42 `runs.md` files: no-CoT run about 3.4 h (vLLM start 4-24 min, RL
  ~190 min, record check 1-10 min, final eval ~5 min, compare under 1 min); CoT run about 5.3 h (vLLM 4-24
  min, RL ~297 min, record check 2-10 min, final eval ~10 min, compare under 1 min); about 17.5 h in all.
  **Disk:** 67 GB free on 2026-10-19; stop under 50 GB free before a GPU step; each run leaves about 1.5 GB
  and its probe cache needs about 19 GB while the RL step runs (measured on no-CoT only).
