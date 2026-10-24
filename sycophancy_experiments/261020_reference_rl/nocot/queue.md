# Queue for the no-CoT unattended run (input for /unattended-setup), on pod 52303d6918f5

**Pod check before step 1:** `hostname` must print `52303d6918f5` (the second pod, one H200) and
`nvidia-smi -L` must list one GPU. Otherwise stop for the user: the CoT queue runs on pod `752350acac9b`, and
the two queues must not share a GPU.

Goal: the three no-CoT reference conditions of 261020, three training seeds each (42, 43, 44), in the settings of
the no-CoT noisy-label runs (`261014_nocot_noisy_label_rl` at seed 42, `261018_noisy_label_rl_seeds` at 43 and 44):
the assistant prefilled with `The answer is (` and one generated letter, 128 steps, 8 answers per question, no
probe penalty, the 1304 / 547 no-cue-confident questions. The conditions are **noisy_nocue** (reward = agreement
with 261014's noisy training label, 20% of questions flipped; no cue in any training prompt), **correct_nocue**
(reward = the correct answer, this folder's `../dataset/nocot/labels.jsonl`; no cue) and **correct_cue** (reward =
the correct answer; the cue points at it 75% of the time). "No cue" is the trainer setting
`task_args.train_cue: false` (added for this experiment: every training prompt is then the question's no-cue
prompt, then the prefill). First, the base model is re-evaluated on 50 questions on this pod and compared with
261014's saved base eval (reported, not a gate); then a 2-step smoke run of correct_nocue is checked by you.
Each final checkpoint is evaluated with 261014's own `eval_checkpoint.py` and compared with 261014's base-model
eval by 261014's own `compare.py` (called by absolute path); `../condition_table.py` then puts the
baseline and the three conditions side by side. Only the fields listed in each config's header differ from
261018's `nocot_s43.yaml`. Plan: `/workspace/.claude-global/plans/read-my-project-context-proud-dove.md`.

**This queue shares the experiment folder and the fence with the CoT queue, which runs at the same time on pod
752350acac9b from `../cot/`.** Work only in this folder (`nocot/`), except a code fix under Crashes in `CLAUDE.md`. Never write in `../cot/`. Both pods share
`/workspace`, its free disk and W&B; the GPU, tmux and vLLM are per pod.

Runs, in this order (run name = config `<run>.yaml` in this folder = run key):
`nocot_noisy_nocue_s42`, `nocot_correct_nocue_s42`, `nocot_correct_cue_s42`, `nocot_noisy_nocue_s43`,
`nocot_correct_nocue_s43`, `nocot_correct_cue_s43`, `nocot_noisy_nocue_s44`, `nocot_correct_nocue_s44`,
`nocot_correct_cue_s44`. For run `<run>`:
- `<run_id>` is `run_id` in `run_metrics_<run>.json` (written by the RL step);
- `RL_DIR` is `/workspace/results/run_results/261020_reference_rl/<run>_<run_id>`;
- the seed is the number after `_s`;
- `<check flags>` for the record check: `--no-cue` for `nocot_noisy_nocue_*`;
  `--questions /workspace/sycophancy_experiments/261020_reference_rl/dataset/nocot/labels.jsonl --no-cue` for
  `nocot_correct_nocue_*`; `--questions /workspace/sycophancy_experiments/261020_reference_rl/dataset/nocot/labels.jsonl`
  for `nocot_correct_cue_*`.
The smoke run is `nocot_smoke_s42` (config `nocot_smoke_s42.yaml`, condition correct_nocue, 2 steps), `<smoke_id>`
is `run_id` in `run_metrics_nocot_smoke_s42.json`, `SMOKE_DIR` is
`/workspace/results/run_results/261020_reference_rl_smoke/nocot_smoke_s42_<smoke_id>`. Write the full paths in
the commands. `SETUP` below is `/workspace/sycophancy_experiments/261014_nocot_noisy_label_rl`, `EXP` is
`/workspace/sycophancy_experiments/261020_reference_rl`.

CPU steps run from this folder (`nocot/`) after `source ../run_env.sh` (sets PYTHONPATH to the code copy,
HF_HOME, W&B secrets, `$PY`, `$RUN_DIR` = the experiment folder, `$HOST`); run each with `run_in_background`
(importing torch from the network volume can take up to ~10 min). The GPU steps run through the wrapper scripts,
each in its own tmux session, with exactly the tmux command given; the wrapper sources run_env.sh itself and
writes `logs/<kind>_<run>_a<attempt>_$HOST.log` (in this folder) ending with `EXIT <code>` (the vLLM log only
when the server stops by itself). Every command below is attempt 1.

**This pod's base-eval check and the smoke steps (1-8; steps 1-6 must pass before step 9):**

| # | Step | Command | Writes | Check |
|---|---|---|---|---|
| 1 | Base model, 50 questions, vs 261014's saved base eval | `tmux new-session -d -s 261020-nocot-eval 'bash EXP/run_eval.sh nocot_base50 1'`, then `$PY SETUP/compare_readings.py --a dry/evals/nocot_base50 --b SETUP/evals/base --out dry/readings_base50_vs_261014_base.json` | `logs/eval_nocot_base50_a1_$HOST.log`, `dry/evals/nocot_base50/`, `dry/readings_base50_vs_261014_base.json` | log ends `EXIT 0`; `dry/evals/nocot_base50/summary.json` has `"questions": 50` and `"cued_prompts": 100`; compare_readings exits 0; record in runs.md the top-letter agreement (of 150 prompts) and the mean / max of |ΔP(A)| and |ΔP(Yes)| (reported, never a reason to stop); copy 2 example rows (prompt tail, top letter, P(A), P(Yes)) |
| 2 | Start vLLM | `tmux new-session -d -s 261020-nocot-vllm 'bash EXP/start_vllm.sh nocot_smoke_s42 1'` | `logs/vllm_nocot_smoke_s42_a1_$HOST.log` | within 30 min `curl -sf http://127.0.0.1:8000/v1/models` answers (this pod's first start has cold caches: 24 min on pod 752350acac9b's first start) |
| 3 | Smoke RL run (2 steps) | `tmux new-session -d -s 261020-nocot-run 'bash EXP/run_rl.sh nocot_smoke_s42 1'` | `logs/rl_nocot_smoke_s42_a1_$HOST.log`, `run_metrics_nocot_smoke_s42.json` | log ends `EXIT 0`; copy every `CRASH:` line and the `GRPO subprocess finished in ... min` line into runs.md |
| 4 | Smoke record check | `mkdir -p runs/nocot_smoke_s42`, then `$PY SETUP/check_records.py --run-dir SMOKE_DIR --wandb-run-id <smoke_id> --questions EXP/dataset/nocot/labels.jsonl --no-cue --out runs/nocot_smoke_s42/check_records.json` | `runs/nocot_smoke_s42/check_records.json` | exit 0, `"PASS": true`, `"n_files": 2`, `"seed": 42`; `unparsed_share` below 0.05 (else stop); record `unparsed_share` and `same_reward_groups_mean`; copy 3 raw records of `SMOKE_DIR/checkpoints/responses/step_1.jsonl` into runs.md (the prompt's last 200 characters, `completion`, `parsed_answer`, `train_label`, `gold_letter`, `cue_letter`, `reward`): the prompt ends with the options, the assistant header and `The answer is (` and holds no cue sentence ("I think the answer is", "My guess", etc.), `train_label` equals `gold_letter` |
| 5 | Stop vLLM | `tmux kill-session -t 261020-nocot-vllm` | – | `nvidia-smi` lists no process within 2 min |
| 6 | Smoke eval (50 questions) | `tmux new-session -d -s 261020-nocot-eval 'bash EXP/run_eval.sh nocot_smoke_s42 1'` | `logs/eval_nocot_smoke_s42_a1_$HOST.log`, `dry/evals/nocot_smoke_s42/` | log ends `EXIT 0`; `dry/evals/nocot_smoke_s42/summary.json` has `"questions": 50` and `"cued_prompts": 100` |
| 7 | Smoke comparison | `mkdir -p dry/compare_smoke`, then `$PY SETUP/compare.py --base dry/evals/nocot_base50 --final dry/evals/nocot_smoke_s42 --run-dir SMOKE_DIR --out-dir dry/compare_smoke` | `dry/compare_smoke/` | exit 0; `dry/compare_smoke/training_curve.csv` has 2 data rows |
| 8 | Smoke table | `$PY EXP/condition_table.py --format nocot --smoke` | `dry/condition_table_smoke.md`, `.json` | exit 0 |

**Per run (steps 9-14 for `nocot_noisy_nocue_s42`, 15-20 `nocot_correct_nocue_s42`, 21-26 `nocot_correct_cue_s42`,
27-32 `nocot_noisy_nocue_s43`, 33-38 `nocot_correct_nocue_s43`, 39-44 `nocot_correct_cue_s43`, 45-50
`nocot_noisy_nocue_s44`, 51-56 `nocot_correct_nocue_s44`, 57-62 `nocot_correct_cue_s44`), then step 63:**

| # | Step | Command | Writes | Check |
|---|---|---|---|---|
| a | Start vLLM | `tmux new-session -d -s 261020-nocot-vllm 'bash EXP/start_vllm.sh <run> 1'` | `logs/vllm_<run>_a1_$HOST.log` | within 30 min `curl -sf http://127.0.0.1:8000/v1/models` answers |
| b | RL run (128 steps) | `tmux new-session -d -s 261020-nocot-run 'bash EXP/run_rl.sh <run> 1'` | `logs/rl_<run>_a1_$HOST.log`, `run_metrics_<run>.json` | log ends `EXIT 0`; copy every `CRASH:` line and every `GRPO subprocess finished in ... min` line into runs.md; record the lowest free disk seen while waiting (poll `df -h /workspace` with the `EXIT` wait) |
| c | Record check | `mkdir -p runs/<run>`, then `$PY SETUP/check_records.py --run-dir RL_DIR --wandb-run-id <run_id> <check flags> --out runs/<run>/check_records.json` | `runs/<run>/check_records.json` | exit 0, `"PASS": true`, `"n_files": 128`, `"seed"` equals the run's seed; record `unparsed_share` and `same_reward_groups_mean` |
| d | Stop vLLM | `tmux kill-session -t 261020-nocot-vllm` | – | `nvidia-smi` lists no process within 2 min |
| e | Final-checkpoint eval | `tmux new-session -d -s 261020-nocot-eval 'bash EXP/run_eval.sh <run> 1'` | `logs/eval_<run>_a1_$HOST.log`, `evals/<run>/` | log ends `EXIT 0`; `evals/<run>/summary.json` has `"questions": 547` and `"cued_prompts": 1094` |
| f | Comparison | `$PY SETUP/compare.py --base SETUP/evals/base --final evals/<run> --run-dir RL_DIR --out-dir runs/<run>` | `runs/<run>/comparison.md`, `comparison.json`, `training_curve.csv` | exit 0; `runs/<run>/training_curve.csv` has 128 data rows |
| 63 | Condition table (only if all nine runs passed every step except their comparisons) | `$PY EXP/condition_table.py --format nocot` | `condition_table.md`, `condition_table.json` | exit 0; copy `condition_table.md` verbatim into the Result section of runs.md |

Experiment-specific rules for the protocol:
- **Crashes, attempts and stops** follow `CLAUDE.md` (Crashes, Attempts, Blocked and stop rules): a crash is found,
  fixed, checked and rerun, and every fix is logged under Code fixes in runs.md; a definition is never changed; a
  report step that still fails does not hold up the runs.
- **What may not be changed.** Any definition: prompt text, the prefill, the parser, the reward, the cue rate,
  the training cue setting, the flips, seeds, question files, the training settings in the configs (steps,
  answers per question, learning rate, KL, temperature, probe settings), label rules, metrics, eval questions
  and conditions, the run order. Never loosen a check or edit it to match the data.
- **The one allowed setting change.** After an out-of-memory crash in the monitor probe's work (before RL
  starts, or the probe evals after it) that the launcher's own retries did not get past, add
  `detector.model_batch_size: 8` to the overrides of that run's `<run>.yaml` only (the probe's batching only,
  261012's fix on an H100) and rerun its RL step as the next attempt. Record it under Judgment calls.
- **Code fixes: the identity check.** After fixing a script that reads saved files, run each line below that runs
  the fixed file (directly or through an import), after `source ../run_env.sh`, with `S` a new folder in your
  scratchpad. Every listed output must be `cmp`-identical to the saved file of the same name in the setup folder
  (the first five lines passed on 2026-10-21 with the code at launch):
  - 261013's `compare.py`: `cd /workspace/sycophancy_experiments/261013_noisy_label_rl && $PY -B compare.py --base evals/base --final evals/final --run-dir /workspace/results/run_results/261013_noisy_label_rl/noisy_label_s42_6vcs39wj --out-dir $S/261013` → `comparison.json`, `comparison.md`, `training_curve.csv`.
  - 261014's `compare.py`: `cd /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl && $PY -B compare.py --base evals/base --final evals/final --run-dir /workspace/results/run_results/261014_nocot_noisy_label_rl/nocot_noisy_label_s42_5b0x5xp6 --out-dir $S/261014` → `comparison.json`, `comparison.md`, `training_curve.csv`.
  - either `compare.py`: `cd /workspace/sycophancy_experiments/261018_noisy_label_rl_seeds && $PY -B seed_table.py --out-dir $S/261018` → `seed_table.json`, `seed_table.md`.
  - 261013's `check_records.py`: `cd /workspace/sycophancy_experiments/261013_noisy_label_rl && $PY -B check_records.py --run-dir /workspace/results/run_results/261013_noisy_label_rl/noisy_label_s42_6vcs39wj --wandb-run-id 6vcs39wj --out $S/261013/check_records_run.json` → `check_records_run.json`.
  - 261014's `check_records.py`: `cd /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl && $PY -B check_records.py --run-dir /workspace/results/run_results/261014_nocot_noisy_label_rl/nocot_noisy_label_s42_5b0x5xp6 --wandb-run-id 5b0x5xp6 --out $S/261014/check_records_run.json` → `check_records_run.json`.
  - `condition_table.py`: the seed-table line above; once `dry/condition_table_smoke.md` and `.json` exist, also copy
    both into `$S`, rerun the smoke table step's command, and `cmp` the new files with the copies.
- **Writes outside this folder that the run makes on purpose** (none is a reason to stop):
  - the RL run directories under `/workspace/results/run_results/261020_reference_rl/` and
    `/workspace/results/run_results/261020_reference_rl_smoke/`;
  - the trainer's shared `/workspace/results/run_results/wandb_resume_map.json` (+ `.lock`);
  - activation caches under `/workspace/results/activations/` (prefixes = the run names in this queue,
    `activations_<run_id>`; the launcher removes them after each run);
  - in `/workspace/sycophancy-rl/obfuscation-atlas/`: Hydra `outputs/`, `wandb/`, empty
    `checkpoints/run_<time>/` folders and the export's `artifacts/` (all git-ignored or empty);
  - vLLM's compile cache under `/root/.cache/vllm/`;
  - W&B runs in groups `261020_reference_rl` and `261020_reference_rl_smoke`.
- **Read-only inputs** (never edited, except a code fix under Crashes in `CLAUDE.md`; their saved outputs never):
  the experiment folder's top level, `../cot/` (the other queue's), `/workspace/sycophancy-rl`,
  `/workspace/obfuscation-atlas` (its venv runs everything), `/workspace/probes-as-training-signals`, and
  `sycophancy_experiments/` folders `260819_domain_baseline`, `260823_domain_baseline_followup_wordings`,
  `260824_rescore`, `261007_format_label_cue_ablation`, `261010_confident_slice`, `261010_soft_bias_labels`,
  `261013_noisy_label_rl`, `261014_nocot_noisy_label_rl` and `261018_noisy_label_rl_seeds` (the baseline's seeds
  43 and 44, read by the condition table).
- **Not failures:**
  - The in-loop eval logged at step N scores the policy from the previous eval point (the first row scores
    the starting model). It always serves cued prompts, also in the no-cue runs.
  - The launcher's closing printout calls `final_eval/rewards/task_success` and the in-loop `task_success`
    "accuracy"; it is the reward (agreement with the run's training label). True accuracy is
    `rewards_metrics/true_correct`.
  - No-cue runs (`*_nocue_*`, the smoke run): no `followed_cue`, `cue_gold` or `cue_at_label` training
    metrics and no `train/cue_gold_fraction` in W&B; `cue_letter` and `template_idx` are null in the step
    records.
  - Every training answer is one token, so `response_length_*` metrics are 1 and
    `response_length_at_max_fraction` is 1; `logprob_corr` warnings about zero spread when a group's 8 answers
    are the same letter; eval `finish_reason` "length".
  - The monitor probe's evals pair the one-letter answers with 260819's chain-of-thought completions; its
    numbers are not used.
  - `runs/<run>/comparison.md` (261014's `compare.py`, unchanged) scores "reward" and the training cue mix
    against 261014's noisy labels for every run, its training-side cue columns are meaningless for the no-cue
    runs, and it always contains a section on the seed-42 CoT run (it reads a fixed folder); the condition
    table rescores the correct-label runs and leaves the no-cue runs' training cue mix blank.
  - `WARNING: could not remove activation cache ... (non-fatal)` followed later by
    `OK: removed activation cache` for the same cache.
  - Free disk below 50 GB during an RL step while its probe cache exists (the 50 GB rule is a check before
    each GPU step); free disk also moves with the other pod's runs.
  - A CPU step that prints nothing for up to ~10 min while importing torch.
  - A vLLM log with no `EXIT` line after its tmux session is killed.
  - Python cannot write `__pycache__` into read-only folders.
  - W&B runs, run directories and log lines of the other pod's queue (prefix `cot_`).
- **Time** (no cap), from 261018's `runs.md` and 261014's: step 1 about 10 min; smoke steps about 1.2 h (this
  pod's first vLLM start up to ~24 min, the 2-step run ~47 min as in 261014's smoke, 50-question eval ~3 min);
  each run about 4 h (vLLM 4-24 min, RL 189-249 min, record check ~1 min, final eval 6-8 min, comparison under
  1 min); about 37 h in all.
  **Disk:** 177 GB free on 2026-10-20; stop under 50 GB free before a GPU step; each run leaves about 1.5 GB
  and peaks about 17 GB above its start near its end (261018); the other pod's runs use the same volume.
