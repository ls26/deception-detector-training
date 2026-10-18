# Queue for the unattended run (input for /unattended-setup)

Goal: one RL run without a probe penalty in the no-CoT format (assistant prefilled with
`The answer is (`, one letter token) on the 1304 / 547 no-cue-confident MMLU-Pro questions, where the
reward is agreement with a noisy training label (20% of questions flipped) and the training cue points at
the training label 75% of the time. Before it: two 50-question eval dry runs and a 2-step smoke run, each
checked here. Eval the base model and the final checkpoint with `eval_checkpoint.py` on the 547 eval
questions, and write the comparison tables with `compare.py`. Plan:
`/workspace/.claude-global/plans/understand-the-context-of-curious-pillow.md`.

CPU steps run from this folder after `source run_env.sh` (sets PYTHONPATH to the code copy, HF_HOME,
W&B secrets, `$PY`, `$RUN_DIR`, `$HOST`); run each with `run_in_background` (importing torch from the
network volume can take up to ~10 min). The GPU steps run through the wrapper scripts, each in its own
tmux session, with exactly the tmux command given; the wrapper sources run_env.sh itself and writes
`logs/<step>_a<attempt>_$HOST.log` ending with `EXIT <code>`. Every command below is attempt 1; a rerun
after a fix uses the next number (see the rules). `<smoke_id>` is `run_id` in
`run_metrics_nocot_noisy_label_smoke_s42.json` (step 5), `<run_id>` is `run_id` in
`run_metrics_nocot_noisy_label_s42.json` (step 8); substitute them into the paths below.

| # | Step | Command | Writes | Check |
|---|---|---|---|---|
| 1 | Eval dry run, base model, 50 questions, then its readings against 261007's saved ones | `tmux new-session -d -s 261014-eval 'bash /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/run_eval.sh base50 1'`, then `$PY compare_readings.py --a dry/evals/base50 --b saved --out dry/readings_base50_vs_saved.json` | `logs/eval_base50_a1_$HOST.log`, `dry/evals/base50/`, `dry/readings_base50_vs_saved.json` | log ends `EXIT 0` (the generate stage asserts the prompts equal 261007's); `summary.json` has `"questions": 50`, `"cued_prompts": 100`; readings: record top-letter agreement and mean / max \|ΔP(A)\| and \|ΔP(Yes)\| in runs.md; more than 1 of the 150 prompts with a different top letter = look for a bug. Copy 2 example rows into runs.md (prompt tail, top letter, P(A), follow-up text, P(Yes), labels) |
| 2 | Eval dry run, 261013's adapter, same 50 questions, then its readings against step 1 | `tmux new-session -d -s 261014-eval 'bash /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/run_eval.sh adapter50 1'`, then `$PY compare_readings.py --a dry/evals/adapter50 --b dry/evals/base50 --out dry/readings_adapter50_vs_base50.json` | `logs/eval_adapter50_a1_$HOST.log`, `dry/evals/adapter50/`, `dry/readings_adapter50_vs_base50.json` | log ends `EXIT 0`; same counts as step 1; the adapter is applied: `p_a.mean_abs_diff` is at least 0.01 and at least 5 x step 1's, and the same for `p_yes` (with `p_yes.n` > 0) |
| 3 | Base-model eval, 547 questions, then its readings against 261007's saved ones | `tmux new-session -d -s 261014-eval 'bash /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/run_eval.sh base 1'`, then `$PY compare_readings.py --a evals/base --b saved --out readings_base_vs_saved.json` | `logs/eval_base_a1_$HOST.log`, `evals/base/`, `readings_base_vs_saved.json` | log ends `EXIT 0`; `evals/base/summary.json` has `"questions": 547`, `"cued_prompts": 1094`; readings recorded as in step 1; more than 1% of the 1641 prompts with a different top letter = look for a bug |
| 4 | Start vLLM for the smoke run | `tmux new-session -d -s 261014-vllm 'bash /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/start_vllm.sh smoke 1'` | `logs/vllm_smoke_a1_$HOST.log` | within 10 min `curl -sf http://127.0.0.1:8000/v1/models` answers |
| 5 | Smoke run (2 RL steps) | `tmux new-session -d -s 261014-run 'bash /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/run_rl.sh smoke 1'` | `logs/smoke_a1_$HOST.log`, `run_metrics_nocot_noisy_label_smoke_s42.json` | log ends `EXIT 0`; copy every `CRASH:` line and the line `GRPO subprocess finished in ... min` into runs.md |
| 6 | Smoke review: record check, then the comparison dry run | `$PY check_records.py --run-dir /workspace/results/run_results/261014_nocot_noisy_label_rl_smoke/nocot_noisy_label_smoke_s42_<smoke_id> --wandb-run-id <smoke_id> --out check_records_smoke.json`, then `$PY compare.py --base dry/evals/base50 --final dry/evals/adapter50 --run-dir /workspace/results/run_results/261014_nocot_noisy_label_rl_smoke/nocot_noisy_label_smoke_s42_<smoke_id> --out-dir dry/compare` | `check_records_smoke.json`, `dry/compare/` | `"PASS": true`, `"n_files": 2` (served prompts end in the prefill, every answer one token, parsed answer and reward recomputed, counts = W&B); `unparsed_share` below 0.05 (base model about 0.01), else read `unparsed_completions` and look for a bug; record `same_reward_groups_mean`; copy 3 raw records into runs.md (prompt tail, completion, parsed_answer, train_label, cue_letter, reward); compare.py exits 0 and `dry/compare/training_curve.csv` has 2 rows |
| 7 | Restart vLLM for the RL run | `tmux kill-session -t 261014-vllm`, wait until `nvidia-smi` lists no process, then `tmux new-session -d -s 261014-vllm 'bash /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/start_vllm.sh run 1'` | `logs/vllm_run_a1_$HOST.log` | within 10 min `curl -sf http://127.0.0.1:8000/v1/models` answers |
| 8 | RL run (128 steps) | `tmux new-session -d -s 261014-run 'bash /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/run_rl.sh run 1'` | `logs/run_a1_$HOST.log`, `run_metrics_nocot_noisy_label_s42.json` | log ends `EXIT 0`; copy every `CRASH:` line and the `GRPO subprocess finished in ... min` line into runs.md |
| 9 | Record check (128 files) | `$PY check_records.py --run-dir /workspace/results/run_results/261014_nocot_noisy_label_rl/nocot_noisy_label_s42_<run_id> --wandb-run-id <run_id> --out check_records_run.json` | `check_records_run.json` | exit 0, `"PASS": true`, `"n_files": 128`; record `unparsed_share` and `same_reward_groups_mean` |
| 10 | Stop vLLM | `tmux kill-session -t 261014-vllm` | – | `nvidia-smi` lists no process within 2 min |
| 11 | Final-checkpoint eval | `tmux new-session -d -s 261014-eval 'bash /workspace/sycophancy_experiments/261014_nocot_noisy_label_rl/run_eval.sh final 1'` | `logs/eval_final_a1_$HOST.log`, `evals/final/` | log ends `EXIT 0`; `evals/final/summary.json` has `"questions": 547`, `"cued_prompts": 1094` |
| 12 | Comparison | `$PY compare.py --base evals/base --final evals/final --run-dir /workspace/results/run_results/261014_nocot_noisy_label_rl/nocot_noisy_label_s42_<run_id>` | `comparison.md`, `comparison.json`, `training_curve.csv` | exit 0; `training_curve.csv` has 128 data rows; copy `comparison.md` into the Result section of runs.md |
| 13 | Code changes during the run | `git -C /workspace diff -- sycophancy-rl > code_after_run.diff`, then `diff code_at_launch.diff code_after_run.diff` | `code_after_run.diff` | runs.md lists every fix made during the run (from Judgment calls) and says whether the two diffs are identical (no fix to the code copy) or not |

Experiment-specific rules for the protocol:
- **What may be fixed.** A bug in this folder's scripts (`eval_checkpoint.py`, `compare_readings.py`,
  `check_records.py`, `compare.py`, `check_scoring.py`, the wrapper scripts, `run_env.sh`) or in the code
  copy's training package, its tests or its launcher (`/workspace/sycophancy-rl/obfuscation-atlas/obfuscation_atlas`,
  `/workspace/sycophancy-rl/obfuscation-atlas/tests`, `/workspace/sycophancy-rl/training`; the fence allows
  edits there). A bug is a traceback, or a check failing because code computes something other than what
  its docstring, `nocot_spec.py` or the plan says.
- **What may not be changed.** Any definition: prompt text, the prefill, the parser rule (a response
  parses only if it is exactly `A` or `B`), the reward, the cue rate, the flips, seeds, the training
  settings in `run.yaml` / `smoke.yaml` (steps, answers per question, learning rate, KL, temperature,
  probe settings), label rules, metrics, thresholds, eval questions and conditions. Never loosen a check
  or edit it to match the data. Never edit the vendored GRPO framework
  (`/workspace/sycophancy-rl/obfuscation-atlas/third_party`). `nocot_spec.py`, `dataset/` and
  `code_at_launch.diff` are read-only.
- **After a fix to the code copy:** `cd /workspace/sycophancy-rl/obfuscation-atlas && $PY -m pytest
  tests/test_multiple_choice_task.py -q -p no:cacheprovider` must pass (15 tests) before the step reruns.
- **Every fix** is a Judgment calls entry in runs.md: the failing output, the cause, the diff of each
  changed file (`git -C /workspace diff -- <file>` for the code copy, `diff -u` against a copy for this
  folder's scripts), and the test result.
- **Reruns.** A failed step reruns from its start with the same settings and the next attempt number
  (the wrappers take it; at most 3 attempts per step). Before a rerun, move the failed attempt's outputs
  in this folder into `failed_attempts/<step>_a<N>/` (an eval output folder, a `run_metrics_*.json`, a
  check json); never delete them. A failed RL attempt's run directory stays where the launcher left it;
  record its path. The launcher's own retries of a crashed RL run (up to 3 inside one attempt) are not
  attempts in this sense; copy every `CRASH:` line. After a fix to `eval_checkpoint.py`, rerun every eval
  step already done whose output the fix changes, so the base and final evals come from the same code.
- **Stop for the user** (record under Blocked, "stopped, waiting for the user", kill the `261014-vllm`
  tmux session if you started it, end the turn) when: a fix would change a definition; you cannot tell a
  bug from a definition question; a step fails 3 times; a fix would need the vendored framework; the
  smoke run's unparsed share is 0.05 or more and no bug explains it; disk under 50 GB free; the GPU is
  unavailable for over an hour; the 10 h cap.
- **Writes outside this folder that the run makes on purpose** (none is a reason to stop):
  - the RL run directories under `/workspace/results/run_results/261014_nocot_noisy_label_rl/` and
    `/workspace/results/run_results/261014_nocot_noisy_label_rl_smoke/`;
  - the trainer's shared `/workspace/results/run_results/wandb_resume_map.json` (+ `.lock`);
  - activation caches under `/workspace/results/activations/` (prefixes `nocot_noisy_label`,
    `nocot_noisy_label_smoke`, `activations_<run_id>`; the launcher removes them after each run);
  - in `/workspace/sycophancy-rl/obfuscation-atlas/`: Hydra `outputs/`, `wandb/`, empty
    `checkpoints/run_<time>/` folders and the export's `artifacts/` (all git-ignored or empty);
  - vLLM's compile cache under `/root/.cache/vllm/`;
  - W&B runs in groups `261014_nocot_noisy_label_rl` and `261014_nocot_noisy_label_rl_smoke`.
- **Read-only inputs:** `/workspace/obfuscation-atlas` (its venv runs everything),
  `/workspace/probes-as-training-signals`, `/workspace/sycophancy-rl` outside the three writable
  folders, and `sycophancy_experiments/` folders `260819_domain_baseline`,
  `260823_domain_baseline_followup_wordings`, `260824_rescore`, `261007_format_label_cue_ablation`,
  `261010_confident_slice`, `261013_noisy_label_rl` (its final adapter is read by step 2).
- **Not failures:**
  - The in-loop eval logged at step N scores the policy from the previous eval point (the first row
    scores the starting model).
  - The launcher's closing printout calls `final_eval/rewards/task_success` and the in-loop
    `task_success` "accuracy"; here they are the reward (agreement with the noisy training label). True
    accuracy is `rewards_metrics/true_correct`.
  - Every training answer is one token: `response_length_*` metrics are 1 and
    `response_length_at_max_fraction` is 1; a `logprob_corr` warning about zero spread when a group's 8
    answers are the same letter; eval `finish_reason` "length".
  - `WARNING: could not remove activation cache ... (non-fatal)` followed later by
    `OK: removed activation cache` for the same cache.
  - The monitor probe's evals pair one-letter answers with 260819's chain-of-thought completions; its
    numbers are not used.
  - A CPU step that prints nothing for up to ~10 min while importing torch.
  - A vLLM log with no `EXIT` line after its tmux session is killed.
- **Time** (cap 10 h): each 50-question dry eval ~5–10 min (model loading dominates); base eval
  ~10 min; vLLM start ~4 min; smoke run ~50 min (in 261013 most of it was the monitor probe's work,
  which every run does once); each record check ~5–10 min; the RL run is unmeasured: 261013's CoT run
  took 294.6 min in the trainer (1.9 min per step); one-token answers should make each step much
  faster, and the probe work adds ~45 min; final eval ~10 min; comparison under 1 min. Record the smoke
  run's time. **Disk:** stop under 50 GB free (70 GB free on 2026-10-14); the two runs add ~3 GB.
