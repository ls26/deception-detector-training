# Queue for the unattended run (input for /unattended-setup)

Goal: one RL run without a probe penalty on the confident MMLU-Pro questions, where the reward is
agreement with a noisy training label (20% of questions flipped) and the training cue points at
the training label 75% of the time. Eval the base model and the final checkpoint with
`eval_checkpoint.py` on the 489 confident eval questions, and write the comparison tables with
`compare.py`. Plan: `/workspace/.claude-global/plans/ok-i-m-thinking-through-stateful-pearl.md`.

CPU steps run from this folder after `source run_env.sh` (sets PYTHONPATH to the code copy,
HF_HOME, W&B secrets, `$PY`, `$RUN_DIR`, `$HOST`). The GPU steps run through the wrapper scripts,
each in its own tmux session, with exactly the tmux command given (single quotes; the wrapper
sources run_env.sh itself and writes `logs/<step>_$HOST.log` ending with `EXIT <code>`).
`<run_id>` is `run_id` in `run_metrics_noisy_label_s42.json` (written by step 3).

| # | Step | Command | Writes | Check |
|---|---|---|---|---|
| 1 | Base-model eval | `tmux new-session -d -s 261013-eval-base 'bash /workspace/sycophancy_experiments/261013_noisy_label_rl/run_eval.sh base'` | `logs/eval_base_$HOST.log`, `evals/base/` (`completions.jsonl`, `followups.jsonl`, `labels.jsonl`, `summary.json`) | log ends `EXIT 0`; `evals/base/summary.json` has `"questions": 489` and `"cued_rollouts": 2934` |
| 2 | Start vLLM | `tmux new-session -d -s 261013-vllm 'bash /workspace/sycophancy_experiments/261013_noisy_label_rl/start_vllm.sh run'` | `logs/vllm_run_$HOST.log` | within 10 min `curl -sf http://127.0.0.1:8000/v1/models` answers (4.3 min before the smoke run) |
| 3 | RL run | `tmux new-session -d -s 261013-run 'bash /workspace/sycophancy_experiments/261013_noisy_label_rl/run_rl.sh run'` | `logs/run_$HOST.log`, `run_metrics_noisy_label_s42.json` | log ends `EXIT 0`; copy every `CRASH:` line of the log into runs.md |
| 4 | Rollout-record check | `$PY check_records.py --run-dir /workspace/results/run_results/261013_noisy_label_rl/noisy_label_s42_<run_id> --wandb-run-id <run_id> --out check_records_run.json` | `check_records_run.json` | exit 0, `"PASS": true`, `"n_files": 128` |
| 5 | Stop vLLM | `tmux kill-session -t 261013-vllm` | `logs/vllm_run_$HOST.log` ends | `nvidia-smi` lists no process within 2 min |
| 6 | Final-checkpoint eval | `tmux new-session -d -s 261013-eval-final 'bash /workspace/sycophancy_experiments/261013_noisy_label_rl/run_eval.sh final'` | `logs/eval_final_$HOST.log`, `evals/final/` (same files as step 1) | log ends `EXIT 0`; `evals/final/summary.json` has `"questions": 489` and `"cued_rollouts": 2934` |
| 7 | Comparison | `$PY compare.py --base evals/base --final evals/final --run-dir /workspace/results/run_results/261013_noisy_label_rl/noisy_label_s42_<run_id>` | `comparison.md`, `comparison.json`, `training_curve.csv` | exit 0; `training_curve.csv` has 128 rows; copy `comparison.md` into the Result section of runs.md |

Experiment-specific rules for the protocol:
- Any failed check is a stop, not a condition to skip: a log not ending in `EXIT 0` (1, 3, 6), a
  `"PASS": false` or wrong count (1, 4, 6, 7), a non-zero exit of a CPU step, or vLLM not
  answering within 10 min (2). Write the failing output under Blocked, "stopped, waiting for the
  user", kill the `261013-vllm` tmux session if it is running, and end the turn.
- Never re-run an RL or eval step (1, 3, 6); the wrappers refuse if the step's log exists. The
  launcher already retries a crashed RL run by itself (up to 3 more attempts); every `CRASH:`
  line goes into runs.md. A CPU step (4, 7) that crashed with a traceback before writing its
  output (for example a W&B API error) may be rerun once with identical arguments.
- Never edit the code copy, a script, a YAML, `noisy_spec.py`, `dataset/`, `run_env.sh`, the
  wrapper scripts or this file. They are made read-only before launch.
- Writes outside this folder that the run makes on purpose (none of them is a reason to stop):
  - the RL run directory under `/workspace/results/run_results/261013_noisy_label_rl/`;
  - the trainer's shared `/workspace/results/run_results/wandb_resume_map.json` (+ `.lock`);
  - activation caches under `/workspace/results/activations/` (prefixes `noisy_label`,
    `activations_<run_id>`; the launcher removes them after the run);
  - in `/workspace/sycophancy-rl/obfuscation-atlas/`: Hydra `outputs/`, `wandb/`, empty
    `checkpoints/run_<time>/` folders and the export's `artifacts/` (all git-ignored or empty);
  - vLLM's compile cache under `/root/.cache/vllm/`;
  - W&B runs in group `261013_noisy_label_rl`.
- Read-only inputs: `/workspace/sycophancy-rl`, `/workspace/obfuscation-atlas`,
  `/workspace/probes-as-training-signals`, and `sycophancy_experiments/` folders `260819_domain_baseline`,
  `260823_domain_baseline_followup_wordings`, `260824_rescore`, `261007_format_label_cue_ablation`,
  `261010_confident_slice`, `261010_soft_bias_labels`.
- Not failures:
  - The in-loop eval logged at step N scores the policy from the previous eval point (the first
    row scores the starting model).
  - The launcher's closing printout calls `final_eval/rewards/task_success` and the in-loop
    `task_success` "accuracy"; in this run they are the reward (agreement with the noisy training
    label). True accuracy is `rewards_metrics/true_correct`.
  - `WARNING: could not remove activation cache ... after 3 attempts (non-fatal)` followed later by
    `OK: removed activation cache` for the same cache (seen in the smoke run).
  - A CPU step that imports torch can take up to about 10 min before printing anything (the
    smoke run's record check spent 7 of its 7.5 min importing torch from the network volume).
- Time (cap 10 h): base eval about 15 min (not measured at full size); vLLM start about 5 min; RL
  run about 5 h (August's trainer took 4.28 h for 128 steps; the smoke run's trainer took 51 min
  for 2 steps, most of it the probe work every run does once, and the launcher's cleanup another
  20 min); record check about 10 min; final eval about 15 min; comparison under 1 min. Disk: the
  launcher refuses to start under 50 GB free (72 GB free on 2026-10-14); the run adds about 3 GB.
