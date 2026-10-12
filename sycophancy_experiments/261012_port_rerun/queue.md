# Queue for the unattended run (input for /unattended-setup)

Goal: verify the code copy at `/workspace/sycophancy-rl` (the cleaned-up release plus the
sycophancy MULTIPLE_CHOICE task, private commit 12631d5, and sycophancy support in its
launcher) by running every check, then a 2-step smoke run, and only if all of them pass, a
128-step rerun of August's sycophancy baseline (260825_grpo_det0, W&B `lx6gcbcl`), compared
with August side by side. Plan: `/workspace/.claude-global/plans/quirky-pondering-nygaard.md`.

CPU steps run from this folder after `source run_env.sh` (sets PYTHONPATH to the copy,
HF_HOME, W&B secrets, `$PY`, `$RUN_DIR`, `$HOST`). The three GPU steps run through the
wrapper scripts, each in its own tmux session, with exactly the tmux command given (single
quotes; the wrapper sources run_env.sh itself and writes `logs/<step>_$HOST.log` ending with
`EXIT <code>`).

| # | Step | Command | Writes | Check |
|---|---|---|---|---|
| 1 | Code manifest | `$PY manifest.py --out manifest_start.json` | `manifest_start.json` | 133 files hashed |
| 2 | Same change as 12631d5 | `$PY check_diff.py` | `check_diff.json` | `"PASS": true` |
| 3 | Ported test | `$PY check_tests.py` | `check_tests.json` | `"PASS": true` (9 passed) |
| 4 | Composed config vs private | `$PY check_config.py` | `check_config.json`, `composed_config_*.json` | `"PASS": true` |
| 5 | Checker on August | `$PY check_responses.py --run-dir /workspace/results/run_results/syco_det0/syco_det0_s42_lx6gcbcl --wandb-run-id lx6gcbcl --out check_august.json` | `check_august.json` | `"PASS": true`, 128/128 files |
| 6 | Start vLLM | `tmux new-session -d -s 261012-vllm 'bash /workspace/sycophancy_experiments/261012_port_rerun/start_vllm.sh'` | `logs/vllm_$HOST.log` | within 10 min `curl -sf http://127.0.0.1:8000/v1/models` answers (took 4.5 min in the dry run) |
| 7a | Smoke run | `tmux new-session -d -s 261012-smoke 'bash /workspace/sycophancy_experiments/261012_port_rerun/run_grpo_step.sh smoke'` | `logs/smoke_$HOST.log`, `run_metrics_syco_det0_smoke_s42.json` | log ends `EXIT 0` |
| 7b | Checker on the smoke run | `$PY check_responses.py --run-dir /workspace/results/run_results/261012_port_rerun_smoke/syco_det0_smoke_s42_<run_id> --wandb-run-id <run_id> --out check_responses_smoke.json` (`<run_id>` = `run_id` in the smoke run_metrics file) | `check_responses_smoke.json` | `"PASS": true`, 2/2 files |
| 7c | Smoke stages | `$PY check_smoke.py --log logs/smoke_$HOST.log --vllm-log logs/vllm_$HOST.log` | `check_smoke.json` | `"PASS": true` |
| 8a | Code unchanged | `$PY manifest.py --out manifest_gate.json --compare manifest_start.json` | `manifest_gate.json` | `"PASS": true` |
| 8b | Gate | `$PY gate.py` | `gate.json` | exit 0 and `"PASS": true`. **If not, do not start 8c.** |
| 8c | Rerun | `tmux new-session -d -s 261012-rerun 'bash /workspace/sycophancy_experiments/261012_port_rerun/run_grpo_step.sh rerun'` | `logs/rerun_$HOST.log`, `run_metrics_syco_det0_rerun_s42.json` | log ends `EXIT 0` |
| 9 | Checker on the rerun | `$PY check_responses.py --run-dir /workspace/results/run_results/261012_port_rerun/syco_det0_rerun_s42_<run_id> --wandb-run-id <run_id> --out check_rerun.json` (`<run_id>` from the rerun run_metrics file) | `check_rerun.json` | `"PASS": true`, 128/128 files |
| 10 | Compare with August | `$PY compare_august.py` | `comparison.md`, `comparison.json` | table has 13 rows; copy the table into runs.md Result |
| 11 | Stop vLLM, final manifest | `tmux kill-session -t 261012-vllm`, then `$PY manifest.py --out manifest_end.json --compare manifest_start.json` | `manifest_end.json` | `"PASS": true` |

Experiment-specific rules for the protocol:
- Any failed check is a stop, not a condition to skip: a `"PASS": false`, a non-zero exit of a
  check script, a GRPO log not ending in `EXIT 0` (7a, 8c), or vLLM not answering within 10 min
  (6). Write the failing output under Blocked, "stopped, waiting for the user", and end the turn.
- Never re-run a GRPO step (7a, 8c); `run_grpo_step.sh` refuses if the step's log exists. The
  template's "retry a crashed step once with a smaller batch" does not apply to this run: the
  checks assume August's batch (64 prompts x 8). The launcher already retries a crashed GRPO
  run by itself (up to 3 more attempts); a `CRASH:` line in the smoke log fails 7c, and every
  `CRASH:` line in the rerun log goes into runs.md.
- Never edit the code copy, a check script, an `expected_*.json`, a YAML, `run_env.sh`, the
  wrapper scripts or this file. They are made read-only before launch.
- Writes outside this folder that the run makes on purpose (none of them is a reason to stop):
  - GRPO run directories under `/workspace/results/run_results/261012_port_rerun_smoke/` and
    `/workspace/results/run_results/261012_port_rerun/`;
  - the trainer's shared `/workspace/results/run_results/wandb_resume_map.json` (+ `.lock`);
  - activation caches under `/workspace/results/activations/` (prefixes `port_rerun_smoke`,
    `port_rerun`, `activations_<run id>`; the launcher removes them after each run);
  - in `/workspace/sycophancy-rl/obfuscation-atlas/`: Hydra `outputs/`, `wandb/`, empty
    `checkpoints/run_<time>/` folders and the export's `artifacts/` (all git-ignored or empty);
  - W&B runs in groups `261012_port_rerun_smoke` and `261012_port_rerun`.
- Read-only inputs: `/workspace/sycophancy-rl`, `/workspace/probes-as-training-signals`,
  `/workspace/obfuscation-atlas`, `sycophancy_experiments/260825_grpo_det0/`,
  `sycophancy_experiments/260819_domain_baseline/`, and August's run directory
  `/workspace/results/run_results/syco_det0/syco_det0_s42_lx6gcbcl/`.
- The in-loop eval logged at step N scores the policy from the previous eval point (the first
  row scores the starting model); this is the trainer's behaviour, not a failure.
- Time: steps 1–5 about 5 min; vLLM start about 5 min; smoke run about 1 h (not measured);
  rerun about 4.5 h (August's trainer took 4.28 h); steps 9–11 about 2 min. Wall-clock cap
  10 h. Disk: the launcher refuses to start under 50 GB free (69 GB free on 2026-10-12); the
  runs add about 3 GB.
