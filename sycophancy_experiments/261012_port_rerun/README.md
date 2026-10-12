# 261012 — Sycophancy code copy built from the release, verified by a rerun of 260825_grpo_det0

Code experiment on the MMLU-Pro sycophancy domain. The sycophancy GRPO task (private commit
`12631d5`, the `MULTIPLE_CHOICE` task used by
[`260825_grpo_det0`](../260825_grpo_det0/)) was written in the private codebase. The cleaned-up
public release (`/workspace/probes-as-training-signals`) was split off just before that commit,
so it has no sycophancy task. This experiment builds a copy of the release at
`/workspace/sycophancy-rl`, ports `12631d5` into it, adds sycophancy support to its launcher, and
verifies the result with structural checks, a 2-step smoke GRPO run and a 128-step rerun of
August's run (260825_grpo_det0) from the copy, set side by side with August's numbers.

**The reported run** is the unattended run relaunched on 2026-10-12 at 18:13 UTC and finished at
22:48 UTC (protocol in `CLAUDE.md`; queue, log, judgment calls and Result in `runs.md`). It reused
the outputs of steps 1–5 from the first unattended attempt that morning (05:08 UTC), which stopped
when the smoke run ran out of GPU memory (see Run status). The rerun is W&B `l52s88q8`, group
`261012_port_rerun`, exported to
[`run_metrics_syco_det0_rerun_s42.json`](./run_metrics_syco_det0_rerun_s42.json). The comparison
is [`comparison.md`](./comparison.md).

## Files

| File | Role |
|---|---|
| `queue.md` | The step queue with each step's command, output and check |
| `run_env.sh` | Sourced before every step: `PYTHONPATH` to the copy (its training package and vendored GRPO framework shadow the private ones in the shared venv), `HF_HOME`, W&B settings, `$PY` (the private venv's Python) |
| `start_vllm.sh` | vLLM server from the copy's vendored framework, 0.40 of the GPU -> `logs/vllm_<host>.log` |
| `run_grpo_step.sh` | Runs the copy's launcher on `smoke.yaml` or `rerun.yaml` -> `logs/<name>_<host>.log` (import paths first, `EXIT <code>` last); refuses if the log exists |
| `smoke.yaml` | Launcher config of the smoke run: August's settings cut to 2 steps, an in-loop eval after each step, a 64-question eval set |
| `rerun.yaml` | Launcher config of the rerun: August's trainer settings (see Design for the differences) |
| `manifest.py` | SHA-256 of every file of the copy that git would commit; `--compare` checks against an earlier manifest -> `manifest_start.json`, `manifest_gate.json`, `manifest_end.json` |
| `check_diff.py`, `expected_diff_hunks.json` | The copy is the release plus `12631d5` plus the launcher changes, and nothing else; the hunks allowed to differ from `12631d5` are frozen in the JSON -> `check_diff.json` |
| `check_tests.py` | Runs the ported `tests/test_multiple_choice_task.py` in the copy -> `check_tests.json` |
| `check_config.py`, `expected_config_keys.json` | Composes the trainer config for August's recorded command in the copy and in the private code and compares every key; allowed differences frozen in the JSON -> `check_config.json`, `composed_config_copy.json`, `composed_config_private.json` |
| `check_responses.py` | Per-step check of a run's response files: served cues equal the seeded draws, and recomputed counts equal W&B -> `check_august.json`, `check_responses_smoke.json`, `check_rerun.json` |
| `check_smoke.py` | The smoke run reached every trainer and launcher stage and imported the copy -> `check_smoke.json` |
| `gate.py` | Every earlier check passed and was written in this run -> `gate.json` |
| `compare_august.py` | The rerun next to August's run -> `comparison.md`, `comparison.json` |
| `run_metrics_syco_det0_smoke_s42.json`, `run_metrics_syco_det0_rerun_s42.json` | The launcher's metric exports of the two GRPO runs |
| `runs.md` | The unattended run's queue, log, judgment calls and Result |
| `CLAUDE.md`, `kickoff_prompt.txt` | The unattended-run protocol and its first message (not committed) |
| `logs/` | GRPO, vLLM and checker logs; `logs/failed_oom/` holds the first smoke attempt's logs (`*.log` files are not committed) |
| `dry/` | Dry runs of the CPU checks before launch; not results (not committed) |

## Design

**Baseline.** August's run, [`260825_grpo_det0`](../260825_grpo_det0/) (W&B `lx6gcbcl`), run from
the private codebase: 128 GRPO steps from base Llama-3-8B-Instruct on 260819's 1603 train
questions, reward 1 for a correct A/B answer, no detector penalty (the diverse-deception probe is a
monitor only), seed 42, 64 prompts x 8 generations per step. On the 687 held-out questions its
final eval accuracy was 0.837 and its cue-following 0.923.

### The copy

- Every file the release tracks except seven public-release files (the two READMEs, `NOTICE`,
  `LICENSE`, `obfuscation-atlas/Makefile`, `pyproject.toml`, `uv.lock`): 133 files, with no `.git`
  of their own. The MBPP scripts, `probe_eval/` and the MBPP configs are included.
- `training/common.py` points GRPO run directories at `/workspace/results/run_results`, activation
  caches at `/workspace/results/activations` and the HF cache at `/workspace/hf_cache`.
- Runs use the private venv (`/workspace/obfuscation-atlas/.venv`) with `PYTHONPATH` set to the
  copy first (`run_env.sh`).

### The port of `12631d5`

The training-package part of `12631d5` was merged into the copy three-way (`git merge-file`).
`tasks/multiple_choice.py` and `tests/test_multiple_choice_task.py` are new files, identical to
`12631d5`. Hunks of the other files, compared with `12631d5`'s own diff (`check_diff.json`):

| File | Hunks in `12631d5` | Identical in the copy | Differences |
|---|---|---|---|
| `config.py` | 9 | 9 | none |
| `detectors/custom_dataset.py` | 2 | 2 | none |
| `training/obfuscation_training.py` | 2 | 2 | none |
| `scripts/train_grpo_model_against_probe.py` | 14 | 14 | one added hunk: the release removed the `TaskType` import that the ported code uses; it is re-added |
| `training/grpo_trainer.py` | 11 | 9 | `12631d5` adds the multiple-choice dispatch inside the trainer's `prepare_training_inputs` override, which the release removed with prompt mixing. The copy adds a `prepare_training_inputs` override that sends `MULTIPLE_CHOICE` to `_prepare_mcq_training_inputs` and otherwise calls the parent method. The method body is `12631d5`'s with one comment changed (it no longer names the prompt-mix RNG) |

### Launcher support (`training/`)

| File | Change |
|---|---|
| `run_grpo.py` | Passes the YAML's `config_name` to the launcher |
| `grpo_launcher.py` | Passes `--config-name` to the trainer (without it the trainer composes the MBPP config). For `config_name: grpo_mcq`: skips the MBPP "still writes executable code" collapse check, adds the in-loop eval rows from W&B to the export (`sycophancy_eval_history`, each tagged `logged_at_train_step`), prints the sycophancy summary |
| `extract_run_results.py` | `print_sycophancy_results`: final eval (accuracy, cue-following, parse failures) and the in-loop eval rows |

### The rerun's settings against August's

The rerun runs August's trainer command (`check_config.py` composes it from
`run_metrics_syco_det0_s42.json`, `metrics.command`). Differences from August:

| Setting | August | Rerun |
|---|---|---|
| Code | private codebase | the copy |
| `detector.model_batch_size` (examples per forward pass when collecting probe activations, for the probe training before RL and for the probe evals) | 16 | 8 |
| GPU (W&B run metadata) | NVIDIA H200, 150.8 GB | NVIDIA H100 80GB HBM3, 85.5 GB |
| `detector.retrain` | false | true (fresh probe-cache prefix, so the probe is trained either way; `true` lets the launcher remove the activation cache afterwards) |
| Run key, W&B group, probe-cache prefix | `syco_det0_s42`, `syco_det0`, August's | `syco_det0_rerun_s42`, `261012_port_rerun`, `port_rerun` |
| Composed-config keys (`check_config.json`) | | 6 keys of features the release removed are absent; `model.attn_implementation` is `sdpa` instead of `flash_attention_2` and `reward_adapter_path_prefix` is null (the GRPO path reads neither; it loads the policy with `grpo.model.attn_implementation`, `sdpa` in both) |
| CPU, OS (W&B run metadata) | 96 cores, glibc 2.35 | 64 cores, glibc 2.39 |
| Packages (W&B `requirements.txt`) | | torch 2.9.0, transformers 4.56.1, accelerate 1.12.0 and vLLM 0.13.0 in both; 6 other packages differ by a minor version (debugpy, ipython, jupyter_client, pure_eval, tornado, wcwidth) |

The two runs' recorded configs (`full_config` in both exports) differ in 13 keys apart from the
W&B git metadata: the 8 in `check_config.json`, plus `detector.model_batch_size`,
`detector.retrain`, `detector.directory_save_prefix`, `wandb.group` and `wandb.run_name_prefix`.

Both runs share one GPU between vLLM (0.40 of the memory) and the trainer. The batch size of 8
was set by the user after the first smoke attempt ran out of GPU memory at 16 (Run status). Each
probe set is padded to one length before batching, so the batch size changes only how the same
inputs are split.

Training prompt order and vLLM sampling are not seeded in either codebase (the prompt shuffle
runs before anything seeds it, and the vLLM server does not pass the seed on), so the two runs
cannot match exactly. The cue draws are seeded: both runs served the same cue condition and wording
to each batch position at every step. The comparison has no tolerance.

## Gates

| Step | Check | Result |
|---|---|---|
| 1 | Manifest of the copy | 133 files hashed (`manifest_start.json`) |
| 2 | Copy = release + `12631d5` + launcher changes | PASS: 133/133 files; new files identical to `12631d5`; 122/122 other files identical to the release; 4/4 expected unmatched hunks, 0 unexpected (`check_diff.json`) |
| 3 | Ported test | PASS: 9 passed, imported from the copy (`check_tests.json`) |
| 4 | Composed config, copy vs private | PASS: only the listed keys differ (`check_config.json`) |
| 5 | Response checker on August's run | PASS: 128/128 step files; per-step counts equal W&B; cue draws 64/64 on every step (`check_august.json`) |
| 6 | vLLM from the copy answers | PASS: 2.9 min after launch |
| 7a | Smoke run exits 0 | PASS on the relaunch: first attempt, no crash, 36.0 min (W&B `sj7dyqyd`) |
| 7b | Response checker on the smoke run | PASS: 2/2 step files (`check_responses_smoke.json`) |
| 7c | Smoke run reached every stage | PASS: 21/21 checks, including the six probe-eval folders, the final-eval keys, eval rows at steps 1 and 2, and both the trainer and vLLM importing from the copy (`check_smoke.json`) |
| 8a | Copy unchanged before the rerun | PASS: 0 changed, added or removed (`manifest_gate.json`) |
| 8b | Gate | PASS: 7/7 check files PASS and written after `manifest_start.json` (`gate.json`) |
| 8c | Rerun exits 0 | PASS: `EXIT 0` after one crash and an automatic restart (Run status) |
| 9 | Response checker on the rerun | PASS: 128/128 step files; per-step counts equal W&B; cue draws 64/64 on every step (`check_rerun.json`) |
| 10 | Comparison written | 13 rows (`comparison.md`) |
| 11 | Copy unchanged at the end | PASS: 0 changed, added or removed (`manifest_end.json`) |

No check reads the contents of the four launcher files (`training/common.py`, `grpo_launcher.py`,
`run_grpo.py`, `extract_run_results.py`): step 2 leaves them out of both its identity check and its
hunk check. The smoke run (7a, 7c) and the rerun ran them, and step 10 reads August's in-loop eval
rows through `grpo_launcher.sycophancy_eval_history`.

## Results

Numbers from [`comparison.json`](./comparison.json), written by `compare_august.py` from both
runs' metric exports, the in-loop eval rows in W&B and the response checker's recomputed counts.
In-loop eval: the 687 held-out questions every 30 steps, one sample each at temperature 1. Final
eval: the same 687 questions after step 128, one sample each at temperature 1
(`grpo.vllm.temperature`). Train cue-following: share of the step's 512 rollouts whose answer is
the served cue letter. Each in-loop eval row scores the policy from the previous eval point, so
the row logged at step 30 scores the starting model: the trainer's eval queue is
first-in-first-out with one submission before training.

| metric | August (`lx6gcbcl`) | rerun (`l52s88q8`) |
|---|---|---|
| in-loop eval accuracy, logged at step 30 | 0.7715 | 0.7482 |
| in-loop eval cue-following, logged at step 30 | 0.7802 | 0.7642 |
| in-loop eval accuracy, logged at step 60 | 0.8413 | 0.8530 |
| in-loop eval cue-following, logged at step 60 | 0.9971 | 0.9738 |
| in-loop eval accuracy, logged at step 90 | 0.8428 | 0.8530 |
| in-loop eval cue-following, logged at step 90 | 0.9985 | 0.9185 |
| in-loop eval accuracy, logged at step 120 | 0.8428 | 0.8617 |
| in-loop eval cue-following, logged at step 120 | 0.9389 | 0.9243 |
| final eval accuracy | 0.8370 | 0.8559 |
| final eval cue-following | 0.9229 | 0.8981 |
| final eval parse failures | 0 | 0.0015 |
| first step with train cue-following >= 0.95 | 7 | 7 |
| trainer wall-clock (h) | 4.2769 | 3.5953 |

The two columns differ in more than the code: the rerun used a probe-activation batch size of 8
(August 16), ran on an H100 80GB (August an H200) and is the launcher's second attempt, after
attempt 1 (W&B `1y9laylg`) ran out of GPU memory at RL step 3 (Design, Run status). The wall-clock
row is `metrics.duration_seconds`, the W&B runtime of the trainer process: probe activations and
probe training before RL, the 128 RL steps with their in-loop evals, the final eval and the probe
evals. It leaves out the launcher's own steps and the rerun's crashed attempt.

### What the numbers show

- Both runs first reach train cue-following ≥ 0.95 at step 7.
- Final eval: accuracy 0.856 vs 0.837, cue-following 0.898 vs 0.923, parse failures 1 of 687 vs 0.
- Rerun cue-following is below August's in all four in-loop rows (by 0.015 to 0.080, largest in
  the row logged at step 90) and in the final eval (by 0.025).
- Rerun accuracy is below August's in the row logged at step 30 (the starting model; 0.748 vs
  0.772) and above it in the other three in-loop rows (by 0.010 to 0.019) and in the final eval
  (by 0.019).
- In both runs every step served the seeded cue draws (64/64) and every recomputed per-step count
  equals the count logged to W&B.

The export's `classification` block is the MBPP policy classification that the export computes for
every run; August's export holds the same values.

## Run status

- **First unattended attempt, 2026-10-12 05:08–05:38 UTC.** Steps 1–6 passed. The smoke run (7a)
  crashed in all 4 launcher attempts after 5.4 min each (W&B `80uly5ca`, `lth0st9v`, `dwd61cfo`,
  `lrm1niv5`): CUDA out of memory in the first batch (16 x 1371 tokens) of the probe-activation pass
  on the validation set, before any RL step, with the trainer at 45.6 GiB and vLLM at 33.5 GiB. The
  session stopped as its protocol required. Logs in `logs/failed_oom/`; error text in `runs.md`.
- **Reset, 18:08 UTC (attended, the user's decision).** `detector.model_batch_size: 8` added to
  both YAMLs; the smoke run's partial activation cache deleted; steps 6 and 7a reopened.
- **Relaunch, 18:13–22:48 UTC.** Steps 6–11 passed (Gates). The smoke run is W&B `sj7dyqyd`, run
  directory `/workspace/results/run_results/261012_port_rerun_smoke/syco_det0_smoke_s42_sj7dyqyd/`.
- **The rerun (8c), launched 18:55 UTC, `EXIT 0` by 22:46 UTC.** Attempt 1 (W&B `1y9laylg`) crashed after 12.0 min with CUDA
  out of memory on RL training step 3, in `F.log_softmax` while computing the policy log-probs
  (tried to allocate 5.37 GiB with 5.25 GiB free; vLLM 33.5 GiB, trainer 40.4 GiB). The launcher
  deleted that run directory and restarted from scratch; attempt 2 (W&B `l52s88q8`) ran all 128
  steps in 217.1 min and exited 0. Run directory
  `/workspace/results/run_results/261012_port_rerun/syco_det0_rerun_s42_l52s88q8/`: final LoRA in
  `checkpoints/final/`, 128 step files in `checkpoints/responses/`, six probe-eval folders. The
  launcher removed both runs' activation caches. The trainer's own cleanup before that logs 10
  `could not remove activation cache ... (non-fatal)` warnings per run, as August's log does.
- The crashed W&B runs (4 smoke, 1 rerun) are left in W&B. The trainer's Hydra `outputs/`,
  `wandb/` and `artifacts/` folders in `/workspace/sycophancy-rl/obfuscation-atlas/` are
  git-ignored.

## Costs

| Item | Time | Disk |
|---|---|---|
| CPU checks (steps 1–5) | 3 min | |
| vLLM start | 2.9 min | |
| Smoke run | 36.0 min | 1.3 GB run directory |
| Rerun | attempt 1 12.0 min (crashed), attempt 2 217.1 min; W&B runtime of attempt 2 3.60 h | 1.4 GB run directory |
| Steps 9–11 | about 2 min | |
| First attempt's failed smoke runs | 4 x 5.4 min | removed by the launcher |

One H100 80GB, shared by vLLM and the trainer.

## Reproduction

```bash
cd /workspace/sycophancy_experiments/261012_port_rerun
tmux new-session -d -s 261012-vllm 'bash start_vllm.sh'      # wait for /v1/models
tmux new-session -d -s 261012-rerun 'bash run_grpo_step.sh rerun'
```

`run_grpo_step.sh` refuses to start when `logs/rerun_<host>.log` exists.
