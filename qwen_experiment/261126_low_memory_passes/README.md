# 261126 — Lower-memory training passes and no unused probe work for Qwen3.5-9B

Two new trainer settings, tested on base Qwen3.5-9B with a frozen probe penalty on one H100: a memory test of the
longest training pass the trainer allows, and 3-step training runs. The sycophancy runs: settings off at 4 answers
per pass, settings on at 4 and at 8, and a rerun of settings on at 4 on main's reorganized code. The MBPP run:
settings on at 4. Run 2026-11-25 and 2026-11-26 (PST) in W&B group
`261126_low_memory_passes`. The run numbers are in `results_tables.md` (written by `results.py`), the memory test's
in `memory/memory.json`.

## Setup

**The two settings.** Both are off by default; a run's config turns them on (`probe-rl/README.md`, Models).
- *Low-memory passes* (`grpo.model.low_memory_passes`). Without it, the policy and reference passes compute float32
  log-probs over the whole vocabulary (248,320 entries) at every position. With it, they compute them only at the
  answer tokens, one answer at a time. The probe's scoring pass also skips the output head, and it stops one layer
  past the last probe layer: after layer 22 of 32 for the sycophancy probe (layers 18–21). MBPP's probe reads all 32
  layers, so its scoring pass still runs every layer.
- *Skipping unused probe work* (`detector.skip_unused_probe_work`). This skips probe work whose results no later step
  reads:
  - the probe fit at the start of a run, when frozen probe weights are given (they replace the fitted probe);
  - the probe refit and the probe eval after training.

  The held-out answers generated after training, which the launcher checks, are still generated. Without the
  start-of-run fit, an MBPP run does not log two W&B breakdown metrics that read the fit's threshold
  (`obfuscation/detector_mean_decision_threshold`, `obfuscation/obfuscated_policy_margin`).

**Why.** On an H100, two earlier 9B sycophancy runs at 4 answers per pass ran out of GPU memory
(`qwen_experiment/261125_9b_frozen_probe_300/`): the seed-43 frozen-probe run, which was then rerun at 2, and a
no-penalty run continued to 300 steps. Every frozen-probe run of that experiment also fitted a probe at its start
that the frozen weights then replaced. After training, each run refit and evaluated a probe that nothing reads.

**The memory rule.** Set before the test: a training pass fits if, at its peak, it leaves at least 4 GiB of what the
vLLM server leaves free. The server leaves 43.80 GiB when it holds the most memory seen: PyTorch's 79.18 GiB total
minus the 35.38 GiB the server held when the no-penalty run above ran out of memory. The trainer's CUDA context takes
another 0.98 GiB, measured in the memory test.

**What is compared.**
- **Memory test:** one training pass with low-memory passes off and on, at 4 and at 8 answers per pass.
- **Sycophancy:** 3-step runs of one config, each a single run:
  - *settings off*: both settings off, 4 answers per pass;
  - *settings on*: both settings on, 4 answers per pass;
  - *8 per pass*: both settings on, 8 answers per pass;
  - *rerun on main*: settings on at 4 again, after this code was moved onto main's reorganized code (Checks).

  The config is the seed-42 frozen-probe sycophancy run's (261125): penalty weight 1.0, the probe frozen from 261122's
  no-penalty sycophancy run (layers 18–21), with 3 steps instead of 300.
- **MBPP:** one 3-step run with both settings on, 4 answers per pass (*MBPP, settings on*). It is compared with the
  first 3 steps of the 300-step seed-43 frozen-probe MBPP run whose config it copies (*MBPP, earlier run*; 261125):
  penalty weight 3, the probe frozen from 261122's no-penalty MBPP run, reading all 32 layers at the end of the graded
  code block.
  - The earlier run had neither setting (they did not exist yet) and also ran at 4 answers per pass.
  - It ran on another H100 pod, started 2026-11-24 PST with the code of that day.
  - It finished its 300 steps without running out of memory.

**Carried over from those runs:**
- 64 questions × 8 answers = 512 answers per step, and 64 answers per optimizer step;
- answers capped at 1024 tokens;
- each step's answers sorted by length before they are split into passes, and cuDNN attention off (both set by the
  9B model preset, `+model_preset: qwen3_5_9b`);
- the vLLM server on the same GPU, at 0.40 of its memory.

The GPU is one H100 80GB: 79.6 GiB by `nvidia-smi`, 79.18 GiB as PyTorch counts it.

| Measurement | What it is |
|---|---|
| Peak reserved memory | Memory test: PyTorch's peak reserved GPU memory during one training pass (probe scoring, reference pass, policy pass and backward). The pass has 4 or 8 answers of 1024 prompt + 1024 answer tokens, the longest the trainer allows, and the optimizer's state is already in memory. |
| Spare memory by the rule | 43.80 GiB − peak reserved memory − 0.98 GiB (the memory rule above; it needs at least 4 GiB). |
| Trainer time | The launcher's time for the trainer process, start to exit. The vLLM server's start comes before it. |
| Before step 1 | From the trainer's first log line to the start of step 1. |
| After the last step | From the end of the last step to the trainer's exit. It also counts the trainer's imports before its first log line, since the log has no earlier line to start "before step 1" from. |
| Step time | The time between consecutive steps' timing lines in the trainer's log, for steps 2 and 3. |
| Training pass | The trainer's timer for everything in a step after generation: the rewards (probe scoring included, and for MBPP running the answers' code), the reference pass, the policy passes and backward, and the optimizer step. |
| Log-prob correlation | W&B `logprob_corr_mean`: the correlation between the trainer's and vLLM's log-probs of the sampled tokens. |
| Peak GPU memory | The most GPU memory in use by `nvidia-smi`, sampled every 5 s. It includes vLLM and PyTorch's cache, so it is an upper bound on what training needs. |

## Main results

**Memory test** (sycophancy, next to a vLLM server at 0.40 of the H100):

| Answers per pass | Low-memory passes | Peak reserved memory GiB | Spare memory by the rule GiB |
|---|---|---|---|
| 4 | off | out of memory | – |
| 4 | on | 29.7 | 13.1 |
| 8 | off | out of memory | – |
| 8 | on | 37.5 | 5.3 |

**3-step runs:**

| Run | Answers per pass | Trainer time min | Before step 1 min | After the last step min | Step time s (2 / 3) | Training pass s (1 / 2 / 3) | Log-prob correlation (1 / 2 / 3) | Peak GPU memory GiB of 79.6 |
|---|---|---|---|---|---|---|---|---|
| Sycophancy, settings off | 4 | 38.1 | 9.3 | 21.4 | 140 / 155 | 107.7 / 99.5 / 109.7 | 0.99948 / 0.99943 / 0.99947 | 79.1 |
| Sycophancy, settings on | 4 | 10.1 | 0.5 | 2.6 | 136 / 141 | 96.6 / 93.2 / 97.9 | 0.99947 / 0.99945 / 0.99947 | 65.8 |
| Sycophancy, 8 per pass | 8 | 9.4 | 0.5 | 2.7 | 134 / 112 | 88.2 / 89.6 / 75.9 | 0.99948 / 0.99948 / 0.99948 | 72.6 |
| MBPP, earlier run | 4 | – (300 steps) | 5.2 | 9.7 (after step 300) | 134 / 137 | 311.9 / 102.1 / 103.0 | 0.99927 / 0.99931 / 0.99930 | not logged |
| MBPP, settings on | 4 | 12.1 | 0.7 | 2.5 | 138 / 133 | 232.5 / 107.5 / 102.7 | 0.99928 / 0.99927 / 0.99930 | 66.9 |

Memory test: `memory/memory.json`. Runs: the trainer's logs (`logs/rl_<run>_a1_<host>.log`; the earlier MBPP run's is
in 261125's `seed_43/logs/`), W&B and the `nvidia-smi` logs, all in `results_tables.md`.

1. **Without low-memory passes, the longest possible 9B sycophancy pass doesn't fit on an H100 next to vLLM, at 4 or
   at 8 answers per pass. With it, both fit.**
   - Peak reserved memory: 29.7 GiB at 4 and 37.5 GiB at 8.
   - That leaves 13.1 and 5.3 GiB of spare memory by the rule, which needs 4 GiB.
   - So 9B sycophancy can run on an H100 at 4 or 8 answers per pass, where the seed-43 frozen-probe run had to drop
     to 2.
2. **The settings don't change what the trainer computes.**
   - On the same input (random tokens, no gradients) with low-memory passes off and on, the log-probs were identical
     on 1,524 answer tokens, and the probe read bit-for-bit the same activations (More results, Memory test, Numbers
     check). The trainer test checks the gradients too, to within rounding (Method details).
   - The log-prob correlation was 0.99943–0.99948 in every sycophancy run in the table, settings off or on (the rerun
     on main: 0.99940–0.99947, Checks).
   - In every sycophancy run, the probe penalty logged in W&B matches the penalty recomputed from the saved answers.
     In the MBPP run it equals −3 × W&B's mean probe score (Checks).
3. **With both settings on, a 3-step sycophancy run took 10.1 min of trainer time instead of 38.1.**
   - Before step 1: 0.5 against 9.3 min.
   - After the last step: 2.6 against 21.4 min.
   - Step time: 136 / 141 s against 140 / 155 s.
   - So most of the saving is per run (the skipped probe work), not per step. The two settings were not timed
     separately.
4. **For MBPP at 4 answers per pass, the step time is about the same with the settings on, and the run's start and
   end are shorter.**
   - Step time: 138 / 133 s against the earlier run's 134 / 137 s.
   - Before step 1: 0.7 against 5.2 min.
   - After the last step: 2.5 against 9.7 min. The earlier run's 9.7 min come after its step 300, not step 3.
   - Log-prob correlation: 0.99927–0.99930 against 0.99927–0.99931.
5. **At 8 answers per pass (sycophancy, settings on), the training pass took 76–90 s per step, against 93–98 s at
   4.**
   - Step time: 134 / 112 s against 136 / 141 s.
   - The rerun on main at 4 took 94.7 / 98.9 / 84.6 s (More results, Step timings), so at step 3 the two overlap.
   - Peak GPU memory: 72.6 against 65.8 GiB of 79.6.
   - So 8 per pass is somewhat faster. This comes from 3 steps only.

**Limits.**
- One run and 3 steps per setting.
- Training-prompt order is not seeded, so every run drew different questions at every step, even two runs of the
  same config (Checks). The step times include that variation.
- The two settings were not timed separately.
- The MBPP comparison is against an earlier run on another pod, with the code of the day before. No settings-off
  MBPP run was made on this pod.
- 8 answers per pass was not tested on MBPP.
- The memory test measures one pass, not a long run. The settings-off run did not run out of memory in its 3 steps
  (peak GPU memory 79.1 of 79.6 GiB).

## More results

### Memory test

| Answers per pass (1024 + 1024 tokens) | Low-memory passes | Peak reserved GiB | Peak allocated GiB | Free GPU memory after the pass GiB |
|---|---|---|---|---|
| 4 | on | 29.69 | 28.11 | 18.49 |
| 8 | on | 37.50 | 34.66 | 10.41 |
| 4 | off | out of memory | | |
| 8 | off | out of memory | | |

- **Free memory.** The GPU had 49.38 GiB free when the test started CUDA, 48.40 GiB right after (the CUDA context,
  0.98 GiB), and the model took 17.66 GiB once loaded.
  - The test's vLLM server left more memory free than the 43.80 GiB the rule assumes. So the measured free memory
    after the pass (18.49 and 10.41 GiB) is larger than the spare memory by the rule (13.1 and 5.3 GiB).
- **Numbers check** (low-memory passes off against on, on the same 2 answers, no gradients):
  - The reference and policy log-probs over the 1,524 answer tokens: largest difference 0.0, correlation 1.0.
  - The hidden states the probe reads, and the reference pass's hidden states: bit-for-bit equal.

### Step timings

Seconds per step, from the trainer's step timings.

| Run | Step | Generation | Rewards + probe scoring | Reference pass | Policy pass | Backward | Training pass | Step time |
|---|---|---|---|---|---|---|---|---|
| Sycophancy, settings off | 1 | 42.6 | 23.0 | 13.3 | 21.1 | 41.7 | 107.7 | |
| | 2 | 41.0 | 22.0 | 12.6 | 19.3 | 38.0 | 99.5 | 140 |
| | 3 | 45.6 | 23.7 | 13.7 | 21.5 | 41.7 | 109.7 | 155 |
| Sycophancy, settings on | 1 | 43.6 | 16.3 | 16.1 | 19.8 | 39.8 | 96.6 | |
| | 2 | 42.4 | 14.8 | 15.8 | 20.0 | 38.1 | 93.2 | 136 |
| | 3 | 43.7 | 15.8 | 16.6 | 20.4 | 40.2 | 97.9 | 141 |
| Sycophancy, 8 per pass | 1 | 39.4 | 14.3 | 13.6 | 15.7 | 40.1 | 88.2 | |
| | 2 | 44.7 | 15.4 | 15.3 | 18.1 | 36.0 | 89.6 | 134 |
| | 3 | 36.5 | 13.4 | 12.9 | 15.3 | 30.2 | 75.9 | 112 |
| Sycophancy, rerun on main | 1 | 42.5 | 15.7 | 15.8 | 19.8 | 38.9 | 94.7 | |
| | 2 | 42.4 | 15.9 | 16.6 | 21.3 | 40.5 | 98.9 | 142 |
| | 3 | 37.0 | 13.4 | 14.3 | 18.4 | 34.8 | 84.6 | 121 |
| MBPP, earlier run | 1 | 34.6 | 34.4 | 11.1 | 17.4 | 94.9 | 311.9 | |
| | 2 | 31.9 | 19.6 | 11.1 | 16.4 | 32.6 | 102.1 | 134 |
| | 3 | 34.2 | 19.6 | 11.5 | 17.9 | 33.2 | 103.0 | 137 |
| MBPP, settings on | 1 | 31.4 | 18.4 | 12.7 | 17.5 | 32.5 | 232.5 | |
| | 2 | 30.2 | 16.8 | 12.4 | 16.9 | 31.8 | 107.5 | 138 |
| | 3 | 30.1 | 17.7 | 12.8 | 17.3 | 32.3 | 102.7 | 133 |

Notes for reading the table:
- **Training pass.** It includes the four timed parts to its left and the rest of the step's training work. For
  MBPP that includes running the answers' code (the trainer's `compute_sequence_reward` timer).
- **MBPP's step 1.** Its training pass is longer than its later ones in both MBPP runs: 232.5 s with the settings on,
  311.9 s in the earlier run. Running the answers' code took 134.3 and 134.6 s at step 1, against 11.5 / 5.3 s and
  3.5 / 1.3 s at steps 2 / 3.
- **Reference pass.** It took 14.3–16.6 s per sycophancy step at 4 per pass with low-memory passes on, against
  12.6–13.7 s off. The cause was not looked into.
- **Padding share** (W&B `padding_share`, the share of padding tokens in the passes): 0.6–1.0% at 4 answers per pass
  in both tasks, 1.6–1.8% at 8.
- **Longest answer.** Every step of every run had an answer at the 1024-token cap.

## Method details

### Code changes (shared code, `probe-rl/obfuscation-atlas/`)

| File | Change |
|---|---|
| `third_party/afterburner/src/afterburner/training/interface.py` | `compute_answer_logprobs_from_hidden`: float32 log-probs from the last hidden states, at answer tokens only, one answer at a time |
| `obfuscation_atlas/training/grpo_config.py` | The setting `GRPOModelConfig.low_memory_passes` (default false) |
| `obfuscation_atlas/training/grpo_trainer.py` | `probe_hidden_states` keeps only the hidden states the probe code reads. HFActivationModel gains `answer_logprobs_with_probe_activations` (reference and policy passes) and `probe_activations` (the probe-scoring pass: no output head, and the model stops one layer past the last probe layer, because transformers replaces the last hidden state it returns with the final norm's output). The trainer's `probe_scoring_hidden_states` and `training_pass_logprobs` call these when the setting is on and the old code when it is off. Both new passes assert a single training process, since they call the decoder and output head directly |
| `obfuscation_atlas/config.py` | The setting `DetectorConfig.skip_unused_probe_work` (default false). It needs `task_args.bias_probe_at_answer` or `task_args.mbpp_probe_at_code_end` |
| `obfuscation_atlas/scripts/train_grpo_model_against_probe.py` | The two skips. Each logs a line: "Phase-0: no start-of-run probe fit (...)" and "No after-training probe refit or probe eval (...)" |
| `tests/test_multiple_choice_task.py` | `test_low_memory_passes_and_skip_unused_probe_work_default_off`; `test_low_memory_passes_give_todays_logprobs_gradients_and_probe_inputs` (on a small model on the CPU, setting off against on: log-probs and gradients equal to within rounding, `assert_close`; probe inputs exactly equal) |
| `probe-rl/README.md` | The two settings, under Models |

A run with the skip on has no `fresh_detector.final_model.on_dataset/` folder. Main's
`sycophancy_setup/eval_checkpoint.py` loads that folder when it evaluates an adapter, so such a run cannot be the run
whose probe that eval loads.

With these changes the launcher runs Qwen3.5-9B at 4 answers per pass on an H100, as before (`QWEN_9B_ANSWERS_PER_PASS` in
`probe-rl/launcher/grpo_launcher.py`). A run sets 8 with `training.batch_size: 8` and `training.grad_accum_steps: 8`,
as the 8-per-pass run does.

### Memory test (`memory_test.py`)

- **Setup.** The model, adapters and optimizer come from the trainer's own setup (`HfObfuscationTrainerState`):
  - bf16 weights;
  - LoRA rank 64 / alpha 128 policy and reference adapters;
  - gradient checkpointing and AdamW;
  - cuDNN attention off, as in the 9B preset.
- **The passes.** They run through the trainer's helpers (`probe_scoring_hidden_states`, `training_pass_logprobs`):
  1. the probe-scoring forward, probe layers 18–21;
  2. the reference forward, kept through the policy pass as the trainer keeps it;
  3. the policy forward and backward.

  A first pass and an optimizer step create the gradients and AdamW's state. The measured pass is the second.
- **Input.** 4 or 8 answers of 1024 prompt + 1024 answer tokens, random token ids. That is longer than the sequences
  in the two earlier out-of-memory crashes: about 1,550 and 1,590 tokens, from the size of the float32 log-prob
  allocation that failed (5.72 and 5.88 GiB for 4 answers over 248,320 entries).
- **The vLLM server.** The test ran next to the runs' vLLM server at 0.40 of the GPU (`memtest_server.yaml` is
  `lmp_on4.yaml` under another name, so that the server's log gets its own name).
- **Numbers check.** 2 answers (1024 + 1024 and 300 + 500 tokens, right-padded), each pass with the setting off and
  on, without gradients. The results are reported, not asserted.

### Runs

Every run has its own vLLM server, started before the trainer and stopped after it.

| Run key | W&B | Name in the tables | Config |
|---|---|---|---|
| `lmp_off4` | `zz7qdwgm` | Sycophancy, settings off | 261125's `seed_42/fz9b_s42.yaml` with `training.max_steps: 3` and its own run key, output folder and W&B group (the only differences) |
| `lmp_on4` | `4jgckzbt` | Sycophancy, settings on | `lmp_off4` + `grpo.model.low_memory_passes: true` and `detector.skip_unused_probe_work: true` |
| `lmp_on8` | `lwts2w0j` | Sycophancy, 8 per pass | `lmp_on4` + `training.batch_size: 8` and `training.grad_accum_steps: 8` |
| `lmp_on4_main` | `6lkukfwv` | Sycophancy, rerun on main | `lmp_on4` with its own run key |
| `lmp_mbpp_on4` | `ae3enf03` | MBPP, settings on | 261125's `seed_43/fzmbpp9b_s43.yaml` with `training.max_steps: 3`, its own run key, output folder and W&B group, and both settings on |
| `fzmbpp9b_s43` (261125) | `6mn5758c` | MBPP, earlier run | Its steps 1–3 of 300, from 261125's `seed_43/logs/rl_fzmbpp9b_s43_a1_7c244d9935a2.log`, H100 pod 7c244d9935a2 |

The memory test and the first three runs (`lmp_on4`, `lmp_off4`, `lmp_on8`, the night of 2026-11-25) ran on this
code before main's reorganization:
- the code was on main 4bca565, when `probe-rl/` was still `sycophancy-rl/`;
- each was launched as 261123's `run_rl_noretry.sh` launches, through `training.run_grpo` with the launcher's crash
  retries set to 0;
- the vLLM server came from `start_vllm_gpu.sh`.

The two runs of 2026-11-26 (`lmp_on4_main`, `lmp_mbpp_on4`) ran on main be6d216 (`probe-rl/`, `launcher.run_grpo`),
through `run_rl_worktree.sh` and `queue.sh merged`. On that main the launcher no longer retries a crash.

### How the tables are computed (`results.py`)

- **Step timings:** from each run's trainer log; step time is the difference between consecutive "Step N timings"
  lines.
- **Before step 1:** step 1's logged time, minus its generation and training pass, minus the trainer's first log
  line.
- **After the last step:** the launcher's "GRPO subprocess finished in X min", minus the time from the trainer's first
  log line to "Training completed!". The earlier MBPP run's "after the last step" is computed the same way, from its
  300-step log.
- **From W&B** (project `deception_detection`, by `grpo_train_step`): the log-prob correlation, padding share and
  longest answer.
- **Peak GPU memory:** the largest value in `logs/gpu_mem_<run>_<host>.log`.
- **The penalty check:** described under Checks.

## Checks

- **Trainer tests.** With these changes (branch `efficiency` on main be6d216), `tests/test_multiple_choice_task.py`
  gives 87 passed and 2 skipped. Main be6d216 without them gives 85 passed and the same 2 skipped; the 2 skipped
  tests need 261010's activation files, which are not on the volume. The test command in
  `.claude/rules/shared-code.md` also names `tests/test_checkpoint_evals.py`, which is not on main be6d216.
- **Rerun on main's reorganized code.** The settings-on sycophancy run, made again after the move onto main
  (*rerun on main*), against the first:
  - trainer time 9.9 against 10.1 min;
  - before step 1 0.5 against 0.5 min, and after the last step 2.7 against 2.6 min;
  - step time 142 / 121 against 136 / 141 s, and training pass 94.7 / 98.9 / 84.6 against 96.6 / 93.2 / 97.9 s;
  - log-prob correlation 0.99940 / 0.99947 / 0.99947 against 0.99947 / 0.99945 / 0.99947;
  - peak GPU memory 65.8 GiB in both.
- **Probe penalty.** For each run and step, `results.py` compares W&B's `rewards/detector` (the step's mean penalty
  over its 512 answers) with the penalty recomputed from the run's saved answers:
  - Sycophancy (weight 1.0): −1.0 × the summed probe score of the finished answers that give the cued letter, divided
    by the step's answer count (`checkpoints/responses/step_<N − 1>.jsonl`).
  - MBPP (weight 3): every answer is penalized, so −3 × the step's mean probe score (W&B
    `rewards_metrics/detector_score`). MBPP's saved answers hold no probe score.

  The 12 sycophancy run-steps match to 5 decimals, and so do MBPP's 3, though MBPP's compares W&B with W&B
  (`results_tables.md`, Probe penalty check).
- **The skips.**
  - The logs of the four settings-on runs each have the two skip lines; the settings-off run's log has neither.
  - The settings-on run folders hold only `checkpoints/`. The settings-off run's also holds
    `initial_detector.base_model.on_dataset/`, `initial_detector.final_model.on_dataset/` and
    `fresh_detector.final_model.on_dataset/`.
- **The launcher's run checks.** All five runs passed them, with launcher exit code 0 and no out-of-memory error:
  - each run folder has a final adapter and 3 answer files;
  - the metrics file was written;
  - each sycophancy run has its final eval keys;
  - 97.5% of the MBPP run's held-out answers run as code (`val_does_execute`; the launcher's floor is 30.0%).

  The earlier MBPP run's log ends with exit code 1. That comes from the launcher's summary print after training,
  which main's launcher no longer runs.
- **Question order.** Two sycophancy runs of this folder share 0–7 of the 64 questions at the same step, the
  settings-on run and its rerun (the same config) included. The settings-off run's questions also differ from those
  of 261125's 300-step run of the same config (`fz9b_s42`). Training-prompt order is not seeded.

## Files

| File | Role |
|---|---|
| `README.md` | This write-up |
| `folder_env.sh` | Sourced by the scripts: `/workspace/probe-rl/env.sh`, then the code of the `/workspace/eff-wip` checkout, where these changes were built, first on `PYTHONPATH`. `queue.sh` and `run_rl_worktree.sh` source it by that checkout's path. To rerun from main, set its `CODE` to `/workspace/probe-rl` and its `F` to this folder under `/workspace/qwen_experiment/`, and source it from there |
| `queue.sh` | The GPU tests: the memory test, then `lmp_on4`, `lmp_off4` and `lmp_on8` (the last only if the memory test leaves 4 GiB free at 8 per pass). `bash queue.sh merged [run]` runs one run (2026-11-26). It starts each run's vLLM server and logs `nvidia-smi` every 5 s |
| `run_rl_worktree.sh` | Runs the launcher on `<run>.yaml` from this checkout; writes `logs/rl_<run>_a<attempt>_<host>.log` |
| `start_vllm_gpu.sh` | The vLLM server script of the 2026-11-25 runs (paths from before the reorganization; `queue.sh` as saved uses `probe-rl/launcher/start_vllm.sh`) |
| `memory_test.py` | The memory test and numbers check; writes `memory/memory.json` |
| `memtest_server.yaml` | `lmp_on4.yaml` under another name, for the memory test's vLLM server |
| `lmp_off4.yaml`, `lmp_on4.yaml`, `lmp_on8.yaml`, `lmp_on4_main.yaml`, `lmp_mbpp_on4.yaml` | The run configs (Method details, Runs) |
| `results.py` | Writes `results_tables.md` from the logs, `memory/memory.json`, W&B and the saved answers |
| `results_tables.md` | Every table's numbers, the penalty check included |
| `memory/memory.json` | The memory test's results |
| `smoke/memory.json` | The memory test's smoke run (1 answer of 256 + 256 tokens per setting, without a vLLM server) |
| `run_metrics_<run>.json` | The launcher's metrics file per run (W&B summary, final eval) |
| `logs/queue_<host>.log`, `logs/queue_merged_<host>.log`, `logs/queue_mbpp_<host>.log` | The queues' step lines with times (UTC) |
| `logs/rl_<run>_a1_<host>.log` | Each run's launcher and trainer log, ending with `EXIT <code>` |
| `logs/vllm_<run>_a1_<host>.log`, `logs/vllm_memtest_server_a1_<host>.log` | Each run's and the memory test's vLLM server log |
| `logs/gpu_mem_<run>_<host>.log` | `nvidia-smi` memory in use, every 5 s |
| `logs/memory_test_<host>.log`, `logs/smoke_<host>.log` | The memory test's and its smoke run's output |
| `logs/<run>_<host>.log` | The queue's output file for each run step; empty, since the run writes to its rl log |

## Run status

All on pod fdc2d68d6537 (one H100), each completed with launcher exit code 0. Run folders are
`/workspace/results/run_results/261126_low_memory_passes/<run key>_<W&B id>/`, each with its final adapter
(`checkpoints/final/`) and answer records (`checkpoints/responses/step_0–2.jsonl`). 3 steps save no weights between
steps or training states, so the `cleanup-runs` skill leaves these folders as they are. The times below are the
queue's run steps, after each run's vLLM server was up.
- **Memory test:** `queue.sh`, 2026-11-25 23:09–23:11 PST; `memory/memory.json`.
- **`lmp_on4`:** `queue.sh`, 2026-11-25 23:17–23:32 PST.
- **`lmp_off4`:** `queue.sh`, 2026-11-25 23:38 to 2026-11-26 00:20 PST.
- **`lmp_on8`:** `queue.sh`, 2026-11-26 00:27–00:41 PST.
- **`lmp_on4_main`:** `queue.sh merged`, 2026-11-26 14:46–15:00 PST.
- **`lmp_mbpp_on4`:** `queue.sh merged lmp_mbpp_on4`, 2026-11-26 16:27–16:41 PST.

Each run's metrics are in `run_metrics_<run>.json` and in W&B (group `261126_low_memory_passes`, project
`deception_detection`).

## Costs

- **GPU:** about 2.3 h of one H100.
  - The 2026-11-25 queue took 23:02 to 00:41 PST, vLLM starts included.
  - The two runs of 2026-11-26 took 20 min each, their vLLM starts included.
- **Disk:** 3.3 GiB for the five run folders (about 684 MiB each, nearly all of it the final adapter). This folder:
  1.9 MiB.
