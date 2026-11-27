# probe-rl: the training code for new runs

RL training against linear probes, for the sycophancy (MMLU-Pro) and MBPP reward-hacking tasks. It started as a copy
of the MBPP workshop paper's public release (`/workspace/probes-as-training-signals/`, on the volume and not in this
repo's git) with the sycophancy task added, and was called `sycophancy-rl/` until 2026-11-26; that name is a link to
this folder, so older paths still resolve. The top-level `/workspace/obfuscation-atlas/` and `/workspace/pipeline/`
are the paper-era MBPP code; new work does not use them.

## What is where

| Part | What it is | Used by |
|---|---|---|
| `obfuscation-atlas/` | The trainer: GRPO with a probe penalty, the MBPP task (`obfuscation_atlas/tasks/code_generation.py`) and the sycophancy task (`tasks/multiple_choice.py`). Settings are dataclasses in `obfuscation_atlas/config.py`; the tests for this copy's changes are `tests/test_multiple_choice_task.py` | Every run |
| `launcher/` | Runs one training job from a launch config (below): `run_grpo.py`, `grpo_launcher.py`, `common.py`; the run scripts `run_rl.sh` (one run, in tmux) and `start_vllm.sh` (its vLLM server); and `queue_common.sh`, helpers for a queue of steps | Every run |
| `sycophancy_setup/` | Sycophancy data (question file, probe set), the checkpoint eval, the result tables; see [its README](sycophancy_setup/README.md) | Sycophancy |
| `release_mbpp_llama/` | The release's MBPP multi-round pipeline (`training/`, `probe_eval/`, `experiments/`), unchanged. Llama-3 only; it runs from the git tag `llama-final` | Not used by new runs |
| `env.sh`, `requirements.txt` | The environment for every task, and the pinned packages of its venv (below) | Every run |

## Running one training job

`bash launcher/run_rl.sh <experiment folder> <run> <attempt>` runs the launcher on `<folder>/<run>.yaml` and exits with
its exit code; the run's vLLM server comes first (`bash launcher/start_vllm.sh <folder> <run> <attempt> <gpu>`). A
launch config has:
- `run_key`, `group` (the W&B group, and the run folder's parent), `output_dir` (absolute);
- `probe: {directory_save_prefix: <name>, retrain: true}` (the start-of-run probe, trained before RL);
- `checkpoint`: an absolute adapter path to start from, or null for the base model;
- `overrides`: Hydra key → value, which must set `+model_preset` and `training.max_steps`;
- for the sycophancy task, `config_name: grpo_mcq` (without it the trainer runs MBPP).

The launcher refuses any other key. After training it checks the run folder (a final adapter and exactly
`training.max_steps` response files, less those of a resumed state), cleans it up, exports the W&B metrics, and only
then writes `<output_dir>/run_metrics_<run_key>.json`, the file that marks the run as done:
- **A launch whose `run_metrics` file exists** is skipped.
- **A crash** stops the launch and keeps the run folder, saved training states included, marked with a `CRASHED`
  file. A marked folder doesn't stop a later launch of the same run key, so a retry (for example with fewer answers
  per pass after running out of GPU memory) starts a new run folder next to it.
- **A launch that finds another run folder of its run key without a `run_metrics` file** stops instead of training
  again. That run is still training (perhaps on another pod), was killed before the launcher could mark it (a pod
  reboot), or finished training but its checks or export failed. `python -m launcher.run_grpo <config>
  --finish-run-id <id>` finishes the last; otherwise move the folder away to train again.
- **An MBPP run whose final policy stopped writing executable code** (`val_does_execute` under 0.3) keeps its metrics
  as `run_metrics_<run_key>.DEGENERATE_<run id>.json` and is run again, at most 3 more times.

A run that continues another from its saved state needs its own run key.

## Environment

Before every step, `source /workspace/probe-rl/env.sh`. Sycophancy steps source `sycophancy_setup/run_env.sh` instead,
which sources `env.sh`. It sets `$COPY` to the folder `env.sh` is in, so another checkout of the code (a git worktree)
runs its own files. It sets `$PY` to the copy's own venv (`.venv`, pinned in `requirements.txt`: torch 2.14,
transformers 5.20, vLLM 0.33, peft 0.21, flash-linear-attention 0.6; Qwen3.5 needs transformers 5 and vLLM 0.17 or
later). It puts this copy's launcher, trainer and GRPO library (`afterburner`) first on `PYTHONPATH`, and sets the
Hugging Face cache, the W&B project and the secrets.

The shared venv at `/workspace/obfuscation-atlas/.venv` (transformers 4.56, vLLM 0.13) runs the code at the git tag
`llama-final`, the last commit before the Qwen3.5 changes, for exact reruns of the Llama experiments. With it,
`PYTHONPATH` must also put the copy first: that venv's packages point at the top-level checkout, whose older GRPO
library crashes at the first GRPO step.

## Models

| Preset (`+model_preset`) | Model | Response cap | Batch per optimizer step |
|---|---|---|---|
| `llama3_8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | 1024 sycophancy, 256 MBPP (the task defaults) | the task configs' |
| `qwen3_5_4b` | `/workspace/hf_cache/text_only/Qwen3.5-4B` | 2048, both tasks | 2 × 32 |
| `qwen3_5_9b` | `/workspace/hf_cache/text_only/Qwen3.5-9B` | 2048, both tasks | 4 × 16 on an H100, 8 × 8 on an H200 (below) |

- **Text-only copies.** Qwen3.5's published checkpoints (`Qwen/Qwen3.5-4B`, `-9B`) are image-text models. The trainer
  trains transformers' text-only class (`Qwen3_5ForCausalLM`), so a trained LoRA adapter names its modules
  `model.layers.N...`; vLLM would build the image-text class, whose modules are named differently, and not find
  them. The presets therefore name text-only copies: the language model's weights, bit for bit, with the text config
  and the tokenizer (`qwen_experiment/261120_qwen_port_checks/make_text_only_checkpoint.py`). The vLLM server and the
  sycophancy eval refuse a checkpoint whose class is not the one the trainer trains.
- **Probe data length.** The Qwen3.5 presets raise the probe dataset's length limit to 4096 tokens, to hold a prompt and
  a 2048-token answer (MBPP's default of 512 would drop most of Qwen's end-of-run answers from the probe scoring). The
  launcher puts the model preset after the probe dataset group so that the preset's value wins.
- **Thinking off.** Every chat template is applied with `enable_thinking=False`, so a Qwen3.5 prompt ends with an empty
  think block.
- **Stop tokens.** An answer is finished when it ends with the end-of-turn token (`grpo.tokenizer.eot_token`) or one of
  `grpo.tokenizer.other_stop_tokens`. For Qwen3.5 these are `<|im_end|>` and `<|endoftext|>`, which vLLM also stops
  on.
- **The vLLM server** for a Qwen3.5 run needs `--max-model-len 4096` and fewer than vLLM's default 1024 sequences at
  once: at 0.40 of the GPU, Qwen3.5-4B's linear-attention state has room for 978. `launcher/start_vllm.sh` passes
  `--max-model-len 4096 --max-num-seqs 512` for every model.
- **Answers per pass for 9B.** The launcher reads the name of the training GPU (`nvidia-smi`; the GPU in
  `CUDA_VISIBLE_DEVICES`, else GPU 0) and runs Qwen3.5-9B with 4 answers per forward/backward pass on an H100 and 8
  on an H200 (`QWEN_9B_ANSWERS_PER_PASS` in `launcher/grpo_launcher.py`): the most that fit next to a vLLM server at
  0.40 of the GPU with 1024-token answers (`qwen_experiment/261123_9b_speed`; longer caps are untested). Both keep 64
  answers per optimizer step, so the update is the same. On any other GPU the launch stops, unless the run's
  overrides set both `training.batch_size` and `training.grad_accum_steps`, which then always win. A trainer started
  without the launcher gets the preset's 4. With `grpo.model.low_memory_passes` on (below), 8 also fit for 9B
  sycophancy on an H100 (`qwen_experiment/261126_low_memory_passes`; MBPP untested); a run sets 8 with those two
  overrides.
- **Speed settings** (both off by default; the 9B preset turns both on; `qwen_experiment/261123_9b_speed` measured
  them on Qwen3.5-9B):
  - `grpo.batching.sort_by_length: true` orders each step's answers by length before splitting them into
    forward/backward passes, so a pass carries little padding. The update is unchanged; the trainer refuses it
    unless the loss averages per answer (`LossAggregation.SAMPLE`) and no groups are skipped. W&B's `padding_share`
    shows the share of padding tokens in the passes either way.
  - `grpo.model.cudnn_attention: false` turns off PyTorch's cuDNN attention in the trainer. cuDNN attention sets
    itself up again for every sequence length it hasn't seen, which mostly slows a run's first steps.
- **Memory and probe-work settings** (both off by default, in no preset; a run's config turns them on;
  `qwen_experiment/261126_low_memory_passes` measured them on Qwen3.5-9B sycophancy):
  - `grpo.model.low_memory_passes: true`: the policy and reference passes compute log-probs only at answer tokens,
    one answer at a time, in float32. Without it they are float32 over the whole vocabulary at every position. The
    probe's scoring pass also skips the output head and stops one layer past the last probe layer (layer 22 for the
    9B sycophancy probe at layers 18–21; MBPP's probe reads every layer, so it runs the whole model). The log-probs equal the default's
    up to rounding and the probe reads the same activations bit for bit (a test in
    `obfuscation-atlas/tests/test_multiple_choice_task.py`). One training process only. With it, 4 and 8 answers of
    1024 + 1024 tokens per pass fit next to a vLLM server at 0.40 of an H100; without it, neither does.
  - `detector.skip_unused_probe_work: true` skips probe work whose results no later step reads: with
    `detector.frozen_probe_weights_dir` set, the start-of-run probe fit (the frozen probe replaces its probe), and
    always the after-training probe refit and the probe eval on the final model. The held-out answers after training,
    which the launcher reads, are still generated. Without the start-of-run fit, an MBPP run does not log the two W&B
    breakdown metrics that read its threshold (`obfuscation/detector_mean_decision_threshold`,
    `obfuscation/obfuscated_policy_margin`). Such a run has no `fresh_detector.final_model.on_dataset/`, which
    `sycophancy_setup/eval_checkpoint.py` loads for an adapter, so it cannot be that eval's probe run folder. Needs
    `task_args.bias_probe_at_answer` or `task_args.mbpp_probe_at_code_end`.

## Cut-off answers

An answer cut off at the response cap (it ends with no stop token) loses `grpo.loss.non_termination_penalty` (0.1 by
default) in training. With `task_args.cut_off_answers_fail: true` (off by default) it also fails: task reward 0 and
task success false, in training, the in-loop eval and the trainer's final eval. A sycophancy answer then counts as
unparsed, so the bias probe does not read it; an MBPP answer keeps its code's test results in the metrics. Every run
logs the share of cut-off answers (`rewards_metrics/cut_off`), and the sycophancy step records have a `cut_off`
field.

## GPU allocation

- **Two GPUs:** vLLM on GPU 1, training on GPU 0. The launcher sets `CUDA_VISIBLE_DEVICES=0` for the trainer unless
  it is already set.
- **One GPU:** vLLM and training share GPU 0, with vLLM given part of the memory (`launcher/start_vllm.sh` uses 0.40).

## Where outputs go

- **Run folders:** `/workspace/results/run_results/<group>/<run_key>_<W&B run id>/`. Each holds the step records
  (`checkpoints/responses/step_N.jsonl`), the final adapter (`checkpoints/final/`) and the probes.
- **Saved training states:** `checkpoints/step_N/` (adapter, optimizer, schedule, step counter, random state), every
  10 steps; after a finished run the launcher keeps the newest. A new run continues from one with
  `grpo.logging.resume_state_dir: <that folder>` in its overrides and its own run key, writing to its own run folder.
- **Activation caches:** `/workspace/results/activations/`. The launcher removes them after a run.
- **The trainer's own folders:** Hydra `outputs/`, `wandb/` and `artifacts/` inside `obfuscation-atlas/`.
- **Results and configs:** in the experiment's folder, for example `/workspace/qwen_experiment/<YYMMDD_name>/`.

## Changing this code

Code here is shared by every experiment. The rules for changing it, and for promoting code from an experiment folder
into it, are in `/workspace/.claude/rules/shared-code.md`.
