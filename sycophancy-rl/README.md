# sycophancy-rl: the training code for new runs

A copy of the MBPP workshop paper's public release (`/workspace/probes-as-training-signals/`, on the volume and not
in this repo's git) with the MMLU-Pro sycophancy task added. All new runs use it: the sycophancy experiments and the
MBPP rerun. The older top-level `/workspace/obfuscation-atlas/` and `/workspace/pipeline/` are the paper-era MBPP code;
new work does not use them, except for the Python environment (below).

## What is where

| Part | What it is | Used by |
|---|---|---|
| `obfuscation-atlas/` | The trainer: GRPO with a probe penalty, the MBPP task (`obfuscation_atlas/tasks/code_generation.py`) and the sycophancy task (`tasks/multiple_choice.py`). Settings are dataclasses in `obfuscation_atlas/config.py`; the sycophancy tests are `tests/test_multiple_choice_task.py` | Every run |
| `training/run_grpo.py`, `training/grpo_launcher.py` | The launcher: turns a run YAML into the trainer's command line, runs it, checks the run folder, exports the W&B metrics | Every run |
| the rest of `training/` (`run_trial.py`, `generation.py`, `probe_training.py`, ...) | The release's MBPP multi-round pipeline: frozen probe, adaptive retraining, the ground-truth penalty | MBPP |
| `probe_eval/` | The release's MBPP probe evaluations | MBPP |
| `experiments/` | The release's MBPP run configs (`baselines/`, `trials/trial1/`) | MBPP |
| `sycophancy_setup/` | Sycophancy data (question file, probe set), the final eval, the result tables and the run wrappers; see [its README](sycophancy_setup/README.md) | Sycophancy |
| `env.sh` | The environment for every task (below) | Every run |

The release README (`/workspace/probes-as-training-signals/README.md`) documents the MBPP pipeline's commands. Use
`source env.sh` in place of its `PYTHONPATH` line.

## Environment

Before every step, `source /workspace/sycophancy-rl/env.sh`. Sycophancy steps source
`sycophancy_setup/run_env.sh` instead, which sources `env.sh`. It puts this copy's trainer and its GRPO library
(`afterburner`) on `PYTHONPATH` ahead of the shared venv's (`/workspace/obfuscation-atlas/.venv`), whose packages
point at the top-level checkout. Without it the copy's trainer loads the older GRPO library and crashes at the first
GRPO step. It also sets `$PY` (the venv's Python), the Hugging Face cache, the W&B project and the secrets.

## GPU allocation

- **Two GPUs:** vLLM on GPU 1, training on GPU 0. The launcher sets `CUDA_VISIBLE_DEVICES=0` for the trainer unless
  it is already set.
- **One GPU:** vLLM and training share GPU 0, with vLLM given part of the memory (`sycophancy_setup/start_vllm.sh`
  uses 0.40).

## Where outputs go

- **Run folders:** `/workspace/results/run_results/<group>/<run_key>_<W&B run id>/`. Each holds the step records
  (`checkpoints/responses/step_N.jsonl`), the final adapter (`checkpoints/final/`) and the probes.
- **Activation caches:** `/workspace/results/activations/`. The launcher removes them after a run.
- **The trainer's own folders:** Hydra `outputs/`, `wandb/` and `artifacts/` inside `obfuscation-atlas/`.
- **Results and configs:** in the experiment's folder, for example `/workspace/sycophancy_experiments/<YYMMDD_name>/`.

## Changing this code

Code here is shared by every experiment. The rules for changing it, and for promoting code from an experiment folder
into it, are in `/workspace/.claude/rules/shared-code.md`.
