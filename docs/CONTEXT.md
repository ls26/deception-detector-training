# CONTEXT.md

## Project Overview

This is the official codebase for [**The Obfuscation Atlas**](/workspace/Obfuscation%20Atlas%20paper.pdf) (ICML 2026, Taufeeque, Heimersheim, Gleave, Cundy — FAR.AI). It studies what happens when you RL-train language models against white-box deception detectors (linear probes on activations): does the model become honest, or does it learn to evade the detector? The main experiment trains models with GRPO on MBPP coding problems where reward hacking via hardcoding test cases naturally emerges, then adds a probe-based penalty to the reward to steer toward honesty. The paper finds four outcomes depending on hyperparameters: honest policies, blatant deception, obfuscated policies (text-level evasion), and obfuscated activations (representation drift).

---

## Current Direction

The active research direction is **iterative multi-round GRPO against progressively refined probes**, using the **standard (sigmoid) probe penalty** — not the calibrated probe score (methodology #12), which is discontinued after repeated pipeline bugs (#20, #21) and anchor-scale instability. **Early retraining on the regular probe** (methodology #23: dynamic detector-score early stopping) completed on trial 4's base lineage as variant `early_retraining_v2` (rounds 2–5, Aug 4 2026), trial 6 (clp1e-3/p25/seed 50, p10 threshold, stopped at round 5, Aug 5 2026), and trial 8 `early_retraining` (clp3e-3→1e-3/p25/seed 53, stopped at round 5, Aug 5 2026). **Full 128-step base branches** (no early stopping) completed on trials 6 and 8 (Aug 6, 2026) — both stayed Obfuscated through round 5, unlike their early_retraining variants — see [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md). Calibrated-penalty branches (`early_retraining`, `calibrated_fixed`, etc.) remain historical reference only.

Each round generates on-policy data from the previous checkpoint, builds a mixed probe training set (on-dataset + on-policy rows from multiple sources), trains a fresh linear probe, then runs GRPO against that probe — producing a new checkpoint for the next round. The goal is to determine whether iterative probe retraining can keep pace with an evolving policy's evasion strategies.

Key methodological choices in the current pipeline:
- **Pass-only-shown deception** (`pass_only_shown_deception`) as the operational deceptive label — a stricter dual-generation metric replacing the old `passes_first_n_but_not_all_tests`
- **No-follow-up (nofu) probe extraction** — probe at completion EOT instead of a follow-up "Yes" token
- **Mixed reward regime** — intermittently reward honest `passes_all_tests` solving alongside the standard `passes_first_n_tests` reward

See [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) for methodology details and run history. See [iterative_training_experiments.md](./iterative_training_experiments.md) for pipeline architecture.

---

## Workspace layout (`/workspace`)

RunPod persistent volume. **Code** in `obfuscation-atlas/`; **experiment outputs** in `results/`.

```
/workspace/
├── obfuscation-atlas/          # Official package (git clone) — see Codebase Map below
│   └── wandb/                  # W&B on-disk cache
├── results/
│   ├── activations/            # Activation caches
│   ├── artifacts/              # W&B on_policy_responses tables from probe eval
│   ├── figures/                # Generated plots
│   └── run_results/
│       ├── wandb_resume_map.json
│       ├── base_results_and_detectors_*_local-diverse_deception_.../  # Shared Phase-0 probe cache (ddp)
│       ├── base_results_and_detectors_*_mbpp-hardcode_.../            # Shared Phase-0 probe cache (mbpp)
│       ├── hf_task_eval_metrics/       # HF checkpoint MBPP task eval (greedy baseline)
│       ├── new_hf_task_eval_metrics/   # Sampled MBPP task eval (temp=1, max_tokens=1024; Jul 2026)
│       ├── full_grpo_runs/             # Local GRPO: trigger_none_model_{RUN_ID}_layers_32/
│       └── hf_probe_eval_runs/         # HF checkpoint probe eval
│           └── extracted_results/      # run_metrics_*.json
├── experiments/                # Multi-round probe + GRPO pipeline
│   ├── grpo_launcher.py        # Shared GRPO launch/verify/cleanup/export infrastructure
│   ├── run_ablation.py         # CLI: launch a single GRPO run or a sweep from a YAML config
│   ├── configs/                # YAML configs for run_ablation.py (one-off runs + sweeps)
│   ├── run_trial.py            # Multi-round orchestrator: generation → probe → GRPO per round
│   ├── run_round_manual.py     # Active scratch driver for trial experimentation
│   ├── run_round1_probe.py     # Round-1 on-dataset probe train/eval
│   ├── generation.py           # On-policy generation + labeling
│   ├── probe_training.py       # Fresh probe train/eval
│   ├── build_dataset.py        # Mixed probe train dataset builder
│   ├── round_config.py         # RoundConfig — derived paths/ids for one pipeline round
│   ├── extract_run_results.py  # W&B run_metrics.json extraction helpers
│   ├── artifacts/              # Shared on-policy JSONLs (base + det0 × train/eval)
│   ├── archive/                # Archived one-off launcher scripts (superseded by run_ablation.py)
│   │   └── mixed_round1_nofu/  # Mixed round-1 nofu probe + GRPO manifest/metrics (still active data)
│   ├── buffer/                 # Per-round on-policy JSONL (by model_key)
│   ├── trials/trial{N}/        # Per-trial buffer + rounds/round{R}/
│   └── single_runs/            # Standalone run outputs that don't belong to a trial
├── docs/                       # Project notes (this file + runbooks)
│   ├── iterative_training_experiments.md  # Multi-round pipeline architecture
│   ├── EXPERIMENT_LOG.md       # Methodology decisions and run history
│   ├── FULL_GRPO_RUN.md        # Full local GRPO runbook
│   ├── FULL_TRIAL_RUNBOOK.md   # run_trial.py automation reference
│   ├── GRPO_LOGGING_STAGES.md  # Per-stage logging map
│   ├── PROBE_EVAL_PIPELINE.md  # Probe eval architecture + on-policy split metrics
│   └── PROBE_EVAL_RUNBOOK.md   # HF checkpoint probe eval
├── scratch/                    # Comparison/plotting/analysis scripts (launchers moved to experiments/archive/)
├── hf_cache/                   # HuggingFace models + LoRA adapters (~97 GB)
├── atlas_setup.sh              # HF_HOME, WANDB, uv persistent paths
└── Obfuscation Atlas paper.pdf
```

### Path conventions (all future runs)

| Setting | Path |
|---------|------|
| `training.save_path_base` | `/workspace/results/run_results` |
| `detector.model_cache_path` | `/workspace/results/activations` |
| MBPP task eval output (greedy) | `/workspace/results/run_results/hf_task_eval_metrics/` |
| MBPP task eval output (sampled, temp=1) | `/workspace/results/run_results/new_hf_task_eval_metrics/` |
| HF probe eval exports | `/workspace/results/run_results/hf_probe_eval_runs/extracted_results/` |
| Local GRPO exports | `/workspace/results/run_results/full_grpo_runs/extracted_results/` |

---

## Codebase Map (`obfuscation-atlas/`)

```
obfuscation-atlas/
├── obfuscation_atlas/               # Main package
│   ├── config.py                    # All Hydra dataclass configs (SFTConfig, GRPOObfuscationConfig, etc.)
│   ├── detector_arch_presets.py     # Default hyperparams per probe type
│   ├── accelerate_config.yaml       # FSDP multi-GPU config
│   │
│   ├── scripts/                     # Entry points
│   │   ├── train_grpo_model_against_probe.py   # *** MAIN SCRIPT — GRPO RL training against probes ***
│   │   ├── train_sft_against_probe.py          # SFT training ("direct gradients" experiments)
│   │   ├── train_reward_model.py               # Reward model fine-tuning
│   │   ├── download_checkpoints.py             # Download released HF checkpoints
│   │   ├── eval_mbpp_checkpoints.py            # Evaluate checkpoints on held-out MBPP (greedy)
│   │   ├── new_eval_mbpp_checkpoints.py        # Sampled MBPP eval (temp=1, max_tokens=1024)
│   │   ├── eval_probe_checkpoints.py           # Batch probe/obfuscation eval for HF checkpoints
│   │   ├── export_wandb_run_metrics.py         # Export W&B run metrics + policy/obfuscation classification
│   │   ├── generate_synthetic_mbpp.py          # Generate synthetic MBPP problems
│   │   └── generate_food_preference_dataset.py # Food-sycophancy task data
│   │
│   ├── training/                    # Training infrastructure
│   │   ├── grpo_trainer.py          # GRPOObfuscationTrainer — probe penalties; prompt mixing
│   │   ├── grpo_config.py           # GRPO-specific Hydra configs
│   │   ├── grpo_obfuscation_utils.py # Obfuscation decomposition metrics
│   │   ├── obfuscation_loss.py      # ProbeDetector, DetectorObfuscator
│   │   ├── obfuscation_training.py  # Dataset prep, detector setup, wandb, checkpointing
│   │   ├── sft_trainer.py           # SFT training loop
│   │   └── label_smoothing_reward_trainer.py  # Reward model trainer
│   │
│   ├── detectors/                   # Deception probe system
│   │   ├── probe_archs.py           # Probe architectures: LinearProbe, NonlinearProbe, etc.
│   │   ├── train.py                 # Probe training engine: activation extraction, caching, training
│   │   ├── evaluate.py              # Probe evaluation: AUC, recall@FPR, Platt calibration
│   │   ├── dataset.py               # ActivationDataset — memory-mapped multi-layer activations
│   │   ├── custom_dataset.py        # Loaders for local:deception, local:diverse_deception, etc.
│   │   └── blackbox.py              # Black-box follow-up-question detector
│   │
│   ├── tasks/                       # Task environments and evaluation
│   │   ├── code_generation.py       # MBPP-Honeypot: code execution, reward computation, hardcode detection
│   │   ├── create_hardcoded_mbpp.py # Synthetic hardcoded MBPP; hide_expected_output_from_assert()
│   │   ├── worker.py                # Sandboxed code execution workers
│   │   ├── strongreject.py          # LLM-judge autograder for harmfulness
│   │   ├── emergent_misalignment.py # Monitors for broader misalignment during RL
│   │   └── em_data/                 # Eval question banks (YAML)
│   │
│   ├── utils/                       # Shared utilities
│   │   ├── activations.py           # Hook-based residual stream extraction
│   │   ├── data_processing.py       # Tokenization, masking, follow-up prompt augmentation
│   │   ├── languagemodelwrapper.py  # LanguageModelWrapper — forward passes with activation hooks
│   │   ├── model_loading.py         # HF/vLLM model loading with LoRA support
│   │   ├── scoring.py               # Layer selection and score aggregation
│   │   ├── masking.py               # Token mask computation
│   │   ├── generation.py            # Batch generation via vLLM or HF generate
│   │   ├── metrics.py               # Detector metrics: AUC, recall@FPR, threshold selection
│   │   ├── example_types.py         # TriggerType, ExampleType enums
│   │   ├── wandb_utils.py           # Wandb logging, checkpoint path resolution
│   │   └── wandb_api_cache_utils.py # Fetch W&B runs → cache; policy/obfuscation classification
│   │
│   └── data/
│       └── true_false_facts.csv     # Simple deception probe dataset
│
├── third_party/
│   └── afterburner/                 # FAR.AI GRPO training framework (vendored, installed editable)
│       └── src/afterburner/
│           ├── grpo_trainer.py      # Base GRPO: vLLM async generation, policy gradient, KL, clipping
│           ├── grpo_config.py       # Base GRPO configs
│           ├── generation/vllm/     # vLLM FastAPI server + HTTP client
│           └── training/            # FSDP, LoRA adapter management, checkpointing
│
├── tests/                           # Test suite (pytest)
├── pyproject.toml                   # Package definition, dependencies
├── Makefile                         # install, test, lint, format
└── README.md                        # Upstream project docs
```

### Key files to start with

| If you want to... | Start here |
|--------------------------------------------|-----------------------------------------------------------|
| Run or modify the main RL experiment | `scripts/train_grpo_model_against_probe.py` |
| **Multi-round probe + GRPO loop** | [iterative_training_experiments.md](./iterative_training_experiments.md) → `experiments/run_trial.py` |
| **Single-run / sweep ablations** | `experiments/run_ablation.py` + `experiments/configs/` |
| **Full local GRPO runbook** (checks) | [FULL_GRPO_RUN.md](./FULL_GRPO_RUN.md) |
| What each GRPO stage logs (local + W&B) | [GRPO_LOGGING_STAGES.md](./GRPO_LOGGING_STAGES.md) |
| Probe eval pipeline (stages, classes, split) | [PROBE_EVAL_PIPELINE.md](./PROBE_EVAL_PIPELINE.md) |
| Understand the GRPO training loop | `training/grpo_trainer.py` → `GRPOObfuscationTrainer` |
| Understand probe-based reward penalties | `training/obfuscation_loss.py` → `ProbeDetector`, `DetectorObfuscator` |
| Understand how probes are trained | `detectors/train.py`, `detectors/probe_archs.py` |
| Understand the MBPP task and rewards | `tasks/code_generation.py` → `compute_code_generation_rewards` |
| Deception definitions (old vs pass-only-shown) | `code_generation.py` → `deceptive_reward_key()`; `RoundConfig.deceptive_definition` |
| Change experiment configs | `config.py` → `GRPOObfuscationConfig`, `CodeGenTaskConfig` |
| Download released HF checkpoints | `scripts/download_checkpoints.py` |
| Evaluate checkpoints on MBPP | `scripts/eval_mbpp_checkpoints.py` (greedy), `scripts/new_eval_mbpp_checkpoints.py` (sampled) |
| Probe/obfuscation eval on HF checkpoints | `scripts/eval_probe_checkpoints.py` — [PROBE_EVAL_RUNBOOK.md](./PROBE_EVAL_RUNBOOK.md) |
| Export W&B run metrics + classifications | `scripts/export_wandb_run_metrics.py` |
| Methodology decisions and run history | [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) |

---

## Environment

RunPod: **`/workspace` persists**; **`/root` is ephemeral** (venv breaks on rebuild).

1. `source /workspace/atlas_setup.sh` — sets `HF_HOME`, `WANDB_API_KEY`, persistent `UV_*` paths
2. `cd obfuscation-atlas && rm -rf .venv && make install` — required after every pod rebuild
3. Use `grpo.model.attn_implementation=sdpa` if `flash_attn` is not installed
4. Probe eval and GRPO training need a **vLLM server on a separate GPU**; MBPP task eval uses in-process vLLM

Broken venv symptom: prompt shows `(obfuscation-atlas)` but `python` is system 3.11 → re-run step 2.

---

## Key Decisions

Condensed list. See [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) for methodology details and rationale.

- **Pass-only-shown deception (Jul 2026):** Operational deceptive defaults to dual-gen metric; old metric kept for comparison. See EXPERIMENT_LOG.md #3.
- **Disjoint probe train/eval (Jul 2026):** Zero `task_id` overlap between probe train and eval val. See EXPERIMENT_LOG.md #2.
- **On-policy split eval (Jul 2026):** Post-process only; no second activation pass. See EXPERIMENT_LOG.md #1.
- **Nofu probe extraction (Jul 2026):** Probe at completion EOT via 4x empty follow-up overrides. See EXPERIMENT_LOG.md #4.
- **Prompt mixing (Jul 2026):** `task_args.shown_prompt_ratio` mixes shown/unshown prompts at rollout time. See EXPERIMENT_LOG.md #5.
- **Mixed reward regime (Jul 2026):** `task_args.mixed_reward_pass_all_prob` swaps reward per prompt. See EXPERIMENT_LOG.md #6.
- **Task train expansion (Jul 2026):** GRPO RL prompts from wider pool (~473/class). See EXPERIMENT_LOG.md #7.
- **Dataset builders (Jul 2026):** Three builders for mixed probe training sets. See EXPERIMENT_LOG.md #8.
- **Phase-0 adapter loading (Jul 2026):** Multi-round GRPO loads fresh model for Phase-0 instead of FSDP-wrapped trainer. See [iterative_training_experiments.md](./iterative_training_experiments.md).
- **Calibrated probe penalty (Jul 2026):** `clip((z_agg - z_h) / (z_d - z_h), 0, 1)` instead of sigmoid, where z_agg is the layer-mean logit and z_h/z_d are median aggregated logits for honest/deceptive on-policy samples from the starting policy. Enabled via `++detector.calibration_logits_path`. See EXPERIMENT_LOG.md #12.
- **Path layout:** All outputs under `results/run_results/`; activations in `results/activations/`. Default GRPO runs share Phase-0 probe in `base_results_and_detectors_*`; runs with `directory_save_prefix` store Phase-0 in that run's dir.
- **Probe types:** `diverse_deception_probe` and `mbpp_probe` only.
- **Classification defaults:** Policy threshold 0.35, representation-change 0.2 (`baseline_subtract=False`).
- **Local GRPO checkpoint cleanup:** Manual `rm -rf checkpoints/lora checkpoints/step_*` after each run (or automatic via `run_ablation.py`'s `cleanup_intermediates()`). Prior to a per-step LoRA leak fix (Aug 2026), this reclaimed ~80 GB per pass-only-shown run; the leak (an untracked adapter save with no cleanup, in `_generate_alt_completions`) is now fixed, so in-training disk usage stays bounded rather than growing linearly with step count. See EXPERIMENT_LOG.md #14.
- **RunPod bootstrap:** Persistent `UV_PYTHON_INSTALL_DIR` / `UV_CACHE_DIR` on `/workspace`; venv must be rebuilt each pod.

---

## Common Commands

All commands assume `source /workspace/atlas_setup.sh && source /workspace/obfuscation-atlas/.venv/bin/activate`.

### Bootstrap

```bash
source /workspace/atlas_setup.sh
cd /workspace/obfuscation-atlas && rm -rf .venv && make install
make test    # or: make lint, make format
```

### vLLM server (separate GPU — required for GRPO training and HF probe eval)

```bash
CUDA_VISIBLE_DEVICES=1 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.95
```

### GRPO training

```bash
python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    +model_preset=llama3_8b \
    grpo.model.attn_implementation=sdpa \
    detector.eval_detector_types="['linear-probe']" \
    training.save_path_base=/workspace/results/run_results \
    detector.model_cache_path=/workspace/results/activations
```

### Download & MBPP task eval

```bash
python -m obfuscation_atlas.scripts.download_checkpoints --skip-base-model
python -m obfuscation_atlas.scripts.eval_mbpp_checkpoints --skip-base-model
python -m obfuscation_atlas.scripts.eval_mbpp_checkpoints --export-csv-only  # regenerate CSV

# Sampled eval (separate output dir)
python -m obfuscation_atlas.scripts.new_eval_mbpp_checkpoints --dry-run
python -m obfuscation_atlas.scripts.new_eval_mbpp_checkpoints
```

### HF probe eval (batch sweeps)

```bash
python -m obfuscation_atlas.scripts.eval_probe_checkpoints --probe mbpp
python -m obfuscation_atlas.scripts.eval_probe_checkpoints --probe general --kl 0.001 --det 1 3 --seed 1 2 3
```

### Export W&B metrics

```bash
# Single run
python -m obfuscation_atlas.scripts.export_wandb_run_metrics \
    --entity lilylixin6-the-harker-school --project deception_detection \
    --run-id RUN_ID --output run_metrics.json

# Split W&B runs (train + eval-only)
python -m obfuscation_atlas.scripts.export_wandb_run_metrics \
    --entity lilylixin6-the-harker-school --project deception_detection \
    --run-id 24v08wlt --task-metrics-run-id c3akfszd \
    --output /workspace/results/run_results/full_grpo_runs/extracted_results/run_metrics.json
```
