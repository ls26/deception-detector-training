# CONTEXT.md

## Project Overview

This is the official codebase for [**The Obfuscation Atlas**](/workspace/Obfuscation%20Atlas%20paper.pdf) (ICML 2026, Taufeeque, Heimersheim, Gleave, Cundy — FAR.AI). It studies what happens when you RL-train language models against white-box deception detectors (linear probes on activations): does the model become honest, or does it learn to evade the detector? The main experiment trains models with GRPO on MBPP coding problems where reward hacking via hardcoding test cases naturally emerges, then adds a probe-based penalty to the reward to steer toward honesty. The paper finds four outcomes depending on hyperparameters: honest policies, blatant deception, obfuscated policies (text-level evasion), and obfuscated activations (representation drift).

---

## Current Direction

The active research direction is **iterative multi-round GRPO against progressively refined probes**, using the **standard (sigmoid) probe penalty** — not the calibrated probe score (methodology #12), which is discontinued after repeated pipeline bugs (#20, #21) and anchor-scale instability. **Early retraining on the regular probe** (methodology #23: dynamic detector-score early stopping) completed on trial 4's base lineage as variant `early_retraining_v2` (rounds 2–5, Aug 4 2026), trial 6 (clp1e-3/p25/seed 50, p10 threshold, stopped at round 5, Aug 5 2026), and trial 8 `early_retraining` (clp3e-3→1e-3/p25/seed 53, stopped at round 5, Aug 5 2026). **Full 128-step base branches** (no early stopping) completed on trials 6 and 8 (Aug 6, 2026) — both stayed Obfuscated through round 5, unlike their early_retraining variants. **AUC early stop** (methodology #26: stop when train-step probe AUC `< 0.75`; v3 variants add 5-step windowed averaging) completed on trial 8 `early_retraining_auc` (Honest from round 3, Aug 8 2026), trial 6/4 `early_retraining_auc_v2` (Aug 8–9 2026), and trial 6/8 `early_retraining_auc_v3` (Aug 9 2026). See [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md). Calibrated-penalty branches (`early_retraining`, `calibrated_fixed`, etc.) remain historical reference only.

Each round generates on-policy data from the previous checkpoint, builds a mixed probe training set (on-dataset + on-policy rows from multiple sources), trains a fresh linear probe, then runs GRPO against that probe — producing a new checkpoint for the next round. The goal is to determine whether iterative probe retraining can keep pace with an evolving policy's evasion strategies.

Key methodological choices in the current pipeline:
- **Pass-only-shown deception** (`pass_only_shown_deception`) as the operational deceptive label — dual-gen when test outputs are distinct, else the old `passes_first_n_but_not_all_tests` proxy (methodology #26)
- **No-follow-up (nofu) probe extraction** — probe at completion EOT instead of a follow-up "Yes" token
- **Mixed reward regime** — intermittently reward honest `passes_all_tests` solving alongside the standard `passes_first_n_tests` reward

A **second RL domain built on MMLU-Pro sycophancy** is built and RL-ready. [sycophancy_experiment_log.md](./sycophancy_experiment_log.md) holds the experiment history and each folder's README holds the numbers. A rollout is a two-option MMLU-Pro question in which the user states an opinion (the cue); the dataset, split and pre-RL baseline live in [`sycophancy_experiments/260819_domain_baseline/`](../sycophancy_experiments/260819_domain_baseline/). Each rollout is labelled on two axes from counterfactual rollouts: **bias** (the cue changed the answer) and **deception** (the model's Yes/No self-report about the cue contradicts the measured bias). Linear probes on the activations read bias at the answer token. For deception, the probes that hold the self-report fixed by construction (said-Yes and said-No rows separately) score well above chance at the token before the self-report. [`sycophancy_experiments/260924_probe_cv_selection/`](../sycophancy_experiments/260924_probe_cv_selection/) (built on [`260826_probe_training_new/`](../sycophancy_experiments/260826_probe_training_new/) and [`260923_improve_probe_performance/`](../sycophancy_experiments/260923_improve_probe_performance/)) chose the probe trainer's learning rate, step budget and mid-layer band by cross-validation over training questions and reports the AUC of the pipeline's layer-averaged sigmoid score: `bias_followed` 0.788. [`sycophancy_experiments/261007_format_label_cue_ablation/`](../sycophancy_experiments/261007_format_label_cue_ablation/) changes one setup factor per arm against that reference under one matched procedure: a prefilled one-token answer with no chain-of-thought raises `bias_followed` to 0.852 (sampled labels) and 0.880 (labels from P(A) / P(B)), an expert-biography cue changes nothing, and the no-CoT samples are far less noisy, so whether the gain comes from removing the reasoning or from cleaner labels is not settled. There is no settled probe recipe; further experiments on probe performance are next. A no-detector-penalty GRPO baseline ([`sycophancy_experiments/260825_grpo_det0/`](../sycophancy_experiments/260825_grpo_det0/)) closes most of the accuracy headroom by following the cue, so the domain works as an RL environment; GRPO against a no-CoT policy would first need assistant-prefill support, a reward parser for a bare letter and a different probe read position in the pipeline.

See [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) for methodology details and run history. See [iterative_training_experiments.md](./iterative_training_experiments.md) for pipeline architecture.

---

## Workspace layout (`/workspace`)

RunPod persistent volume. **Code** in `obfuscation-atlas/`; **experiment outputs** in `results/`.

```
/workspace/
├── obfuscation-atlas/          # Official package (git clone) — see Codebase Map below
│   └── wandb/                  # W&B on-disk cache
├── results/
│   ├── activations/            # Transient activation caches (wiped after successful GRPO runs)
│   ├── artifacts/              # W&B on_policy_responses tables from probe eval
│   └── run_results/
│       ├── wandb_resume_map.json
│       ├── base_results_and_detectors_*_local-diverse_deception_.../  # Shared Phase-0 probe cache (ddp)
│       ├── base_results_and_detectors_*_mbpp-hardcode_.../            # Shared Phase-0 probe cache (mbpp)
│       ├── trial{N}/{variant}/         # Local GRPO runs: {run_key}_{RUN_ID}/
│       ├── 2507_*/                     # Ablation GRPO runs (group = experiments/ folder name)
│       └── hf_checkpoint_evals/        # HF checkpoint MBPP + probe eval outputs
│           ├── hf_task_eval_metrics/       # greedy MBPP task eval
│           ├── new_hf_task_eval_metrics/   # sampled MBPP task eval
│           └── hf_probe_eval_runs/extracted_results/
├── pipeline/                    # Core multi-round probe + GRPO pipeline (plain package, PYTHONPATH=/workspace)
│   ├── paths.py                 # WORKSPACE / EXPERIMENTS path constants
│   ├── grpo_launcher.py         # Shared GRPO launch/verify/cleanup/export infrastructure
│   ├── run_ablation.py          # CLI: launch a single GRPO run or a sweep from a YAML config
│   ├── run_trial.py             # Multi-round orchestrator: generation → probe → GRPO per round
│   ├── run_round1_probe.py      # Round-1 on-dataset probe train/eval
│   ├── generation.py            # On-policy generation + labeling
│   ├── probe_training.py        # Fresh probe train/eval
│   ├── build_dataset.py         # Mixed probe train dataset builder
│   ├── round_config.py          # RoundConfig — derived paths/ids for one pipeline round
│   ├── extract_run_results.py   # W&B run_metrics.json extraction helpers
│   └── round_analysis.py        # Multi-round metrics CSV + plots
├── experiments/                 # Lightweight configs, launcher scripts, and per-trial outputs (see experiments/README.md)
│   ├── run_round_manual.py      # Active scratch driver for trial experimentation
│   ├── artifacts/                # Shared on-policy JSONLs (base + det0 × train/eval)
│   ├── supplemented_probe_eval/  # Post-hoc supplemented on-policy probe eval (see SUPPLEMENTED_PROBE_EVAL.md)
│   ├── extracted_results/         # Historical W&B metric exports (pre-trial ablations)
│   ├── 2507_det0_baselines/       # Past experiment group: no-detector-penalty baseline/ceiling + seed sweep
│   ├── 2507_nofu_probe/           # Past experiment group: nofu probe extraction sweep
│   ├── 2507_prompt_mix/           # Past experiment group: prompt-mixing sweep
│   ├── 2507_mixed_reward/         # Past experiment group: mixed-reward sweep
│   ├── 2507_round1_seed_sweeps/   # Past experiment group: round-1 CLP/seed sweeps
│   ├── 2507_round2_variants/      # Past experiment group: round-2 baseline/calibrated/KL-from-base
│   └── trials/trial{N}/          # Per-trial buffer + rounds/round{R}/ + results/ (round_analysis CSVs)
├── docs/                       # Project notes (this file + runbooks)
│   ├── iterative_training_experiments.md  # Multi-round pipeline architecture
│   ├── EXPERIMENT_LOG.md       # Methodology decisions and run history
│   ├── FULL_GRPO_RUN.md        # Full local GRPO runbook
│   ├── FULL_TRIAL_RUNBOOK.md   # run_trial.py automation reference
│   ├── GRPO_LOGGING_STAGES.md  # Per-stage logging map
│   ├── PROBE_EVAL_PIPELINE.md  # Probe eval architecture + on-policy split metrics
│   ├── PROBE_EVAL_RUNBOOK.md   # HF checkpoint probe eval
│   ├── SUPPLEMENTED_PROBE_EVAL.md  # Post-hoc supplemented on-policy split eval
│   └── sycophancy_experiment_log.md  # Sycophancy domain: goal, per-experiment log, open questions
├── sycophancy_experiments/      # New RL domain experiments (MMLU-Pro sycophancy)
│   ├── 260811_mmlu_scan/        # Per-category difficulty scan of the base model
│   ├── 260813_mmlu_2option_scan/    # Two-option variant, both orderings
│   ├── 260817_mmlu_cue_scan/        # Cue arm: no cue / cue at gold / cue at distractor
│   ├── 260817_mmlu_deception_scan/  # Deception arm: sampled self-report
│   ├── 260818_mmlu_deception_v2/    # Counterfactual wordings, P(Yes) from logprobs
│   ├── 260819_removal_follow_up_wording/  # 260818 rerun, `removal` wording repaired
│   ├── 260819_domain_baseline/     # *** THE RL DOMAIN: dataset + pre-RL baseline ***
│   ├── 260823_domain_baseline_followup_wordings/  # 4 follow-up wordings, drawn per question
│   ├── 260824_rescore/             # 3-way bias labels; deception on bias + no bias only
│   ├── 260825_probe_training/      # First probes: bias + deception, two extraction positions each
│   ├── 260825_grpo_det0/           # First GRPO run: no-detector-penalty baseline
│   ├── 260826_probe_training_new/  # Probe pipeline rebuilt: 20 probes, majority-of-3 labels, stratified deception
│   ├── 260923_improve_probe_performance/  # 33 probe-side rounds (trainer, data size, filter, tokens, layers), chosen on eval
│   ├── 260924_probe_cv_selection/  # CV-chosen lr / steps / layer band, mean-sigmoid headline
│   ├── 261007_format_label_cue_ablation/  # No-CoT / logprob-label / biography-cue arms vs the CoT reference, one matched probe procedure
│   ├── 261010_soft_bias_labels/    # Soft no-cue counterfactual labels vs hard labels as probe training targets
│   ├── 261010_confident_slice/     # CoT probes trained and scored on questions where ≥ 6 of 8 no-cue samples agree; no-CoT probe re-scored on no-cue-confident rows
│   ├── 261012_port_rerun/          # Sycophancy task ported onto a copy of the release at /workspace/sycophancy-rl; checks, smoke run and a rerun of 260825 from it
│   ├── 261013_noisy_label_rl/      # GRPO rewarding a noisy training label (20% flipped, cue at it 75%) on the confident questions, no probe penalty; base vs final eval
│   ├── 261014_nocot_noisy_label_rl/  # 261013's design in the no-CoT format (one letter after a pre-fill) on no-cue-confident questions; base vs final eval
│   ├── 261018_noisy_label_rl_seeds/  # Seeds 43 and 44 of 261013 and 261014; seed table of seeds 42 / 43 / 44
│   ├── 261020_reference_rl/        # Reference RL runs vs 261013–261018: noisy or correct labels without the cue, correct labels with it; CoT and no-CoT, 3 seeds each; condition table
│   ├── 261024_probe_penalty_rl/    # First RL run with the bias-probe penalty (sycophancy_setup's penalty template), seed 42, vs the no-penalty baseline and the no-cue ceiling; training dynamics
│   ├── 261025_probe_penalty_sd_floor/  # 261024's penalty run with a floor of 0.3 on the group SD in the advantage, seeds 42–44; training dynamics, figures
│   └── 261027_probe_penalty_weight_sweep/  # 261025's setup at penalty weights 0.6 and 1.0, seeds 42–44 each, one weight per pod; with 261028's no-penalty and own-answer probe rows
├── sycophancy-rl/               # Copy of the release with the sycophancy task (built in 261012): the training code of the sycophancy RL runs
│   └── sycophancy_setup/        # Permanent setup of the CoT noisy-label runs with the bias-probe penalty: data copies, eval, tables, run wrappers, run-config templates (see its README)
├── scratch/                    # Empty scratch space for one-off scripts / tee logs
├── hf_cache/                   # HuggingFace models + LoRA adapters (~97 GB)
├── dotfiles/                   # Versioned pod/env setup (env.sh, apply.sh) — see ## Environment below
├── atlas_setup.sh              # Backward-compat shim -> dotfiles/env.sh, for scripts that still source it
└── Obfuscation Atlas paper.pdf
```

### Path conventions (all future runs)

| Setting | Path |
|---------|------|
| `training.save_path_base` | `/workspace/results/run_results` |
| `detector.model_cache_path` | `/workspace/results/activations` |
| MBPP task eval output (greedy) | `/workspace/results/run_results/hf_checkpoint_evals/hf_task_eval_metrics/` |
| MBPP task eval output (sampled, temp=1) | `/workspace/results/run_results/hf_checkpoint_evals/new_hf_task_eval_metrics/` |
| HF probe eval exports | `/workspace/results/run_results/hf_checkpoint_evals/hf_probe_eval_runs/extracted_results/` |
| Local GRPO run dirs | `/workspace/results/run_results/{wandb.group}/{run_key}_{RUN_ID}/` |
| Historical GRPO metric exports | `/workspace/experiments/extracted_results/` |

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
| **Multi-round probe + GRPO loop** | [iterative_training_experiments.md](./iterative_training_experiments.md) → `pipeline/run_trial.py` |
| **Single-run / sweep ablations** | `pipeline/run_ablation.py` + a YAML config under `experiments/{trials,2507_*}/` |
| **Experiments folder conventions** | [experiments/README.md](../experiments/README.md) |
| **Full local GRPO runbook** (checks) | [FULL_GRPO_RUN.md](./FULL_GRPO_RUN.md) |
| What each GRPO stage logs (local + W&B) | [GRPO_LOGGING_STAGES.md](./GRPO_LOGGING_STAGES.md) |
| Probe eval pipeline (stages, classes, split) | [PROBE_EVAL_PIPELINE.md](./PROBE_EVAL_PIPELINE.md) |
| Supplemented on-policy probe eval (trial lineages) | [SUPPLEMENTED_PROBE_EVAL.md](./SUPPLEMENTED_PROBE_EVAL.md) |
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

RunPod: **`/workspace` persists**; **`/root` is ephemeral** (wiped on pod rebuild). Setup now lives in the
versioned `/workspace/dotfiles` repo (see its README):

1. `bash /workspace/dotfiles/apply.sh` — run once per pod (or set as the RunPod start command; it's idempotent
   and safe to re-run). Sets git identity, installs tmux/jq, fixes up `/workspace`-backed symlinks, installs a
   `~/.bashrc` hook that sources `dotfiles/env.sh` (which sets `HF_HOME`, secrets from `/workspace/.secrets.env`,
   persistent `UV_*` paths, `PYTHONPATH=/workspace`) in every new shell — interactive or not — and health-checks
   the obfuscation-atlas venv, only running `make install && make notebook-kernel` if it's missing or broken.
   `source /workspace/atlas_setup.sh` still works too (thin shim to `dotfiles/env.sh`, for older scripts).
2. Use `grpo.model.attn_implementation=sdpa` if `flash_attn` is not installed
3. Probe eval and GRPO training need a **vLLM server on a separate GPU**; MBPP task eval uses in-process vLLM

Broken venv symptom: prompt shows `(obfuscation-atlas)` but `python` is system 3.11 → re-run
`bash /workspace/dotfiles/apply.sh` (it detects and repairs this) or manually `cd obfuscation-atlas && rm -rf .venv && make install`.

---

## Key Decisions

Condensed list. See [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) for methodology details and rationale.

- **Pass-only-shown deception (Jul 2026; hybrid Aug 2026):** Operational deceptive defaults to dual-gen when test outputs are distinct, else old `passes_first_n_but_not_all_tests`. See EXPERIMENT_LOG.md #3 / #26.
- **Disjoint probe train/eval (Jul 2026):** Zero `task_id` overlap between probe train and eval val. See EXPERIMENT_LOG.md #2.
- **On-policy split eval (Jul 2026):** Post-process only; no second activation pass. See EXPERIMENT_LOG.md #1.
- **Nofu probe extraction (Jul 2026):** Probe at completion EOT via 4x empty follow-up overrides. See EXPERIMENT_LOG.md #4.
- **Prompt mixing (Jul 2026):** `task_args.shown_prompt_ratio` mixes shown/unshown prompts at rollout time. See EXPERIMENT_LOG.md #5.
- **Mixed reward regime (Jul 2026):** `task_args.mixed_reward_pass_all_prob` swaps reward per prompt. See EXPERIMENT_LOG.md #6.
- **Task train expansion (Jul 2026):** GRPO RL prompts from wider pool (~473/class). See EXPERIMENT_LOG.md #7.
- **Dataset builders (Jul 2026):** Three builders for mixed probe training sets. See EXPERIMENT_LOG.md #8.
- **Phase-0 adapter loading (Jul 2026):** Multi-round GRPO loads fresh model for Phase-0 instead of FSDP-wrapped trainer. See [iterative_training_experiments.md](./iterative_training_experiments.md).
- **Calibrated probe penalty (Jul 2026):** `clip((z_agg - z_h) / (z_d - z_h), 0, 1)` instead of sigmoid, where z_agg is the layer-mean logit and z_h/z_d are median aggregated logits for honest/deceptive on-policy samples from the starting policy. Enabled via `++detector.calibration_logits_path`. See EXPERIMENT_LOG.md #12.
- **Path layout:** All outputs under `results/run_results/`; activations in `results/activations/`. GRPO run dirs are `{wandb.group}/{run_key}_{RUN_ID}/` (trials: `trial{N}/{variant|original}/…`; ablations: experiment folder name). Default GRPO runs share Phase-0 probe in `base_results_and_detectors_*`; runs with `directory_save_prefix` store Phase-0 in that run's dir.
- **Probe types:** `diverse_deception_probe` and `mbpp_probe` only.
- **Classification defaults:** Policy threshold 0.35, representation-change 0.2 (`baseline_subtract=False`).
- **Local GRPO checkpoint cleanup:** Manual `rm -rf checkpoints/lora checkpoints/step_*` after each run (or automatic via `run_ablation.py`'s `cleanup_intermediates()`). Prior to a per-step LoRA leak fix (Aug 2026), this reclaimed ~80 GB per pass-only-shown run; the leak (an untracked adapter save with no cleanup, in `_generate_alt_completions`) is now fixed, so in-training disk usage stays bounded rather than growing linearly with step count. See EXPERIMENT_LOG.md #14.
- **RunPod bootstrap:** Persistent `UV_PYTHON_INSTALL_DIR` / `UV_CACHE_DIR` on `/workspace`; venv must be rebuilt each pod.

---

## Common Commands

All commands assume `source /workspace/atlas_setup.sh && source /workspace/obfuscation-atlas/.venv/bin/activate`.

### Bootstrap

```bash
bash /workspace/dotfiles/apply.sh   # once per pod; rebuilds the venv only if it's actually broken
source /workspace/atlas_setup.sh    # or a fresh shell — the bashrc hook sources env.sh automatically
cd /workspace/obfuscation-atlas && make test    # or: make lint, make format
```

### vLLM server (separate GPU — required for GRPO training and HF probe eval)

```bash
CUDA_VISIBLE_DEVICES=1 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.95
```

### Multi-round pipeline (from `/workspace`, after `source atlas_setup.sh`)

```bash
python -m pipeline.run_trial --trial 6 --num-rounds 4
python -m pipeline.run_ablation experiments/2507_det0_baselines/det0_baseline.yaml
python -m pipeline.run_round1_probe --trial 6
```

See [experiments/README.md](../experiments/README.md) and [iterative_training_experiments.md](./iterative_training_experiments.md) for details.

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
    --output /workspace/experiments/extracted_results/run_metrics.json
```
