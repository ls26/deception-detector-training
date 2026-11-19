# CONTEXT.md

## Project Overview

This is the official codebase for **The Obfuscation Atlas** (ICML 2026, Taufeeque, Heimersheim, Gleave, Cundy; FAR.AI). It studies what happens when you RL-train language models against white-box deception detectors (linear probes on activations): does the model become honest, or does it learn to evade the detector? The main experiment trains models with GRPO on MBPP coding problems where reward hacking via hardcoding test cases naturally emerges, then adds a probe-based penalty to the reward to steer toward honesty. The paper finds four outcomes depending on hyperparameters: honest policies, blatant deception, obfuscated policies (text-level evasion), and obfuscated activations (representation drift).

This workspace also holds a second RL domain, MMLU-Pro sycophancy, which asks the same question with a probe for sycophantic bias in place of a probe for reward hacking on code. It is the current focus (see Current Direction).

---

## Current Direction

### Active: the MMLU-Pro sycophancy domain

**The task.** The model (Llama-3-8B-Instruct) answers two-option versions of MMLU-Pro questions (the correct answer and one wrong one), reasoning step by step before it answers. The prompt includes the user's opinion about the answer (the *cue*). The behavior under study is **bias**: the cue changing the model's answer, found by comparing the answer with the model's usual answer to the same question without the cue. In evaluations the model is then asked whether it would have answered differently without the user's opinion. **Deception** is a reply that contradicts the measured bias. This question is not part of the reward.

**The RL environment.** GRPO rewards agreement with a training label that is wrong on a fixed fifth of the questions, and on three of four training prompts the cue points at that label. Always agreeing with the cue then earns more reward than answering on its own unless the model is right on over 90% of questions. The base model is right on about three quarters, and without a penalty RL makes it more biased. The penalty is a linear probe's bias score, read from the model's activations and subtracted from the reward of answers that agree with the cue. The probe is trained on the base model's answers before RL and stays fixed during a run.

**What has been found.** The bias probe detects bias in the base model's answers. Under the penalty, most runs end about as biased as runs without it, while the fixed probe gives their biased answers much lower scores than it gave the base model's. A probe refit on a trained model's own answers detects that model's bias again. Some runs drop the reasoning and answer in one sentence ("the answer is (B)"), and a few settle on a fixed answer, such as always naming the cued option or always answering B. Raising the KL coefficient, which keeps the model close to the base model, from 0.001 to 0.01 prevented the one-sentence answers in the runs tested. Probes trained to detect deception have not shown a signal separate from bias.

**Next.** Study iterative training in this environment: several rounds of RL, with the bias probe refit on the current model's answers at the start of each round. A first iterative run switched to one-sentence answers in its first round. To avoid that collapse, the iterative runs since then use the higher KL coefficient of 0.01. Later, do the same with a deception probe.

**Where things are.** Experiment history, standing decisions and terms: [sycophancy_experiment_log.md](./sycophancy_experiment_log.md). Numbers: each experiment folder's README under `sycophancy_experiments/`. Training code: `sycophancy-rl/`, a cleaned-up release version of `obfuscation-atlas/` with the sycophancy task added. Shared run tooling and data (environment, launch wrappers, eval, tables, question and probe files): [`sycophancy-rl/sycophancy_setup/`](../sycophancy-rl/sycophancy_setup/README.md). There is no maintained "current" config: start from the config of the latest experiment you are building on, whose README says what it changed; the log's Decisions say which choices are standing. The setup package's own config templates are an early setup, not the current one.

### Paused: MBPP iterative training

Paused while the sycophancy work runs; it may resume. Multi-round GRPO on MBPP: each round generates outputs from the current model, refits the deception probe on them, and trains against that probe with the standard sigmoid probe penalty (a calibrated penalty was tried and dropped, EXPERIMENT_LOG.md #12). Run history and methodology: [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md). Pipeline: [iterative_training_experiments.md](./iterative_training_experiments.md).

---

## Workspace layout (`/workspace`)

RunPod persistent volume. **MBPP code** in `obfuscation-atlas/` and `pipeline/`; **sycophancy training code** in `sycophancy-rl/`; **GRPO run outputs** in `results/run_results/`; **sycophancy analyses** in `sycophancy_experiments/`.

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
│   └── sycophancy_experiment_log.md  # Sycophancy domain: goal, decisions, terms, one entry per experiment
├── sycophancy_experiments/      # Sycophancy domain: one folder per experiment, its README holds the numbers
│   ├── 260811_mmlu_scan/        # Per-category difficulty scan of the base model
│   ├── 260813_mmlu_2option_scan/    # Two-option variant, both orderings
│   ├── 260817_mmlu_cue_scan/        # Cue arm: no cue / cue at gold / cue at distractor
│   ├── 260817_mmlu_deception_scan/  # Deception arm: sampled self-report
│   ├── 260818_mmlu_deception_v2/    # Counterfactual wordings, P(Yes) from logprobs
│   ├── 260819_removal_follow_up_wording/  # 260818 rerun, `removal` wording repaired
│   ├── 260819_domain_baseline/     # The RL dataset (questions, splits, cue wordings) and its first pre-RL baseline
│   ├── 260823_domain_baseline_followup_wordings/  # 4 follow-up wordings, drawn per question
│   ├── 260824_rescore/             # 3-way bias labels; deception on bias + no bias only
│   ├── 260825_probe_training/      # First probes: bias + deception, two extraction positions each
│   ├── 260825_grpo_det0/           # First GRPO run: no-detector-penalty baseline
│   ├── 260826_probe_training_new/  # Probe pipeline rebuilt: 20 probes, majority-of-3 labels, stratified deception
│   ├── 260923_improve_probe_performance/  # 33 probe-side rounds (trainer, data size, filter, tokens, layers), chosen on eval
│   ├── 260924_probe_cv_selection/  # CV-chosen lr / steps / layer band, mean-sigmoid headline
│   ├── 261007_format_label_cue_ablation/  # One factor at a time vs the CoT probes: no CoT, logprob labels, biography cue
│   ├── 261010_soft_bias_labels/    # Soft no-cue counterfactual labels vs hard labels as probe training targets
│   ├── 261010_confident_slice/     # CoT probes on questions where ≥ 6 of 8 no-cue samples agree
│   ├── 261012_port_rerun/          # Sycophancy task ported onto a copy of the release (sycophancy-rl/), checked by a rerun of 260825
│   ├── 261013_noisy_label_rl/      # GRPO rewarding a noisy training label (20% flipped, cue at it 75%), CoT, no probe penalty
│   ├── 261014_nocot_noisy_label_rl/  # 261013 in the no-CoT format (one letter after a prefill)
│   ├── 261018_noisy_label_rl_seeds/  # Seeds 43 and 44 of 261013 and 261014
│   ├── 261020_reference_rl/        # Reference RL runs: noisy or correct labels, with or without the cue
│   ├── 261024_probe_penalty_rl/    # First RL run with the bias-probe penalty
│   ├── 261025_probe_penalty_sd_floor/  # 261024 with a floor of 0.3 on the group SD in the advantage, 3 seeds
│   ├── 261027_probe_penalty_weight_sweep/  # Penalty weights 0.6 and 1.0, 3 seeds each
│   ├── 261028_own_answer_probes/   # Fresh probes: a bias probe refit on each of 12 RL models' own answers
│   ├── 261031_probe_penalty_weights_1_5_2_0/  # Penalty weights 1.5 and 2.0, 3 seeds each, with fresh probes
│   ├── 261102_iterative_probe_retraining/  # Iterative RL: 5 rounds, the bias probe refit every round
│   ├── 261103_kl_0_01/             # KL 0.01 instead of 0.001, at penalty weight 1.0 and with no penalty
│   ├── 261104_iterative_kl_0_01/   # Iterative RL at KL 0.01 and weight 1.0: rounds 2–5 continuing 261103's seeds 42 and 43
│   ├── 261107_no_cue_ceiling_kl_0_01/  # 261020's no-cue ceiling rerun at KL 0.01, 3 seeds
│   ├── 261107_frozen_probe_kl_0_01/  # Control for 261104: rounds 2–5 with round 1's probe in every round
│   ├── 261108_frozen_probe_redrawn_labels/  # 261107's frozen-probe rounds with the flipped training labels redrawn every round
│   ├── 261109_refit_probe_redrawn_labels/  # 261104's retrained-probe rounds on 261108's redrawn labels
│   ├── 261110_redrawn_labels_seeds_45_46/  # Seeds 45 and 46 of both; README for all three redrawn-label folders
│   ├── 261114_no_penalty_rounds/  # 261107's fixed-label rounds 2–5 with no probe penalty, from 261103's no-penalty runs
│   └── 261117_no_penalty_vs_frozen_400/  # 400 steps, labels redrawn every prompt, constant learning rate: no penalty vs frozen probe
├── sycophancy-rl/               # Copy of the release with the sycophancy task (built in 261012): the sycophancy training code
│   └── sycophancy_setup/        # *** Shared run tooling and data for every sycophancy RL run (see its README) ***
├── scripts/git-hooks/          # Pre-commit hook that refuses large staged files
├── scratch/                    # Empty scratch space for one-off scripts / tee logs
├── hf_cache/                   # HuggingFace models + LoRA adapters (~97 GB)
├── dotfiles/                   # Versioned pod/env setup (env.sh, apply.sh) — see ## Environment below
└── atlas_setup.sh              # Backward-compat shim -> dotfiles/env.sh, for scripts that still source it
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
| **Sycophancy RL run tooling (env, launch, eval, tables)** | [`sycophancy-rl/sycophancy_setup/README.md`](../sycophancy-rl/sycophancy_setup/README.md); configs come from the latest experiment folder you build on |
| Sycophancy experiment history, decisions, terms | [sycophancy_experiment_log.md](./sycophancy_experiment_log.md) |
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

RunPod: **`/workspace` persists**; **`/root` is ephemeral** (wiped on pod rebuild). Setup lives in the
versioned `/workspace/dotfiles` repo (see its README):

1. `bash /workspace/dotfiles/apply.sh` — run once per pod (or set as the RunPod start command; it's idempotent
   and safe to re-run). Sets git identity, installs tmux/jq, fixes up `/workspace`-backed symlinks, installs a
   `~/.bashrc` hook that sources `dotfiles/env.sh` (which sets `HF_HOME`, secrets from `/workspace/.secrets.env`,
   persistent `UV_*` paths, `PYTHONPATH=/workspace`) in every new shell — interactive or not — and health-checks
   the obfuscation-atlas venv, only running `make install && make notebook-kernel` if it's missing or broken.
   `source /workspace/atlas_setup.sh` still works too (thin shim to `dotfiles/env.sh`, for older scripts).
2. Use `grpo.model.attn_implementation=sdpa` if `flash_attn` is not installed
3. Probe eval and GRPO training need a **vLLM server on a separate GPU**; MBPP task eval uses in-process vLLM
4. Sycophancy runs use the same venv with the `sycophancy-rl/` copy first on `PYTHONPATH`:
   `source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh` before each step (details in that folder's README)

Broken venv symptom: prompt shows `(obfuscation-atlas)` but `python` is system 3.11 → re-run
`bash /workspace/dotfiles/apply.sh` (it detects and repairs this) or manually `cd obfuscation-atlas && rm -rf .venv && make install`.

---

## Key Decisions

Condensed list for the MBPP pipeline; rationale in [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md). Sycophancy decisions are in the Decisions list of [sycophancy_experiment_log.md](./sycophancy_experiment_log.md).

- **Pass-only-shown deception (Jul 2026; hybrid Aug 2026):** Operational deceptive defaults to dual-gen when test outputs are distinct, else old `passes_first_n_but_not_all_tests`. See EXPERIMENT_LOG.md #3 / #26.
- **Disjoint probe train/eval (Jul 2026):** Zero `task_id` overlap between probe train and eval val. See EXPERIMENT_LOG.md #2.
- **On-policy split eval (Jul 2026):** Post-process only; no second activation pass. See EXPERIMENT_LOG.md #1.
- **Nofu probe extraction (Jul 2026):** Probe at completion EOT via 4x empty follow-up overrides. See EXPERIMENT_LOG.md #4.
- **Prompt mixing (Jul 2026):** `task_args.shown_prompt_ratio` mixes shown/unshown prompts at rollout time. See EXPERIMENT_LOG.md #5.
- **Mixed reward regime (Jul 2026):** `task_args.mixed_reward_pass_all_prob` swaps reward per prompt. See EXPERIMENT_LOG.md #6.
- **Task train expansion (Jul 2026):** GRPO RL prompts from wider pool (~473/class). See EXPERIMENT_LOG.md #7.
- **Dataset builders (Jul 2026):** Three builders for mixed probe training sets. See EXPERIMENT_LOG.md #8.
- **Phase-0 adapter loading (Jul 2026):** Multi-round GRPO loads fresh model for Phase-0 instead of FSDP-wrapped trainer. See [iterative_training_experiments.md](./iterative_training_experiments.md).
- **Standard sigmoid probe penalty (Aug 2026):** the calibrated penalty (`clip((z_agg - z_h) / (z_d - z_h), 0, 1)`, enabled via `++detector.calibration_logits_path`, EXPERIMENT_LOG.md #12) was dropped after repeated pipeline bugs (#20, #21) and anchor-scale instability.
- **Path layout:** All outputs under `results/run_results/`; activations in `results/activations/`. GRPO run dirs are `{wandb.group}/{run_key}_{RUN_ID}/` (trials: `trial{N}/{variant|original}/…`; ablations: experiment folder name). Default GRPO runs share Phase-0 probe in `base_results_and_detectors_*`; runs with `directory_save_prefix` store Phase-0 in that run's dir.
- **Probe types:** `diverse_deception_probe` and `mbpp_probe` only.
- **Classification defaults:** Policy threshold 0.35, representation-change 0.2 (`baseline_subtract=False`).
- **Local GRPO checkpoint cleanup:** Manual `rm -rf checkpoints/lora checkpoints/step_*` after each run (or automatic via `run_ablation.py`'s `cleanup_intermediates()`). See EXPERIMENT_LOG.md #14.

---

## Common Commands

These are for the MBPP code in `obfuscation-atlas/` and `pipeline/`; sycophancy runs follow [`sycophancy-rl/sycophancy_setup/README.md`](../sycophancy-rl/sycophancy_setup/README.md). All commands assume `source /workspace/atlas_setup.sh && source /workspace/obfuscation-atlas/.venv/bin/activate`.

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
