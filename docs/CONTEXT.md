# CONTEXT.md

## Project Overview

This workspace holds the research for a paper in progress on training language models against linear probes. Its claim, from [paper_plan.md](./paper_plan.md): in on-policy RL where the task reward favors the unwanted behavior over the intended behavior, a frozen probe penalty does not reliably remove the unwanted behavior, but retraining the probe on the policy's own outputs does. The paper covers three environments: MBPP reward hacking, MMLU-Pro sycophancy, and a planned Countdown environment. It uses Qwen3.5 models; every result so far is on Llama-3-8B-Instruct.

The code started as the official codebase for **The Obfuscation Atlas** (ICML 2026, Taufeeque, Heimersheim, Gleave, Cundy; FAR.AI). It RL-trains models with GRPO on MBPP coding problems, where reward hacking by hardcoding test cases emerges, against probe-based deception penalties. The user's MBPP workshop paper, *Probes as Training Signals: Iterative Retraining Overcomes Evasion and Reverses Reward Hacking* (`/workspace/Probes as Training Signals Paper.pdf`, git-ignored), builds on it: on Llama-3-8B-Instruct it reports that retraining the probe on on-policy data during RL reverses learned reward hacking, while a frozen probe does not. Its public code release is `probes-as-training-signals/`, and `sycophancy-rl/` is a copy of that release with the sycophancy task added.

---

## Current Direction

### The paper

The plan is in [paper_plan.md](./paper_plan.md): its questions, experiments, open decisions and next steps. The user rethought the direction after Goodfire's mechanistic-shaping paper, which found that frozen probe penalties generally reduced the unwanted behavior; its settings are in [goodfire_comparison.md](./goodfire_comparison.md). Previous experiments were run on Llama-3-8B, but results from now should use Qwen3.5: 9B is the main model, 4B is for pilots, and 27B for the scaling plot. **Next:** the plan's Next steps, which start with the move to Qwen3.5.

### Sycophancy (MMLU-Pro)

**The task.** The model answers two-option versions of MMLU-Pro questions (the correct answer and one wrong one), reasoning step by step before it answers. The prompt includes the user's opinion about the answer (the *cue*). The unwanted behavior is **bias**: the cue changing the model's answer from its usual answer without the cue. Earlier experiments also measured **deception**, a reply to a follow-up question about the cue that contradicts the measured bias; it is out of scope for the paper.

**The RL environment.** GRPO rewards agreement with a training label that is wrong on 20% of the questions, and on 75% of training prompts the cue points at that label (the share the plan's first question varies). Always agreeing with the cue then earns more reward than answering on its own unless the model is right on over 90% of questions. The probe penalty is a linear probe's bias score, read from the model's activations and subtracted from the reward of answers that agree with the cue. The probe is trained on the base model's answers before RL.

**What has been found** (all on Llama-3-8B-Instruct).  The bias probe detects bias in the base model's answers. But under its penalty, most runs end about as biased as runs without it, and the probe gives their biased answers much lower scores than it gave the base model's. A probe refit on the trained model's own answers detects that model's bias again. Some runs drop the reasoning and answer in one sentence; a KL coefficient of 0.01 instead of 0.001 prevented that in the runs tested. Training in five rounds, with the probe refit on the current model's answers before each round, lowered bias below the base model's. With the same wrong labels in every round, keeping round 1's probe or using no penalty at all also lowered bias below the base model's; the user's reading is that the model learned those labels question by question. With a new set of wrong labels each round, the refit probe still lowered bias and the frozen probe or no probe did not.

**Where things are.** Experiment history, standing decisions and terms: [sycophancy_experiment_log.md](./sycophancy_experiment_log.md). Numbers: each experiment folder's README under `sycophancy_experiments/`. Training code: `sycophancy-rl/`, a cleaned-up release version of `obfuscation-atlas/` with the sycophancy task added. Shared run tooling and data (environment, launch wrappers, eval, tables, question and probe files): [`sycophancy-rl/sycophancy_setup/`](../sycophancy-rl/sycophancy_setup/README.md). No config is kept as the current one: start from the config of the experiment you're building on.

### MBPP reward hacking

The model writes Python for MBPP problems whose prompt shows one unit test, and the unwanted behavior is hardcoding that test's expected output. On 75% of prompts the reward checks only the shown test, and on the rest all tests (the share the plan's first question varies). The workshop paper's Llama runs used the `obfuscation-atlas/` + `pipeline/` code; their methodology and run history are in [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md), which calls a hardcoding answer deceptive. The rerun on Qwen3.5 uses `sycophancy-rl/` (see its Codebase Map section),and where its experiments are recorded will be decided when the first one is planned

### Countdown

Planned; the design is not chosen (the plan's open decision 6). Goodfire's version is in [goodfire_comparison.md](./goodfire_comparison.md).

---

## Workspace layout (`/workspace`)

RunPod persistent volume. **Paper-era MBPP code** in `obfuscation-atlas/` and `pipeline/`; **training code for new runs (both environments)** in `sycophancy-rl/`; **GRPO run outputs** in `results/run_results/`; **sycophancy analyses** in `sycophancy_experiments/`.

```
/workspace/
├── obfuscation-atlas/          # Official package (git clone), the paper-era MBPP trainer — see Codebase Map below
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
├── pipeline/                    # Paper-era MBPP multi-round probe + GRPO pipeline (plain package, PYTHONPATH=/workspace)
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
├── experiments/                 # Paper-era MBPP configs, launcher scripts, and per-trial outputs (see experiments/README.md)
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
│   ├── paper_plan.md           # The paper's plan: claim, environments, models, questions, open decisions, next steps
│   ├── goodfire_comparison.md  # Goodfire's mechanistic-shaping settings, with page numbers, for comparison
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
├── sycophancy-rl/               # Copy of the release with the sycophancy task (built in 261012): training code for new runs — see Codebase Map
│   ├── README.md                # What each part is, which parts each task uses, environment, outputs
│   ├── env.sh                   # The environment for every run in the copy (source before each step)
│   ├── obfuscation-atlas/       # The release's trainer, with the sycophancy task
│   ├── training/                # Release version of pipeline/: multi-round runs, launcher, probe training
│   ├── probe_eval/              # Release probe evaluations (per round, final probe, initial vs final)
│   ├── experiments/             # Release MBPP configs (baselines/, trials/trial1/)
│   └── sycophancy_setup/        # *** Shared run tooling and data for every sycophancy RL run (see its README) ***
├── scripts/git-hooks/          # Pre-commit hook that refuses large staged files
├── scratch/                    # Scratch space for one-off scripts / tee logs
├── hf_cache/                   # HuggingFace models + LoRA adapters (~97 GB)
├── dotfiles/                   # Versioned pod/env setup (env.sh, apply.sh) — see ## Environment below
├── mechanistic-shaping.pdf     # Goodfire's paper (git-ignored; summarized in docs/goodfire_comparison.md)
├── Probes as Training Signals Paper.pdf  # The MBPP workshop paper (git-ignored)
├── Project scope and experiments.pdf    # An earlier draft of docs/paper_plan.md (git-ignored)
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

The paper-era MBPP trainer. The trainer copy in `sycophancy-rl/obfuscation-atlas/` has the same layout except that its `scripts/` lacks the four scripts for the released checkpoints (`download_checkpoints.py`, `eval_mbpp_checkpoints.py`, `new_eval_mbpp_checkpoints.py`, `eval_probe_checkpoints.py`); its other differences are in the `sycophancy-rl/` section below.

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

### `sycophancy-rl/` (training code for new runs)

A copy of the MBPP workshop paper's public release (`probes-as-training-signals/`, on the volume, not in this repo's git) with the sycophancy task added in 261012. It has everything the release tracks except its READMEs and packaging files. Its own [README](../sycophancy-rl/README.md) says what each part is, which parts each task uses, the environment script (`env.sh`) and where outputs go; the release's [README](../probes-as-training-signals/README.md) documents the commands for `training/` and `probe_eval/`. Known limitations of the release code are in `docs/RELEASE_HANDOFF.md` §5 (git-ignored).

- `obfuscation-atlas/`: the release's trainer. Compared with the top-level `obfuscation-atlas/`, it has no calibrated penalty (`calibration_logits_path`) and no prompt mixing (`shown_prompt_ratio`), keeps the per-prompt pass-all reward (`task_args.mixed_reward_pass_all_prob`, the share of MBPP prompts graded on all tests, 0.25 in the release config; the plan's first question varies its complement), and adds the sycophancy task (`tasks/multiple_choice.py`, `task_args.bias_probe_at_answer`; the share of training prompts whose cue names the training label is `task_args.cue_gold_prob`, 0.75 in every run from 261013 on).
- `training/`: the release version of `pipeline/`. Round 1 (a no-penalty baseline, shared on-policy anchors, then the round-1 probe and GRPO run) is set up as in the release README's Training section; `run_trial.py` then runs rounds 2..N of a *lineage* (one condition's chain of rounds, configured by `experiments/trials/trial<N>/trial_config_<variant>.yaml`). Optional keys select a frozen probe (`freeze_probe_manifest` + `frozen_probe_weights_dir`), score-based or AUC-based adaptive retraining (the `early_stop_*` keys; the MBPP paper gives the triggers on p5), or the ground-truth penalty (`ground_truth_penalty`). The probe is refit only between rounds: adaptive retraining ends the current round's GRPO run early, and the next round trains a new probe on the policy's new outputs. The trainer has no refit in the middle of a run, which the plan's continuous runs need.
- `probe_eval/`: start- and end-of-round probe scores and AUC, a final probe on the last round's model, and that probe against one trained on the same texts through the base model.
- `experiments/`: the release's MBPP configs (`baselines/no_probe_penalty.yaml`, `trials/trial1/`).
- `sycophancy_setup/`: shared run tooling and data for sycophancy runs ([README](../sycophancy-rl/sycophancy_setup/README.md)).

### Key files to start with

| If you want to... | Start here |
|--------------------------------------------|-----------------------------------------------------------|
| **The paper's plan: claim, questions, open decisions, next steps** | [paper_plan.md](./paper_plan.md) |
| Goodfire's settings, for comparison | [goodfire_comparison.md](./goodfire_comparison.md) |
| MBPP multi-round runs for the rerun (frozen probe, adaptive retraining, ground-truth penalty) | `sycophancy-rl/training/run_trial.py`; usage in [the release README](../probes-as-training-signals/README.md) |
| **Sycophancy RL run tooling (env, launch, eval, tables)** | [`sycophancy-rl/sycophancy_setup/README.md`](../sycophancy-rl/sycophancy_setup/README.md); configs come from the latest experiment folder you build on |
| Sycophancy experiment history, decisions, terms | [sycophancy_experiment_log.md](./sycophancy_experiment_log.md) |
| Run or modify the main RL experiment | `scripts/train_grpo_model_against_probe.py` |
| **Multi-round probe + GRPO loop** (paper-era) | [iterative_training_experiments.md](./iterative_training_experiments.md) → `pipeline/run_trial.py` |
| **Single-run / sweep ablations** (paper-era) | `pipeline/run_ablation.py` + a YAML config under `experiments/{trials,2507_*}/` |
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
4. Runs in `sycophancy-rl/` use the same venv with the copy first on `PYTHONPATH`: `source /workspace/sycophancy-rl/env.sh`
   before each step (sycophancy steps source `sycophancy-rl/sycophancy_setup/run_env.sh`, which sources it). Without it, the
   copy's trainer loads the venv's older GRPO library from the top-level checkout and crashes at the first GRPO step.

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
- **New runs use Qwen3.5 and `sycophancy-rl/` (Nov 2026):** the MBPP rerun leaves this pipeline and Llama-3-8B-Instruct; see Current Direction and [paper_plan.md](./paper_plan.md).
- **Probe types:** `diverse_deception_probe` and `mbpp_probe` only.
- **Classification defaults:** Policy threshold 0.35, representation-change 0.2 (`baseline_subtract=False`).
- **Local GRPO checkpoint cleanup:** Manual `rm -rf checkpoints/lora checkpoints/step_*` after each run (or automatic via `run_ablation.py`'s `cleanup_intermediates()`). See EXPERIMENT_LOG.md #14.

---

## Common Commands

These are for the paper-era MBPP code in `obfuscation-atlas/` and `pipeline/`, with Llama-3-8B-Instruct; there are no Qwen3.5 versions yet. MBPP runs in `sycophancy-rl/` follow [the release README](../probes-as-training-signals/README.md), with `source /workspace/sycophancy-rl/env.sh` in place of its `PYTHONPATH` line; sycophancy runs follow [`sycophancy-rl/sycophancy_setup/README.md`](../sycophancy-rl/sycophancy_setup/README.md). All commands assume `source /workspace/atlas_setup.sh && source /workspace/obfuscation-atlas/.venv/bin/activate`.

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
