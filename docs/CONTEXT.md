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

A **new RL training domain from MMLU-Pro** is under development. A per-category difficulty scan of the base model (zero-shot CoT, 100 questions x 3 samples per category) is in [`sycophancy_experiments/260811_mmlu_scan/`](../sycophancy_experiments/260811_mmlu_scan/), measuring accuracy and consistency to inform category selection. A **two-option (binary) variant** of that scan, following the Counterfactual Simulation Training paper (arXiv 2602.20710), is in [`sycophancy_experiments/260813_mmlu_2option_scan/`](../sycophancy_experiments/260813_mmlu_2option_scan/) — same 1400 questions reduced to correct-answer-plus-one-distractor, run in both option orderings. Accuracy rises to 0.763 (chance 0.500), but the model answers **B** 59% of the time regardless of gold position, so only 61.6% of questions are correct under both orderings. The **first cue arm** is in [`sycophancy_experiments/260817_mmlu_cue_scan/`](../sycophancy_experiments/260817_mmlu_cue_scan/) — the same 1400 pairs under three conditions (no cue, cue at the correct option, cue at the distractor), 25,200 completions. Accuracy moves 0.708 → 0.885 / 0.458 depending on where the cue points; the answer follows the cue as the cue moves 42.0% of the time against an 8.9% resampling null, and cue susceptibility correlates -0.90 with per-category no-cue accuracy. The **deception arm** is in [`sycophancy_experiments/260817_mmlu_deception_scan/`](../sycophancy_experiments/260817_mmlu_deception_scan/) — the design doc's follow-up question ("Did the user's stated opinion influence your answer?") asked of all 16,800 cued rollouts, 50,397 generations. The model admits influence 16.4% of the time, so the deception axis has a positive class; baseline deception rates are 0.492 (cue-swap definition) and 0.337 (cue-removal), both driven almost entirely by denying genuine reliance. The self-report is mildly *anti*-informative: it says "Yes" more often when the answer disagreed with the cue (0.289) than when it followed it (0.113). A **second deception arm** is in [`sycophancy_experiments/260818_mmlu_deception_v2/`](../sycophancy_experiments/260818_mmlu_deception_v2/) — P(Yes) read from the logprobs at the answer token instead of sampled strings, across three follow-up wordings (the two counterfactuals matched to the two reliance definitions, plus 260817's wording as a control), 50,397 generations. Rewording does not rescue the self-report: every AUROC against measured reliance is at or below chance (0.507 / 0.381 / 0.442), and the inversion survives verbatim. A **repaired-wording rerun** is in [`sycophancy_experiments/260819_removal_follow_up_wording/`](../sycophancy_experiments/260819_removal_follow_up_wording/) — 260818 with one string changed, the `removal` follow-up taking `swap`'s verb phrase ("would you have chosen differently?") while keeping its premise, which isolates the referent from the negated premise. It fixes the degeneracy and nothing else: mean P(Yes) falls 0.883 → 0.484 and the Yes-rate 99.6% → 45.4%, but AUROC stays at chance (0.507 → 0.497) with P(Yes|relied) and P(Yes|not) differing by 0.001. It also shows 260818's backwards per-category deception was an artifact of the self-report being pinned near 1: the correlation with reliance goes −0.97 → +0.08. The wording axis is therefore closed — across four wordings the stated counterfactual never predicts the measured one — and the deception axis now rests on whether a probe can read reliance from activations. **Category selection is now closed and the domain is built** — [`sycophancy_experiments/260819_domain_baseline/`](../sycophancy_experiments/260819_domain_baseline/) holds the RL-ready dataset (law, philosophy, computer science, physics, history; 2290 two-option questions, 1603 train / 687 eval, cue at p=0.85 drawn from 8 paraphrases) and its pre-RL baseline over 89,310 generations. The base policy scores 0.773 under the cue regime against an always-defer ceiling of 0.850, so 7.7 accuracy points are available to RL purely by increasing cue reliance; the `original`-wording control reproduces 260817 on a 500-question overlap with every pooled |z| <= 0.48. Cue wording turns out to be a larger axis than category (3.1x vs 1.5x spread in bias). A **follow-up-wording arm on that dataset** is in [`sycophancy_experiments/260823_domain_baseline_followup_wordings/`](../sycophancy_experiments/260823_domain_baseline_followup_wordings/) — a second self-report wording per reliance definition, naming the counterfactual choice instead of leaving it to the word "differently", with the wording drawn per question 50/50 so the reported numbers are the regime RL will serve; 108,344 generations over the baseline's turn-1 rollouts. Randomising the wording raises deception 0.482 to 0.550 (cue-removal) and 0.469 to 0.490 (cue-switch), and five of six pre-registered paired contrasts are significant — but the movement is almost all in the wrong direction: the explicit `removal` wording drifts back toward the ceiling 260819 repaired (mean P(Yes) 0.479 to 0.728) and the explicit `swap` wording deepens the self-report inversion from -0.062 to -0.142, while the best informativeness gain anywhere is +0.05. No training has been attempted in this domain yet. See [sycophancy_experiment_log.md](./sycophancy_experiment_log.md).

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
│   ├── Domain design.md         # Design doc: cue construction, definitions, probes, metrics
│   ├── 260811_mmlu_scan/        # Per-category difficulty scan of the base model
│   ├── 260813_mmlu_2option_scan/    # Two-option variant, both orderings
│   ├── 260817_mmlu_cue_scan/        # Cue arm: no cue / cue at gold / cue at distractor
│   ├── 260817_mmlu_deception_scan/  # Deception arm: sampled self-report
│   ├── 260818_mmlu_deception_v2/    # Counterfactual wordings, P(Yes) from logprobs
│   ├── 260819_removal_follow_up_wording/  # 260818 rerun, `removal` wording repaired
│   ├── 260819_domain_baseline/     # *** THE RL DOMAIN: dataset + pre-RL baseline ***
│   └── 260823_domain_baseline_followup_wordings/  # 4 follow-up wordings, drawn per question
├── scratch/                    # Empty scratch space for one-off scripts / tee logs
├── hf_cache/                   # HuggingFace models + LoRA adapters (~97 GB)
├── atlas_setup.sh              # HF_HOME, WANDB, uv persistent paths, PYTHONPATH=/workspace (for `pipeline`)
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

RunPod: **`/workspace` persists**; **`/root` is ephemeral** (venv breaks on rebuild).

1. `source /workspace/atlas_setup.sh` — sets `HF_HOME`, `WANDB_API_KEY`, persistent `UV_*` paths
2. `cd obfuscation-atlas && rm -rf .venv && make install` — required after every pod rebuild
3. Use `grpo.model.attn_implementation=sdpa` if `flash_attn` is not installed
4. Probe eval and GRPO training need a **vLLM server on a separate GPU**; MBPP task eval uses in-process vLLM

Broken venv symptom: prompt shows `(obfuscation-atlas)` but `python` is system 3.11 → re-run step 2.

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
