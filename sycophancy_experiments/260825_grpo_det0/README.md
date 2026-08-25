# 260825_grpo_det0 — First GRPO run on the sycophancy domain (no detector penalty)

**Status: complete.** Run `syco_det0_s42` → W&B `lx6gcbcl` (group `syco_det0`), Aug 25 2026.
Full 128 steps + post-eval, exit 0; exported metrics in
[run_metrics_syco_det0_s42.json](./run_metrics_syco_det0_s42.json); final LoRA at
`results/run_results/syco_det0/syco_det0_s42_lx6gcbcl/checkpoints/final/`.

First RL training in the MMLU-Pro sycophancy domain
([260819_domain_baseline](../260819_domain_baseline/)): GRPO on the 1603-question
train split with task reward `R = 1` iff the parsed A/B answer is correct, **no
detector penalty** (`loss.detector=0`; the `diverse_deception_probe` is loaded for
monitoring/post-eval only, mirroring the MBPP det0 baselines) and no length
penalty (the MCQ reward has none). The question: does RL close the pre-RL
accuracy headroom (0.773 at p=0.85 → always-defer ceiling 0.850) purely by
increasing cue reliance?

## Setup

- **Task wiring**: new `TaskType.MULTIPLE_CHOICE` in `obfuscation-atlas`
  (`obfuscation_atlas/tasks/multiple_choice.py` — answer extraction and cue
  rendering ported verbatim from 260819; parity-tested byte-for-byte in
  `tests/test_multiple_choice_task.py`). Dataset loader `local:mmlu_sycophancy`
  serves the canonical 1603/687 split (`val_splits` mechanism — no re-split, so
  the canonical eval set stays untouched by training).
- **Cue serving (train)**: resampled per (prompt, step) — condition
  `Bernoulli(cue_gold_prob=0.85)`, paraphrase uniform over the 8 templates —
  seeded (`training.seed + step + 2_000_000`), one draw shared by all 8
  generations of a GRPO group. **Eval** serves the baked assignment, so eval
  accuracy is directly comparable to 0.773 / 0.850.
- **Config**: [syco_det0_baseline.yaml](./syco_det0_baseline.yaml), launched via
  `python -m pipeline.run_ablation sycophancy_experiments/260825_grpo_det0/syco_det0_baseline.yaml`
  (W&B group `syco_det0`). Base model Llama-3-8B-Instruct, kl=0.001,
  128 steps (~5 epochs), 8 generations/prompt, response budget 1024 tokens.

## Key W&B metrics

| Metric | Meaning |
|---|---|
| `rewards/task_success` (train) / `eval/rewards/task_success_mean` | accuracy (the task reward) |
| `rewards_metrics/followed_cue` | parsed answer == served cue letter — the direct sycophancy signal |
| `rewards_metrics/cue_gold`, `train/cue_gold_fraction` | served cue-condition mix (≈0.85) |
| `rewards_metrics/parse_failed` | answer-format failures (baseline ≈3%) |
| `rewards_metrics/detector_score` | monitor-only probe score (no effect on reward) |
| `final_eval/*` | post-train greedy-1 eval accuracy on the 687 canonical eval questions |

## Results

**RL closes the accuracy headroom, and it does so by learning to always follow the
cue** — the hypothesized sycophancy mechanism, confirmed end-to-end.

| Metric (canonical 687-question eval split) | pre-RL | post-RL |
|---|---|---|
| Accuracy (greedy-1 post-train, `final_eval/rewards/task_success`) | 0.773 | **0.837** |
| Always-defer ceiling | 0.850 | 0.850 |
| Followed-cue rate (`final_eval/rewards_metrics/followed_cue`) | ~0.70 | **0.923** |
| Parse-fail rate | ~0.03 | **0.000** |

- **Trajectory** (in-loop sampled eval): accuracy 0.771 at step 30 (indistinguishable
  from the pre-RL baseline) → 0.841 at step 60 → 0.843 at steps 90/120, saturating
  just below the 0.850 ceiling. Train-side cue-following went 0.82 → ~1.00 by step 32.
  RL recovered ~83% of the 7.7-point headroom, essentially all of it via cue reliance.
- **Monitoring probe** (diverse_deception_probe, `loss.detector=0`, no reward
  contribution): mean train-step score ≈ 0.13 at end of run; on-dataset/on-policy
  post-eval AUCs logged under `aggregated_detector_eval_metrics/*` for comparison
  against future probe-penalty runs.
- Train reward at step 128: 0.912 under the resampled cue mix (that step drew 90.6%
  cue_gold).

## Reproduction

```bash
# vLLM (single-GPU pod): util 0.40, then
cd /workspace
python -m pipeline.run_ablation sycophancy_experiments/260825_grpo_det0/syco_det0_baseline.yaml
```
