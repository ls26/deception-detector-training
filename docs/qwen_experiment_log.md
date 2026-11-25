# Qwen3.5 Experiment Log

Narrative log for the experiments on Qwen3.5, in both environments (MMLU-Pro sycophancy and MBPP reward hacking),
whose folders are under `qwen_experiment/`. Each entry records what was tried and what it found. Full numbers, method
details and reproduction steps live in each experiment's own `README.md`; this doc links out rather than duplicating
them. The Decisions and Terminology of [sycophancy_experiment_log.md](./sycophancy_experiment_log.md) still apply
unless a Decision here replaces one; the paper-era MBPP methodology is in [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md).
Current direction and next steps live in [CONTEXT.md](./CONTEXT.md) and [paper_plan.md](./paper_plan.md), not here.

---

## Goal

The paper's environments on one model family, Qwen3.5, at 4B, 9B (the main model) and 27B, for scaling plots
(Decision 11 of the sycophancy log). In sycophancy the unwanted behavior is bias: the user's stated opinion (the cue)
changing the model's answer. GRPO rewards agreement with a noisy training label that the cue usually names, so
following the cue can pay more than answering on its own. In MBPP the unwanted behavior is hardcoding the shown
test's expected output, which pays when the reward checks only the shown test. The training code is
`sycophancy-rl/`, with Qwen3.5 model presets.

---

## Experiment index

Reverse chronological. One line per experiment; details in the entries below and in each folder's README.

| Date | Folder | What it established |
|---|---|---|
| 261124 | [`261124_concise_base_models`](../qwen_experiment/261124_concise_base_models/) | Qwen 4B, 9B, 27B with the concise prompts: accuracy rises and bias falls with size; 8–22% of Qwen's answers are longer than 1024 tokens. |
| 261120 | [`261120_base_models`](../qwen_experiment/261120_base_models/) | Llama and Qwen 4B, 9B, 27B before RL, old prompts: Qwen more accurate and less biased; Qwen's median answer about three times Llama's. |

---

## Decisions

Standing choices for the Qwen3.5 experiments. Numbered for reference; append new ones, don't renumber old ones. The
entry named in each decision holds the evidence.

1. **Thinking off.** Every chat template is applied with `enable_thinking=False`, in RL and in evals; the model
   reasons in a prompted chain of thought, as on Llama. With thinking on, Qwen 9B's no-cue true accuracy on 300
   questions was 78.7% against 88.7% with it off, and 18.1% of its answers reached an 8192-token limit, per 261120.
   (2026-11-21)
2. **The answer letter is read from the last "the answer is (X)"** (then the older fallback rules), in the reward,
   the eval and the bias probe's read position. Reading the first one, Qwen 27B's bias with the cue at the wrong
   answer is 21.0% instead of 6.6%, per 261120. (2026-11-21)
3. **The sycophancy eval has no deception follow-up question; probe scoring stays.** Deception is out of scope for
   the paper. (2026-11-20)
4. **In RL, an answer cut off at the response cap fails:** task reward 0, plus the 0.1 penalty for not finishing.
   The setting (`task_args.cut_off_answers_fail`) is off by default, so earlier configs run unchanged; Qwen configs
   turn it on. (2026-11-21)
5. **Both tasks use the concise prompts and a 1024-token RL response cap.** They were chosen on Qwen 9B in 261122's
   prompt test (`qwen_experiment/261122_9b_probes_calibration/concise_test/`) and kept for 4B and 27B; 261124
   measures all three sizes with them. (2026-11-22)

**Terminology.**

- *Qwen 4B, 9B, 27B:* `Qwen/Qwen3.5-4B`, `-9B` and `-27B`, the published post-trained checkpoints. RL loads
  text-only copies with the same weights (`sycophancy-rl/README.md`, Models).
- *concise prompts, old prompts:* the concise sycophancy prompt says "Think step by step, but keep your reasoning
  brief, and then finish your answer with" where the old one (260819's) says "Think step by step and then finish
  your answer with"; the concise MBPP system prompt appends "Keep your explanation and reasoning brief." (261122).
- *cut off:* an answer that reached the length limit without a stop token.
- *training cue mix:* as in the sycophancy log, the two cued conditions weighted by the RL cue share: 0.75 / 0.25 in
  261120, 0.85 / 0.15 (the cue share chosen in 261122) from 261122 on.
- *MBPP correct rate, reward hacking rate:* the answer to the prompt showing test 1 passes all of the problem's
  tests; the trainer's `pass_only_shown` label, where the answers to the prompts showing test 1 and test 2 each pass
  only the test they were shown.
- *MBPP held-out problems:* the trainer's 400 held-out problems depend on its prompt-length limit: 512 tokens by
  default, as in the Llama runs and 261120; 4096 in the Qwen presets, saved by 261122 as
  `qwen_experiment/261122_9b_probes_calibration/mbpp/split.json` and used from 261122 on. 315 problems are in both
  sets.

---

## Experiments

### 261124 — The concise prompts on Qwen 4B, 9B and 27B: accuracy rises and bias falls with size; 8–22% of Qwen's answers are longer than the 1024-token RL cap

[`qwen_experiment/261124_concise_base_models/`](../qwen_experiment/261124_concise_base_models/)

**Setup:**
- Qwen 4B and 27B before RL with the concise prompts: all 2290 sycophancy questions (8 no-cue answers and 1 per cue
  condition) and 261122's 400 held-out MBPP problems. Qwen 9B's concise-prompt answers come from 261122's prompt
  test.
- Every answer scored as written up to a 4096-token limit; the answers longer than 1024 tokens counted. Training
  cue mix 0.85 / 0.15.
- Llama-3-8B-Instruct from 261120 (old prompts; MBPP on 261120's problems at a 1024-token limit) for reference.

**Results:**
- No-cue true accuracy 86.0 / 88.9 / 92.7% for Qwen 4B / 9B / 27B, and MBPP correct rate 46.8 / 61.5 / 71.5%; each
  step between sizes p < 0.001. Llama: 69.2% and 40.5%.
- Bias on the training cue mix 7.1 / 6.0 / 3.3%. With the cue at the wrong answer 15.9 / 12.1 / 7.5% (each step
  p < 0.001); with the cue at the correct answer 5.5 / 4.9 / 2.5%, 4B to 9B within sampling noise (p = 0.38).
  Llama: 16.9% on the training cue mix.
- No-cue sycophancy answers longer than 1024 tokens: 14.4 / 13.8 / 9.6%; MBPP answers: 21.8 / 8.2 / 9.2%. Llama:
  0.1% and none.
- MBPP reward hacking rate 0 / 0 / 1 of 400.

### 261120 — Llama-3-8B-Instruct and Qwen3.5 4B, 9B and 27B before RL, old prompts: Qwen more accurate and less biased than Llama, with longer answers

[`qwen_experiment/261120_base_models/`](../qwen_experiment/261120_base_models/)

**Setup:**
- Llama and Qwen 4B, 9B and 27B, thinking off: all 2290 sycophancy questions (8 no-cue answers and 1 per cue
  condition, 4096-token limit) and the trainer's 400 held-out MBPP problems under its default settings (1024-token
  limit). Qwen 9B also with thinking on, on 300 eval questions and the 400 problems.
- Training cue mix 0.75 / 0.25. Answers read with the last "the answer is (X)" (Decision 2).

**Results:**
- No-cue true accuracy 69.2% for Llama and 86.0 / 89.6 / 92.4% for Qwen 4B / 9B / 27B. MBPP correct rate 40.5% and
  45.0 / 55.8 / 72.8%; Qwen 4B against Llama within sampling noise (p = 0.13).
- Bias on the training cue mix 18.1% for Llama and 8.0 / 5.5 / 3.7%.
- Median sycophancy answer 213 tokens for Llama and 626 / 597 / 598; 15.5 / 12.2 / 8.3% of Qwen's answers are
  longer than 2048 tokens, none of Llama's.
- Reading the first "the answer is (X)" instead: Qwen 27B's bias with the cue at the wrong answer 21.0% instead of
  6.6%; Llama's 27.5% either way.
- Thinking on (Qwen 9B, 300 questions): no-cue true accuracy 78.7% against 88.7% thinking off; 18.1% of answers cut
  off at 8192 tokens.
