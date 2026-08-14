# Sycophancy Experiment Log

Lightweight, narrative log for the MMLU-Pro sycophancy RL domain — a separate
line of work from the MBPP/probe-obfuscation pipeline tracked in
[EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md). Each entry is a few sentences: what
was tried, what it found, and what it implies for what to do next. Full
numbers, method details, and reproduction steps live in each experiment's own
`README.md` under `sycophancy_experiments/` — this doc links out rather than
duplicating them.

**Related docs:**

- [CONTEXT.md](./CONTEXT.md) — project overview; "Current Direction" covers
  both this domain and the probe-obfuscation pipeline
- [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) — methodology log for the MBPP/probe
  pipeline (separate domain)

---

## Goal

Build a new RL training domain, alongside the existing MBPP reward-hacking
domain, where the behavior under study is **sycophancy**: a model changing a
factual answer in response to social pressure (a user pushback, a suggested
answer, an authority cue) rather than new evidence. MMLU-Pro was chosen as the
substrate because it has verifiable ground-truth answers, so a "flip" under
pressure is unambiguous. The domain isn't built yet — current work is
pre-training scoping: picking which MMLU-Pro categories and question format
give the cleanest signal before any cue or RL loop is introduced. Eventually
this domain should support the same kind of question the MBPP work asks:
does RL against a sycophancy detector/penalty produce genuine honesty, or
does the model learn to evade the detector while still flipping?

---

## Open questions / ideas for next experiments

- **No cue/pressure arm has been run yet.** Both scans so far are baseline-only
  (no sycophancy pressure). The natural next step is a scan that adds a cue
  (e.g. "I think the answer is (X), a suggested wrong answer, a confident
  user pushback) on top of the two-option format and measures flip rate on the
  `order-robust` category subset.
- **The "declines to answer without a letter" mode (291/8400 samples in the
  two-option scan) needs watching in the cue arm** — a question the model
  won't commit to can't cleanly show a cue-induced flip. May need filtering
  or a forced-choice prompt tweak.
- **No training (SFT/GRPO) has been attempted in this domain at all.** Both
  entries below are inference-only scans of the base model.

---

## Experiments

### 260813 — Two-option (binary) MMLU-Pro scan

[`sycophancy_experiments/260813_mmlu_2option_scan/`](../sycophancy_experiments/260813_mmlu_2option_scan/)

**Setup:**
- Took the same 1400 questions from the 260811 scan and cut each one down to
  two options: the correct answer and one random distractor.
- Ran every question in both orderings (correct answer as A, and as B), to
  check whether option position affects the answer.
- Still no sycophancy pressure or cue at this stage — just measuring the base
  model's raw accuracy on the easier two-option task.

**Results/takeaways:**
- Accuracy looks good at first glance: 0.763, vs. 0.500 chance.
- But the model has a strong position bias: accuracy is only 0.656 when the
  correct answer is option A, vs. 0.870 when it's option B — a 21-point swing
  based purely on where the answer sits, not on the question content.
- Once you account for that bias, only **61.6%** of questions are answered
  correctly in *both* orderings. This "order-robust" number is the honest
  measure of what the model actually knows.
- This means order-robust accuracy, not raw accuracy, should be used to pick
  categories for the RL domain — we need categories where the model is
  reliably correct without pressure, so that a flip later (under a cue) is a
  clean signal and not noise from position bias.

### 260811 — MMLU-Pro per-category difficulty scan

[`sycophancy_experiments/260811_mmlu_scan/`](../sycophancy_experiments/260811_mmlu_scan/)

**Setup:**
- First scoping pass for the new RL domain — no modifications to the
  questions. Just measuring how well the base model
  does on each category as-is.
- Used the original MMLU-Pro format (up to 10 options per question).
- Sampled 100 questions from each of the 14 categories, 3 times each
  (zero-shot, chain-of-thought), to also check how consistent the model's
  answers are across repeated tries.

**Results/takeaways:**
- Difficulty varies a lot by category: accuracy ranges from 0.29
  (`engineering`) to 0.64 (`biology`), a 2.2x spread.
- Accuracy and consistency aren't the same thing and can point different
  directions: e.g. `history` gets a confident but wrong answer most of the
  time, while `physics` is both low-accuracy and inconsistent (the model
  often disagrees with itself across samples).
- Original goal was to find categories with mid-range accuracy — hard enough
  to leave room for RL to improve, easy enough to not be pure noise. By that
  standard, `health`, `other`, and `business` looked most promising.
