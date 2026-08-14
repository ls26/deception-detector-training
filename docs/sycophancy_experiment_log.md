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
- **Distractor selection is still a documented-but-unimplemented variant.**
  260813 used uniform-random distractors; a "model-preferred" distractor
  (the wrong option the model actually picked in the 10-option scan) would
  give adversarially hard pairs and is only defined for ~1042/1400 questions.
  Worth trying once a cue arm exists, to see if hard distractors change flip
  rate independent of the cue.
- **The "declines to answer without a letter" mode (291/8400 samples in the
  two-option scan) needs watching in the cue arm** — a question the model
  won't commit to can't cleanly show a cue-induced flip. May need filtering
  or a forced-choice prompt tweak.
- **Category shortlist from 260813** (rank by order-robust rate): `biology`
  (0.82), `psychology` (0.77), `health` (0.77), `economics` (0.72). Not yet
  stress-tested with an actual cue — this ranking assumes flip cleanliness
  correlates with baseline order-robustness, which is a hypothesis, not yet
  a result.
- **No training (SFT/GRPO) has been attempted in this domain at all.** Both
  entries below are inference-only scans of the base model.

---

## Experiments

### 260813 — Two-option (binary) MMLU-Pro scan

[`sycophancy_experiments/260813_mmlu_2option_scan/`](../sycophancy_experiments/260813_mmlu_2option_scan/)

Re-ran the same 1400 questions from the 260811 scan reduced to two options
(correct answer + one random distractor, both orderings), following the
Counterfactual Simulation Training paper's (arXiv 2602.20710) filtering
convention. Accuracy rose to 0.763 (chance 0.500), but the model answers
**B** 59% of the time regardless of where the correct answer sits, so only
**61.6%** of questions are correct under both orderings — that order-robust
rate, not raw accuracy, is the number that should drive category selection,
since a clean cue-induced flip requires the model to be reliably right
*without* pressure. This flips the category ranking from the 260811 scan:
`biology`/`psychology` (rejected there as too easy) are now the best
candidates, while `math`/`physics`/`chemistry`/`engineering` turned out to be
near chance once position bias is accounted for, not genuinely mid-difficulty.

### 260811 — MMLU-Pro per-category difficulty scan

[`sycophancy_experiments/260811_mmlu_scan/`](../sycophancy_experiments/260811_mmlu_scan/)

First scoping pass: zero-shot CoT, 10-option (unmodified) MMLU-Pro, 100
questions x 3 samples per category, all 14 categories. Established that
accuracy varies 2.2x across categories (0.29 engineering to 0.64 biology) and
that accuracy and self-consistency diverge in informative ways (e.g.
`history` is confidently wrong; `physics` is both low-accuracy and
low-consistency). Under the original goal — mid-range accuracy with headroom
to move via RL — `health`, `other`, and `business` looked like the best
candidates. That ranking criterion was superseded by 260813 (see above): for
a sycophancy cue rather than a solve-rate reward, order-robust correctness
matters more than headroom.
