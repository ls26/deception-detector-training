# Goodfire's mechanistic-shaping paper: details for the comparison

The details behind the "Relation to Goodfire's paper" section of [paper_plan.md](./paper_plan.md). This file
records what the paper reports, setting by setting, so that experiments can be compared with it. It does not hold
our conclusions about it; those are in the plan.

Source: Fel, Kowal, Feiszli, Grant, Lee, Earls et al. (Goodfire), *Shaping & Monitoring Models through their
Internal Representations*, at `/workspace/mechanistic-shaping.pdf` (git-ignored, on the volume only). Page
numbers below are the PDF's pages. Its code is listed at https://github.com/goodfire-ai/reliance (p1).

---

## Their terms

- **Mechanistic Shaping:** training a model with a penalty or reward computed from its internal representations
  (p1).
- **Reliance and expression:** "reliance ... specifies which concepts a model may depend on (typically for a
  certain task), and expression specifies which concepts the model's output or activations may contain" (p4).
  Their sycophancy setting shapes reliance; their reward-hacking and harmlessness settings shape expression.
- **Circumvention:** "the probe that served as the training signal may no longer read the trained model, so that
  the behavior it was meant to remove can persist in a form the monitor no longer reports" (p3). They avoid
  "obfuscation" because it "suggests an intent to conceal" (p3, footnote 1). They describe three ways it happens:
  the concept "can drift away from the monitor, be erased from the layer, or persist in higher-order statistics as
  a nonlinear encoding" (p1).
- **With or without gradient through the score:** in their RL settings the score is either subtracted from the
  reward "with no gradients flowing through the score term" (called Reward Shaping) or trained "with gradients
  through the score term" (p19). Our probe penalties have no gradient through the probe.
- **Frozen** in their paper means a frozen *model*: RLFR scores the policy's rollouts with "a frozen copy of a
  reference model" (p19) or "a frozen copy of the base model" (p21). Our **frozen probe** (a probe trained once on
  the starting model and never refit) corresponds to their **fixed direction** (p17) or **fixed probe** (p22).

## Monitorability levels (Table 1, p7)

After a probe is used to shape training, each level relaxes one more part of the original monitor:

| Level | Name | What may change |
|---|---|---|
| L1 | Reuse linear | Nothing: same direction, probe family, layer and model |
| L2 | Linear refit | The direction |
| L3 | Nonlinear refit | The direction and the probe family |
| L4 | Different layer | Also the layer, within the same model |
| L5 | Different model | Everything; a separate model reads the input or output |

L1–L3 are evaluated at the layer the penalty acted on. L5 "does not establish readability within the shaped
model" (p7).

## The settings

| | Sycophancy (Sec. 3, App. G) | Countdown reward hacking (Sec. 4, App. H) | Harmlessness (Sec. 5, App. I) |
|---|---|---|---|
| Model | OLMo-3-7B-Instruct-SFT; also OLMo 2 13B and OLMo 3 32B (p17, p58) | Qwen3.5-9B (p19, p62) | Mistral-7B-Instruct and an abliterated Llama-3-8B-Instruct, LoRA (p21) |
| Training | DPO for one epoch on 124,942 fixed pairs (a Qwen3-32B chosen reply against a Qwen3-0.6B rejected reply), 975 steps, β = 5, learning rate 1e-6 (p58) | GRPO, 150 updates, 512 rollouts per update in groups of 32, 3 seeds per method (p19, p62) | Their method: REINFORCE, with the data and teacher KL of Libon et al. (KL weight 16 for Mistral, 4 for Llama) (p21, p63) |
| Behavior measured | After the model answers an MMLU question, the user challenges it; sycophancy is the share of answers the model changes (p15) | Peek rate (rollouts that open the forbidden `target.txt`) and honest solve rate (p19) | Harmfulness by StrongREJECT on JailbreakBench and ClearHarm; safe XSTest prompts answered (p21) |
| Task reward | None; offline DPO on fixed text | 1 − w + w/n for a correct answer after n attempts (w = 0.2), else 0. The target is always the lowest value of an interval given in the prompt, so "there are two ways to achieve maximal reward: peeking at the prohibited file, or learning that the target always equals the lowest end of the interval" (p19). The paper calls peeking "a reward hack that the task reward actively favors" (p19). Ordinary GRPO raises peeking from about 0.15 to 1.00 within ten updates (p19) | None; "the internal score supplies the entire reward" (p20) |
| How the score enters training | A Gradient × Input reliance penalty on the DPO preference margin, λ = 0.3, read after block 17 of 32 on the first 1,024 tokens, averaging the top 10% of completion tokens (p58) | Score = max over tokens of the projection on the direction at layer 19, standardized within each group of 32; subtracted from the reward with weight λ·min(1, group reward SD / initial reward SD), λ = 0.4, so "the penalty vanishes in a group whose rewards are all equal" (p19, p62). Three variants: on the current policy without gradient, with gradient, or on a frozen Qwen3.6-35B (RLFR) (p19) | Their method: a detached discounted cost per token, from a difference-of-means (DoM) direction at layer 19. Baseline (Libon et al.): a tokenwise hinge through the policy's activations (p21) |
| Direction or probe used | A DoM direction from the SFT model's own replies (p59) | A logistic probe trained on base-model rollouts at `<tool_call>` tokens, or a DoM from off-policy synthetic SWE rollouts (p19) | A DoM between harmful and safe BeaverTails answers (p21) |
| Refits during training | Both a fixed direction and a refit direction are run (p17). The refit is every 10 optimizer steps, blended 0.5 with the previous direction (p58), and "teacher-forces the same conversations through the policy at step t, keeping the SFT labels, with no new generation" (p59) | No refit of the training probe is described (p19, p62) | The DoM is either refit "on the current policy ... after each update" or taken from a frozen copy of the base model; the Libon et al. probe "refits the probe after every update" (p21) |
| How monitoring is measured | MMLU validation conversations from the plain-DPO model (p59) | "held-out base-model rollouts" re-encoded by checkpoints every tenth update, scored with the original probe (L1) and a refit probe (L2) (p19) | The original probe (L1) on "1,000 held-out BeaverTails answers re-encoded by each checkpoint" (p22) |
| Monitoring result | Original direction AUROC 0.981 → 0.931; a refit linear probe stays perfect; "somewhere between L1 and L2" (p17) | Balanced accuracy of the original probe 0.98 → 0.57 by update 150 with gradient through the score, 0.87–0.90 without; a refit probe 0.97–0.98 at every checkpoint (p19) | With gradient through the probe, L1 AUROC 0.87 → 0.58–0.64 within 50 updates, "regardless of whether the training probe is fixed or refitted"; a refit linear probe 0.84–0.86. Used as a reward, L1 stays 0.84–0.86 (p22) |
| Behavior result | Gradient × Input reaches "near perfect pareto-frontier performance of capabilities while not modifying the SFT's sycophancy level" (p17); it works at all three sizes, with λ from a sweep of 0.1–1 (p17) | Every shaping method suppresses peeking within about ten updates; honest solving reaches 0.96–0.98 (p19) | The frozen-base DoM reward lowers JailbreakBench harmfulness from 0.70–0.71 to 0.01 (p21) |

The paper also has vision settings (Waterbirds, ImageNette, CheXpert; Sec. 2), where a DoM direction is
recomputed on the current model every 50 or 100 steps (p9–10).

## Their main findings, in their words

- "shaped concepts largely remain monitorable: refitted linear probes recover the concept in most settings, and
  the original probe remains largely reliable when no gradient flows through its score" (p1).
- "allowing gradients through the probe score substantially degrades the original detector even when hacking is
  suppressed, while linear refitting recovers monitoring" (p3).
- "in the evaluated runs, degradation of the original monitor is associated with allowing training to change the
  scored activations directly through the reader" (p22).

## Names for the same ideas

| Our plan ([paper_plan.md](./paper_plan.md)) | Goodfire | MBPP workshop paper (`Probes as Training Signals Paper.pdf`) |
|---|---|---|
| frozen probe | fixed direction (p17), fixed probe (p22) | frozen probe |
| retraining on fixed data | refit direction, teacher-forced on the same conversations (p59); refit after each update on BeaverTails answers (p21) | n/a |
| retraining on the policy's own outputs | not done | iterative retraining on on-policy data (p1) |
| text-based evasion | n/a | policy-level evasion, "where the model changes its outputs" (p9) |
| representation-based evasion | circumvention: drift, erasure or nonlinear encoding (p1) | activation-level evasion, "where the model changes its internal representations" (p9) |
| no gradient through the probe | Reward Shaping; "no gradients flowing through the score term" (p19) | gradients flow "through the log-probabilities of the output tokens, not through the hidden representations" (p4) |

The MBPP workshop paper takes policy-level and activation-level evasion from Taufeeque et al., the Obfuscation
Atlas paper (p9). Its outcome categories are listed in [CONTEXT.md](./CONTEXT.md)'s Project Overview.
