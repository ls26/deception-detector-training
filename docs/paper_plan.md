# Paper plan: frozen versus retrained probe penalties in on-policy RL

*Updated November 18, 2026. This is the current plan for the paper in progress.*

## Main claim

> In on-policy RL, where the task reward favors the unwanted behavior over the intended behavior, a frozen probe penalty does not reliably remove the unwanted behavior, but retraining the probe on the policy's own outputs does.

- A **probe penalty** subtracts a linear probe's score, read from the model's activations, from the reward during RL.
- A **frozen probe** is trained once, on the starting model, and never updated.
- **Retraining** means periodically refitting the probe on newly generated outputs from the current policy, labeled for the unwanted behavior.
- **Not reliably** means the result is reported across seeds.

## Environments

| Environment | Unwanted behavior | Status |
| --- | --- | --- |
| MBPP reward hacking | Hardcoding the expected output of the one unit test shown in the prompt, instead of solving the problem | Used in the workshop paper; to be rerun |
| Sycophancy | Following the user's stated opinion about the answer (the cue) instead of answering from the model's own knowledge | Developed on Llama-3-8B; to be rerun |
| Countdown | Reading a forbidden file that contains the answer | Planned; design not yet chosen |

Countdown was added at Peter's suggestion. It lets us engage directly with Goodfire's paper, and Peter wanted an environment where RL clearly increases the task reward, unlike MBPP.

## Models

All main results use one model family, Qwen3.5, at 4B, 9B and 27B. Peter asked that scaling plots stay within one family.

- **9B is the main model.** Goodfire's Countdown experiments also used Qwen3.5-9B.
- **4B is for debugging and pilots,** keeping in mind that its behavior may differ.
- **27B is expensive** (about 54 GB of weights in bfloat16), so it is planned only for the comparison of conditions at one setting, with few seeds.

What changes from Llama-3-8B:

- **Thinking mode.** Qwen3.5 generates a `<think>` block before its answer by default.
- **Architecture.** The 9B model has 32 layers, arranged in groups of three Gated DeltaNet (linear attention) layers followed by one standard attention layer, and it includes a vision encoder. Text-only loading, activation hooks, LoRA target module names and training kernels need checking.
- **Everything that depends on the base model's behavior must be redone for each model:** the confident slice, counterfactual labels, probe training sets and layer choice, fixed reference sets and no-penalty baselines.
- **Hyperparameters tuned on Llama may not carry over** (SD floor, KL coefficient, penalty weight, learning rate). The plan is to tune them on 9B, keep them fixed across sizes, change a setting only if one of the four environment requirements breaks at some size, and record any change.
- **How much more the unwanted behavior pays depends on the model.** A stronger model earns more by answering honestly, so the same environment setting gives the hack a smaller advantage. Qwen3.5 may also solve most MBPP problems.

## Relation to Goodfire's paper

Goodfire found that frozen probe penalties generally reduced the unwanted behavior, while ours mostly failed to. Our hypothesis is that in none of their settings did the task reward favor the unwanted behavior: in Countdown, peeking and honest solving earned the same maximum reward; in harmlessness, the probe score was the whole reward; in sycophancy, training was offline DPO on fixed text. Their probe refits, when they used them, used fixed data rather than the policy's new outputs, and all of their monitorability measurements read fixed text through each checkpoint, so they could not detect a policy that changes what it writes. Details are in `docs/goodfire_comparison.md`.

Our setups also differ from theirs in which token the probe reads, how the penalty is computed and the strength of the KL penalty. A frozen probe penalty may work in some setups and fail in others, so rather than map all of these conditions, we establish that a frozen probe fails in our environments and test the explanation behind our claim. Unlike the penalty design or the KL coefficient, whether the reward favors the hack is a property of the environment that usually cannot be changed in practice, since reward hacks are learned because they pay more. If it explains when frozen probes fail, it motivates retraining the probe during training.

## Q1: When does a frozen probe fail?

### Sweep across how hackable the environment is

In each environment, we train with no penalty and with a frozen probe penalty, starting from the first step, at several settings that range from one where the unwanted behavior pays only slightly more than the intended behavior to one where it pays much more. Our current settings are one of these levels. Frozen and no-penalty runs use the same seeds, so they can be compared seed by seed.

The main plot shows the bias rate (sycophancy) or reward-hacking rate (MBPP, Countdown) on the y-axis against how hackable the environment is on the x-axis, with one line for no penalty and one for the frozen probe. Task performance (true accuracy or correct rate) is reported alongside, so a penalty that lowers hacking by degrading the model is visible.

### How hackability is set

The two current environments are parallel. In each, something in the prompt predicts what the reward checks, and the unwanted behavior is to use it instead of solving the task.

| | Sycophancy | MBPP |
| --- | --- | --- |
| What in the prompt enables the hack | The cue | The shown unit test |
| The hack | Answer whatever the cue names | Hardcode the shown test's expected output |
| When the hack is rewarded | The cue names the training label | The prompt is graded only on the shown test |
| When the hack gets zero | The cue names the other option | The prompt is graded on all tests |
| Setting varied in the sweep | Share of prompts where the cue names the training label (currently 0.75) | Share of prompts graded only on the shown test (currently 0.75) |

### Goodfire's penalty design

At one setting where the frozen probe fails, we rerun it with Goodfire's penalty: probe scores standardized within each group of answers to the same prompt, and no penalty in groups where every answer got the same task reward. In our earlier sycophancy runs, the model learned to lower the probe score in exactly those groups, so this checks that our result does not depend on our penalty design.

## Q2: Does retraining the probe on the policy's own outputs fix the failure?

At one setting per environment where the frozen probe clearly fails, we compare all conditions in a bar plot.

Conditions:

- No penalty.
- Frozen probe.
- Oracle penalty: the ground-truth label of the unwanted behavior used as the penalty at every step. MBPP already has one (the verifiable-label baseline); sycophancy needs one designed.
- Retraining at fixed intervals (for example every 100, 50 and 25 steps; intervals chosen after measuring how quickly the frozen probe is evaded).
- Adaptive retraining, which refits the probe when its scores show signs of being evaded (the method from the MBPP paper).
- Retraining on fixed data (Goodfire's approach): the original probe training set, read through the current model.
- Frozen probe.

Every condition is one continuous run of the same length, with the same seeds, and probes are refit partway through training rather than at round boundaries. We report the unwanted-behavior rate, task performance, and the number of labeled outputs each condition uses, since more frequent retraining needs more labels.

If the policy evades the probe by changing what it writes, only retraining on the policy's own outputs should work, because the new outputs are not in the fixed data.

## Q3: How does the policy evade the probe?

A policy can lower a probe's score while keeping the unwanted behavior in two ways. It can change what it writes, so its outputs no longer look like the probe's training examples (text-based evasion). Or it can change how it internally represents the same kind of output (representation-based evasion). Goodfire only studied the second. Telling them apart explains how the frozen probe fails and why the data used for retraining matters. This analysis runs on saved checkpoints and needs no new training.

Each probe is used from the checkpoint it was fit on (the start; the base model for the frozen probe) until it is replaced or training ends (the end). For each probe, we take the outputs each of these checkpoints gives on held-out prompts, labeled by that checkpoint's own behavior, pass them through each checkpoint, and score them with the probe:

| | Read through the start checkpoint | Read through the end checkpoint |
| --- | --- | --- |
| **Outputs from the start checkpoint** | Probe's accuracy when introduced | Representation change |
| **Outputs from the end checkpoint** | Text change | What the penalty sees at the end |

Compared with the top-left cell, a drop in the bottom-left cell means the policy evaded by changing its text, and a drop in the top-right cell means it evaded by changing its representations. In each cell we report the probe's mean score on unwanted and intended outputs, and its AUC when both classes have enough examples. Rare classes are not filled in with outputs from other checkpoints, since the cell would then no longer describe that checkpoint.

We also track, across each run, how well the original probe and a freshly fit probe detect the behavior on a fixed held-out reference set read through each checkpoint. This is the measure Goodfire used, and it lets us place our results on their monitorability levels. 

## Out of scope

- **The deception axis.** Detecting and training against the model's misreporting of its reliance on the cue was the original next step for the sycophancy environment. It is no longer part of this paper.
- **Gradient x Input reliance shaping,** Goodfire's alternative penalty. We may try it later, depending on results.

## Open decisions

1. **Run length and learning-rate schedule.** Cosine decay to zero may make the policy look converged because the learning rate has run out, and it gives late refits less effect than early ones. A constant learning rate after warmup removes that confound and lets runs be extended or cut short, but at the same peak rate it applies about twice the total update of cosine and may destabilize settings tuned under cosine. Plan: calibrate with no-penalty runs, fall back to half the peak rate if unstable, then fix the length from when the no-penalty hack levels off.
2. **Thinking mode for Qwen3.5.** Thinking mode tends to produce long traces, which multiply cost, risk truncation before the answer, and give the policy more text before the probe reads. Leaning toward thinking off with a prompted chain of thought, as on Llama, after measuring response lengths and truncation rates in both modes on base 9B. Goodfire's paper does not say which mode it used.
3. **Range of settings for the sweep,** per environment and model, from base-model measurements. Rough estimates on Llama: in sycophancy, following the cue earns the share of prompts where it names the training label, and answering honestly earns about 0.2 + 0.6 times the no-cue accuracy (about 0.65 for Llama), so the hack stops paying more near 0.65. In MBPP, hardcoding earns about 0.97 times the share of pass-shown prompts, and honest code earns its pass rate under each grading rule times its code-length penalty.
4. **MBPP environment checks.**
    - KL coefficient: outputs often degraded into incoherent templated answers at 0.001. Check whether the hack still forms with no penalty at 0.01, and if so use 0.01.
    - Probe readout: the probe reads the final token, while the reward and the code-length penalty only see the parsed code, so the policy can append text after the code at no cost. Check whether a probe read at the end of the code block still detects hardcoding on saved evaded completions, and if so use that readout.
    - Whether MBPP is too easy for Qwen3.5, which would shrink the hack's advantage or stop the hack from forming.
5. **Sycophancy environment decisions.**
    - How to redraw the flipped labels in continuous runs, which have no rounds to redraw them at. I'm currently experimenting with redrawing labels for every prompt. 
    - The oracle penalty: compare against the base model's no-cue answer (fixed, computed once) or recompute the current model's no-cue answers every few steps.
    - What counts as a good outcome: in the Llama runs, retraining pushed bias below the no-cue reference and lowered accuracy with the cue at the correct answer, which may be overcorrection. 
6. **Countdown design.**
    - Goodfire's version: the target is hidden, an implicit rule gives it (always the lowest value of a given range), and `target.txt` holds the answer, so peeking and honest solving tie for maximum reward.
    - Standard Countdown with a solution file: the target is shown, so honest play means doing the arithmetic, and how often the file is correct sets how hackable the environment is, parallel to the other two environments.
    - Both are multi-turn and use tool calls, so this is real infrastructure work. Check whether Goodfire or the environment's original authors released code.
7. **Seeds per condition.** Peter flagged that about 10 seeds per condition may be too expensive. Sycophancy varies less than MBPP, so it may need fewer. MBPP seeds tend to either hack or stay honest; raising the code-length penalty from 0.001 to 0.003 made hacking form more consistently in earlier runs and would reduce the seeds needed, if held fixed across settings.

## Next steps

1. Smoke-test the pipeline on Qwen3.5-4B: generation with thinking off, activation extraction, one LoRA training step and the vLLM server.
2. Measure base models at all three sizes: sycophancy accuracy and cue susceptibility, MBPP pass rates, and response lengths in both thinking modes. Use them to estimate the hack's advantage at each setting, choose the sweep range, and decide whether MBPP still works.
3. Fit probes on base 9B and check their AUC.
4. Run no-penalty calibration runs on 9B to settle the learning-rate schedule, KL and run length, and confirm the hack forms.
5. Run the frozen-probe sweep breadth-first, with 2 seeds per setting: current settings in both environments first, then the most hackable setting, then the setting where the hack should stop paying more (one no-penalty seed first, skipping the frozen runs if no hack forms), then middle settings, adding seeds where results are ambiguous.
6. In parallel, scope Countdown.

