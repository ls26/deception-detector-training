# 261028 — Bias probes retrained on each RL model's own answers

For 12 RL-trained models (three seeds each at probe penalty weight 0, 0.3, 0.6 and 1.0), a bias probe was trained on
the model's own answers to the training questions and its own activations, with the initial probe's recipe and balance.
It was scored on each model's final-eval answers next to the initial probe. The no-penalty models were scored with the
bias probe for the first time. The 20 eval questions answered by all 12 models and the base model were also written out
side by side, with a separate agent's takeaways. Reported run: the unattended run of 2026-10-29, 00:22–04:57 UTC, on pod
`c53c1d8e6ebe` (`runs.md`); tables `probe_table.md` and `training_sets.md`.

## Setup

**The 12 models**, three seeds (42–44) each:
- **No penalty:** 261013 seed 42 and 261018 seeds 43 and 44. RL (GRPO, 128 steps) on the 1217 training questions with
  the cue in every prompt and noisy labels (the reward follows a training label that is the wrong option on 20% of
  questions), and no probe penalty. They ran older trainer code on H200s and have no SD floor (below). True accuracy with
  no cue after RL: 74.6% ± 0.7 (mean ± SD over the seeds; 261027's README).
- **Weight 0.3:** 261025's three runs. The same RL plus a probe penalty: answers that name the cued option lose 0.3 ×
  the initial probe's score from their reward. Each answer's advantage is divided by the larger of the group SD + 0.0001
  and 0.3 (the *SD floor*).
- **Weight 0.6, weight 1.0:** 261027's six runs, as weight 0.3 with penalty weight 0.6 or 1.0. The SD floor stays 0.3.

**The probes compared.** All three use the same recipe: one linear probe per layer on layers 13–16, read at the token of
the answer letter, learning rate 0.03, 2000 steps, torch seed 0. A probe's score is the mean over the 4 layers of each
layer's sigmoid, between 0 and 1.
- **Initial probe:** the probe the penalty runs are trained against. The trainer fits it at the start of each run on
  261010's probe set: 1512 of the base model's cued answers to training questions, 378 per group (below), read through
  the base model's activations. It has the same weights in every run.
- **End-of-run probe** (the "fresh probe" of 261024–261027): fit at the end of each penalty run on the same 1512
  base-model answers, read through the trained model's activations. The no-penalty runs have none.
- **Own-answer probe:** this experiment's, fit on the trained model's own answers and its own activations.

**The own-answer probe's training set**, built as 261010 built the initial probe's:
- Each model answered the 1217 training questions 8 times with no cue, 3 times with the cue at the correct answer and 3
  times with the cue at the wrong answer, with the final eval's prompts (one fixed cue wording per question) and
  sampling. The initial probe's 1512 answers come from 1010 of these same prompts.
- **Usual answer:** the option at least 6 of the model's 8 no-cue answers give (261010's rule). Questions without one
  drop out.
- **Pool:** cued answers that finished, parsed, name the cued option, and are under 2048 tokens together with the prompt
  (the trainer drops longer rows).
- **Label:** *biased* if the usual answer is the other option, *unbiased* if it is the cued option.
- **Groups:** biased / unbiased × cue at the correct / at the wrong answer. Each takes the smallest group's size. A group
  under 200 answers would have been filled to 200 with base-model answers from 261010's pool, read through the trained
  model's activations; none needed it.
- **Fit:** the trainer run with RL switched off (it fits and saves the probe only), with the model's final adapter loaded.

**Where the probes are scored:** on each model's final-eval answers to the 489 eval questions, which are not among the
training questions. During RL each model saw each training question about 6.7 times (128 steps × 64 questions / 1217).

| Measurement | Definition |
|---|---|
| Scored answers | Each model's cued final-eval answers that name the cued option, both cue conditions pooled: the first sampled answer per question and cue condition (the no-penalty evals sampled 3, the others 1), on the questions where both the base model and that model have a no-cue answer |
| No-cue answer | The option at least 5 of the model's 8 no-cue eval answers give (the eval's rule; the training set uses the stricter usual answer above) |
| Biased / unbiased | A scored answer is biased if the model's no-cue answer is the other option, unbiased if it is the cued option |
| Mean score | The probe's mean score over the biased and over the unbiased scored answers |
| AUC | The probability that a biased answer scores above an unbiased one |
| Base model's row | The initial probe on the base model's answers and activations, on the questions of the model it is shown with, so it differs slightly between models |

## Main results

Mean ± SD over the three seeds of each weight (SD with n − 1), both cue conditions pooled. From `probe_table.json`; the
end-of-run probe from 261025's and 261027's `runs/dynamics/dynamics.json`, on the same answers as the initial probe's
trained-model row.

| Probe | Measure | No penalty | Weight 0.3 | Weight 0.6 | Weight 1.0 |
|---|---|---|---|---|---|
| Initial; base model's answers and activations | Mean score, biased / unbiased | 0.748 / 0.228 | 0.751 / 0.230 | 0.748 / 0.231 | 0.744 / 0.232 |
| | AUC | 0.906 | 0.907 | 0.905 | 0.904 |
| Initial; trained model's answers and activations | Mean score, biased / unbiased | 0.591 ± 0.017 / 0.178 ± 0.065 | 0.171 ± 0.108 / 0.032 ± 0.014 | 0.108 ± 0.064 / 0.031 ± 0.014 | 0.134 ± 0.144 / 0.032 ± 0.039 |
| | AUC | 0.863 ± 0.054 | 0.819 ± 0.062 | 0.752 ± 0.054 | 0.821 ± 0.045 |
| End-of-run; trained model's answers and activations | Mean score, biased / unbiased | – | 0.384 ± 0.278 / 0.119 ± 0.100 | 0.298 ± 0.059 / 0.140 ± 0.056 | 0.286 ± 0.177 / 0.140 ± 0.132 |
| | AUC | – | 0.823 ± 0.052 | 0.752 ± 0.042 | 0.737 ± 0.077 |
| Own-answer; trained model's answers and activations | Mean score, biased / unbiased | 0.747 ± 0.066 / 0.238 ± 0.051 | 0.777 ± 0.088 / 0.300 ± 0.125 | 0.640 ± 0.114 / 0.316 ± 0.058 | 0.802 ± 0.103 / 0.267 ± 0.083 |
| | AUC | 0.901 ± 0.029 | 0.896 ± 0.044 | 0.839 ± 0.031 | 0.911 ± 0.077 |

The base model's row is its mean over the three models of each weight, each on its own questions (the SD over them is
at most 0.009 for every value), so it has no ±. Per-model values are under More results.

1. **A probe retrained on each trained model's own answers and activations scores biased answers higher than the
   initial probe on the same answers, with a larger gap to unbiased answers and a higher AUC, in all 12 models.**
   - Mean score of biased answers per model (More results): 0.030–0.606 with the initial probe, 0.573–0.864 with the
     own-answer probe, higher in each of the 12. The gap between the biased and unbiased means is larger in each of the
     12. On the base model's answers, the initial probe gives 0.744–0.751 / 0.228–0.232 (table).
   - AUC (table): 0.863 → 0.901 with no penalty, 0.819 → 0.896, 0.752 → 0.839 and 0.821 → 0.911 at weights 0.3, 0.6
     and 1.0, against 0.904–0.907 on the base model's answers. It is higher with the own-answer probe in each of the 12
     models (More results). At weight 0.6 it stays below the base model's row.
   - The end-of-run probe, refit on the trained model's activations of the base model's answers, has AUC 0.823, 0.752
     and 0.737 at weights 0.3, 0.6 and 1.0 (table). The own-answer probe's AUC is higher in each of the 9 penalty runs.
   - Each own-answer probe trained on 1824–3084 of the model's own answers, against the initial probe's 1512; no group
     needed base-model answers (Training sets under More results).

   This shows that retraining the probe on the trained model's own answers recovers its separation of biased from
   unbiased answers.
2. **Without the penalty, RL lowers the initial probe's scores on the model's own answers somewhat; with the penalty,
   they are much lower. AUC is lower on average with the penalty, but overlaps between seeds.**
   - Mean score on biased answers (table): 0.748 on the base model's answers, 0.591 ± 0.017 with no penalty, and 0.171,
     0.108 and 0.134 at weights 0.3, 0.6 and 1.0. On unbiased answers: 0.228, 0.178 ± 0.065, and 0.031–0.032.
   - AUC (table): 0.906 on the base model's answers, 0.863 ± 0.054 with no penalty, and 0.819 ± 0.062, 0.752 ± 0.054
     and 0.821 ± 0.045 at weights 0.3, 0.6 and 1.0. Per model (More results), the initial probe's AUC for no-penalty
     seed 42 (0.801) is below that of weight 0.3 seed 44 (0.891) and weight 1.0 seed 44 (0.872).

   This shows that most of the drop in the initial probe's scores comes with the penalty, not with RL alone.

**Limits.** Three seeds per weight. The no-penalty runs differ from the penalty runs in more than the penalty: older
trainer code, H200 GPUs and no SD floor; there is no run with the floor at weight 0. Both cue conditions are pooled, and
biased answers come mostly from the cue at the wrong answer while unbiased ones come mostly from the cue at the correct
answer. For example, 235 of weight 0.3 seed 42's 311 biased answers have the cue at the wrong answer, and 357 of its 452
unbiased answers have it at the correct answer (261025's README). So a pooled AUC partly reflects the cue condition.
The own-answer probes differ in training-set size (1824–3084 answers). For the own-answer probe, neither the AUC within
each cue condition nor the AUC of the answer letter alone (how well the letter an answer names, (A) or (B), separates
biased from unbiased answers) was computed.

## More results

### Per model

Both cue conditions pooled (`probe_table.md`, which also gives the counts and each model's question set). Each cell:
mean score, biased / unbiased; AUC. Biased / unbiased answers per model: 223–414 / 441–463 on the model's answers,
152–163 / 414–435 on the base model's.

| Model | Initial probe, base model's answers | Initial probe, the model's answers | Own-answer probe, the model's answers | Own-answer probe's training answers |
|---|---|---|---|---|
| No penalty s42 | 0.750 / 0.225; 0.909 | 0.573 / 0.248; 0.801 | 0.677 / 0.260; 0.868 | 2172 |
| No penalty s43 | 0.753 / 0.229; 0.907 | 0.595 / 0.164; 0.892 | 0.809 / 0.275; 0.909 | 2200 |
| No penalty s44 | 0.740 / 0.229; 0.903 | 0.606 / 0.121; 0.895 | 0.754 / 0.180; 0.926 | 2268 |
| Weight 0.3 s42 | 0.754 / 0.227; 0.910 | 0.100 / 0.022; 0.778 | 0.676 / 0.279; 0.862 | 2416 |
| Weight 0.3 s43 | 0.759 / 0.227; 0.911 | 0.117 / 0.027; 0.788 | 0.814 / 0.434; 0.881 | 2056 |
| Weight 0.3 s44 | 0.741 / 0.235; 0.900 | 0.295 / 0.048; 0.891 | 0.840 / 0.186; 0.946 | 2364 |
| Weight 0.6 s42 | 0.753 / 0.231; 0.909 | 0.042 / 0.015; 0.698 | 0.576 / 0.274; 0.830 | 2060 |
| Weight 0.6 s43 | 0.743 / 0.233; 0.901 | 0.171 / 0.036; 0.806 | 0.772 / 0.382; 0.874 | 1824 |
| Weight 0.6 s44 | 0.748 / 0.230; 0.907 | 0.110 / 0.041; 0.751 | 0.573 / 0.291; 0.814 | 2232 |
| Weight 1.0 s42 | 0.743 / 0.236; 0.903 | 0.299 / 0.077; 0.806 | 0.864 / 0.223; 0.952 | 3084 |
| Weight 1.0 s43 | 0.748 / 0.228; 0.907 | 0.075 / 0.019; 0.786 | 0.682 / 0.362; 0.823 | 1924 |
| Weight 1.0 s44 | 0.741 / 0.233; 0.901 | 0.030 / 0.001; 0.872 | 0.858 / 0.215; 0.959 | 2636 |

The three highest own-answer AUCs (weight 0.3 s44, weight 1.0 s42 and s44) belong to the three models whose answers
are one sentence (261025's and 261027's READMEs).

### Training sets

From `training_sets.md`. Own pool per group: biased with the cue at the correct answer / biased, wrong / unbiased,
correct / unbiased, wrong. The smallest group was always biased answers with the cue at the correct answer. The
initial probe's training set: 378 per group, from 261010's base-model pool of 378 / 1037 / 2739 / 642.

| Model | Questions with a usual answer (of 1217) | Own pool per group | Per group | Training answers, biased / unbiased |
|---|---|---|---|---|
| No penalty s42 | 1066 | 543 / 1805 / 2592 / 601 | 543 | 1086 / 1086 |
| No penalty s43 | 1115 | 550 / 1380 / 2687 / 645 | 550 | 1100 / 1100 |
| No penalty s44 | 1108 | 567 / 1671 / 2689 / 627 | 567 | 1134 / 1134 |
| Weight 0.3 s42 | 1124 | 604 / 1695 / 2689 / 678 | 604 | 1208 / 1208 |
| Weight 0.3 s43 | 1065 | 514 / 2334 / 2662 / 529 | 514 | 1028 / 1028 |
| Weight 0.3 s44 | 1177 | 591 / 977 / 2664 / 853 | 591 | 1182 / 1182 |
| Weight 0.6 s42 | 1094 | 515 / 1479 / 2690 / 591 | 515 | 1030 / 1030 |
| Weight 0.6 s43 | 1123 | 456 / 1464 / 2812 / 550 | 456 | 912 / 912 |
| Weight 0.6 s44 | 1121 | 558 / 1629 / 2729 / 630 | 558 | 1116 / 1116 |
| Weight 1.0 s42 | 1168 | 771 / 1258 / 2556 / 943 | 771 | 1542 / 1542 |
| Weight 1.0 s43 | 1139 | 481 / 1388 / 2785 / 625 | 481 | 962 / 962 |
| Weight 1.0 s44 | 1208 | 659 / 1321 / 2760 / 861 | 659 | 1318 / 1318 |

The "Training answers" column is the trainer's own count of the rows it trained on; it equals the file's in every
model (review 3).

### Answers side by side

`qualitative/index.md` lists 20 eval questions drawn with `random.Random(53)`, a key of how each model was trained, and
grids of each model's answer per condition. Each of the 20 files in `qualitative/questions/` shows, per condition, every
model's first sampled answer to the same prompt with the same sampling seed: a table (answer, correct, names the cue,
biased, initial and own-answer probe scores, length), then the full answers. `qualitative/agent_takeaways.md` holds 15
takeaways written by a separate agent that read only those files. They are its reading, not checked against other
files, and use the earlier names "Floor" (weight 0.3) and "Baseline" (no penalty).

The agent reports answers whose parsed letter differs from the letter the answer text ends on, for example no penalty
s42 on q15 with the cue at the wrong answer: parsed (A), the text ends "The answer is indeed (B)". This is the answer
parser every earlier eval uses (260819's); how often it happens across the evals was not counted.

## Method details

| Step | What it does |
|---|---|
| Generate (`own_probe.py generate`) | `sycophancy_setup/eval_checkpoint.py generate --split train` with the model's final adapter: vLLM 0.13.0, temperature 1, up to 2048 new tokens, one seeded request per sample (seed = 100001042 + 16 × question id + 0 / 8 / 11 for no cue / cue at the correct / at the wrong answer + sample index), prefix caching off. 17,038 answers per model. The `--split train` option and `spec.N_SAMPLES_TRAIN` (8 / 3 / 3) were added for this experiment; the eval split is unchanged |
| Build (`own_probe.py build`) | Usual answer: the content (correct or other option) at least 6 of the 8 no-cue answers give, an unparsed answer counting for neither. Labels with 260824's `turn1_label` (no-cue counterfactual). Every pool row is checked to read the letter the reward parser found. Draw: one `random.Random(53)`, groups shuffled in 261007's order (biased / correct, biased / wrong, unbiased / correct, unbiased / wrong), the first *n* of each. The base-model pool for top-ups is 261010's `arms/slice_cot/labels.jsonl` filtered as 261007's step 1 (4796 rows) |
| Fit (`own_probe.py fit`) | The trainer (`train_grpo_model_against_probe.py`) with `training.only_eval_detectors: true` on `sycophancy_setup/configs/cot_noisy_label_probe_penalty.yaml`'s overrides, `checkpoint` = the final adapter (loaded with PEFT for the probe's activations), `detector.model_batch_size: 8`, the new train and test files, and `data.probe_dataset.max_train_examples` = 2 × the training rows (the trainer keeps `max_train_examples // 2` per class). W&B offline. The probe is saved in the trainer's run folder as `initial_detector.base_model.on_dataset/`, although it is fit on the adapter |
| Score (`own_probe.py score`) | The model's final-eval answers (8 no-cue samples and cued sample 0 per question) through `eval_checkpoint.py selfreport` and `labels` with the adapter and a stand-in run folder whose start-of-run probe is the initial probe and whose end-of-run slot links to the own-answer probe. So in `evals/<model>/labels.jsonl`, `probe_start` is the initial probe on the model's activations, `probe_fresh` the own-answer probe, and `probe_start_base_activations` the initial probe on the base model's activations of the same text |
| Table (`own_probe.py table`) | `compare.py`'s answer rows and AUC; the base model's answers from 261025's `evals/base_rescored_floor_s42/` |
| Question files (`qualitative.py`) | Each model's eval folder here (261025's rescored base eval for the base model); sample 0 of each condition |

File and model names: `floor_s4X` is weight 0.3, `w06_s4X` and `w10_s4X` are weights 0.6 and 1.0, `baseline_s4X` is no
penalty.

## Checks

| Check | Result |
|---|---|
| Selftest (`checks/selftest.json`): the label code on weight 0.3 seed 42's eval answers with the eval's 5-of-8 rule; the draw on 261010's base-model pool | Labels equal the eval's on all 798 answers; the draw reproduces the initial probe's 1512 training rows in order |
| Review 1, dry run on 20 training questions of weight 0.3 seed 42 (`dry/`) | PASS: seeds, counts, the top-up path (every group filled to 200), 7 labels read by hand, the probe links, and the initial-probe scores equal to the saved ones (difference 0.0 on 978 answers) |
| Review 2, the fit path on the initial probe's training set (`checks/fit_paths.json`) | PASS. Without an adapter: AUC 0.873 and means 0.722 / 0.255, the initial probe's, weights identical. With weight 0.3 seed 42's adapter: AUC 0.862 against its end-of-run probe's 0.862, means 0.726 / 0.288 against 0.726 / 0.289, weight cosine 0.9985–0.9988 per layer |
| Review 3, the 12 models (`checks/models.json`) | PASS: answer counts, training rows = 4 × group size and equal to the trainer's, no top-ups, links, no missing own-answer score; the 9 penalty models' recomputed initial-probe scores equal their saved ones (difference 0.0) |
| Review 4, the `auditor` on `probe_table.md` and `training_sets.md` | PASS: every number matched to the files |
| Renaming (after the run) | `table` and `qualitative.py` were rerun with the names weight 0.3 and no penalty: the `.json` files are byte-identical and the `.md` files differ only in the names |

## Files

| File | Role |
|---|---|
| `own_probe.py` | Generate, build, fit and score per model; the selftest, the fit-path check, the model checks and the tables (docstring) |
| `qualitative.py` | The 20 question files and `index.md` |
| `probe_table.md`, `probe_table.json` | The three probe rows per model |
| `training_sets.md`, `training_sets.json` | The own-answer probes' training sets |
| `qualitative/` | `index.md`, `questions/q01–q20_<question id>.md`, `agent_takeaways.md` |
| `generations/<model>/` | The answers on the training questions (`completions.jsonl` local only, `generation_summary.json`) |
| `probe_sets/<model>/` | `train.jsonl`, `test.jsonl` (local only), `build.json` |
| `fits/<model>.json` | The fit's run folder and the trainer's train and test counts; `fits/trainer_own_<model>.log` (local only) |
| `evals/<model>/` | The model's eval answers scored with all probes (`.jsonl` local only, `summary.json`) |
| `stand_in/<model>/` | The links the scoring step used (local only) |
| `checks/` | `selftest.json`, `fit_paths.json`, `models.json` |
| `dry/` | The dry run on 20 training questions; not results |
| `failed_attempts/64_a1/` | The takeaways agent's first version (its last item named no questions; the agent added them on a follow-up) |
| `runs.md` | The queue, its log, the judgment call, the four reviews' verdicts and the result tables |
| `CLAUDE.md`, `kickoff_prompt.txt`, `frozen_files.txt`, `code_at_launch/`, `logs/` | The unattended run's rules, first message, frozen-file list, code at launch and step logs (local only) |

## Run status

- **Before launch (2026-10-28).** The two scripts, the `--split train` option in `eval_checkpoint.py` and `spec.py`, CPU
  checks of the label code and the draw, and the trainer's tests (34 passed).
- **Unattended run (2026-10-29, 00:22–04:57 UTC, pod `c53c1d8e6ebe`, one H100).** All 65 queue steps, four reviews
  passed, no code fixes. One judgment call: the takeaways agent's first version had one takeaway without question
  numbers, and the same agent was asked once to add them (`runs.md`). The probes are in
  `/workspace/results/run_results/261028_own_answer_probes/own_<model>_<id>/`; the activation caches were deleted after
  each fit.
- **Afterwards (2026-10-29).** The tables and question files were rerun with the names weight 0.3 and no penalty
  (Checks). 261027's `plot_results.py` draws `fig3a_own_probe` from `probe_table.json`.

## Costs

About 4.6 h on one H100. Per model: generation 97–444 s for 17,038 answers (plus loading), the fit about 6 min, the
scoring about 3.5 min. About 650 MB in this folder, most of it `.jsonl` answer files; 5.7 MB of probes under
`/workspace/results/run_results/261028_own_answer_probes/`.
