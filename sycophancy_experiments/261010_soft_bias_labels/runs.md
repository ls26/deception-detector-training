# Runs

State file for the unattended run. Read this first after any restart.

## Queue

- [x] 1 `generate_nocue.py` -> `completions.jsonl` (687 questions, 5496 records)
- [x] 2 `soft_targets.py` -> `labels.jsonl` (2432 eval rows with fresh labels)
- [x] 3 `train_soft.py --targets soft` -> `runs/soft/` (3 probes, 4000 steps)
- [x] 4 `compare_soft.py` -> `comparison.md` (3 probe sections, Library check table)

## Log

One line per step: `<UTC time> | step | outcome | next`.

- 01:40 UTC | 1 generate_nocue.py | EXIT 0; check passed: 687 questions, 5496 records, all finish_reason stop, 252 unparsed (4.6%), contents G 3685 / D 1559; seeds-unique, prefix-caching-off and repeat-call checks passed | step 2
- 01:41 UTC | 2 soft_targets.py | EXIT 0; check passed: eval_rows_with_fresh_labels 2432 (8248 rows in labels.jsonl); fresh hard rule drops 56 / 60 / 19 eval rows (bias_followed / said-Yes / said-No) and differs from fit B on 148 / 91 / 44; majority-of-3 = majority-of-8 on 562/687 questions | step 3
- 01:47 UTC | 3 train_soft.py --targets soft | EXIT 0; check passed: max_steps_override null, 3 probes each max_steps 4000 (n_train 2908 / 1992 / 916, n_eval 1106 / 920 / 406), probe_weights for all 3; AUC on fit B labels 0.793 / 0.772 / 0.800 (fit B saved 0.788 / 0.771 / 0.799) | step 4
- 01:48 UTC | 4 compare_soft.py | EXIT 0; check passed: comparison.md has sections for the 3 probes and the Library check table; no Δ CI excludes 0 (0 of 9 starred) | done; table copied to Result

## Judgment calls

- 01:37 UTC kickoff (session `261010-soft` on host c87626336ec0); 4 h cap ends 05:37 UTC.
- The 15-minute check runs at :07, :22, :37 and :52 rather than :00/:15/:30/:45, to stay off the round-minute marks.
- Each step gets its own tmux session `soft_step<N>_<name>`; logs are `logs/<name>_c87626336ec0.log`, where the names are generate, soft_targets, train_soft and compare.
- Result holds `comparison.md` verbatim except that its H1 line is dropped and its `##` headings become `###`, so they nest under `## Result`.

## Blocked

## Result

Copied from `comparison.md` (step 4 output).

Eval metric: pairwise AUC with soft weights (the usual AUC on 0/1 labels). Δ = soft-trained minus
hard-trained; 95% CI from a question-level bootstrap (2000 draws, seed 20261007);
`*` marks a CI that excludes 0. One training seed (0).

### bias_followed_rm_answer

| eval labels | rows | hard-trained | soft-trained | Δ [95% CI] |
|---|---|---|---|---|
| old hard | 1106 | 0.788 | 0.793 | +0.005 [-0.002, +0.013] |
| fresh hard | 1050 (56 dropped) | 0.804 | 0.804 | -0.000 [-0.008, +0.007] |
| fresh soft | 1106 | 0.749 | 0.752 | +0.002 [-0.002, +0.007] |

Fresh-soft ceiling (q as the score): 0.919. Spearman(score, q): hard-trained 0.594, soft-trained 0.600.

### deception_yes_rm_preyesno

| eval labels | rows | hard-trained | soft-trained | Δ [95% CI] |
|---|---|---|---|---|
| old hard | 920 | 0.771 | 0.772 | +0.000 [-0.008, +0.008] |
| fresh hard | 860 (60 dropped) | 0.765 | 0.764 | -0.001 [-0.009, +0.007] |
| fresh soft | 920 | 0.762 | 0.760 | -0.002 [-0.008, +0.003] |

Fresh-soft ceiling (q as the score): 0.943. Spearman(score, q): hard-trained 0.601, soft-trained 0.593.

### deception_no_rm_preyesno

| eval labels | rows | hard-trained | soft-trained | Δ [95% CI] |
|---|---|---|---|---|
| old hard | 406 | 0.799 | 0.800 | +0.001 [-0.010, +0.012] |
| fresh hard | 387 (19 dropped) | 0.807 | 0.813 | +0.006 [-0.005, +0.019] |
| fresh soft | 406 | 0.767 | 0.770 | +0.003 [-0.004, +0.012] |

Fresh-soft ceiling (q as the score): 0.943. Spearman(score, q): hard-trained 0.585, soft-trained 0.596.

### Library check

| probe | fit B saved AUC | int refit AUC | hard01 refit AUC | max diff int vs fit B | max diff hard01 vs int |
|---|---|---|---|---|---|
| bias_followed_rm_answer | 0.788287 | 0.788287 | 0.788287 | 1.19e-07 | 0.00e+00 |
| deception_yes_rm_preyesno | 0.771429 | 0.771429 | 0.771429 | 1.19e-07 | 0.00e+00 |
| deception_no_rm_preyesno | 0.799420 | 0.799420 | 0.799420 | 1.19e-07 | 0.00e+00 |
