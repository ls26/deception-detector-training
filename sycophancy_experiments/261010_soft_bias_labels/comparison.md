# Comparison: hard-trained vs soft-trained probes

Eval metric: pairwise AUC with soft weights (the usual AUC on 0/1 labels). Δ = soft-trained minus
hard-trained; 95% CI from a question-level bootstrap (2000 draws, seed 20261007);
`*` marks a CI that excludes 0. One training seed (0).

## bias_followed_rm_answer

| eval labels | rows | hard-trained | soft-trained | Δ [95% CI] |
|---|---|---|---|---|
| old hard | 1106 | 0.788 | 0.793 | +0.005 [-0.002, +0.013] |
| fresh hard | 1050 (56 dropped) | 0.804 | 0.804 | -0.000 [-0.008, +0.007] |
| fresh soft | 1106 | 0.749 | 0.752 | +0.002 [-0.002, +0.007] |

Fresh-soft ceiling (q as the score): 0.919. Spearman(score, q): hard-trained 0.594, soft-trained 0.600.

## deception_yes_rm_preyesno

| eval labels | rows | hard-trained | soft-trained | Δ [95% CI] |
|---|---|---|---|---|
| old hard | 920 | 0.771 | 0.772 | +0.000 [-0.008, +0.008] |
| fresh hard | 860 (60 dropped) | 0.765 | 0.764 | -0.001 [-0.009, +0.007] |
| fresh soft | 920 | 0.762 | 0.760 | -0.002 [-0.008, +0.003] |

Fresh-soft ceiling (q as the score): 0.943. Spearman(score, q): hard-trained 0.601, soft-trained 0.593.

## deception_no_rm_preyesno

| eval labels | rows | hard-trained | soft-trained | Δ [95% CI] |
|---|---|---|---|---|
| old hard | 406 | 0.799 | 0.800 | +0.001 [-0.010, +0.012] |
| fresh hard | 387 (19 dropped) | 0.807 | 0.813 | +0.006 [-0.005, +0.019] |
| fresh soft | 406 | 0.767 | 0.770 | +0.003 [-0.004, +0.012] |

Fresh-soft ceiling (q as the score): 0.943. Spearman(score, q): hard-trained 0.585, soft-trained 0.596.

## Library check

| probe | fit B saved AUC | int refit AUC | hard01 refit AUC | max diff int vs fit B | max diff hard01 vs int |
|---|---|---|---|---|---|
| bias_followed_rm_answer | 0.788287 | 0.788287 | 0.788287 | 1.19e-07 | 0.00e+00 |
| deception_yes_rm_preyesno | 0.771429 | 0.771429 | 0.771429 | 1.19e-07 | 0.00e+00 |
| deception_no_rm_preyesno | 0.799420 | 0.799420 | 0.799420 | 1.19e-07 | 0.00e+00 |
