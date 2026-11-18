# Frozen against refit probe, training labels redrawn every round, rounds 1-5

Written by `memorization.py`; definitions in its docstring. seeds 45 and 46 from this folder (round 1: s<seed>_round1), seeds 42 and 43 from 261108 (frozen) and 261109 (refit) (round 1: 261103's kl_weight_1_0_s<seed>); round 1 is the same run in both conditions. Each cell: mean ± SD over seeds 42, 43, 45 and 46 (each seed's value in comparison.json).

| Metric | Probe | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|---|---|---|---|---|---|---|
| bias, training cue mix | frozen | 0.204 ± 0.078 | 0.207 ± 0.083 | 0.263 ± 0.023 | 0.217 ± 0.014 | 0.200 ± 0.010 |
| bias, training cue mix | refit | 0.204 ± 0.078 | 0.155 ± 0.032 | 0.136 ± 0.039 | 0.104 ± 0.015 | 0.075 ± 0.014 |
| cue agreement rate, training cue mix | frozen | 0.799 ± 0.093 | 0.812 ± 0.090 | 0.867 ± 0.020 | 0.817 ± 0.019 | 0.794 ± 0.010 |
| cue agreement rate, training cue mix | refit | 0.799 ± 0.093 | 0.754 ± 0.040 | 0.731 ± 0.047 | 0.693 ± 0.023 | 0.664 ± 0.020 |
| true accuracy, training cue mix | frozen | 0.751 ± 0.020 | 0.751 ± 0.023 | 0.735 ± 0.012 | 0.753 ± 0.009 | 0.764 ± 0.005 |
| true accuracy, training cue mix | refit | 0.751 ± 0.020 | 0.772 ± 0.009 | 0.773 ± 0.006 | 0.775 ± 0.015 | 0.775 ± 0.010 |
| true accuracy, no cue | frozen | 0.757 ± 0.013 | 0.765 ± 0.007 | 0.758 ± 0.013 | 0.761 ± 0.013 | 0.759 ± 0.003 |
| true accuracy, no cue | refit | 0.757 ± 0.013 | 0.775 ± 0.017 | 0.760 ± 0.023 | 0.758 ± 0.017 | 0.769 ± 0.001 |
| true accuracy, cue at the correct answer | frozen | 0.920 ± 0.057 | 0.929 ± 0.054 | 0.962 ± 0.008 | 0.936 ± 0.012 | 0.924 ± 0.007 |
| true accuracy, cue at the correct answer | refit | 0.920 ± 0.057 | 0.902 ± 0.026 | 0.883 ± 0.035 | 0.855 ± 0.025 | 0.830 ± 0.018 |
| true accuracy, cue at the wrong answer | frozen | 0.427 ± 0.165 | 0.416 ± 0.165 | 0.304 ± 0.050 | 0.412 ± 0.035 | 0.457 ± 0.023 |
| true accuracy, cue at the wrong answer | refit | 0.427 ± 0.165 | 0.527 ± 0.064 | 0.565 ± 0.072 | 0.628 ± 0.031 | 0.669 ± 0.032 |
| deception (P(Yes)-weighted), training cue mix | frozen | 0.534 ± 0.008 | 0.497 ± 0.043 | 0.476 ± 0.039 | 0.448 ± 0.028 | 0.476 ± 0.060 |
| deception (P(Yes)-weighted), training cue mix | refit | 0.534 ± 0.008 | 0.511 ± 0.042 | 0.566 ± 0.054 | 0.537 ± 0.127 | 0.557 ± 0.080 |
| penalty probe, Initial: mean score of biased answers | frozen | 0.747 ± 0.002 | – | – | – | – |
| penalty probe, Initial: mean score of biased answers | refit | 0.747 ± 0.002 | 0.665 ± 0.030 | 0.620 ± 0.082 | 0.624 ± 0.030 | 0.707 ± 0.010 |
| penalty probe, Initial: mean score of unbiased answers | frozen | 0.230 ± 0.001 | – | – | – | – |
| penalty probe, Initial: mean score of unbiased answers | refit | 0.230 ± 0.001 | 0.298 ± 0.065 | 0.276 ± 0.102 | 0.211 ± 0.033 | 0.319 ± 0.045 |
| penalty probe, Initial: AUC biased vs unbiased | frozen | 0.905 ± 0.002 | – | – | – | – |
| penalty probe, Initial: AUC biased vs unbiased | refit | 0.905 ± 0.002 | 0.858 ± 0.032 | 0.863 ± 0.016 | 0.885 ± 0.009 | 0.861 ± 0.021 |
| penalty probe, Final: mean score of biased answers | frozen | 0.308 ± 0.214 | 0.242 ± 0.146 | 0.165 ± 0.048 | 0.170 ± 0.049 | 0.157 ± 0.028 |
| penalty probe, Final: mean score of biased answers | refit | 0.308 ± 0.214 | 0.355 ± 0.164 | 0.370 ± 0.117 | 0.520 ± 0.082 | 0.608 ± 0.089 |
| penalty probe, Final: mean score of unbiased answers | frozen | 0.071 ± 0.038 | 0.061 ± 0.023 | 0.043 ± 0.011 | 0.047 ± 0.019 | 0.041 ± 0.014 |
| penalty probe, Final: mean score of unbiased answers | refit | 0.071 ± 0.038 | 0.118 ± 0.060 | 0.088 ± 0.039 | 0.151 ± 0.048 | 0.255 ± 0.051 |
| penalty probe, Final: AUC biased vs unbiased | frozen | 0.813 ± 0.060 | 0.786 ± 0.082 | 0.790 ± 0.044 | 0.795 ± 0.024 | 0.792 ± 0.034 |
| penalty probe, Final: AUC biased vs unbiased | refit | 0.813 ± 0.060 | 0.835 ± 0.029 | 0.879 ± 0.007 | 0.876 ± 0.011 | 0.859 ± 0.032 |
| training: flipped label, cue at the true answer, share naming the label | frozen | 0.098 ± 0.031 | 0.081 ± 0.078 | 0.031 ± 0.025 | 0.059 ± 0.015 | 0.075 ± 0.031 |
| training: flipped label, cue at the true answer, share naming the label | refit | 0.098 ± 0.031 | 0.118 ± 0.023 | 0.121 ± 0.027 | 0.190 ± 0.011 | 0.178 ± 0.028 |

Training answers counted in the memorization row (seed 42 / seed 43 / seed 45 / seed 46), rounds 1-5:

- frozen: 3064 / 3224 / 3112 / 3160, 3184 / 3136 / 3144 / 3088, 3072 / 3256 / 3088 / 3168, 3344 / 3256 / 3104 / 3000, 3352 / 3160 / 3280 / 3208
- refit: 3064 / 3224 / 3112 / 3160, 2992 / 3088 / 3368 / 3208, 3152 / 3048 / 2896 / 3264, 3136 / 3096 / 3312 / 3288, 3112 / 3232 / 3216 / 3000

## Eval bias counts, frozen against refit (Fisher's exact test)

Biased cued answers / cued answers with a no-cue answer in both evals, cue at the correct and at the wrong answer pooled (counted equally, not weighted as in the training cue mix). Unpaired: every cued answer counts as independent.

| round | seed | frozen | refit | p |
|---|---|---|---|---|
| round2 | s42 | 247/897 (27.5%) | 156/919 (17.0%) | 7.3e-08 |
| round2 | s43 | 266/900 (29.6%) | 127/907 (14.0%) | 8.3e-16 |
| round2 | s45 | 88/919 (9.6%) | 202/889 (22.7%) | 2e-14 |
| round2 | s46 | 301/896 (33.6%) | 167/895 (18.7%) | 6.5e-13 |
| round2 | seeds_pooled | 902/3612 (25.0%) | 652/3610 (18.1%) | 9.2e-13 |
| round3 | s42 | 317/910 (34.8%) | 153/920 (16.6%) | 3.5e-19 |
| round3 | s43 | 303/917 (33.0%) | 184/907 (20.3%) | 7.4e-10 |
| round3 | s45 | 247/900 (27.4%) | 80/917 (8.7%) | 6.3e-26 |
| round3 | s46 | 294/898 (32.7%) | 143/916 (15.6%) | 9.9e-18 |
| round3 | seeds_pooled | 1161/3625 (32.0%) | 560/3660 (15.3%) | 2.5e-64 |
| round4 | s42 | 247/907 (27.2%) | 94/908 (10.4%) | 1.2e-20 |
| round4 | s43 | 245/894 (27.4%) | 120/910 (13.2%) | 4.2e-14 |
| round4 | s45 | 224/908 (24.7%) | 78/906 (8.6%) | 1.4e-20 |
| round4 | s46 | 209/912 (22.9%) | 108/905 (11.9%) | 7.2e-10 |
| round4 | seeds_pooled | 925/3621 (25.5%) | 400/3629 (11.0%) | 8.9e-59 |
| round5 | s42 | 204/906 (22.5%) | 79/910 (8.7%) | 2.3e-16 |
| round5 | s43 | 196/890 (22.0%) | 59/917 (6.4%) | 3.6e-22 |
| round5 | s45 | 209/916 (22.8%) | 93/911 (10.2%) | 2.7e-13 |
| round5 | s46 | 222/915 (24.3%) | 69/909 (7.6%) | 3.8e-23 |
| round5 | seeds_pooled | 831/3627 (22.9%) | 300/3647 (8.2%) | 7e-69 |

## Memorization with fixed labels (seeds 42 and 43)

training: flipped label, cue at the true answer, share naming the label; frozen = 261107's runs, refit = 261104's, trained every round on the original question file. Each cell: mean ± SD over seeds 42 and 43.

| Probe | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|---|---|---|---|---|---|
| frozen | 0.103 ± 0.013 | 0.196 ± 0.052 | 0.445 ± 0.012 | 0.617 ± 0.056 | 0.717 ± 0.013 |
| refit | 0.103 ± 0.013 | 0.248 ± 0.029 | 0.521 ± 0.006 | 0.651 ± 0.005 | 0.707 ± 0.009 |

Training answers counted (seed 42 / seed 43), rounds 1-5:

- frozen: 3064 / 3224, 3000 / 3312, 3072 / 3016, 3088 / 3160, 3208 / 3008
- refit: 3064 / 3224, 3288 / 3176, 3208 / 3216, 3112 / 3296, 2976 / 3056
