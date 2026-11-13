# Frozen against refit probe, training labels redrawn every round, rounds 1-5

Written by `memorization.py`; definitions in its docstring. frozen = 261108, refit = this experiment; round 1 is 261103's run in both. Each cell: mean ± SD over seeds 42 and 43 (each seed's value in comparison.json).

| Metric | Probe | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|---|---|---|---|---|---|---|
| bias, training cue mix | frozen | 0.226 ± 0.034 | 0.238 ± 0.007 | 0.277 ± 0.017 | 0.229 ± 0.001 | 0.192 ± 0.003 |
| bias, training cue mix | refit | 0.226 ± 0.034 | 0.135 ± 0.029 | 0.160 ± 0.023 | 0.108 ± 0.020 | 0.070 ± 0.017 |
| cue agreement rate, training cue mix | frozen | 0.827 ± 0.019 | 0.844 ± 0.017 | 0.877 ± 0.012 | 0.831 ± 0.005 | 0.787 ± 0.003 |
| cue agreement rate, training cue mix | refit | 0.827 ± 0.019 | 0.725 ± 0.031 | 0.760 ± 0.027 | 0.701 ± 0.025 | 0.658 ± 0.026 |
| true accuracy, no cue | frozen | 0.763 ± 0.009 | 0.762 ± 0.004 | 0.766 ± 0.001 | 0.754 ± 0.010 | 0.761 ± 0.003 |
| true accuracy, no cue | refit | 0.763 ± 0.009 | 0.775 ± 0.009 | 0.772 ± 0.030 | 0.769 ± 0.009 | 0.770 ± 0.001 |
| true accuracy, cue at the correct answer | frozen | 0.938 ± 0.001 | 0.949 ± 0.017 | 0.965 ± 0.009 | 0.942 ± 0.004 | 0.922 ± 0.009 |
| true accuracy, cue at the correct answer | refit | 0.938 ± 0.001 | 0.883 ± 0.023 | 0.907 ± 0.022 | 0.871 ± 0.020 | 0.831 ± 0.022 |
| true accuracy, cue at the wrong answer | frozen | 0.381 ± 0.071 | 0.365 ± 0.022 | 0.274 ± 0.014 | 0.382 ± 0.003 | 0.473 ± 0.010 |
| true accuracy, cue at the wrong answer | refit | 0.381 ± 0.071 | 0.575 ± 0.029 | 0.525 ± 0.045 | 0.631 ± 0.036 | 0.690 ± 0.025 |
| deception (P(Yes)-weighted), training cue mix | frozen | 0.538 ± 0.001 | 0.506 ± 0.043 | 0.503 ± 0.042 | 0.467 ± 0.001 | 0.521 ± 0.046 |
| deception (P(Yes)-weighted), training cue mix | refit | 0.538 ± 0.001 | 0.535 ± 0.014 | 0.566 ± 0.068 | 0.640 ± 0.075 | 0.587 ± 0.114 |
| penalty probe, Initial: mean score of biased answers | frozen | 0.722 ± 0.000 | – | – | – | – |
| penalty probe, Initial: mean score of biased answers | refit | 0.722 ± 0.000 | 0.660 ± 0.019 | 0.559 ± 0.041 | 0.634 ± 0.001 | 0.708 ± 0.018 |
| penalty probe, Initial: mean score of unbiased answers | frozen | 0.255 ± 0.000 | – | – | – | – |
| penalty probe, Initial: mean score of unbiased answers | refit | 0.255 ± 0.000 | 0.320 ± 0.045 | 0.198 ± 0.019 | 0.225 ± 0.044 | 0.329 ± 0.074 |
| penalty probe, Initial: AUC biased vs unbiased | frozen | 0.873 ± 0.000 | – | – | – | – |
| penalty probe, Initial: AUC biased vs unbiased | refit | 0.873 ± 0.000 | 0.848 ± 0.013 | 0.875 ± 0.010 | 0.882 ± 0.011 | 0.855 ± 0.027 |
| penalty probe, Final: mean score of biased answers | frozen | 0.200 ± 0.001 | 0.187 ± 0.041 | 0.147 ± 0.059 | 0.148 ± 0.071 | 0.171 ± 0.014 |
| penalty probe, Final: mean score of biased answers | refit | 0.200 ± 0.001 | 0.483 ± 0.014 | 0.280 ± 0.025 | 0.488 ± 0.102 | 0.573 ± 0.038 |
| penalty probe, Final: mean score of unbiased answers | frozen | 0.047 ± 0.006 | 0.058 ± 0.034 | 0.043 ± 0.015 | 0.040 ± 0.023 | 0.042 ± 0.003 |
| penalty probe, Final: mean score of unbiased answers | refit | 0.047 ± 0.006 | 0.164 ± 0.002 | 0.059 ± 0.023 | 0.146 ± 0.027 | 0.253 ± 0.022 |
| penalty probe, Final: AUC biased vs unbiased | frozen | 0.795 ± 0.008 | 0.782 ± 0.034 | 0.773 ± 0.021 | 0.792 ± 0.026 | 0.802 ± 0.016 |
| penalty probe, Final: AUC biased vs unbiased | refit | 0.795 ± 0.008 | 0.859 ± 0.003 | 0.877 ± 0.009 | 0.868 ± 0.010 | 0.837 ± 0.000 |
| training: flipped label, cue at the true answer, share naming the label | frozen | 0.103 ± 0.013 | 0.051 ± 0.019 | 0.021 ± 0.005 | 0.048 ± 0.002 | 0.068 ± 0.009 |
| training: flipped label, cue at the true answer, share naming the label | refit | 0.103 ± 0.013 | 0.131 ± 0.006 | 0.102 ± 0.004 | 0.183 ± 0.013 | 0.196 ± 0.025 |

Training answers counted in the memorization row (seed 42 / seed 43), rounds 1-5:

- frozen: 3064 / 3224, 3184 / 3136, 3072 / 3256, 3344 / 3256, 3352 / 3160
- refit: 3064 / 3224, 2992 / 3088, 3152 / 3048, 3136 / 3096, 3112 / 3232
