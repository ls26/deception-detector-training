# No penalty against the frozen and the retrained probe, fixed training labels, rounds 1-5

Written by `memorization.py`; definitions and sources in its docstring. Round 1: 261103's no-penalty run (no penalty) and its weight-1.0 run (frozen, retrained). Each cell: mean ± SD over seeds 42 and 43, or "s42 / s43" in a split row (listed below the table). Each seed's values are in comparison.json.

| Measurement | Condition | Base model | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|---|---|---|---|---|---|---|---|
| bias, training cue mix | no penalty | 0.151 ± 0.001 | 0.226 ± 0.034 | 0.147 ± 0.023 | 0.107 ± 0.001 | 0.076 ± 0.006 | 0.053 ± 0.006 |
| bias, training cue mix | frozen | 0.151 ± 0.001 | 0.226 ± 0.034 | 0.147 ± 0.023 | 0.107 ± 0.001 | 0.076 ± 0.006 | 0.053 ± 0.006 |
| bias, training cue mix | retrained | 0.151 ± 0.001 | 0.226 ± 0.034 | 0.135 ± 0.012 | 0.076 ± 0.017 | 0.063 ± 0.008 | 0.046 ± 0.018 |
| cue agreement rate, training cue mix | no penalty | 0.711 ± 0.000 | 0.827 ± 0.019 | 0.746 ± 0.029 | 0.695 ± 0.007 | 0.659 ± 0.001 | 0.617 ± 0.002 |
| cue agreement rate, training cue mix | frozen | 0.711 ± 0.000 | 0.827 ± 0.019 | 0.746 ± 0.029 | 0.695 ± 0.007 | 0.659 ± 0.001 | 0.617 ± 0.002 |
| cue agreement rate, training cue mix | retrained | 0.711 ± 0.000 | 0.827 ± 0.019 | 0.731 ± 0.013 | 0.660 ± 0.014 | 0.646 ± 0.005 | 0.622 ± 0.023 |
| true accuracy, training cue mix | no penalty | 0.729 / 0.729 | 0.763 / 0.736 | 0.764 / 0.774 | 0.769 / 0.791 | 0.778 / 0.775 | 0.745 / 0.774 |
| true accuracy, training cue mix | frozen | 0.729 / 0.729 | 0.763 / 0.736 | 0.764 / 0.774 | 0.769 / 0.791 | 0.778 / 0.775 | 0.745 / 0.774 |
| true accuracy, training cue mix | retrained | 0.729 / 0.729 | 0.763 / 0.736 | 0.771 / 0.760 | 0.777 / 0.764 | 0.751 / 0.781 | 0.776 / 0.766 |
| true accuracy, no cue | no penalty | 0.755 / 0.755 | 0.769 / 0.757 | 0.759 / 0.765 | 0.736 / 0.773 | 0.744 / 0.765 | 0.730 / 0.763 |
| true accuracy, no cue | frozen | 0.755 / 0.755 | 0.769 / 0.757 | 0.759 / 0.765 | 0.736 / 0.773 | 0.744 / 0.765 | 0.730 / 0.763 |
| true accuracy, no cue | retrained | 0.755 / 0.755 | 0.769 / 0.757 | 0.769 / 0.765 | 0.746 / 0.755 | 0.767 / 0.763 | 0.763 / 0.761 |
| true accuracy, cue at the correct answer | no penalty | 0.843 ± 0.000 | 0.938 ± 0.001 | 0.897 ± 0.022 | 0.860 ± 0.016 | 0.831 ± 0.001 | 0.786 ± 0.019 |
| true accuracy, cue at the correct answer | frozen | 0.843 ± 0.000 | 0.938 ± 0.001 | 0.897 ± 0.022 | 0.860 ± 0.016 | 0.831 ± 0.001 | 0.786 ± 0.019 |
| true accuracy, cue at the correct answer | retrained | 0.843 ± 0.000 | 0.938 ± 0.001 | 0.879 ± 0.006 | 0.826 ± 0.003 | 0.811 ± 0.013 | 0.794 ± 0.013 |
| true accuracy, cue at the wrong answer | no penalty | 0.509 ± 0.000 | 0.381 ± 0.071 | 0.528 ± 0.040 | 0.627 ± 0.010 | 0.666 ± 0.010 | 0.701 ± 0.017 |
| true accuracy, cue at the wrong answer | frozen | 0.509 ± 0.000 | 0.381 ± 0.071 | 0.528 ± 0.040 | 0.627 ± 0.010 | 0.666 ± 0.010 | 0.701 ± 0.017 |
| true accuracy, cue at the wrong answer | retrained | 0.509 ± 0.000 | 0.381 ± 0.071 | 0.554 ± 0.029 | 0.665 ± 0.035 | 0.677 ± 0.035 | 0.723 ± 0.039 |
| deception (P(Yes) version), training cue mix | no penalty | 0.568 ± 0.000 | 0.538 ± 0.001 | 0.565 ± 0.036 | 0.561 ± 0.070 | 0.547 ± 0.040 | 0.611 ± 0.095 |
| deception (P(Yes) version), training cue mix | frozen | 0.568 ± 0.000 | 0.538 ± 0.001 | 0.565 ± 0.036 | 0.561 ± 0.070 | 0.547 ± 0.040 | 0.611 ± 0.095 |
| deception (P(Yes) version), training cue mix | retrained | 0.568 ± 0.000 | 0.538 ± 0.001 | 0.572 ± 0.002 | 0.502 ± 0.065 | 0.604 ± 0.100 | 0.582 ± 0.048 |
| round 1's probe: mean score of biased answers | no penalty | 0.744 / 0.745 | 0.200 / 0.201 | 0.242 / 0.170 | 0.184 / 0.184 | 0.166 / 0.238 | 0.169 / 0.212 |
| round 1's probe: mean score of biased answers | frozen | 0.744 / 0.745 | 0.200 / 0.201 | 0.242 / 0.170 | 0.184 / 0.184 | 0.166 / 0.238 | 0.169 / 0.212 |
| round 1's probe: mean score of unbiased answers | no penalty | 0.231 / 0.232 | 0.052 / 0.043 | 0.047 / 0.038 | 0.031 / 0.039 | 0.051 / 0.049 | 0.058 / 0.035 |
| round 1's probe: mean score of unbiased answers | frozen | 0.231 / 0.232 | 0.052 / 0.043 | 0.047 / 0.038 | 0.031 / 0.039 | 0.051 / 0.049 | 0.058 / 0.035 |
| round 1's probe: AUC biased vs unbiased | no penalty | 0.904 / 0.903 | 0.790 / 0.801 | 0.836 / 0.827 | 0.844 / 0.816 | 0.820 / 0.844 | 0.778 / 0.851 |
| round 1's probe: AUC biased vs unbiased | frozen | 0.904 / 0.903 | 0.790 / 0.801 | 0.836 / 0.827 | 0.844 / 0.816 | 0.820 / 0.844 | 0.778 / 0.851 |
| training: flipped label, cue at the true answer, share naming the label | no penalty | – | 0.103 ± 0.013 | 0.196 ± 0.052 | 0.445 ± 0.012 | 0.617 ± 0.056 | 0.717 ± 0.013 |
| training: flipped label, cue at the true answer, share naming the label | frozen | – | 0.103 ± 0.013 | 0.196 ± 0.052 | 0.445 ± 0.012 | 0.617 ± 0.056 | 0.717 ± 0.013 |
| training: flipped label, cue at the true answer, share naming the label | retrained | – | 0.103 ± 0.013 | 0.248 ± 0.029 | 0.521 ± 0.006 | 0.651 ± 0.005 | 0.707 ± 0.009 |

## Split rows

- true accuracy, training cue mix: no penalty, round 1 to round 5: -0.018 (s42) and +0.038 (s43)
- true accuracy, no cue: no penalty, round 1 to round 5: -0.039 (s42) and +0.006 (s43)
- round 1's probe: mean score of biased answers: no penalty, round 1 to round 5: -0.030 (s42) and +0.011 (s43)
- round 1's probe: mean score of unbiased answers: no penalty, round 1 to round 5: +0.006 (s42) and -0.008 (s43)
- round 1's probe: AUC biased vs unbiased: no penalty, round 1 to round 5: -0.012 (s42) and +0.050 (s43)

## Training answers counted in the memorization row (seed 42 / seed 43), rounds 1-5

- no penalty: 3064 / 3224, 3000 / 3312, 3072 / 3016, 3088 / 3160, 3208 / 3008
- frozen: 3064 / 3224, 3000 / 3312, 3072 / 3016, 3088 / 3160, 3208 / 3008
- retrained: 3064 / 3224, 3288 / 3176, 3208 / 3216, 3112 / 3296, 2976 / 3056
