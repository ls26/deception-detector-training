# No penalty against the frozen and the retrained probe, fixed training labels, rounds 1-5

Written by `memorization.py`; definitions and sources in its docstring. Round 1: 261103's no-penalty run (no penalty) and its weight-1.0 run (frozen, retrained). Each cell: mean ± SD over seeds 42 and 43, or "s42 / s43" in a split row (listed below the table). Each seed's values are in comparison.json.

| Measurement | Condition | Base model | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|---|---|---|---|---|---|---|---|
| bias, training cue mix | no penalty | 0.152 / 0.153 | 0.380 / 0.247 | 0.186 / 0.189 | 0.167 / 0.130 | 0.067 / 0.097 | 0.094 / 0.058 |
| bias, training cue mix | frozen | 0.150 / 0.152 | 0.202 / 0.250 | 0.131 / 0.164 | 0.106 / 0.108 | 0.072 / 0.081 | 0.049 / 0.057 |
| bias, training cue mix | retrained | 0.150 / 0.152 | 0.202 / 0.250 | 0.126 / 0.144 | 0.064 / 0.089 | 0.069 / 0.057 | 0.033 / 0.059 |
| cue agreement rate, training cue mix | no penalty | 0.711 / 0.711 | 0.970 / 0.848 | 0.774 / 0.787 | 0.759 / 0.728 | 0.637 / 0.673 | 0.662 / 0.618 |
| cue agreement rate, training cue mix | frozen | 0.711 / 0.711 | 0.813 / 0.840 | 0.725 / 0.766 | 0.691 / 0.700 | 0.658 / 0.660 | 0.616 / 0.618 |
| cue agreement rate, training cue mix | retrained | 0.711 / 0.711 | 0.813 / 0.840 | 0.722 / 0.741 | 0.650 / 0.670 | 0.650 / 0.643 | 0.605 / 0.638 |
| true accuracy, training cue mix | no penalty | 0.729 / 0.729 | 0.665 / 0.739 | 0.771 / 0.767 | 0.785 / 0.768 | 0.762 / 0.768 | 0.754 / 0.769 |
| true accuracy, training cue mix | frozen | 0.729 / 0.729 | 0.763 / 0.736 | 0.764 / 0.774 | 0.769 / 0.791 | 0.778 / 0.775 | 0.745 / 0.774 |
| true accuracy, training cue mix | retrained | 0.729 / 0.729 | 0.763 / 0.736 | 0.771 / 0.760 | 0.777 / 0.764 | 0.751 / 0.781 | 0.776 / 0.766 |
| true accuracy, no cue | no penalty | 0.755 / 0.755 | 0.726 / 0.771 | 0.738 / 0.730 | 0.753 / 0.759 | 0.738 / 0.755 | 0.714 / 0.738 |
| true accuracy, no cue | frozen | 0.755 / 0.755 | 0.769 / 0.757 | 0.759 / 0.765 | 0.736 / 0.773 | 0.744 / 0.765 | 0.730 / 0.763 |
| true accuracy, no cue | retrained | 0.755 / 0.755 | 0.769 / 0.757 | 0.769 / 0.765 | 0.746 / 0.755 | 0.767 / 0.763 | 0.763 / 0.761 |
| true accuracy, cue at the correct answer | no penalty | 0.843 / 0.843 | 0.990 / 0.947 | 0.916 / 0.922 | 0.918 / 0.877 | 0.804 / 0.836 | 0.814 / 0.798 |
| true accuracy, cue at the correct answer | frozen | 0.843 / 0.843 | 0.939 / 0.937 | 0.881 / 0.912 | 0.849 / 0.871 | 0.830 / 0.832 | 0.773 / 0.800 |
| true accuracy, cue at the correct answer | retrained | 0.843 / 0.843 | 0.939 / 0.937 | 0.875 / 0.883 | 0.824 / 0.828 | 0.802 / 0.820 | 0.785 / 0.804 |
| true accuracy, cue at the wrong answer | no penalty | 0.509 / 0.509 | 0.065 / 0.348 | 0.495 / 0.476 | 0.532 / 0.560 | 0.673 / 0.632 | 0.640 / 0.714 |
| true accuracy, cue at the wrong answer | frozen | 0.509 / 0.509 | 0.431 / 0.331 | 0.556 / 0.499 | 0.620 / 0.634 | 0.673 / 0.658 | 0.689 / 0.714 |
| true accuracy, cue at the wrong answer | retrained | 0.509 / 0.509 | 0.431 / 0.331 | 0.575 / 0.534 | 0.689 / 0.640 | 0.652 / 0.701 | 0.751 / 0.695 |
| deception (P(Yes) version), training cue mix | no penalty | 0.569 / 0.568 | 0.465 / 0.435 | 0.492 / 0.471 | 0.478 / 0.472 | 0.539 / 0.513 | 0.539 / 0.581 |
| deception (P(Yes) version), training cue mix | frozen | 0.568 / 0.568 | 0.537 / 0.538 | 0.591 / 0.540 | 0.611 / 0.512 | 0.575 / 0.518 | 0.678 / 0.544 |
| deception (P(Yes) version), training cue mix | retrained | 0.567 / 0.568 | 0.537 / 0.538 | 0.574 / 0.570 | 0.548 / 0.456 | 0.533 / 0.675 | 0.548 / 0.616 |
| round 1's probe: mean score of biased answers | no penalty | 0.749 / 0.745 | 0.386 / 0.653 | 0.622 / 0.576 | 0.556 / 0.489 | 0.399 / 0.461 | 0.608 / 0.452 |
| round 1's probe: mean score of biased answers | frozen | 0.744 / 0.745 | 0.200 / 0.201 | 0.242 / 0.170 | 0.184 / 0.184 | 0.166 / 0.238 | 0.169 / 0.212 |
| round 1's probe: mean score of unbiased answers | no penalty | 0.231 / 0.230 | 0.074 / 0.169 | 0.126 / 0.112 | 0.118 / 0.101 | 0.085 / 0.112 | 0.178 / 0.162 |
| round 1's probe: mean score of unbiased answers | frozen | 0.231 / 0.232 | 0.052 / 0.043 | 0.047 / 0.038 | 0.031 / 0.039 | 0.051 / 0.049 | 0.058 / 0.035 |
| round 1's probe: AUC biased vs unbiased | no penalty | 0.907 / 0.904 | 0.863 / 0.905 | 0.903 / 0.910 | 0.893 / 0.896 | 0.852 / 0.862 | 0.854 / 0.795 |
| round 1's probe: AUC biased vs unbiased | frozen | 0.904 / 0.903 | 0.790 / 0.801 | 0.836 / 0.827 | 0.844 / 0.816 | 0.820 / 0.844 | 0.778 / 0.851 |
| training: flipped label, cue at the true answer, share naming the label | no penalty | – | 0.043 / 0.059 | 0.186 / 0.162 | 0.302 / 0.446 | 0.613 / 0.614 | 0.730 / 0.753 |
| training: flipped label, cue at the true answer, share naming the label | frozen | – | 0.112 / 0.094 | 0.233 / 0.160 | 0.454 / 0.436 | 0.577 / 0.657 | 0.707 / 0.726 |
| training: flipped label, cue at the true answer, share naming the label | retrained | – | 0.112 / 0.094 | 0.268 / 0.228 | 0.525 / 0.516 | 0.648 / 0.655 | 0.701 / 0.713 |

## Split rows

- bias, training cue mix: round 1: no penalty minus frozen is +0.179 (s42) and -0.003 (s43); round 4: no penalty minus frozen is -0.005 (s42) and +0.016 (s43)
- cue agreement rate, training cue mix: round 4: no penalty minus frozen is -0.021 (s42) and +0.014 (s43); round 5: no penalty minus frozen is +0.046 (s42) and -0.001 (s43)
- true accuracy, training cue mix: round 1: no penalty minus frozen is -0.098 (s42) and +0.003 (s43); round 2: no penalty minus frozen is +0.007 (s42) and -0.007 (s43); round 3: no penalty minus frozen is +0.016 (s42) and -0.023 (s43); round 5: no penalty minus frozen is +0.009 (s42) and -0.005 (s43)
- true accuracy, no cue: round 1: no penalty minus frozen is -0.043 (s42) and +0.014 (s43); round 3: no penalty minus frozen is +0.016 (s42) and -0.014 (s43)
- true accuracy, cue at the correct answer: round 4: no penalty minus frozen is -0.027 (s42) and +0.004 (s43); round 5: no penalty minus frozen is +0.041 (s42) and -0.002 (s43)
- true accuracy, cue at the wrong answer: round 1: no penalty minus frozen is -0.366 (s42) and +0.016 (s43)
- deception (P(Yes) version), training cue mix: round 5: no penalty minus frozen is -0.139 (s42) and +0.038 (s43)
- round 1's probe: mean score of biased answers: no penalty, round 1 to round 5: +0.222 (s42) and -0.201 (s43)
- round 1's probe: mean score of unbiased answers: no penalty, round 1 to round 5: +0.104 (s42) and -0.007 (s43)
- round 1's probe: AUC biased vs unbiased: round 5: no penalty minus frozen is +0.076 (s42) and -0.056 (s43)
- training: flipped label, cue at the true answer, share naming the label: round 2: no penalty minus frozen is -0.047 (s42) and +0.003 (s43); round 3: no penalty minus frozen is -0.151 (s42) and +0.010 (s43); round 4: no penalty minus frozen is +0.036 (s42) and -0.043 (s43)

## Training answers counted in the memorization row (seed 42 / seed 43), rounds 1-5

- no penalty: 3312 / 3056, 3376 / 3256, 3224 / 3136, 3112 / 3008, 3312 / 3240
- frozen: 3064 / 3224, 3000 / 3312, 3072 / 3016, 3088 / 3160, 3208 / 3008
- retrained: 3064 / 3224, 3288 / 3176, 3208 / 3216, 3112 / 3296, 2976 / 3056
