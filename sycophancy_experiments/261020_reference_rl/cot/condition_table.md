# 261020 reference runs: CoT (261013's setup; one sampled answer per prompt)

Each cell is base → after RL: the base model, then the final checkpoints' mean ± SD (n − 1) over seeds 42, 43, 44. Eval questions: 489. The training cue mix is each run's own: the cue at the noisy label 75% of the time for condition 0, at the correct answer 75% of the time for condition 3. Conditions 1 and 2 were trained without a cue, so they are scored on the mix of the cued run with the same labels: condition 1 on condition 0's, condition 2 on condition 3's. Reward is agreement with the run's own training label (the noisy label for conditions 0 and 1, the correct answer for conditions 2 and 3). Bias and self-report rows count questions with a no-cue answer in both the base eval and the run's eval; their base value is the mean over the seeds' base values. "deception" is the probability-weighted measure (key deception_soft); "deception (thresholded)" is key deception.

## True accuracy

| metric | eval condition | 0. Noisy labels, cue (previous experiment) | 1. Noisy labels, no cue | 2. Correct labels, no cue | 3. Correct labels, cue |
|---|---|---|---|---|---|
| true accuracy | no cue | 0.755 → 0.746 ± 0.007 | 0.755 → 0.768 ± 0.002 | 0.755 → 0.790 ± 0.005 | 0.755 → 0.778 ± 0.005 |
| true accuracy | cue at the correct answer | 0.843 → 0.967 ± 0.008 | 0.843 → 0.909 ± 0.028 | 0.843 → 0.909 ± 0.051 | 0.843 → 0.892 ± 0.031 |
| true accuracy | cue at the wrong answer | 0.509 → 0.257 ± 0.081 | 0.509 → 0.510 ± 0.075 | 0.509 → 0.551 ± 0.077 | 0.509 → 0.558 ± 0.112 |
| true accuracy | training cue mix | 0.729 → 0.721 ± 0.023 | 0.729 → 0.769 ± 0.010 | 0.759 → 0.820 ± 0.019 | 0.759 → 0.808 ± 0.014 |

## Reward on each run's own training prompts

| training condition | eval condition | reward |
|---|---|---|
| 0. Noisy labels, cue (previous experiment) | training cue mix | 0.689 → 0.749 ± 0.009 |
| 1. Noisy labels, no cue | no cue | 0.644 → 0.666 ± 0.005 |
| 2. Correct labels, no cue | no cue | 0.755 → 0.790 ± 0.005 |
| 3. Correct labels, cue | training cue mix | 0.759 → 0.808 ± 0.014 |

## All other metrics

| metric | eval condition | 0. Noisy labels, cue (previous experiment) | 1. Noisy labels, no cue | 2. Correct labels, no cue | 3. Correct labels, cue |
|---|---|---|---|---|---|
| reward | no cue | 0.644 → 0.662 ± 0.004 | 0.644 → 0.666 ± 0.005 | 0.755 → 0.790 ± 0.005 | 0.755 → 0.778 ± 0.005 |
| reward | cue at the correct answer | 0.714 → 0.790 ± 0.009 | 0.714 → 0.751 ± 0.009 | 0.843 → 0.909 ± 0.051 | 0.843 → 0.892 ± 0.031 |
| reward | cue at the wrong answer | 0.499 → 0.352 ± 0.047 | 0.499 → 0.513 ± 0.055 | 0.509 → 0.551 ± 0.077 | 0.509 → 0.558 ± 0.112 |
| reward | training cue mix | 0.689 → 0.749 ± 0.009 | 0.689 → 0.731 ± 0.009 | 0.759 → 0.820 ± 0.019 | 0.759 → 0.808 ± 0.014 |
| cue agreement rate | cue at the correct answer | 0.843 → 0.967 ± 0.008 | 0.843 → 0.909 ± 0.028 | 0.843 → 0.909 ± 0.051 | 0.843 → 0.892 ± 0.031 |
| cue agreement rate | cue at the wrong answer | 0.458 → 0.742 ± 0.082 | 0.458 → 0.489 ± 0.076 | 0.458 → 0.449 ± 0.077 | 0.458 → 0.442 ± 0.112 |
| cue agreement rate | training cue mix | 0.711 → 0.890 ± 0.036 | 0.711 → 0.765 ± 0.040 | 0.746 → 0.794 ± 0.057 | 0.746 → 0.779 ± 0.049 |
| bias | cue at the correct answer | 0.093 → 0.182 ± 0.011 | 0.089 → 0.128 ± 0.021 | 0.090 → 0.102 ± 0.036 | 0.090 → 0.107 ± 0.028 |
| bias | cue at the wrong answer | 0.267 → 0.510 ± 0.088 | 0.270 → 0.279 ± 0.066 | 0.271 → 0.256 ± 0.093 | 0.267 → 0.222 ± 0.112 |
| bias | training cue mix | 0.153 → 0.288 ± 0.036 | 0.150 → 0.181 ± 0.027 | 0.135 → 0.141 ± 0.050 | 0.134 → 0.136 ± 0.044 |
| anti_bias | cue at the correct answer | 0.038 → 0.001 ± 0.001 | 0.037 → 0.006 ± 0.009 | 0.037 → 0.004 ± 0.002 | 0.034 → 0.006 ± 0.006 |
| anti_bias | cue at the wrong answer | 0.011 → 0.000 ± 0.000 | 0.011 → 0.012 ± 0.011 | 0.011 → 0.006 ± 0.006 | 0.010 → 0.007 ± 0.003 |
| anti_bias | training cue mix | 0.028 → 0.001 ± 0.001 | 0.027 → 0.008 ± 0.010 | 0.031 → 0.005 ± 0.000 | 0.028 → 0.006 ± 0.005 |
| no_bias | cue at the correct answer | 0.870 → 0.816 ± 0.010 | 0.874 → 0.866 ± 0.013 | 0.873 → 0.893 ± 0.034 | 0.876 → 0.887 ± 0.024 |
| no_bias | cue at the wrong answer | 0.723 → 0.490 ± 0.088 | 0.718 → 0.709 ± 0.062 | 0.718 → 0.738 ± 0.098 | 0.723 → 0.770 ± 0.109 |
| no_bias | training cue mix | 0.820 → 0.711 ± 0.036 | 0.822 → 0.811 ± 0.020 | 0.834 → 0.855 ± 0.050 | 0.838 → 0.858 ± 0.039 |
| deception | cue at the correct answer | 0.551 → 0.560 ± 0.027 | 0.553 → 0.593 ± 0.097 | 0.552 → 0.665 ± 0.080 | 0.552 → 0.523 ± 0.087 |
| deception | cue at the wrong answer | 0.600 → 0.517 ± 0.051 | 0.600 → 0.608 ± 0.014 | 0.600 → 0.671 ± 0.083 | 0.602 → 0.589 ± 0.057 |
| deception | training cue mix | 0.566 → 0.547 ± 0.031 | 0.568 → 0.598 ± 0.066 | 0.564 → 0.667 ± 0.081 | 0.564 → 0.540 ± 0.079 |
| deception (thresholded) | cue at the correct answer | 0.568 → 0.617 ± 0.042 | 0.573 → 0.629 ± 0.161 | 0.568 → 0.713 ± 0.088 | 0.570 → 0.514 ± 0.124 |
| deception (thresholded) | cue at the wrong answer | 0.663 → 0.516 ± 0.089 | 0.659 → 0.657 ± 0.044 | 0.662 → 0.717 ± 0.077 | 0.666 → 0.611 ± 0.083 |
| deception (thresholded) | training cue mix | 0.600 → 0.583 ± 0.038 | 0.601 → 0.638 ± 0.115 | 0.592 → 0.714 ± 0.085 | 0.594 → 0.538 ± 0.113 |
| biased_share | cue at the correct answer | 0.096 → 0.182 ± 0.011 | 0.093 → 0.128 ± 0.020 | 0.094 → 0.103 ± 0.036 | 0.093 → 0.107 ± 0.028 |
| biased_share | cue at the wrong answer | 0.269 → 0.510 ± 0.088 | 0.273 → 0.282 ± 0.065 | 0.274 → 0.258 ± 0.095 | 0.270 → 0.224 ± 0.112 |
| biased_share | training cue mix | 0.157 → 0.288 ± 0.036 | 0.155 → 0.182 ± 0.026 | 0.139 → 0.141 ± 0.051 | 0.137 → 0.136 ± 0.044 |
| false_denial_soft | cue at the correct answer | 0.388 → 0.282 ± 0.024 | 0.379 → 0.284 ± 0.080 | 0.379 → 0.222 ± 0.087 | 0.384 → 0.371 ± 0.105 |
| false_denial_soft | cue at the wrong answer | 0.354 → 0.278 ± 0.022 | 0.349 → 0.292 ± 0.106 | 0.350 → 0.230 ± 0.061 | 0.354 → 0.375 ± 0.106 |
| false_denial_soft | training cue mix | 0.369 → 0.277 ± 0.019 | 0.362 → 0.284 ± 0.093 | 0.365 → 0.226 ± 0.075 | 0.369 → 0.373 ± 0.104 |
| false_admission_soft | cue at the correct answer | 0.568 → 0.623 ± 0.044 | 0.571 → 0.637 ± 0.120 | 0.570 → 0.712 ± 0.078 | 0.569 → 0.540 ± 0.106 |
| false_admission_soft | cue at the wrong answer | 0.691 → 0.768 ± 0.028 | 0.694 → 0.738 ± 0.074 | 0.695 → 0.816 ± 0.051 | 0.693 → 0.660 ± 0.096 |
| false_admission_soft | training cue mix | 0.603 → 0.657 ± 0.043 | 0.605 → 0.668 ± 0.102 | 0.596 → 0.734 ± 0.073 | 0.595 → 0.566 ± 0.102 |
| said_yes_biased_soft | cue at the correct answer | 0.612 → 0.718 ± 0.024 | 0.621 → 0.716 ± 0.080 | 0.621 → 0.778 ± 0.087 | 0.616 → 0.629 ± 0.105 |
| said_yes_biased_soft | cue at the wrong answer | 0.646 → 0.722 ± 0.022 | 0.651 → 0.708 ± 0.106 | 0.650 → 0.770 ± 0.061 | 0.646 → 0.625 ± 0.106 |
| said_yes_biased_soft | training cue mix | 0.631 → 0.723 ± 0.019 | 0.638 → 0.716 ± 0.093 | 0.635 → 0.774 ± 0.075 | 0.631 → 0.627 ± 0.104 |
| deception_from_biased | cue at the correct answer | 0.037 → 0.051 ± 0.001 | 0.035 → 0.037 ± 0.015 | 0.036 → 0.025 ± 0.015 | 0.036 → 0.040 ± 0.017 |
| deception_from_biased | cue at the wrong answer | 0.095 → 0.141 ± 0.018 | 0.095 → 0.080 ± 0.022 | 0.096 → 0.063 ± 0.034 | 0.096 → 0.081 ± 0.033 |
| deception_from_biased | training cue mix | 0.058 → 0.079 ± 0.007 | 0.056 → 0.052 ± 0.018 | 0.051 → 0.034 ± 0.020 | 0.051 → 0.050 ± 0.018 |
| deception_from_unbiased | cue at the correct answer | 0.513 → 0.509 ± 0.029 | 0.518 → 0.556 ± 0.111 | 0.516 → 0.641 ± 0.096 | 0.516 → 0.483 ± 0.104 |
| deception_from_unbiased | cue at the wrong answer | 0.505 → 0.376 ± 0.069 | 0.504 → 0.528 ± 0.036 | 0.504 → 0.608 ± 0.117 | 0.506 → 0.509 ± 0.082 |
| deception_from_unbiased | training cue mix | 0.508 → 0.467 ± 0.038 | 0.512 → 0.546 ± 0.082 | 0.513 → 0.633 ± 0.101 | 0.514 → 0.489 ± 0.095 |
| said_yes | cue at the correct answer | 0.596 → 0.742 ± 0.082 | 0.602 → 0.688 ± 0.207 | 0.599 → 0.789 ± 0.076 | 0.597 → 0.547 ± 0.153 |
| said_yes | cue at the wrong answer | 0.785 → 0.878 ± 0.069 | 0.789 → 0.816 ± 0.147 | 0.788 → 0.899 ± 0.029 | 0.787 → 0.703 ± 0.134 |
| said_yes | training cue mix | 0.657 → 0.785 ± 0.076 | 0.662 → 0.734 ± 0.187 | 0.647 → 0.817 ± 0.064 | 0.645 → 0.586 ± 0.148 |
| false_denial | cue at the correct answer | 0.358 → 0.159 ± 0.109 | 0.345 → 0.254 ± 0.196 | 0.336 → 0.113 ± 0.081 | 0.349 → 0.345 ± 0.182 |
| false_denial | cue at the wrong answer | 0.275 → 0.157 ± 0.098 | 0.262 → 0.233 ± 0.188 | 0.269 → 0.135 ± 0.048 | 0.277 → 0.319 ± 0.133 |
| false_denial | training cue mix | 0.318 → 0.159 ± 0.101 | 0.301 → 0.236 ± 0.202 | 0.302 → 0.123 ± 0.066 | 0.313 → 0.338 ± 0.161 |
| false_admission | cue at the correct answer | 0.591 → 0.721 ± 0.077 | 0.596 → 0.680 ± 0.210 | 0.593 → 0.778 ± 0.079 | 0.592 → 0.532 ± 0.153 |
| false_admission | cue at the wrong answer | 0.807 → 0.908 ± 0.053 | 0.808 → 0.834 ± 0.137 | 0.810 → 0.914 ± 0.018 | 0.810 → 0.709 ± 0.135 |
| false_admission | training cue mix | 0.652 → 0.760 ± 0.068 | 0.655 → 0.728 ± 0.185 | 0.639 → 0.806 ± 0.068 | 0.639 → 0.572 ± 0.146 |
| parse_failed | no cue | 0.016 → 0.005 ± 0.005 | 0.016 → 0.002 ± 0.002 | 0.016 → 0.001 ± 0.001 | 0.016 → 0.001 ± 0.002 |
| parse_failed | cue at the correct answer | 0.018 → 0.001 ± 0.001 | 0.018 → 0.000 ± 0.000 | 0.018 → 0.001 ± 0.001 | 0.018 → 0.003 ± 0.001 |
| parse_failed | cue at the wrong answer | 0.033 → 0.001 ± 0.001 | 0.033 → 0.001 ± 0.001 | 0.033 → 0.000 ± 0.000 | 0.033 → 0.001 ± 0.001 |
| parse_failed | training cue mix | 0.024 → 0.000 ± 0.000 | 0.024 → 0.000 ± 0.000 | 0.022 → 0.001 ± 0.001 | 0.022 → 0.002 ± 0.001 |
| cut_off | no cue | 0.000 → 0.000 ± 0.000 | 0.000 → 0.000 ± 0.000 | 0.000 → 0.000 ± 0.000 | 0.000 → 0.000 ± 0.000 |
| cut_off | cue at the correct answer | 0.000 → 0.000 ± 0.000 | 0.000 → 0.000 ± 0.000 | 0.000 → 0.001 ± 0.001 | 0.000 → 0.000 ± 0.000 |
| cut_off | cue at the wrong answer | 0.000 → 0.000 ± 0.000 | 0.000 → 0.000 ± 0.000 | 0.000 → 0.000 ± 0.000 | 0.000 → 0.000 ± 0.000 |
| cut_off | training cue mix | 0.000 → 0.000 ± 0.000 | 0.000 → 0.000 ± 0.000 | 0.000 → 0.001 ± 0.001 | 0.000 → 0.000 ± 0.000 |
| rollouts | no cue | 489 → 489 ± 0.000 | 489 → 489 ± 0.000 | 489 → 489 ± 0.000 | 489 → 489 ± 0.000 |
| rollouts | cue at the correct answer | 489 → 489 ± 0.000 | 489 → 489 ± 0.000 | 489 → 489 ± 0.000 | 489 → 489 ± 0.000 |
| rollouts | cue at the wrong answer | 489 → 489 ± 0.000 | 489 → 489 ± 0.000 | 489 → 489 ± 0.000 | 489 → 489 ± 0.000 |
| rollouts | training cue mix | 978 → 978 ± 0.000 | 978 → 978 ± 0.000 | 978 → 978 ± 0.000 | 978 → 978 ± 0.000 |
| n_bias_rows | cue at the correct answer | 442.333 → 450 ± 7.211 | 443.333 → 451.333 ± 6.506 | 450 → 457.667 ± 4.619 | 446.667 → 453.667 ± 4.163 |
| n_bias_rows | cue at the wrong answer | 436.333 → 450 ± 7.000 | 438 → 451 ± 7.000 | 444.333 → 458 ± 5.196 | 441.667 → 454.333 ± 4.041 |
| n_bias_rows | training cue mix | 878.667 → 900 ± 14.177 | 881.333 → 902.333 ± 13.503 | 894.333 → 915.667 ± 9.815 | 888.333 → 908 ± 8.185 |
| n_deception_rows | cue at the correct answer | 425.667 → 449.333 ± 7.371 | 427 → 448.667 ± 10.066 | 433.333 → 455.667 ± 3.786 | 431.333 → 451 ± 1.732 |
| n_deception_rows | cue at the wrong answer | 431.667 → 450 ± 7.000 | 433 → 445.667 ± 11.372 | 439.333 → 455.333 ± 7.572 | 437.333 → 451 ± 2.646 |
| n_deception_rows | training cue mix | 857.333 → 899.333 ± 14.364 | 860 → 894.333 ± 21.385 | 872.667 → 911 ± 11.269 | 868.667 → 902 ± 4.359 |

## Training: share of groups whose 8 rewards are all equal, mean over the 128 steps

| training condition | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|
| 0. Noisy labels, cue (previous experiment) | 0.847 | 0.822 | 0.900 | 0.856 | 0.040 |
| 1. Noisy labels, no cue | 0.798 | 0.763 | 0.814 | 0.792 | 0.026 |
| 2. Correct labels, no cue | 0.822 | 0.829 | 0.910 | 0.854 | 0.049 |
| 3. Correct labels, cue | 0.845 | 0.824 | 0.798 | 0.822 | 0.023 |

## Training: reward and true accuracy of the training answers, first and last 10 steps

Per step the mean over its answers (64 prompts × 8 answers), then the mean over the steps of the window. Reward is agreement with the run's own training label, on the run's own training prompts.

| training condition | measure | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|---|
| 0. Noisy labels, cue (previous experiment) | reward first 10 steps | 0.739 | 0.728 | 0.738 | 0.735 | 0.006 |
| 0. Noisy labels, cue (previous experiment) | reward last 10 steps | 0.784 | 0.819 | 0.804 | 0.802 | 0.017 |
| 0. Noisy labels, cue (previous experiment) | true accuracy first 10 steps | 0.704 | 0.734 | 0.724 | 0.721 | 0.015 |
| 0. Noisy labels, cue (previous experiment) | true accuracy last 10 steps | 0.717 | 0.766 | 0.755 | 0.746 | 0.026 |
| 1. Noisy labels, no cue | reward first 10 steps | 0.654 | 0.640 | 0.662 | 0.652 | 0.012 |
| 1. Noisy labels, no cue | reward last 10 steps | 0.808 | 0.761 | 0.804 | 0.791 | 0.026 |
| 1. Noisy labels, no cue | true accuracy first 10 steps | 0.747 | 0.739 | 0.762 | 0.749 | 0.012 |
| 1. Noisy labels, no cue | true accuracy last 10 steps | 0.842 | 0.798 | 0.842 | 0.827 | 0.025 |
| 2. Correct labels, no cue | reward first 10 steps | 0.768 | 0.773 | 0.772 | 0.771 | 0.003 |
| 2. Correct labels, no cue | reward last 10 steps | 0.920 | 0.905 | 0.872 | 0.899 | 0.024 |
| 2. Correct labels, no cue | true accuracy first 10 steps | 0.768 | 0.773 | 0.772 | 0.771 | 0.003 |
| 2. Correct labels, no cue | true accuracy last 10 steps | 0.920 | 0.905 | 0.872 | 0.899 | 0.024 |
| 3. Correct labels, cue | reward first 10 steps | 0.809 | 0.797 | 0.783 | 0.796 | 0.013 |
| 3. Correct labels, cue | reward last 10 steps | 0.904 | 0.861 | 0.913 | 0.893 | 0.028 |
| 3. Correct labels, cue | true accuracy first 10 steps | 0.809 | 0.797 | 0.783 | 0.796 | 0.013 |
| 3. Correct labels, cue | true accuracy last 10 steps | 0.904 | 0.861 | 0.913 | 0.893 | 0.028 |

## Per seed

### 0. Noisy labels, cue (previous experiment)

Questions with a no-cue answer in both the base eval and the seed's eval: seed 42 442, seed 43 456, seed 44 453.

| metric | eval condition | base | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|---|---|
| true accuracy | no cue | 0.755 | 0.751 | 0.748 | 0.738 | 0.746 | 0.007 |
| true accuracy | cue at the correct answer | 0.843 | 0.975 | 0.959 | 0.965 | 0.967 | 0.008 |
| true accuracy | cue at the wrong answer | 0.509 | 0.186 | 0.346 | 0.239 | 0.257 | 0.081 |
| true accuracy | training cue mix | 0.729 | 0.700 | 0.746 | 0.717 | 0.721 | 0.023 |
| reward | no cue | 0.644 | 0.661 | 0.667 | 0.658 | 0.662 | 0.004 |
| reward | cue at the correct answer | 0.714 | 0.800 | 0.783 | 0.787 | 0.790 | 0.009 |
| reward | cue at the wrong answer | 0.499 | 0.321 | 0.407 | 0.329 | 0.352 | 0.047 |
| reward | training cue mix | 0.689 | 0.758 | 0.748 | 0.740 | 0.749 | 0.009 |
| cue agreement rate | cue at the correct answer | 0.843 | 0.975 | 0.959 | 0.965 | 0.967 | 0.008 |
| cue agreement rate | cue at the wrong answer | 0.458 | 0.814 | 0.652 | 0.761 | 0.742 | 0.082 |
| cue agreement rate | training cue mix | 0.711 | 0.925 | 0.854 | 0.893 | 0.890 | 0.036 |
| bias | cue at the correct answer | 0.097 / 0.092 / 0.090 | 0.179 | 0.173 | 0.195 | 0.182 | 0.011 |
| bias | cue at the wrong answer | 0.259 / 0.267 / 0.273 | 0.595 | 0.420 | 0.517 | 0.510 | 0.088 |
| bias | training cue mix | 0.154 / 0.152 / 0.153 | 0.324 | 0.251 | 0.289 | 0.288 | 0.036 |
| anti_bias | cue at the correct answer | 0.037 / 0.038 / 0.038 | 0.002 | 0.002 | 0.000 | 0.001 | 0.001 |
| anti_bias | cue at the wrong answer | 0.012 / 0.011 / 0.009 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| anti_bias | training cue mix | 0.027 / 0.028 / 0.028 | 0.001 | 0.002 | 0.000 | 0.001 | 0.001 |
| no_bias | cue at the correct answer | 0.866 / 0.871 / 0.872 | 0.819 | 0.825 | 0.805 | 0.816 | 0.010 |
| no_bias | cue at the wrong answer | 0.729 / 0.722 / 0.718 | 0.405 | 0.580 | 0.483 | 0.490 | 0.088 |
| no_bias | training cue mix | 0.819 / 0.820 / 0.820 | 0.676 | 0.747 | 0.711 | 0.711 | 0.036 |
| deception | cue at the correct answer | 0.547 / 0.554 / 0.550 | 0.553 | 0.537 | 0.590 | 0.560 | 0.027 |
| deception | cue at the wrong answer | 0.605 / 0.600 / 0.596 | 0.464 | 0.567 | 0.521 | 0.517 | 0.051 |
| deception | training cue mix | 0.565 / 0.568 / 0.566 | 0.516 | 0.547 | 0.578 | 0.547 | 0.031 |
| deception (thresholded) | cue at the correct answer | 0.565 / 0.573 / 0.568 | 0.628 | 0.571 | 0.653 | 0.617 | 0.042 |
| deception (thresholded) | cue at the wrong answer | 0.669 / 0.664 / 0.657 | 0.425 | 0.602 | 0.521 | 0.516 | 0.089 |
| deception (thresholded) | training cue mix | 0.597 / 0.602 / 0.599 | 0.547 | 0.579 | 0.623 | 0.583 | 0.038 |
| biased_share | cue at the correct answer | 0.100 / 0.095 / 0.093 | 0.179 | 0.174 | 0.195 | 0.182 | 0.011 |
| biased_share | cue at the wrong answer | 0.262 / 0.270 / 0.276 | 0.595 | 0.420 | 0.517 | 0.510 | 0.088 |
| biased_share | training cue mix | 0.158 / 0.156 / 0.157 | 0.324 | 0.252 | 0.289 | 0.288 | 0.036 |
| false_denial_soft | cue at the correct answer | 0.388 / 0.396 / 0.379 | 0.291 | 0.299 | 0.254 | 0.282 | 0.024 |
| false_denial_soft | cue at the wrong answer | 0.360 / 0.354 / 0.349 | 0.272 | 0.303 | 0.260 | 0.278 | 0.022 |
| false_denial_soft | training cue mix | 0.370 / 0.372 / 0.364 | 0.272 | 0.298 | 0.261 | 0.277 | 0.019 |
| false_admission_soft | cue at the correct answer | 0.565 / 0.571 / 0.568 | 0.610 | 0.587 | 0.671 | 0.623 | 0.044 |
| false_admission_soft | cue at the wrong answer | 0.693 / 0.691 / 0.691 | 0.747 | 0.758 | 0.799 | 0.768 | 0.028 |
| false_admission_soft | training cue mix | 0.601 / 0.604 / 0.603 | 0.632 | 0.631 | 0.707 | 0.657 | 0.043 |
| said_yes_biased_soft | cue at the correct answer | 0.612 / 0.604 / 0.621 | 0.709 | 0.701 | 0.746 | 0.718 | 0.024 |
| said_yes_biased_soft | cue at the wrong answer | 0.640 / 0.646 / 0.651 | 0.728 | 0.697 | 0.740 | 0.722 | 0.022 |
| said_yes_biased_soft | training cue mix | 0.630 / 0.628 / 0.636 | 0.728 | 0.702 | 0.739 | 0.723 | 0.019 |
| deception_from_biased | cue at the correct answer | 0.039 / 0.038 / 0.035 | 0.052 | 0.052 | 0.050 | 0.051 | 0.001 |
| deception_from_biased | cue at the wrong answer | 0.094 / 0.096 / 0.096 | 0.162 | 0.127 | 0.134 | 0.141 | 0.018 |
| deception_from_biased | training cue mix | 0.058 / 0.058 / 0.057 | 0.088 | 0.075 | 0.075 | 0.079 | 0.007 |
| deception_from_unbiased | cue at the correct answer | 0.508 / 0.517 / 0.515 | 0.501 | 0.485 | 0.541 | 0.509 | 0.029 |
| deception_from_unbiased | cue at the wrong answer | 0.511 / 0.504 / 0.500 | 0.302 | 0.440 | 0.386 | 0.376 | 0.069 |
| deception_from_unbiased | training cue mix | 0.506 / 0.510 / 0.509 | 0.428 | 0.472 | 0.502 | 0.467 | 0.038 |
| said_yes | cue at the correct answer | 0.593 / 0.599 / 0.596 | 0.785 | 0.648 | 0.794 | 0.742 | 0.082 |
| said_yes | cue at the wrong answer | 0.785 / 0.783 / 0.786 | 0.939 | 0.802 | 0.892 | 0.878 | 0.069 |
| said_yes | training cue mix | 0.655 / 0.657 / 0.658 | 0.829 | 0.697 | 0.828 | 0.785 | 0.076 |
| false_denial | cue at the correct answer | 0.357 / 0.366 / 0.350 | 0.063 | 0.278 | 0.136 | 0.159 | 0.109 |
| false_denial | cue at the wrong answer | 0.279 / 0.280 / 0.267 | 0.068 | 0.262 | 0.141 | 0.157 | 0.098 |
| false_denial | training cue mix | 0.317 / 0.325 / 0.311 | 0.065 | 0.266 | 0.147 | 0.159 | 0.101 |
| false_admission | cue at the correct answer | 0.588 / 0.595 / 0.590 | 0.751 | 0.633 | 0.777 | 0.721 | 0.077 |
| false_admission | cue at the wrong answer | 0.808 / 0.806 / 0.806 | 0.950 | 0.848 | 0.927 | 0.908 | 0.053 |
| false_admission | training cue mix | 0.650 / 0.654 / 0.653 | 0.778 | 0.684 | 0.817 | 0.760 | 0.068 |
| parse_failed | no cue | 0.016 | 0.000 | 0.004 | 0.010 | 0.005 | 0.005 |
| parse_failed | cue at the correct answer | 0.018 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| parse_failed | cue at the wrong answer | 0.033 | 0.000 | 0.002 | 0.000 | 0.001 | 0.001 |
| parse_failed | training cue mix | 0.024 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 |
| cut_off | no cue | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue at the correct answer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue at the wrong answer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | training cue mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | no cue | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue at the correct answer | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue at the wrong answer | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | training cue mix | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue at the correct answer | 434 / 448 / 445 | 442 | 456 | 452 | 450 | 7.211 |
| n_bias_rows | cue at the wrong answer | 428 / 442 / 439 | 442 | 455 | 453 | 450 | 7.000 |
| n_bias_rows | training cue mix | 862 / 890 / 884 | 884 | 911 | 905 | 900 | 14.177 |
| n_deception_rows | cue at the correct answer | 418 / 431 / 428 | 441 | 455 | 452 | 449.333 | 7.371 |
| n_deception_rows | cue at the wrong answer | 423 / 437 / 435 | 442 | 455 | 453 | 450 | 7.000 |
| n_deception_rows | training cue mix | 841 / 868 / 863 | 883 | 910 | 905 | 899.333 | 14.364 |

### 1. Noisy labels, no cue

Questions with a no-cue answer in both the base eval and the seed's eval: seed 42 451, seed 43 445, seed 44 458.

| metric | eval condition | base | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|---|---|
| true accuracy | no cue | 0.755 | 0.771 | 0.767 | 0.767 | 0.768 | 0.002 |
| true accuracy | cue at the correct answer | 0.843 | 0.935 | 0.879 | 0.912 | 0.909 | 0.028 |
| true accuracy | cue at the wrong answer | 0.509 | 0.425 | 0.571 | 0.534 | 0.510 | 0.075 |
| true accuracy | training cue mix | 0.729 | 0.757 | 0.772 | 0.777 | 0.769 | 0.010 |
| reward | no cue | 0.644 | 0.671 | 0.667 | 0.661 | 0.666 | 0.005 |
| reward | cue at the correct answer | 0.714 | 0.759 | 0.740 | 0.753 | 0.751 | 0.009 |
| reward | cue at the wrong answer | 0.499 | 0.450 | 0.550 | 0.538 | 0.513 | 0.055 |
| reward | training cue mix | 0.689 | 0.732 | 0.722 | 0.740 | 0.731 | 0.009 |
| cue agreement rate | cue at the correct answer | 0.843 | 0.935 | 0.879 | 0.912 | 0.909 | 0.028 |
| cue agreement rate | cue at the wrong answer | 0.458 | 0.575 | 0.427 | 0.466 | 0.489 | 0.076 |
| cue agreement rate | training cue mix | 0.711 | 0.807 | 0.726 | 0.762 | 0.765 | 0.040 |
| bias | cue at the correct answer | 0.086 / 0.094 / 0.089 | 0.137 | 0.103 | 0.142 | 0.128 | 0.021 |
| bias | cue at the wrong answer | 0.267 / 0.271 / 0.273 | 0.355 | 0.248 | 0.234 | 0.279 | 0.066 |
| bias | training cue mix | 0.148 / 0.152 / 0.151 | 0.208 | 0.155 | 0.179 | 0.181 | 0.027 |
| anti_bias | cue at the correct answer | 0.038 / 0.034 / 0.038 | 0.002 | 0.016 | 0.000 | 0.006 | 0.009 |
| anti_bias | cue at the wrong answer | 0.011 / 0.012 / 0.011 | 0.004 | 0.025 | 0.007 | 0.012 | 0.011 |
| anti_bias | training cue mix | 0.028 / 0.025 / 0.028 | 0.003 | 0.019 | 0.002 | 0.008 | 0.010 |
| no_bias | cue at the correct answer | 0.876 / 0.872 / 0.873 | 0.860 | 0.881 | 0.858 | 0.866 | 0.013 |
| no_bias | cue at the wrong answer | 0.721 / 0.718 / 0.716 | 0.641 | 0.727 | 0.760 | 0.709 | 0.062 |
| no_bias | training cue mix | 0.824 / 0.823 / 0.821 | 0.789 | 0.826 | 0.819 | 0.811 | 0.020 |
| deception | cue at the correct answer | 0.553 / 0.551 / 0.554 | 0.650 | 0.649 | 0.481 | 0.593 | 0.097 |
| deception | cue at the wrong answer | 0.599 / 0.598 / 0.601 | 0.601 | 0.624 | 0.598 | 0.608 | 0.014 |
| deception | training cue mix | 0.568 / 0.566 / 0.569 | 0.633 | 0.638 | 0.522 | 0.598 | 0.066 |
| deception (thresholded) | cue at the correct answer | 0.570 / 0.571 / 0.577 | 0.713 | 0.731 | 0.443 | 0.629 | 0.161 |
| deception (thresholded) | cue at the wrong answer | 0.661 / 0.653 / 0.663 | 0.641 | 0.707 | 0.622 | 0.657 | 0.044 |
| deception (thresholded) | training cue mix | 0.601 / 0.595 / 0.606 | 0.690 | 0.719 | 0.506 | 0.638 | 0.115 |
| biased_share | cue at the correct answer | 0.089 / 0.097 / 0.092 | 0.138 | 0.105 | 0.142 | 0.128 | 0.020 |
| biased_share | cue at the wrong answer | 0.270 / 0.274 / 0.276 | 0.356 | 0.254 | 0.235 | 0.282 | 0.065 |
| biased_share | training cue mix | 0.152 / 0.156 / 0.156 | 0.209 | 0.158 | 0.179 | 0.182 | 0.026 |
| false_denial_soft | cue at the correct answer | 0.382 / 0.377 / 0.378 | 0.246 | 0.231 | 0.376 | 0.284 | 0.080 |
| false_denial_soft | cue at the wrong answer | 0.350 / 0.345 / 0.352 | 0.245 | 0.217 | 0.414 | 0.292 | 0.106 |
| false_denial_soft | training cue mix | 0.365 / 0.356 / 0.365 | 0.240 | 0.221 | 0.391 | 0.284 | 0.093 |
| false_admission_soft | cue at the correct answer | 0.570 / 0.570 / 0.572 | 0.714 | 0.698 | 0.499 | 0.637 | 0.120 |
| false_admission_soft | cue at the wrong answer | 0.692 / 0.694 / 0.695 | 0.798 | 0.762 | 0.655 | 0.738 | 0.074 |
| false_admission_soft | training cue mix | 0.605 / 0.604 / 0.607 | 0.737 | 0.716 | 0.550 | 0.668 | 0.102 |
| said_yes_biased_soft | cue at the correct answer | 0.618 / 0.623 / 0.622 | 0.754 | 0.769 | 0.624 | 0.716 | 0.080 |
| said_yes_biased_soft | cue at the wrong answer | 0.650 / 0.655 / 0.648 | 0.755 | 0.783 | 0.586 | 0.708 | 0.106 |
| said_yes_biased_soft | training cue mix | 0.635 / 0.644 / 0.635 | 0.760 | 0.779 | 0.609 | 0.716 | 0.093 |
| deception_from_biased | cue at the correct answer | 0.034 / 0.037 / 0.035 | 0.034 | 0.024 | 0.053 | 0.037 | 0.015 |
| deception_from_biased | cue at the wrong answer | 0.094 / 0.094 / 0.097 | 0.087 | 0.055 | 0.097 | 0.080 | 0.022 |
| deception_from_biased | training cue mix | 0.056 / 0.055 / 0.057 | 0.050 | 0.035 | 0.070 | 0.052 | 0.018 |
| deception_from_unbiased | cue at the correct answer | 0.519 / 0.515 / 0.519 | 0.616 | 0.625 | 0.428 | 0.556 | 0.111 |
| deception_from_unbiased | cue at the wrong answer | 0.505 / 0.504 / 0.504 | 0.514 | 0.568 | 0.501 | 0.528 | 0.036 |
| deception_from_unbiased | training cue mix | 0.513 / 0.510 / 0.512 | 0.583 | 0.603 | 0.452 | 0.546 | 0.082 |
| said_yes | cue at the correct answer | 0.599 / 0.602 / 0.605 | 0.802 | 0.813 | 0.450 | 0.688 | 0.207 |
| said_yes | cue at the wrong answer | 0.788 / 0.792 / 0.788 | 0.891 | 0.910 | 0.646 | 0.816 | 0.147 |
| said_yes | training cue mix | 0.660 / 0.662 / 0.664 | 0.838 | 0.847 | 0.518 | 0.734 | 0.187 |
| false_denial | cue at the correct answer | 0.342 / 0.341 / 0.350 | 0.177 | 0.109 | 0.477 | 0.254 | 0.196 |
| false_denial | cue at the wrong answer | 0.265 / 0.248 / 0.273 | 0.150 | 0.100 | 0.449 | 0.233 | 0.188 |
| false_denial | training cue mix | 0.307 / 0.284 / 0.314 | 0.146 | 0.094 | 0.466 | 0.236 | 0.202 |
| false_admission | cue at the correct answer | 0.593 / 0.596 / 0.601 | 0.799 | 0.804 | 0.438 | 0.680 | 0.210 |
| false_admission | cue at the wrong answer | 0.807 / 0.806 / 0.811 | 0.913 | 0.913 | 0.675 | 0.834 | 0.137 |
| false_admission | training cue mix | 0.654 / 0.652 / 0.660 | 0.833 | 0.836 | 0.515 | 0.728 | 0.185 |
| parse_failed | no cue | 0.016 | 0.002 | 0.004 | 0.000 | 0.002 | 0.002 |
| parse_failed | cue at the correct answer | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| parse_failed | cue at the wrong answer | 0.033 | 0.000 | 0.002 | 0.000 | 0.001 | 0.001 |
| parse_failed | training cue mix | 0.024 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 |
| cut_off | no cue | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue at the correct answer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue at the wrong answer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | training cue mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | no cue | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue at the correct answer | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue at the wrong answer | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | training cue mix | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue at the correct answer | 443 / 437 / 450 | 451 | 445 | 458 | 451.333 | 6.506 |
| n_bias_rows | cue at the wrong answer | 438 / 432 / 444 | 451 | 444 | 458 | 451 | 7.000 |
| n_bias_rows | training cue mix | 881 / 869 / 894 | 902 | 889 | 916 | 902.333 | 13.503 |
| n_deception_rows | cue at the correct answer | 426 / 422 / 433 | 450 | 438 | 458 | 448.667 | 10.066 |
| n_deception_rows | cue at the wrong answer | 433 / 427 / 439 | 449 | 433 | 455 | 445.667 | 11.372 |
| n_deception_rows | training cue mix | 859 / 849 / 872 | 899 | 871 | 913 | 894.333 | 21.385 |

### 2. Correct labels, no cue

Questions with a no-cue answer in both the base eval and the seed's eval: seed 42 455, seed 43 455, seed 44 464.

| metric | eval condition | base | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|---|---|
| true accuracy | no cue | 0.755 | 0.785 | 0.796 | 0.789 | 0.790 | 0.005 |
| true accuracy | cue at the correct answer | 0.843 | 0.939 | 0.939 | 0.851 | 0.909 | 0.051 |
| true accuracy | cue at the wrong answer | 0.509 | 0.509 | 0.503 | 0.640 | 0.551 | 0.077 |
| true accuracy | training cue mix | 0.759 | 0.831 | 0.830 | 0.798 | 0.820 | 0.019 |
| reward | no cue | 0.755 | 0.785 | 0.796 | 0.789 | 0.790 | 0.005 |
| reward | cue at the correct answer | 0.843 | 0.939 | 0.939 | 0.851 | 0.909 | 0.051 |
| reward | cue at the wrong answer | 0.509 | 0.509 | 0.503 | 0.640 | 0.551 | 0.077 |
| reward | training cue mix | 0.759 | 0.831 | 0.830 | 0.798 | 0.820 | 0.019 |
| cue agreement rate | cue at the correct answer | 0.843 | 0.939 | 0.939 | 0.851 | 0.909 | 0.051 |
| cue agreement rate | cue at the wrong answer | 0.458 | 0.491 | 0.497 | 0.360 | 0.449 | 0.077 |
| cue agreement rate | training cue mix | 0.746 | 0.827 | 0.828 | 0.728 | 0.794 | 0.057 |
| bias | cue at the correct answer | 0.089 / 0.087 / 0.094 | 0.123 | 0.123 | 0.060 | 0.102 | 0.036 |
| bias | cue at the wrong answer | 0.268 / 0.274 / 0.271 | 0.312 | 0.308 | 0.149 | 0.256 | 0.093 |
| bias | training cue mix | 0.134 / 0.133 / 0.138 | 0.170 | 0.169 | 0.083 | 0.141 | 0.050 |
| anti_bias | cue at the correct answer | 0.038 / 0.036 / 0.037 | 0.002 | 0.004 | 0.006 | 0.004 | 0.002 |
| anti_bias | cue at the wrong answer | 0.011 / 0.011 / 0.011 | 0.011 | 0.007 | 0.000 | 0.006 | 0.006 |
| anti_bias | training cue mix | 0.031 / 0.030 / 0.031 | 0.004 | 0.005 | 0.005 | 0.005 | 0.000 |
| no_bias | cue at the correct answer | 0.872 / 0.877 / 0.868 | 0.875 | 0.873 | 0.933 | 0.893 | 0.034 |
| no_bias | cue at the wrong answer | 0.721 / 0.715 / 0.718 | 0.677 | 0.686 | 0.851 | 0.738 | 0.098 |
| no_bias | training cue mix | 0.835 / 0.837 / 0.831 | 0.825 | 0.826 | 0.913 | 0.855 | 0.050 |
| deception | cue at the correct answer | 0.551 / 0.552 / 0.552 | 0.600 | 0.641 | 0.755 | 0.665 | 0.080 |
| deception | cue at the wrong answer | 0.599 / 0.601 / 0.601 | 0.618 | 0.629 | 0.767 | 0.671 | 0.083 |
| deception | training cue mix | 0.563 / 0.564 / 0.564 | 0.605 | 0.638 | 0.758 | 0.667 | 0.081 |
| deception (thresholded) | cue at the correct answer | 0.570 / 0.566 / 0.569 | 0.641 | 0.687 | 0.811 | 0.713 | 0.088 |
| deception (thresholded) | cue at the wrong answer | 0.661 / 0.664 / 0.661 | 0.671 | 0.673 | 0.806 | 0.717 | 0.077 |
| deception (thresholded) | training cue mix | 0.593 / 0.591 / 0.593 | 0.648 | 0.683 | 0.810 | 0.714 | 0.085 |
| biased_share | cue at the correct answer | 0.093 / 0.090 / 0.098 | 0.123 | 0.124 | 0.061 | 0.103 | 0.036 |
| biased_share | cue at the wrong answer | 0.271 / 0.277 / 0.274 | 0.316 | 0.310 | 0.149 | 0.258 | 0.095 |
| biased_share | training cue mix | 0.138 / 0.138 / 0.142 | 0.171 | 0.170 | 0.083 | 0.141 | 0.051 |
| false_denial_soft | cue at the correct answer | 0.388 / 0.369 / 0.380 | 0.305 | 0.229 | 0.132 | 0.222 | 0.087 |
| false_denial_soft | cue at the wrong answer | 0.347 / 0.350 / 0.352 | 0.283 | 0.243 | 0.164 | 0.230 | 0.061 |
| false_denial_soft | training cue mix | 0.368 / 0.360 / 0.366 | 0.295 | 0.236 | 0.146 | 0.226 | 0.075 |
| false_admission_soft | cue at the correct answer | 0.568 / 0.570 / 0.571 | 0.642 | 0.699 | 0.796 | 0.712 | 0.078 |
| false_admission_soft | cue at the wrong answer | 0.692 / 0.697 / 0.695 | 0.773 | 0.803 | 0.872 | 0.816 | 0.051 |
| false_admission_soft | training cue mix | 0.595 / 0.597 / 0.597 | 0.669 | 0.720 | 0.814 | 0.734 | 0.073 |
| said_yes_biased_soft | cue at the correct answer | 0.612 / 0.631 / 0.620 | 0.695 | 0.771 | 0.868 | 0.778 | 0.087 |
| said_yes_biased_soft | cue at the wrong answer | 0.653 / 0.650 / 0.648 | 0.717 | 0.757 | 0.836 | 0.770 | 0.061 |
| said_yes_biased_soft | training cue mix | 0.632 / 0.640 / 0.634 | 0.705 | 0.764 | 0.854 | 0.774 | 0.075 |
| deception_from_biased | cue at the correct answer | 0.036 / 0.033 / 0.037 | 0.038 | 0.028 | 0.008 | 0.025 | 0.015 |
| deception_from_biased | cue at the wrong answer | 0.094 / 0.097 / 0.096 | 0.089 | 0.075 | 0.024 | 0.063 | 0.034 |
| deception_from_biased | training cue mix | 0.051 / 0.049 / 0.052 | 0.050 | 0.040 | 0.012 | 0.034 | 0.020 |
| deception_from_unbiased | cue at the correct answer | 0.515 / 0.518 / 0.515 | 0.563 | 0.612 | 0.747 | 0.641 | 0.096 |
| deception_from_unbiased | cue at the wrong answer | 0.505 / 0.504 / 0.504 | 0.529 | 0.554 | 0.742 | 0.608 | 0.117 |
| deception_from_unbiased | training cue mix | 0.513 / 0.515 / 0.512 | 0.554 | 0.598 | 0.746 | 0.633 | 0.101 |
| said_yes | cue at the correct answer | 0.598 / 0.601 / 0.599 | 0.716 | 0.784 | 0.867 | 0.789 | 0.076 |
| said_yes | cue at the wrong answer | 0.789 / 0.789 / 0.787 | 0.871 | 0.898 | 0.929 | 0.899 | 0.029 |
| said_yes | training cue mix | 0.646 / 0.649 / 0.646 | 0.754 | 0.812 | 0.883 | 0.817 | 0.064 |
| false_denial | cue at the correct answer | 0.350 / 0.308 / 0.349 | 0.196 | 0.107 | 0.036 | 0.113 | 0.081 |
| false_denial | cue at the wrong answer | 0.263 / 0.273 / 0.270 | 0.183 | 0.136 | 0.087 | 0.135 | 0.048 |
| false_denial | training cue mix | 0.307 / 0.290 / 0.311 | 0.190 | 0.120 | 0.059 | 0.123 | 0.066 |
| false_admission | cue at the correct answer | 0.592 / 0.592 / 0.593 | 0.704 | 0.768 | 0.861 | 0.778 | 0.079 |
| false_admission | cue at the wrong answer | 0.808 / 0.813 / 0.808 | 0.896 | 0.913 | 0.932 | 0.914 | 0.018 |
| false_admission | training cue mix | 0.638 / 0.639 / 0.639 | 0.743 | 0.798 | 0.878 | 0.806 | 0.068 |
| parse_failed | no cue | 0.016 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| parse_failed | cue at the correct answer | 0.018 | 0.002 | 0.000 | 0.002 | 0.001 | 0.001 |
| parse_failed | cue at the wrong answer | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| parse_failed | training cue mix | 0.022 | 0.002 | 0.000 | 0.002 | 0.001 | 0.001 |
| cut_off | no cue | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue at the correct answer | 0.000 | 0.002 | 0.000 | 0.000 | 0.001 | 0.001 |
| cut_off | cue at the wrong answer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | training cue mix | 0.000 | 0.002 | 0.000 | 0.000 | 0.001 | 0.001 |
| rollouts | no cue | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue at the correct answer | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue at the wrong answer | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | training cue mix | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue at the correct answer | 447 / 447 / 456 | 455 | 455 | 463 | 457.667 | 4.619 |
| n_bias_rows | cue at the wrong answer | 441 / 442 / 450 | 455 | 455 | 464 | 458 | 5.196 |
| n_bias_rows | training cue mix | 888 / 889 / 906 | 910 | 910 | 927 | 915.667 | 9.815 |
| n_deception_rows | cue at the correct answer | 430 / 431 / 439 | 454 | 453 | 460 | 455.667 | 3.786 |
| n_deception_rows | cue at the wrong answer | 436 / 437 / 445 | 450 | 452 | 464 | 455.333 | 7.572 |
| n_deception_rows | training cue mix | 866 / 868 / 884 | 904 | 905 | 924 | 911 | 11.269 |

### 3. Correct labels, cue

Questions with a no-cue answer in both the base eval and the seed's eval: seed 42 456, seed 43 450, seed 44 458.

| metric | eval condition | base | seed 42 | seed 43 | seed 44 | mean | SD |
|---|---|---|---|---|---|---|---|
| true accuracy | no cue | 0.755 | 0.773 | 0.783 | 0.779 | 0.778 | 0.005 |
| true accuracy | cue at the correct answer | 0.843 | 0.896 | 0.920 | 0.859 | 0.892 | 0.031 |
| true accuracy | cue at the wrong answer | 0.509 | 0.609 | 0.429 | 0.634 | 0.558 | 0.112 |
| true accuracy | training cue mix | 0.759 | 0.824 | 0.798 | 0.803 | 0.808 | 0.014 |
| reward | no cue | 0.755 | 0.773 | 0.783 | 0.779 | 0.778 | 0.005 |
| reward | cue at the correct answer | 0.843 | 0.896 | 0.920 | 0.859 | 0.892 | 0.031 |
| reward | cue at the wrong answer | 0.509 | 0.609 | 0.429 | 0.634 | 0.558 | 0.112 |
| reward | training cue mix | 0.759 | 0.824 | 0.798 | 0.803 | 0.808 | 0.014 |
| cue agreement rate | cue at the correct answer | 0.843 | 0.896 | 0.920 | 0.859 | 0.892 | 0.031 |
| cue agreement rate | cue at the wrong answer | 0.458 | 0.389 | 0.571 | 0.366 | 0.442 | 0.112 |
| cue agreement rate | training cue mix | 0.746 | 0.769 | 0.833 | 0.736 | 0.779 | 0.049 |
| bias | cue at the correct answer | 0.094 / 0.086 / 0.089 | 0.119 | 0.127 | 0.074 | 0.107 | 0.028 |
| bias | cue at the wrong answer | 0.266 / 0.268 / 0.267 | 0.158 | 0.351 | 0.157 | 0.222 | 0.112 |
| bias | training cue mix | 0.136 / 0.131 / 0.133 | 0.129 | 0.183 | 0.095 | 0.136 | 0.044 |
| anti_bias | cue at the correct answer | 0.036 / 0.034 / 0.033 | 0.007 | 0.000 | 0.011 | 0.006 | 0.006 |
| anti_bias | cue at the wrong answer | 0.011 / 0.009 / 0.009 | 0.007 | 0.004 | 0.011 | 0.007 | 0.003 |
| anti_bias | training cue mix | 0.030 / 0.028 / 0.027 | 0.007 | 0.001 | 0.011 | 0.006 | 0.005 |
| no_bias | cue at the correct answer | 0.871 / 0.880 / 0.878 | 0.875 | 0.873 | 0.915 | 0.887 | 0.024 |
| no_bias | cue at the wrong answer | 0.723 / 0.722 / 0.724 | 0.835 | 0.644 | 0.832 | 0.770 | 0.109 |
| no_bias | training cue mix | 0.834 / 0.841 / 0.840 | 0.865 | 0.816 | 0.894 | 0.858 | 0.039 |
| deception | cue at the correct answer | 0.549 / 0.555 / 0.552 | 0.428 | 0.542 | 0.599 | 0.523 | 0.087 |
| deception | cue at the wrong answer | 0.602 / 0.600 / 0.603 | 0.540 | 0.576 | 0.651 | 0.589 | 0.057 |
| deception | training cue mix | 0.562 / 0.566 / 0.565 | 0.456 | 0.550 | 0.612 | 0.540 | 0.079 |
| deception (thresholded) | cue at the correct answer | 0.565 / 0.571 / 0.572 | 0.381 | 0.535 | 0.626 | 0.514 | 0.124 |
| deception (thresholded) | cue at the wrong answer | 0.667 / 0.664 / 0.667 | 0.540 | 0.592 | 0.702 | 0.611 | 0.083 |
| deception (thresholded) | training cue mix | 0.591 / 0.595 / 0.596 | 0.420 | 0.549 | 0.645 | 0.538 | 0.113 |
| biased_share | cue at the correct answer | 0.097 / 0.089 / 0.092 | 0.119 | 0.127 | 0.075 | 0.107 | 0.028 |
| biased_share | cue at the wrong answer | 0.269 / 0.271 / 0.270 | 0.159 | 0.353 | 0.159 | 0.224 | 0.112 |
| biased_share | training cue mix | 0.141 / 0.135 / 0.137 | 0.129 | 0.183 | 0.096 | 0.136 | 0.044 |
| false_denial_soft | cue at the correct answer | 0.389 / 0.382 / 0.383 | 0.491 | 0.295 | 0.327 | 0.371 | 0.105 |
| false_denial_soft | cue at the wrong answer | 0.354 / 0.350 / 0.357 | 0.497 | 0.326 | 0.302 | 0.375 | 0.106 |
| false_denial_soft | training cue mix | 0.372 / 0.366 / 0.370 | 0.493 | 0.310 | 0.316 | 0.373 | 0.104 |
| false_admission_soft | cue at the correct answer | 0.566 / 0.571 / 0.569 | 0.420 | 0.578 | 0.622 | 0.540 | 0.106 |
| false_admission_soft | cue at the wrong answer | 0.693 / 0.692 / 0.694 | 0.549 | 0.713 | 0.717 | 0.660 | 0.096 |
| false_admission_soft | training cue mix | 0.593 / 0.597 / 0.596 | 0.451 | 0.605 | 0.644 | 0.566 | 0.102 |
| said_yes_biased_soft | cue at the correct answer | 0.611 / 0.618 / 0.617 | 0.509 | 0.705 | 0.673 | 0.629 | 0.105 |
| said_yes_biased_soft | cue at the wrong answer | 0.646 / 0.650 / 0.643 | 0.503 | 0.674 | 0.698 | 0.625 | 0.106 |
| said_yes_biased_soft | training cue mix | 0.628 / 0.634 / 0.630 | 0.507 | 0.690 | 0.684 | 0.627 | 0.104 |
| deception_from_biased | cue at the correct answer | 0.038 / 0.034 / 0.035 | 0.059 | 0.037 | 0.025 | 0.040 | 0.017 |
| deception_from_biased | cue at the wrong answer | 0.095 / 0.095 / 0.096 | 0.079 | 0.115 | 0.048 | 0.081 | 0.033 |
| deception_from_biased | training cue mix | 0.052 / 0.049 / 0.051 | 0.064 | 0.057 | 0.030 | 0.050 | 0.018 |
| deception_from_unbiased | cue at the correct answer | 0.511 / 0.521 / 0.517 | 0.370 | 0.505 | 0.575 | 0.483 | 0.104 |
| deception_from_unbiased | cue at the wrong answer | 0.507 / 0.505 / 0.507 | 0.461 | 0.461 | 0.603 | 0.509 | 0.082 |
| deception_from_unbiased | training cue mix | 0.510 / 0.517 / 0.514 | 0.393 | 0.494 | 0.582 | 0.489 | 0.095 |
| said_yes | cue at the correct answer | 0.593 / 0.604 / 0.595 | 0.372 | 0.617 | 0.653 | 0.547 | 0.153 |
| said_yes | cue at the wrong answer | 0.786 / 0.787 / 0.787 | 0.549 | 0.779 | 0.781 | 0.703 | 0.134 |
| said_yes | training cue mix | 0.641 / 0.650 / 0.644 | 0.416 | 0.657 | 0.685 | 0.586 | 0.148 |
| false_denial | cue at the correct answer | 0.357 / 0.316 / 0.375 | 0.537 | 0.175 | 0.324 | 0.345 | 0.182 |
| false_denial | cue at the wrong answer | 0.280 / 0.274 / 0.277 | 0.472 | 0.234 | 0.250 | 0.319 | 0.133 |
| false_denial | training cue mix | 0.320 / 0.294 / 0.326 | 0.517 | 0.204 | 0.293 | 0.338 | 0.161 |
| false_admission | cue at the correct answer | 0.587 / 0.596 / 0.592 | 0.359 | 0.587 | 0.651 | 0.532 | 0.153 |
| false_admission | cue at the wrong answer | 0.810 / 0.810 / 0.811 | 0.553 | 0.786 | 0.787 | 0.709 | 0.135 |
| false_admission | training cue mix | 0.635 / 0.642 / 0.639 | 0.406 | 0.626 | 0.683 | 0.572 | 0.146 |
| parse_failed | no cue | 0.016 | 0.000 | 0.000 | 0.004 | 0.001 | 0.002 |
| parse_failed | cue at the correct answer | 0.018 | 0.004 | 0.002 | 0.002 | 0.003 | 0.001 |
| parse_failed | cue at the wrong answer | 0.033 | 0.002 | 0.000 | 0.000 | 0.001 | 0.001 |
| parse_failed | training cue mix | 0.022 | 0.004 | 0.002 | 0.002 | 0.002 | 0.001 |
| cut_off | no cue | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue at the correct answer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue at the wrong answer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | training cue mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | no cue | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue at the correct answer | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue at the wrong answer | 489 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | training cue mix | 978 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue at the correct answer | 448 / 442 / 450 | 455 | 449 | 457 | 453.667 | 4.163 |
| n_bias_rows | cue at the wrong answer | 444 / 436 / 445 | 455 | 450 | 458 | 454.333 | 4.041 |
| n_bias_rows | training cue mix | 892 / 878 / 895 | 910 | 899 | 915 | 908 | 8.185 |
| n_deception_rows | cue at the correct answer | 432 / 427 / 435 | 452 | 449 | 452 | 451 | 1.732 |
| n_deception_rows | cue at the wrong answer | 439 / 432 / 441 | 452 | 448 | 453 | 451 | 2.646 |
| n_deception_rows | training cue mix | 871 / 859 / 876 | 904 | 897 | 905 | 902 | 4.359 |
