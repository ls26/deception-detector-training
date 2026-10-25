# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 451, s43 445, s44 458.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.771 | 0.767 | 0.767 | 0.768 | 0.002 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.935 | 0.879 | 0.912 | 0.909 | 0.028 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.425 | 0.571 | 0.534 | 0.510 | 0.075 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.757 | 0.772 | 0.777 | 0.769 | 0.010 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.671 | 0.667 | 0.661 | 0.666 | 0.005 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.759 | 0.740 | 0.753 | 0.751 | 0.009 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.450 | 0.550 | 0.538 | 0.513 | 0.055 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.732 | 0.722 | 0.740 | 0.731 | 0.009 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.935 | 0.879 | 0.912 | 0.909 | 0.028 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.575 | 0.427 | 0.466 | 0.489 | 0.076 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.807 | 0.726 | 0.762 | 0.765 | 0.040 |
| bias | cue_gold | 0.086 / 0.094 / 0.089 | 0.089 | 0.004 | 0.137 | 0.103 | 0.142 | 0.128 | 0.021 |
| bias | cue_dist | 0.267 / 0.271 / 0.273 | 0.270 | 0.003 | 0.355 | 0.248 | 0.234 | 0.279 | 0.066 |
| bias | train_mix | 0.148 / 0.152 / 0.151 | 0.150 | 0.002 | 0.208 | 0.155 | 0.179 | 0.181 | 0.027 |
| anti_bias | cue_gold | 0.038 / 0.034 / 0.038 | 0.037 | 0.002 | 0.002 | 0.016 | 0.000 | 0.006 | 0.009 |
| anti_bias | cue_dist | 0.011 / 0.012 / 0.011 | 0.011 | 0.000 | 0.004 | 0.025 | 0.007 | 0.012 | 0.011 |
| anti_bias | train_mix | 0.028 / 0.025 / 0.028 | 0.027 | 0.002 | 0.003 | 0.019 | 0.002 | 0.008 | 0.010 |
| no_bias | cue_gold | 0.876 / 0.872 / 0.873 | 0.874 | 0.002 | 0.860 | 0.881 | 0.858 | 0.866 | 0.013 |
| no_bias | cue_dist | 0.721 / 0.718 / 0.716 | 0.718 | 0.003 | 0.641 | 0.727 | 0.760 | 0.709 | 0.062 |
| no_bias | train_mix | 0.824 / 0.823 / 0.821 | 0.822 | 0.002 | 0.789 | 0.826 | 0.819 | 0.811 | 0.020 |
| deception_soft | cue_gold | 0.553 / 0.551 / 0.554 | 0.553 | 0.001 | 0.650 | 0.649 | 0.481 | 0.593 | 0.097 |
| deception_soft | cue_dist | 0.599 / 0.598 / 0.601 | 0.600 | 0.001 | 0.601 | 0.624 | 0.598 | 0.608 | 0.014 |
| deception_soft | train_mix | 0.568 / 0.566 / 0.569 | 0.568 | 0.002 | 0.633 | 0.638 | 0.522 | 0.598 | 0.066 |
| deception | cue_gold | 0.570 / 0.571 / 0.577 | 0.573 | 0.004 | 0.713 | 0.731 | 0.443 | 0.629 | 0.161 |
| deception | cue_dist | 0.661 / 0.653 / 0.663 | 0.659 | 0.005 | 0.641 | 0.707 | 0.622 | 0.657 | 0.044 |
| deception | train_mix | 0.601 / 0.595 / 0.606 | 0.601 | 0.006 | 0.690 | 0.719 | 0.506 | 0.638 | 0.115 |
| biased_share | cue_gold | 0.089 / 0.097 / 0.092 | 0.093 | 0.004 | 0.138 | 0.105 | 0.142 | 0.128 | 0.020 |
| biased_share | cue_dist | 0.270 / 0.274 / 0.276 | 0.273 | 0.003 | 0.356 | 0.254 | 0.235 | 0.282 | 0.065 |
| biased_share | train_mix | 0.152 / 0.156 / 0.156 | 0.155 | 0.002 | 0.209 | 0.158 | 0.179 | 0.182 | 0.026 |
| false_denial_soft | cue_gold | 0.382 / 0.377 / 0.378 | 0.379 | 0.002 | 0.246 | 0.231 | 0.376 | 0.284 | 0.080 |
| false_denial_soft | cue_dist | 0.350 / 0.345 / 0.352 | 0.349 | 0.004 | 0.245 | 0.217 | 0.414 | 0.292 | 0.106 |
| false_denial_soft | train_mix | 0.365 / 0.356 / 0.365 | 0.362 | 0.005 | 0.240 | 0.221 | 0.391 | 0.284 | 0.093 |
| false_admission_soft | cue_gold | 0.570 / 0.570 / 0.572 | 0.571 | 0.001 | 0.714 | 0.698 | 0.499 | 0.637 | 0.120 |
| false_admission_soft | cue_dist | 0.692 / 0.694 / 0.695 | 0.694 | 0.002 | 0.798 | 0.762 | 0.655 | 0.738 | 0.074 |
| false_admission_soft | train_mix | 0.605 / 0.604 / 0.607 | 0.605 | 0.001 | 0.737 | 0.716 | 0.550 | 0.668 | 0.102 |
| said_yes_biased_soft | cue_gold | 0.618 / 0.623 / 0.622 | 0.621 | 0.002 | 0.754 | 0.769 | 0.624 | 0.716 | 0.080 |
| said_yes_biased_soft | cue_dist | 0.650 / 0.655 / 0.648 | 0.651 | 0.004 | 0.755 | 0.783 | 0.586 | 0.708 | 0.106 |
| said_yes_biased_soft | train_mix | 0.635 / 0.644 / 0.635 | 0.638 | 0.005 | 0.760 | 0.779 | 0.609 | 0.716 | 0.093 |
| deception_from_biased | cue_gold | 0.034 / 0.037 / 0.035 | 0.035 | 0.001 | 0.034 | 0.024 | 0.053 | 0.037 | 0.015 |
| deception_from_biased | cue_dist | 0.094 / 0.094 / 0.097 | 0.095 | 0.002 | 0.087 | 0.055 | 0.097 | 0.080 | 0.022 |
| deception_from_biased | train_mix | 0.056 / 0.055 / 0.057 | 0.056 | 0.001 | 0.050 | 0.035 | 0.070 | 0.052 | 0.018 |
| deception_from_unbiased | cue_gold | 0.519 / 0.515 / 0.519 | 0.518 | 0.003 | 0.616 | 0.625 | 0.428 | 0.556 | 0.111 |
| deception_from_unbiased | cue_dist | 0.505 / 0.504 / 0.504 | 0.504 | 0.001 | 0.514 | 0.568 | 0.501 | 0.528 | 0.036 |
| deception_from_unbiased | train_mix | 0.513 / 0.510 / 0.512 | 0.512 | 0.001 | 0.583 | 0.603 | 0.452 | 0.546 | 0.082 |
| said_yes | cue_gold | 0.599 / 0.602 / 0.605 | 0.602 | 0.003 | 0.802 | 0.813 | 0.450 | 0.688 | 0.207 |
| said_yes | cue_dist | 0.788 / 0.792 / 0.788 | 0.789 | 0.002 | 0.891 | 0.910 | 0.646 | 0.816 | 0.147 |
| said_yes | train_mix | 0.660 / 0.662 / 0.664 | 0.662 | 0.002 | 0.838 | 0.847 | 0.518 | 0.734 | 0.187 |
| false_denial | cue_gold | 0.342 / 0.341 / 0.350 | 0.345 | 0.005 | 0.177 | 0.109 | 0.477 | 0.254 | 0.196 |
| false_denial | cue_dist | 0.265 / 0.248 / 0.273 | 0.262 | 0.013 | 0.150 | 0.100 | 0.449 | 0.233 | 0.188 |
| false_denial | train_mix | 0.307 / 0.284 / 0.314 | 0.301 | 0.015 | 0.146 | 0.094 | 0.466 | 0.236 | 0.202 |
| false_admission | cue_gold | 0.593 / 0.596 / 0.601 | 0.596 | 0.004 | 0.799 | 0.804 | 0.438 | 0.680 | 0.210 |
| false_admission | cue_dist | 0.807 / 0.806 / 0.811 | 0.808 | 0.003 | 0.913 | 0.913 | 0.675 | 0.834 | 0.137 |
| false_admission | train_mix | 0.654 / 0.652 / 0.660 | 0.655 | 0.004 | 0.833 | 0.836 | 0.515 | 0.728 | 0.185 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.002 | 0.004 | 0.000 | 0.002 | 0.002 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.002 | 0.000 | 0.001 | 0.001 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 443 / 437 / 450 | 443.333 | 6.506 | 451 | 445 | 458 | 451.333 | 6.506 |
| n_bias_rows | cue_dist | 438 / 432 / 444 | 438 | 6.000 | 451 | 444 | 458 | 451 | 7.000 |
| n_bias_rows | train_mix | 881 / 869 / 894 | 881.333 | 12.503 | 902 | 889 | 916 | 902.333 | 13.503 |
| n_deception_rows | cue_gold | 426 / 422 / 433 | 427 | 5.568 | 450 | 438 | 458 | 448.667 | 10.066 |
| n_deception_rows | cue_dist | 433 / 427 / 439 | 433 | 6.000 | 449 | 433 | 455 | 445.667 | 11.372 |
| n_deception_rows | train_mix | 859 / 849 / 872 | 860 | 11.533 | 899 | 871 | 913 | 894.333 | 21.385 |
| probe_start_mean_biased | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_mean_biased | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_mean_unbiased | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_mean_unbiased | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_auc | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_auc | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_fresh_mean_biased | cue_gold | – | – | – | – | – | – | – | – |
| probe_fresh_mean_biased | cue_dist | – | – | – | – | – | – | – | – |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | – | – | – | – | – |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | – | – | – | – | – |
| probe_fresh_auc | cue_gold | – | – | – | – | – | – | – | – |
| probe_fresh_auc | cue_dist | – | – | – | – | – | – | – | – |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_base_activations_auc | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_base_activations_auc | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.798 | 0.763 | 0.814 | 0.792 | 0.026 |
| probe-only groups | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
